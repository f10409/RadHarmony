#!/usr/bin/env python
"""Zero-shot MedPMC specialty/modality tags using MedPMC-CLIP. The tags come from pixels.

Complements the caption lexicon (scripts/build_medpmc_tags.py). The lexicon is
fast and easy to audit, but it fails when a caption is a bare fragment ("(e) Axial
section") or has no modality. This is about a third of the corpus. This
tagger looks at the image, so it can tag those cases.

This is the authors' own method: they used MedPMC-CLIP zero-shot to pull the skin
photograph subset for their dermatology analysis.

Writes image_id, specialty, modality, clip_prob, clip_margin, tag_method to a
parquet with the same schema as the lexicon tagger, so the harmonizer can
read either. `clip_prob` is the softmax over class prompts. `clip_margin`
is top1 minus top2. A low margin means the model cannot choose between two
classes, so treat the tag as weak evidence.

Run one process per GPU. The --rank and --world flags split the shards.
Usage:
    python scripts/build_medpmc_tags_clip.py --base-dir <D> --rank 0 --world 6 --device cuda:0
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import tarfile

import numpy as np
import pandas as pd
import torch
from PIL import Image

TAG_METHOD = "medpmc-clip-zeroshot-v1"
REPO = "Yale-BIDS-Chen/medpmc-clip-l-14_jun24_v1"

# Use several prompts per class and average them. This is common CLIP practice and is more stable than one prompt.
PROMPTS: dict[tuple[str, str | None], list[str]] = {
    ("radiology", "ct"):            ["a computed tomography scan", "a CT image of the body", "an axial CT slice"],
    ("radiology", "mri"):           ["a magnetic resonance imaging scan", "an MRI image", "a T2-weighted MRI slice"],
    ("radiology", "xray"):          ["a chest radiograph", "an x-ray image", "a plain film radiograph"],
    ("radiology", "ultrasound"):    ["an ultrasound image", "a sonogram", "an echocardiogram"],
    ("radiology", "pet_spect"):     ["a PET scan", "a nuclear medicine scintigraphy image"],
    ("radiology", "angiography"):   ["an angiogram", "a catheter angiography image"],
    ("radiology", "mammography"):   ["a mammogram"],
    ("pathology", None):            ["a histopathology image", "an H&E stained tissue section",
                                     "a microscopic image of stained tissue"],
    ("microscopy_nonpathology", None): ["a fluorescence microscopy image", "an electron micrograph",
                                        "a confocal microscopy image of cells"],
    ("ophthalmology", None):        ["a fundus photograph of the retina", "an optical coherence tomography scan of the eye"],
    ("endoscopy", None):            ["an endoscopic image", "a colonoscopy image"],
    ("dermatology", None):          ["a dermoscopy image of a skin lesion", "a photograph of a skin rash"],
    ("clinical_photography", None): ["a clinical photograph of a patient", "an intraoperative photograph of surgery",
                                     "a photograph of a resected specimen"],
    ("physiological_signals", None): ["an electrocardiogram tracing", "an EEG waveform recording"],
    ("non_medical", None):          ["a bar chart", "a flow diagram", "a scatter plot", "a line graph"],
}


def _load_model(device):
    import open_clip
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=None)
    # The authors' inference_example.py names open_clip_pytorch_model.safetensors,
    # but the repo actually ships open_clip_model.safetensors. Try both.
    last = None
    for fn in ("open_clip_model.safetensors", "open_clip_pytorch_model.safetensors"):
        try:
            ckpt = hf_hub_download(REPO, fn)
            model.load_state_dict(load_file(ckpt), strict=True)
            last = None
            break
        except Exception as e:
            last = e
    if last is not None:
        raise RuntimeError(f"could not load MedPMC-CLIP weights: {last}")
    model = model.to(device).eval()
    tok = open_clip.get_tokenizer("ViT-L-14")

    keys = list(PROMPTS)
    with torch.no_grad():
        embs = []
        for k in keys:
            t = tok(PROMPTS[k]).to(device)
            f = model.encode_text(t)
            f = f / f.norm(dim=-1, keepdim=True)
            embs.append(f.mean(0) / f.mean(0).norm())     # average the prompt ensemble
        text = torch.stack(embs)
    return model, preprocess, text, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--shards", type=int, default=0)
    ap.add_argument("--subsets", nargs="*", default=None,
                    help="restrict to multipanel/singlepanel/subfigure. Zero-shot "
                         "tagging of MULTIPANEL composites is ill-posed — a 9-panel "
                         "figure has no single class, which is why the paper excluded "
                         "compound figures from its own composition audit.")
    a = ap.parse_args()

    subs = a.subsets or ["multipanel", "singlepanel", "subfigure"]
    tars = sorted(t for s in subs
                  for t in glob.glob(os.path.join(a.base_dir, "webdataset", s, "*.tar")))
    if a.shards:
        tars = tars[: a.shards]
    tars = tars[a.rank :: a.world]
    out_dir = os.path.join(a.base_dir, "tags", "_clip_parts")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[rank {a.rank}] {len(tars)} shards on {a.device}", flush=True)

    model, preprocess, text, keys = _load_model(a.device)

    for ti, tar_path in enumerate(tars, 1):
        stem = os.path.relpath(tar_path, a.base_dir).replace(os.sep, "_")
        out = os.path.join(out_dir, f"{stem}.parquet")
        if os.path.isfile(out):
            continue
        ids, keys_b, batch, rows = [], [], [], []

        def flush(ids=ids, keys_b=keys_b, batch=batch, rows=rows):
            if not batch:
                return
            x = torch.stack(batch).to(a.device)
            with torch.no_grad():
                f = model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
                p = (100.0 * f @ text.T).softmax(dim=-1).float().cpu().numpy()
            top = p.argmax(1)
            srt = np.sort(p, axis=1)
            for i, k in enumerate(top):
                spec, mod = keys[k]
                rows.append({"key": keys_b[i], "image_id": ids[i],
                             "specialty": spec, "modality": mod,
                             "clip_prob": float(p[i, k]),
                             "clip_margin": float(srt[i, -1] - srt[i, -2]),
                             "tag_method": TAG_METHOD})
            ids.clear(); keys_b.clear(); batch.clear()

        try:
            with tarfile.open(tar_path, "r|*") as tf:
                pending = {}
                for m in tf:
                    fh = tf.extractfile(m)
                    if fh is None:
                        continue
                    stem_m = os.path.splitext(os.path.basename(m.name))[0]
                    if m.name.endswith(".json"):
                        pending[stem_m] = json.load(fh).get("image_id")
                        continue
                    try:
                        im = Image.open(io.BytesIO(fh.read()))
                        # Decode at a smaller scale where the format allows (JPEG DCT
                        # scaling). The transform resizes to 224 anyway.
                        try:
                            im.draft("RGB", (256, 256))
                        except Exception:
                            pass
                        img = im.convert("RGB")
                    except Exception:
                        continue
                    # The json for a key can arrive after its image. Then use the key.
                    ids.append(pending.get(stem_m, stem_m))
                    keys_b.append(stem_m)          # tar member stem == the index `key`
                    batch.append(preprocess(img))
                    if len(batch) >= a.batch_size:
                        flush()
                flush()
        except Exception as e:
            print(f"  [warn] {tar_path}: {type(e).__name__}: {e}", flush=True)
            continue
        pd.DataFrame(rows).to_parquet(out, index=False)
        if ti % 5 == 0 or ti == len(tars):
            print(f"[rank {a.rank}] {ti}/{len(tars)} shards", flush=True)
    print(f"[rank {a.rank}] DONE", flush=True)


if __name__ == "__main__":
    main()
