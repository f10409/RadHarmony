#!/usr/bin/env python
"""Build the MedPMC specialty/modality tag table.

MedPMC ships no specialty field, so RadHarmony derives one. This reads the caption
JSON out of every webdataset shard and writes one parquet keyed by ``image_id``:

    image_id, caption, specialty, modality, tag_confidence, tag_evidence,
    tag_source, tag_method

`tag_method` is stamped on every row so a downstream reader can always tell these
tags apart from anything the MedPMC authors published. They published none.

Cost note: a tar must be read from the start to reach its members. So extracting
captions needs a full read of the corpus (~695 GB), but we keep only the
JSON. This cost happens once. Shards run in parallel and the result is cached. Per-shard outputs are written first so an interrupted run resumes.

Usage:
    python scripts/build_medpmc_tags.py --base-dir /data/MedPMC
    python scripts/build_medpmc_tags.py --base-dir ... --shards 4   # smoke test
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radharmony.harmonizer.medpmc.tagging import TAG_METHOD, tag_caption  # noqa: E402


def _one_shard(args) -> str:
    tar_path, out_dir = args
    stem = tar_path.replace(os.sep, "_").strip("_")
    out = os.path.join(out_dir, f"{stem}.parquet")
    if os.path.isfile(out):
        return out
    rows = []
    try:
        with tarfile.open(tar_path, "r|*") as tf:      # streaming mode, no seeking
            for m in tf:
                if not m.name.endswith(".json"):
                    continue
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                try:
                    j = json.load(fh)
                except Exception:
                    continue
                cap = j.get("caption")
                t = tag_caption(cap, j.get("parent_caption"))
                rows.append({"image_id": j.get("image_id"), "caption": cap,
                             "specialty": t["specialty"], "modality": t["modality"],
                             "tag_confidence": t["tag_confidence"],
                             "tag_evidence": t["tag_evidence"],
                             "tag_source": t.get("tag_source"),
                             "tag_method": TAG_METHOD})
    except Exception as e:                              # a bad shard must not kill the run
        print(f"  [warn] {tar_path}: {type(e).__name__}: {e}", flush=True)
        return ""
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shards", type=int, default=0, help="cap shard count (smoke test)")
    a = ap.parse_args()

    wds = os.path.join(a.base_dir, "webdataset")
    tars = sorted(glob.glob(os.path.join(wds, "*", "*.tar")))
    if a.shards:
        tars = tars[: a.shards]
    if not tars:
        raise SystemExit(f"no .tar shards under {wds} — has the download finished?")
    part_dir = os.path.join(a.base_dir, "tags", "_parts")
    os.makedirs(part_dir, exist_ok=True)
    print(f"tagging {len(tars)} shards with {a.workers} workers", flush=True)

    done = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_one_shard, (t, part_dir)): t for t in tars}
        for i, f in enumerate(as_completed(futs), 1):
            p = f.result()
            if p:
                done.append(p)
            if i % 10 == 0 or i == len(tars):
                print(f"  {i}/{len(tars)} shards", flush=True)

    df = pd.concat([pd.read_parquet(p) for p in done], ignore_index=True)
    df = df.drop_duplicates("image_id")
    out = os.path.join(a.base_dir, "tags", "medpmc_tags.parquet")
    df.to_parquet(out, index=False)

    print(f"\nwrote {out}  rows={len(df)}")
    print("\nspecialty distribution (compare against the paper's manual audit:")
    print("  radiology 25.6 · pathology 26.6 · microscopy-nonpath 31.9 · clinical photo 6.2")
    print("  ophthalmology 2.3 · endoscopy 1.4 · dermatology 1.0 · signals 0.3 · non-medical 4.7)")
    print((df["specialty"].value_counts(normalize=True) * 100).round(1).to_string())
    rad = df[df["specialty"] == "radiology"]
    if len(rad):
        print("\nradiology modality split (paper: MRI 31.7 · CT 23.4 · X-ray 15.1 · US 10.3"
              " · PET/SPECT 7.5 · fMRI 5.6 · angio 4.8 · mammo 0.7):")
        print((rad["modality"].value_counts(normalize=True) * 100).round(1).to_string())


if __name__ == "__main__":
    main()
