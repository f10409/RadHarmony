#!/usr/bin/env python
"""Per-image accuracy for the MedPMC caption tagger, scored on ROCOv2.

WHY. Tuning the tagger by matching MedPMC's published aggregate class proportions
licenses nothing about per-image correctness: two opposite errors cancel and the
histogram still matches. ROCOv2 gives what that metric cannot: per-image labels.

WHY THIS IS NOT CIRCULAR. The labels are ROCOv2's MANUALLY curated UMLS concepts
(test_concepts_manual.csv), and CUI codes are resolved to concept names through
Zenodo's cui_mapping.csv. Neither comes from our caption lexicon. Inferring "which
CUI means CT" from the captions that say "CT" WOULD be circular.
This script avoids that shortcut.

SCOPE LIMITS, stated because they bound every number below:
  * ROCOv2 is radiology-ONLY. There are no negatives, so this measures modality
    accuracy and radiology recall. It CANNOT measure whether the tagger wrongly
    calls microscopy "radiology". We suspect the error is there.
  * ROCOv2 images are PMC figures like MedPMC's, but not the same corpus, so this
    is a near-domain estimate rather than an in-corpus one.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_paths import paths  # noqa: E402
from radharmony.harmonizer.medpmc.tagging import TAG_METHOD, tag_caption  # noqa: E402

# CUI -> our modality taxonomy. Each mapping follows the canonical name in
# Zenodo's cui_mapping.csv, never what a caption says.
CUI2MOD = {
    "C0040405": "ct",           # X-Ray Computed Tomography
    "C0729619": "ct",           # Computed tomography imaging - action
    "C1306645": "xray",         # Plain x-ray
    "C1962945": "xray",         # Radiographic imaging procedure
    "C0024485": "mri",          # Magnetic Resonance Imaging
    "C0994163": "mri",          # Cholangiopancreatography, Magnetic Resonance
    "C0041618": "ultrasound",   # Ultrasonography
    "C0002978": "angiography",  # angiogram
    "C0243032": "angiography",  # Magnetic Resonance Angiography
    "C0032743": "pet_spect",    # Positron-Emission Tomography
    "C0040399": "pet_spect",    # Tomography, Emission-Computed, Single-Photon
    "C3472245": "pet_spect",    # SPECT-CT
    "C0024671": "mammography",  # Mammography
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roco-dir", default=paths("rocov2", missing_ok=True).get("base_dir"),
                    help="The ROCOv2 folder. Default: the data paths file.")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    man = pd.read_csv(f"{a.roco_dir}/{a.split}_concepts_manual.csv")
    man.columns = ["image_id", "cuis"]
    def truth(s):
        mods = {CUI2MOD[c.strip()] for c in str(s).split(";") if c.strip() in CUI2MOD}
        return mods.pop() if len(mods) == 1 else None      # skip ambiguous multi-modality
    man["true_modality"] = man["cuis"].map(truth)

    caps = []
    for f in sorted(glob.glob(f"{a.roco_dir}/hf/data/{a.split}-*.parquet")):
        caps.append(pd.read_parquet(f, columns=["image_id", "caption"]))
    caps = pd.concat(caps, ignore_index=True)

    d = caps.merge(man[["image_id", "true_modality"]], on="image_id", how="inner")
    d = d[d["true_modality"].notna()].reset_index(drop=True)
    print(f"scored on {len(d):,} ROCOv2 {a.split} images with an unambiguous manual modality")
    print(f"tagger under test: {TAG_METHOD}\n")

    pred = d["caption"].map(lambda c: tag_caption(c))
    d["pred_specialty"] = [p["specialty"] for p in pred]
    d["pred_modality"] = [p["modality"] for p in pred]
    d["conf"] = [p["tag_confidence"] for p in pred]

    rec = (d["pred_specialty"] == "radiology").mean()
    print(f"RADIOLOGY RECALL (every image here IS radiology): {rec*100:.1f}%")
    print(f"  missed entirely (specialty=unknown): {(d['pred_specialty']=='unknown').mean()*100:.1f}%")
    wrong = d[~d["pred_specialty"].isin(["radiology", "unknown"])]
    if len(wrong):
        print(f"  called something else ({len(wrong)/len(d)*100:.1f}%): "
              f"{wrong['pred_specialty'].value_counts().head(4).to_dict()}")

    # Every scored image counts. No tag, or 'radiology_other', is a miss (FN).
    print("\nPER-MODALITY (every scored image, a missing or other tag is a miss):")
    print(f"{'modality':<14}{'support':>9}{'precision':>11}{'recall':>9}{'F1':>8}")
    f1s = []
    for m in sorted(set(d["true_modality"])):
        tp = ((d["pred_modality"] == m) & (d["true_modality"] == m)).sum()
        fp = ((d["pred_modality"] == m) & (d["true_modality"] != m)).sum()
        fn = ((d["pred_modality"] != m) & (d["true_modality"] == m)).sum()
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / (tp + fn) if tp + fn else float("nan")
        # Each m has support (tp + fn >= 1), so F1 is defined. No hit gives 0.
        f1 = 2 * tp / (2 * tp + fp + fn)
        f1s.append(f1)
        print(f"{m:<14}{int((d['true_modality']==m).sum()):>9}{p:>11.3f}{r:>9.3f}{f1:>8.3f}")
    print(f"\nmacro-F1 over modalities: {sum(f1s)/len(f1s):.3f}")

    print("\nCONFUSION (true -> most common prediction):")
    cm = pd.crosstab(d["true_modality"], d["pred_modality"].fillna("(none)"))
    print(cm.to_string())
    out = f"{a.roco_dir}/tagger_eval_{a.split}.csv"
    d.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
