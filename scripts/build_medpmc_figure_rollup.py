#!/usr/bin/env python
"""Roll subfigure tags up to their parent multipanel figure.

WHY THIS IS A JOIN AND NOT A MODEL. The 3,156,144 multipanel figures have no CLIP
tag. A 9-panel composite is called "non-medical" 47.6% of the time. The question
"what modality is this figure?" has no clear answer when it holds three.
Measured on 2026-08-22: 1,788,906 of them already have panels extracted as
`subfigure` rows. Every subfigure has a tag (4.1 panels per parent on average).
So a figure-level label for most figures is a groupby over labels that exist.

The remaining 1,367,238 figures have no panels at all. They are reported here and
left alone -- decomposing them is a separate project (layout detection, panel
extraction, QC), not something this script guesses at.

Output: <base_dir>/tags/medpmc_figure_rollup.parquet, one row per decomposed
multipanel figure:

    image_id          the parent figure
    n_panels          how many panels it was cut into
    n_radiology       how many of those are radiology
    frac_radiology    n_radiology / n_panels
    figure_verdict    all_radiology | mixed | no_radiology
    modalities        ";"-joined distinct modalities across radiology panels
    n_modalities      how many distinct modalities appear
    top_modality      most common modality among radiology panels
    specialties       ";"-joined distinct panel specialties

The verdict inherits panel-level tagger error (measured elsewhere: 75.2% precision
for the caption lexicon, better for CLIP). A figure called all_radiology is only as
trustworthy as its panels.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_paths import paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=paths("medpmc", missing_ok=True).get("base_dir"),
                help="The MedPMC folder. Default: the data paths file.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    M = a.base_dir

    idx = pd.read_parquet(f"{M}/webdataset/metadata/medpmc_index.parquet",
                          columns=["source_type", "image_id", "parent_image_id", "key"])
    clip = pd.read_parquet(f"{M}/tags/medpmc_tags_clip.parquet",
                           columns=["key", "specialty", "modality"])

    sub = idx[idx.source_type == "subfigure"].merge(clip, on="key", how="left")
    sub = sub[sub.parent_image_id.notna()]
    mp = idx[idx.source_type == "multipanel"]
    print(f"multipanel figures : {len(mp):,}", flush=True)
    print(f"tagged subfigures  : {len(sub):,}", flush=True)

    sub["is_rad"] = sub.specialty.eq("radiology")
    g = sub.groupby("parent_image_id")

    def _join(s):
        v = sorted({x for x in s.dropna().unique()})
        return ";".join(v) if v else None

    out = pd.DataFrame({
        "n_panels":    g.size(),
        "n_radiology": g.is_rad.sum(),
        "specialties": g.specialty.agg(_join),
    })
    rad = sub[sub.is_rad]
    rg = rad.groupby("parent_image_id")
    out["modalities"]   = rg.modality.agg(_join)
    out["n_modalities"] = rg.modality.nunique()
    # mode() is empty when every value is NaN, so check for that.
    out["top_modality"] = rg.modality.agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) else None)

    out["frac_radiology"] = out.n_radiology / out.n_panels
    out["figure_verdict"] = pd.cut(
        out.frac_radiology, bins=[-0.01, 0.0, 0.999, 1.0],
        labels=["no_radiology", "mixed", "all_radiology"]).astype(str)
    out["n_modalities"] = out.n_modalities.fillna(0).astype(int)
    out = out.reset_index().rename(columns={"parent_image_id": "image_id"})

    p = a.out or f"{M}/tags/medpmc_figure_rollup.parquet"
    out.to_parquet(p, index=False)

    print(f"\nwrote {p}  n={len(out):,}")
    print(out.figure_verdict.value_counts().to_string())
    print(f"\nmultipanel figures with NO panels extracted: "
          f"{(~mp.image_id.isin(set(out.image_id))).sum():,}  (left alone -- Phase 2)")
    print("\ntop modality among all_radiology figures:")
    print(out[out.figure_verdict == "all_radiology"].top_modality
          .value_counts().head(8).to_string())
    print("\npanels per figure:", round(out.n_panels.mean(), 1),
          "| figures with >1 distinct modality:",
          f"{(out.n_modalities > 1).sum():,}")


if __name__ == "__main__":
    main()
