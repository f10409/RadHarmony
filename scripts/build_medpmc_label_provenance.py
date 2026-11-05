#!/usr/bin/env python
"""Resolve one modality/specialty per MedPMC image, and record WHERE it came from.

Four sources disagree with each other. Before, the harmonizer picked one without
a note. This script shows the disagreement. It keeps every source in its own
column and picks a winner by a fixed precedence. It records the winning source
next to the value.

PRECEDENCE (Judy, 2026-08-22): human > clip > figure_rollup > caption.

Earned rather than assumed. On the 151 tagger conflicts a clinician adjudicated the
same day, CLIP named the right modality 72.8% of the time and the caption lexicon
10.6%. The clinician's own 872 judgements outrank both -- they are the only labels
in the corpus produced by a person looking at the image.

WHAT "human_verified" MEANS HERE: strict. It is True only when a person judged
that exact image. A figure's panels do not inherit it, and a panel's parent does
not either. A clinician who checked a composite did not check its six panels.
Passing the flag on would claim a check that never happened.

Output: <base_dir>/tags/medpmc_labels_resolved.parquet

    image_id
    modality_caption / modality_clip / modality_human   per-source, canonical keys
    modality_figure_rollup                               top panel modality (multipanel)
    modality_final                                       winner
    modality_source                                      which source won
    modality_display                                     clinical code (MR, CT, XR)
    specialty_caption / specialty_clip / specialty_human
    specialty_final / specialty_source
    human_verified          bool, strict
    human_annotator         initials, when verified
    sources_disagree        bool, for auditing
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
from radharmony.harmonizer.medpmc.vocab import (  # noqa: E402
    normalize, display, SOURCE_PRECEDENCE,
)


def _human(review_dir: str) -> pd.DataFrame:
    """Clinician labels from the blind labeling app.

    Modality is stored joined with ";" because a composite can have several.
    A single clear modality is the verified value. A multi-modality answer is
    recorded but left out of modality_human. "This figure contains CT and MR"
    is not one modality.
    """
    rows = []
    for f in sorted(glob.glob(f"{review_dir}/labels/*__*.parquet")):
        d = pd.read_parquet(f)
        if "image_id" not in d.columns:
            continue
        rows.append(d)
    if not rows:
        return pd.DataFrame(columns=["image_id", "modality_human", "specialty_human",
                                     "human_annotator", "n_human_modalities"])
    h = pd.concat(rows, ignore_index=True)
    h = h.sort_values("ts").drop_duplicates("image_id", keep="last")

    def _mod(v):
        parts = [p for p in str(v).split(";") if p and p != "None"]
        return normalize(parts[0]) if len(parts) == 1 else None

    h["n_human_modalities"] = h.modality.fillna("").apply(
        lambda v: len([p for p in str(v).split(";") if p and p != "None"]))
    h["modality_human"] = h.modality.apply(_mod)
    # "mixed" means the figure holds radiology and non-radiology.
    # It is a real verdict, not a specialty, so keep it as is.
    h["specialty_human"] = h.is_radiology.map(
        {"radiology": "radiology", "not_radiology": "non_radiology",
         "mixed": "mixed", "unclear": None})
    return h[["image_id", "modality_human", "specialty_human", "annotator",
              "n_human_modalities"]].rename(columns={"annotator": "human_annotator"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=paths("medpmc", missing_ok=True).get("base_dir"),
                help="The MedPMC folder. Default: the data paths file.")
    ap.add_argument("--review-dir", required=True, help="the review app folder, with labels/*__*.parquet")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    M = a.base_dir

    idx = pd.read_parquet(f"{M}/webdataset/metadata/medpmc_index.parquet",
                          columns=["image_id", "key", "source_type"])
    print(f"corpus images        : {len(idx):,}", flush=True)

    clip = pd.read_parquet(f"{M}/tags/medpmc_tags_clip.parquet",
                           columns=["key", "specialty", "modality"]).rename(
        columns={"specialty": "specialty_clip", "modality": "modality_clip"})
    cap = pd.read_parquet(f"{M}/tags/medpmc_tags.parquet",
                          columns=["image_id", "specialty", "modality"]).rename(
        columns={"specialty": "specialty_caption", "modality": "modality_caption"})
    hum = _human(a.review_dir)
    # Built by build_medpmc_figure_rollup.py. Multipanel figures have no CLIP tag,
    # so this is the source between clip and caption in SOURCE_PRECEDENCE.
    roll_path = f"{M}/tags/medpmc_figure_rollup.parquet"
    if os.path.isfile(roll_path):
        roll = pd.read_parquet(roll_path, columns=["image_id", "top_modality"]).rename(
            columns={"top_modality": "modality_figure_rollup"})
    else:
        print(f"no {roll_path}: run build_medpmc_figure_rollup.py first", flush=True)
        roll = pd.DataFrame(columns=["image_id", "modality_figure_rollup"])
    print(f"clip-tagged          : {len(clip):,}", flush=True)
    print(f"caption-tagged       : {len(cap):,}", flush=True)
    print(f"figure roll-up       : {len(roll):,}", flush=True)
    print(f"human-verified       : {len(hum):,}", flush=True)

    df = (idx.merge(clip, on="key", how="left")
             .merge(cap, on="image_id", how="left")
             .merge(roll, on="image_id", how="left")
             .merge(hum, on="image_id", how="left"))

    for c in ("modality_caption", "modality_clip", "modality_figure_rollup", "modality_human"):
        df[c] = df[c].map(normalize)

    # ---- resolve by precedence. Vectorised: fill down the precedence order.
    mod = pd.Series(pd.NA, index=df.index, dtype="object")
    src = pd.Series(pd.NA, index=df.index, dtype="object")
    for s in SOURCE_PRECEDENCE:
        col = f"modality_{s}"
        if col not in df.columns:
            continue
        take = mod.isna() & df[col].notna()
        mod[take] = df.loc[take, col]
        src[take] = s
    df["modality_final"] = mod
    df["modality_source"] = src
    df["modality_display"] = df.modality_final.map(display)

    spec = pd.Series(pd.NA, index=df.index, dtype="object")
    ssrc = pd.Series(pd.NA, index=df.index, dtype="object")
    for s in SOURCE_PRECEDENCE:
        col = f"specialty_{s}"
        if col not in df.columns:
            continue
        take = spec.isna() & df[col].notna()
        spec[take] = df.loc[take, col]
        ssrc[take] = s
    df["specialty_final"] = spec
    df["specialty_source"] = ssrc

    df["human_verified"] = df.modality_human.notna() | df.specialty_human.notna()
    both = df.modality_caption.notna() & df.modality_clip.notna()
    df["sources_disagree"] = both & (df.modality_caption != df.modality_clip)

    keep = ["image_id", "source_type",
            "modality_caption", "modality_clip", "modality_figure_rollup",
            "modality_human", "modality_final", "modality_source", "modality_display",
            "specialty_caption", "specialty_clip", "specialty_human",
            "specialty_final", "specialty_source",
            "human_verified", "human_annotator", "n_human_modalities",
            "sources_disagree"]
    out = df[[c for c in keep if c in df.columns]]
    p = a.out or f"{M}/tags/medpmc_labels_resolved.parquet"
    out.to_parquet(p, index=False)

    print(f"\nwrote {p}  n={len(out):,}")
    print("\nmodality_source (who won):")
    print(out.modality_source.value_counts(dropna=False).to_string())
    print("\nmodality_display:")
    print(out.modality_display.value_counts(dropna=False).head(10).to_string())
    print(f"\nhuman_verified rows  : {int(out.human_verified.sum()):,}")
    print(f"caption/clip disagree : {int(out.sources_disagree.sum()):,} "
          f"({100*out.sources_disagree.mean():.1f}% of corpus)")


if __name__ == "__main__":
    main()
