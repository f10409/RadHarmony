#!/usr/bin/env python
"""Radiology-focused annotation frame for MedPMC.

The general frame sampled disagreements across all specialties. Judy's interest is
radiology, so the strata below are chosen to estimate the things a radiology filter
actually gets judged on:

  A  lexicon=radiology, CLIP!=radiology   -> images the filter would WRONGLY INCLUDE
  B  lexicon!=radiology, CLIP=radiology   -> radiology the filter would WRONGLY EXCLUDE
  C  both radiology, MODALITY differs     -> whether ct/mri/xray/... can be trusted
  D  lexicon unknown, CLIP=radiology      -> the 462k the caption tagger is blind to
  E  both agree fully                     -> control (without it, precision cannot be measured)

A and B give the precision/recall of `radiology_only`. C gives modality accuracy,
which is what `modalities=["ct","mri"]` depends on. E is not optional: a sample drawn
only from disagreements cannot estimate how often the taggers are jointly right.

Horvitz-Thompson weights are attached so corpus-level rates are recoverable despite
the deliberate oversampling.

ANNOTATION RULE: the interface must show the IMAGE ONLY -- never the caption, never
either tagger's label. Showing the caption measures caption-agreement rather than
what the image is, and would have concealed the occlusal-photograph error that
started this.
"""
from __future__ import annotations
import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_paths import paths  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base-dir", default=paths("medpmc", missing_ok=True).get("base_dir"),
                help="The MedPMC folder. Default: the data paths file.")
ap.add_argument("--n", type=int, default=800)
ap.add_argument("--seed", type=int, default=1337)
a = ap.parse_args()
M = a.base_dir

idx = pd.read_parquet(f"{M}/webdataset/metadata/medpmc_index.parquet",
                      columns=["image_id", "key", "source_type", "shard"])
clip = pd.read_parquet(f"{M}/tags/medpmc_tags_clip.parquet",
                       columns=["key", "specialty", "modality", "clip_prob", "clip_margin"]
                       ).rename(columns={"specialty": "clip_spec", "modality": "clip_mod"})
lex = pd.read_parquet(f"{M}/tags/medpmc_tags.parquet",
                      columns=["image_id", "specialty", "modality", "tag_confidence"]
                      ).rename(columns={"specialty": "lex_spec", "modality": "lex_mod",
                                        "tag_confidence": "lex_conf"})
m = idx.merge(clip, on="key", how="inner").merge(lex, on="image_id", how="left")
known = m.lex_spec.notna() & (m.lex_spec != "unknown")
R = "radiology"

strata = {
    "A_false_positive_risk": m[known & (m.lex_spec == R) & (m.clip_spec != R)],
    "B_missed_radiology":    m[known & (m.lex_spec != R) & (m.clip_spec == R)],
    "C_modality_conflict":   m[(m.lex_spec == R) & (m.clip_spec == R)
                               & m.lex_mod.notna() & m.clip_mod.notna()
                               & (m.lex_mod != m.clip_mod)],
    "D_lexicon_blind":       m[(~known) & (m.clip_spec == R)],
    "E_both_agree_control":  m[(m.lex_spec == R) & (m.clip_spec == R) & (m.lex_mod == m.clip_mod)],
}
# A and B decide the precision and recall of the filter, so they get the most labels.
# C decides the modality sub-filter.
# E is the control and needs enough labels to be useful.
alloc = {"A_false_positive_risk": 0.30, "B_missed_radiology": 0.20,
         "C_modality_conflict": 0.25, "D_lexicon_blind": 0.10,
         "E_both_agree_control": 0.15}

out = []
for name, df in strata.items():
    k = min(len(df), int(round(a.n * alloc[name])))
    s = df.sample(k, random_state=a.seed).copy()
    s["stratum"] = name
    s["stratum_size"] = len(df)
    w = len(df) / k if k else float("nan")
    s["sample_weight"] = w
    out.append(s)
    print(f"  {name:<26} population {len(df):>9,}   sampled {k:>4}   weight {w:>8.1f}")

frame = pd.concat(out, ignore_index=True)
p = f"{M}/tags/annotation_frame_radiology.parquet"
frame.to_parquet(p, index=False)
print(f"\nwrote {p}  n={len(frame)}")
print("\nwhat each stratum answers:")
print("  A -> precision of radiology_only     B -> recall of radiology_only")
print("  C -> accuracy of modalities=[...]    D -> how much the caption tagger misses")
print("  E -> the joint-correct rate, without which A and B cannot be turned into rates")
