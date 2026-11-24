#!/usr/bin/env python
"""Merge zero-shot CLIP tag shards, compare against the caption lexicon, and build
the blind annotation sampling frame.

The two taggers are NOT independent evidence: MedPMC-CLIP was trained on MedPMC
captions, so both ultimately read the same text and their errors correlate. Agreement
between them is therefore reassuring-looking and weak. What agreement IS good for is
sampling. The cells where they DISAGREE are where the truth is unclear. A label
spent there gives much more than a label on an image that both taggers call CT.
"""
from __future__ import annotations
import argparse, glob
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--base-dir", required=True)
ap.add_argument("--n-annotate", type=int, default=800)
a = ap.parse_args()
M = a.base_dir

parts = sorted(glob.glob(f"{M}/tags/_clip_parts/*.parquet"))
print(f"merging {len(parts)} CLIP shard parts", flush=True)
clip = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
clip = clip.drop_duplicates("key")
clip.to_parquet(f"{M}/tags/medpmc_tags_clip.parquet", index=False)
print(f"  clip rows: {len(clip):,}")
print((clip.specialty.value_counts(normalize=True)*100).round(1).to_string())

idx = pd.read_parquet(f"{M}/webdataset/metadata/medpmc_index.parquet",
                      columns=["image_id","key","source_type","shard"])
lex = pd.read_parquet(f"{M}/tags/medpmc_tags.parquet",
                      columns=["image_id","specialty","modality","tag_confidence"])
lex = lex.rename(columns={"specialty":"lex_specialty","modality":"lex_modality",
                          "tag_confidence":"lex_conf"})
c = clip.rename(columns={"specialty":"clip_specialty","modality":"clip_modality"})
m = idx.merge(c[["key","clip_specialty","clip_modality","clip_prob","clip_margin"]],
              on="key", how="inner").merge(lex, on="image_id", how="left")
print(f"\njoined rows: {len(m):,}")

agree = (m.lex_specialty == m.clip_specialty)
known = m.lex_specialty.notna() & (m.lex_specialty != "unknown")
print(f"  lexicon tagged:            {known.mean()*100:.1f}%")
print(f"  agreement where both tag:  {agree[known].mean()*100:.1f}%")
print(f"  CLIP rescues (lex unknown):{(~known).sum():,}")
print("\nconfusion where they disagree (lexicon -> clip), top 10:")
dis = m[known & ~agree]
print(dis.groupby(["lex_specialty","clip_specialty"]).size()
        .sort_values(ascending=False).head(10).to_string())

# stratified frame: disagreement cells oversampled, plus lex-unknown, plus agreement
rng = 1337
strata = []
strata.append(dis.assign(stratum="disagree").sample(min(len(dis), a.n_annotate//2), random_state=rng))
unk = m[~known]
strata.append(unk.assign(stratum="lex_unknown").sample(min(len(unk), a.n_annotate//4), random_state=rng))
agr = m[known & agree]
strata.append(agr.assign(stratum="agree").sample(min(len(agr), a.n_annotate//4), random_state=rng))
frame = pd.concat(strata, ignore_index=True)
# sampling weights so corpus-level rates can be recovered (Horvitz-Thompson)
sizes = {"disagree": len(dis), "lex_unknown": len(unk), "agree": len(agr)}
frame["stratum_size"] = frame.stratum.map(sizes)
frame["sample_weight"] = frame.stratum_size / frame.groupby("stratum").stratum.transform("size")
frame.to_parquet(f"{M}/tags/annotation_frame.parquet", index=False)
print(f"\nwrote {M}/tags/annotation_frame.parquet  n={len(frame)}")
print(frame.stratum.value_counts().to_string())
print("\nNOTE: annotators must see the IMAGE ONLY. Showing the caption measures")
print("caption-agreement, not modality truth, and would have hidden the")
print("occlusal-photograph bug rather than exposing it.")
