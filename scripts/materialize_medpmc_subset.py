#!/usr/bin/env python
"""Extract a filtered MedPMC subset out of the webdataset tars into image files.

11M single image files would be too much for network storage. So RadHarmony
keeps the corpus in its shards and extracts only the subset you ask for. Images land at ``<base_dir>/images/<source_type>/<key><ext>``,
which is exactly the ``image_path`` the harmonizer writes.

Usage:
    python scripts/materialize_medpmc_subset.py --base-dir /data/MedPMC \\
        --radiology-only --modalities ct mri --min-tag-confidence high --limit 20000
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radharmony.harmonizer.medpmc import MedPMCHarmonizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--subsets", nargs="*", default=None)
    ap.add_argument("--radiology-only", action="store_true")
    ap.add_argument("--specialties", nargs="*", default=None)
    ap.add_argument("--modalities", nargs="*", default=None)
    ap.add_argument("--min-tag-confidence", default=None, choices=["low", "high"],
                    help="minimum tagger confidence (default: no filter)")
    # The app preset "high-precision radiology" is radiology_only, tagger agreement
    # and CLIP p>=0.9. Without all three, the script cannot make the cohort the
    # UI shows. That is how only 419 of 648,694 images were on disk.
    ap.add_argument("--require-tagger-agreement", action="store_true",
                    help="keep only rows where the caption tagger and CLIP agree")
    ap.add_argument("--min-clip-prob", type=float, default=None,
                    help="minimum CLIP probability (the preset uses 0.9)")
    ap.add_argument("--clip-tags-path", default=None,
                    help="override the CLIP tag table (default: <base-dir>/tags/...)")
    ap.add_argument("--subsets-default-singlepanel-subfigure", action="store_true",
                    help="shorthand for --subsets singlepanel subfigure (the app default)")
    ap.add_argument("--high-precision-radiology", action="store_true",
                    help="exact equivalent of the app preset: singlepanel+subfigure, "
                         "radiology_only, tagger agreement, CLIP p>=0.9")
    ap.add_argument("--limit", type=int, default=0, help="cap images extracted")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--skip-existing", action="store_true",
                    help="do not re-extract images already on disk (resumable)")
    a = ap.parse_args()

    subsets = a.subsets
    if a.high_precision_radiology or a.subsets_default_singlepanel_subfigure:
        subsets = subsets or ["singlepanel", "subfigure"]
    if a.high_precision_radiology:
        a.radiology_only = True
        a.require_tagger_agreement = True
        if a.min_clip_prob is None:
            a.min_clip_prob = 0.9

    h = MedPMCHarmonizer(base_dir=a.base_dir, subsets=subsets,
                         radiology_only=a.radiology_only, specialties=a.specialties,
                         modalities=a.modalities, min_tag_confidence=a.min_tag_confidence,
                         require_tagger_agreement=a.require_tagger_agreement,
                         min_clip_prob=a.min_clip_prob,
                         clip_tags_path=a.clip_tags_path)
    df = h.harmonize()
    if a.limit:
        df = df.head(a.limit)
    print(f"selected {len(df)} images", flush=True)
    if df.empty:
        return

    out_root = a.out_dir or os.path.join(a.base_dir, "images")
    by_shard = defaultdict(list)
    for shard, key, path in zip(df["shard"], df["key"], df["image_path"], strict=False):
        by_shard[shard].append((key, path))

    n = 0
    skipped = 0
    done_shards = 0
    print(f"{len(by_shard)} shards to scan", flush=True)
    for shard, items in sorted(by_shard.items()):
        tar_path = os.path.join(a.base_dir, "webdataset", shard)
        if not os.path.isfile(tar_path):
            print(f"  [warn] missing shard {shard}", flush=True)
            continue
        want = {k: p for k, p in items}
        with tarfile.open(tar_path, "r|*") as tf:       # one sequential pass per shard
            for m in tf:
                if m.name.endswith(".json"):
                    continue
                stem = os.path.splitext(os.path.basename(m.name))[0]
                if stem not in want:
                    continue
                dest = os.path.join(out_root, want[stem])
                if a.skip_existing and os.path.isfile(dest) and os.path.getsize(dest) > 0:
                    skipped += 1
                    continue
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                # Write to a temp name and rename, so an interrupted run never leaves
                # a truncated image that --skip-existing would then treat as done.
                tmp = dest + ".part"
                with open(tmp, "wb") as w:
                    w.write(fh.read())
                os.replace(tmp, dest)
                n += 1
        done_shards += 1
        print(f"  [{done_shards}/{len(by_shard)}] {shard}: {n} extracted, "
              f"{skipped} already present", flush=True)
    print(f"\nmaterialised {n} new images ({skipped} already present) under {out_root}")


if __name__ == "__main__":
    main()
