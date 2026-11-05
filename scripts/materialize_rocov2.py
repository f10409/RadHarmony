#!/usr/bin/env python
"""Write ROCOv2 images out of the HF parquet into files RadHarmony can read.

The HF mirror stores image bytes inline in parquet. RadHarmony's VQA datasets read
image_path from disk. So decode once to <base_dir>/images/<split>/<image_id>.jpg.
There are 80k files here (MedPMC has 11M), so plain files are fine.

Usage:
    python scripts/materialize_rocov2.py --base-dir /data/ROCOv2
    python scripts/materialize_rocov2.py --base-dir ... --splits test --limit 500
"""
from __future__ import annotations

import argparse
import glob
import io
import os

import pandas as pd
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--splits", nargs="*", default=["train", "validation", "test"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    total = 0
    for split in a.splits:
        out_dir = os.path.join(a.base_dir, "images", split)
        os.makedirs(out_dir, exist_ok=True)
        n = 0
        for p in sorted(glob.glob(os.path.join(a.base_dir, "hf", "data", f"{split}-*.parquet"))):
            df = pd.read_parquet(p, columns=["image_id", "image"])
            for iid, img in zip(df["image_id"], df["image"], strict=False):
                dest = os.path.join(out_dir, f"{iid}.jpg")
                if os.path.isfile(dest):
                    n += 1
                    continue
                b = img.get("bytes") if isinstance(img, dict) else img
                if b is None:
                    continue
                try:
                    Image.open(io.BytesIO(b)).convert("RGB").save(dest, quality=95)
                except Exception as e:
                    print(f"  [warn] {iid}: {type(e).__name__}: {e}", flush=True)
                    continue
                n += 1
                if a.limit and n >= a.limit:
                    break
            print(f"  {split}: {n} images", flush=True)
            if a.limit and n >= a.limit:
                break
        total += n
    print(f"materialised {total} images under {a.base_dir}/images")


if __name__ == "__main__":
    main()
