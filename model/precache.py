"""Pre-cache RadHarmony datasets for DDP-safe training.

Run this once on a single process before launching multi-GPU training
so that every sample is already cached and workers don't race on writes.

Accepts the same data arguments as train.py.

Example:
    python precache.py \
        --chexpert_dir /path/to/CheXpert-v1.0/train/ \
        --mimic_dir /path/to/mimic \
        --mimic_label_csv /path/to/mimic-cxr-2.0.0-chexpert.csv \
        --chestxray14_dir /path/to/CXR14 \
        --cache_dir ./cache \
        --num_workers 8
"""

from __future__ import annotations

import argparse
import os

from radharmony.dataset import (
    CheXpertDataset,
    ChestXray14Dataset,
    MIMICCXRDataset,
    RadiologyTransform2D,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-cache datasets for LeJEPA 2D")

    p.add_argument("--chexpert_dir", type=str, default=None)
    p.add_argument("--chexpert_csv", type=str, default=None)
    p.add_argument("--mimic_dir", type=str, default=None)
    p.add_argument("--mimic_csv", type=str, default=None)
    p.add_argument("--mimic_label_csv", type=str, default=None)
    p.add_argument("--chestxray14_dir", type=str, default=None)
    p.add_argument("--chestxray14_csv", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default="./cache")
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--probe_label", type=str, default="pneumothorax",
        help="Target label column (used to validate label availability)",
    )

    return p.parse_args()


def main():
    args = parse_args()

    transform = RadiologyTransform2D(
        img_size=args.img_size, output_keys={"img", "cls"}
    ).get_transform()

    dataset_configs = []

    if args.chexpert_dir:
        kwargs = dict(
            base_image_dir=args.chexpert_dir,
            transform=transform,
            output_cls=True,
            cache_dir=os.path.join(args.cache_dir, "chexpert"),
        )
        if args.chexpert_csv:
            kwargs["csv_path"] = args.chexpert_csv
        dataset_configs.append(("CheXpert", CheXpertDataset, kwargs))

    if args.mimic_dir:
        kwargs = dict(
            base_image_dir=args.mimic_dir,
            transform=transform,
            output_cls=True,
            cache_dir=os.path.join(args.cache_dir, "mimic"),
        )
        if args.mimic_csv:
            kwargs["csv_path"] = args.mimic_csv
        if args.mimic_label_csv:
            kwargs["label_csv_path"] = args.mimic_label_csv
        dataset_configs.append(("MIMIC-CXR", MIMICCXRDataset, kwargs))

    if args.chestxray14_dir:
        kwargs = dict(
            base_image_dir=args.chestxray14_dir,
            transform=transform,
            output_cls=True,
            cache_dir=os.path.join(args.cache_dir, "chestxray14"),
        )
        if args.chestxray14_csv:
            kwargs["csv_path"] = args.chestxray14_csv
        dataset_configs.append(("ChestX-ray14", ChestXray14Dataset, kwargs))

    if not dataset_configs:
        raise SystemExit("No dataset directories provided.")

    for name, DatasetCls, kwargs in dataset_configs:
        print(f"\n{'='*60}")
        print(f"Processing {name}")
        print(f"{'='*60}")

        ds = DatasetCls(**kwargs)

        # Validate probe label
        target_col = next(
            (c for c in ds.LABEL_COLS if c.lower() == args.probe_label.lower()),
            None,
        )
        if target_col is None:
            print(f"  Skipping — '{args.probe_label}' not in LABEL_COLS")
            continue
        ds.LABEL_COLS = [target_col]

        df = ds.get_harmonized_df()
        if target_col not in df.columns:
            print(f"  Skipping — '{target_col}' column missing from harmonized DataFrame")
            continue

        # Verify images (drops missing / multichannel)
        ds.verify_images()

        # Build train + val PersistentDatasets
        train_ds, val_ds = ds.get_datasets(n_splits=10, num_cores=4)
        print(f"  {name}: train={len(train_ds):,}  val={len(val_ds):,}")

        # Pre-cache both splits
        print(f"  Caching train split …")
        ds.pre_cache(train_ds, num_workers=args.num_workers)
        print(f"  Caching val split …")
        ds.pre_cache(val_ds, num_workers=args.num_workers)

        print(f"  {name} done.")

    print(f"\nAll caching complete. Cache dir: {args.cache_dir}")


if __name__ == "__main__":
    main()
