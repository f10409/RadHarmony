"""Linear probe evaluation on MIMIC-CXR-JPG with the 14 CheXpert labels.

Mirror of ``eval_linear_probe_emory.py`` but on MIMIC-CXR-JPG (the full JPEG
copy, not the partial DICOM copy). Uses MIMIC's official
``mimic-cxr-2.0.0-chexpert.csv`` for finding labels — full 14 CheXpert columns,
not the gpt-oss-120b / TAIX-transformed labels.

Cohort
------
- 25 000-image deterministic subset of MIMIC-CXR-JPG frontals
  (seed=42, sampled from rows that have ≥1 non-NaN CheXpert label after
  U-Ignore binarization).
- 80/20 train/test split, same seed → identical split across backbones.

Usage
-----
  python eval_linear_probe_mimic.py --backbone raddino   --device cuda:0
  python eval_linear_probe_mimic.py --backbone eva_x     --device cuda:0
  python eval_linear_probe_mimic.py --backbone chexfound --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from monai.utils.enums import TraceKeys

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([TraceKeys])

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from eval_linear_probe_taix import OUTPUT_ROOT, make_backbone  # noqa: E402
from dataset_paths import paths  # noqa: E402  (the taix import puts scripts/ on sys.path)

# Registry key of MIMIC-CXR-JPG 2.0.0. The data paths file gives the image folder (base_image_dir),
# the metadata CSV (csv_path) and the CheXpert label CSV (label_csv_path).
MIMIC_KEY = "mimic_cxr_jpg"

DEFAULT_CACHE_DIR = os.path.join(OUTPUT_ROOT, "mimic_lp_cache")
DEFAULT_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "mimic_lp_results")

# 14 standard CheXpert labels — column names in mimic-cxr-2.0.0-chexpert.csv
# use TitleCase ("Atelectasis", "Pleural Effusion" with space, etc.). The
# harmonizer in MIMICCXRJPGDataset normalises these to lowercase-snakecase.
MIMIC_PROBE_LABELS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "enlarged_cardiomediastinum",
    "fracture",
    "lung_lesion",
    "lung_opacity",
    "no_finding",
    "pleural_effusion",
    "pleural_other",
    "pneumonia",
    "pneumothorax",
    "support_devices",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MIMIC-CXR-JPG linear probe — 25K cohort, 14 CheXpert labels"
    )
    p.add_argument(
        "--backbone",
        choices=["chexagent", "medsiglip", "biomed_clip", "lejepa", "navit",
                 "raddino", "medimageinsights", "eva_x", "dinov3", "chexfound"],
        required=True,
    )
    p.add_argument("--ckpt", type=str, default=None,
                   help="Checkpoint path (required for navit/lejepa)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--n_bootstrap", type=int, default=200)
    p.add_argument("--n_train_samples", type=int, nargs="+", default=None)
    p.add_argument("--cohort_size", type=int, default=25000)
    p.add_argument("--cohort_seed", type=int, default=42)
    p.add_argument("--test_frac", type=float, default=0.20)
    p.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--force-extract", action="store_true")
    return p.parse_args()


def build_cohort_dfs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) — fixed 25K MIMIC frontal cohort.

    Joins MIMIC's official metadata (file paths + ViewPosition) with the
    official CheXpert label CSV. Frontal-only filter via ViewPosition in
    {"PA", "AP"}.  CheXpert U-Ignore binarisation: 1→1, 0→0, -1→NaN, NaN→NaN.
    """
    mimic = paths(MIMIC_KEY)
    meta = pd.read_csv(mimic["csv_path"])
    labels = pd.read_csv(mimic["label_csv_path"])
    # CheXpert CSV columns are TitleCase with spaces → normalise to snake_case
    rename = {c: c.lower().replace(" ", "_") for c in labels.columns}
    labels = labels.rename(columns=rename)

    # Build per-image rows: (subject_id, study_id, dicom_id, ViewPosition,
    # image_path). MIMIC's metadata has one row per DICOM; JPG mirror uses
    # the same dicom_id.jpg layout.
    if "ViewPosition" in meta.columns:
        meta = meta[meta["ViewPosition"].isin(["PA", "AP"])].copy()
    print(f"  MIMIC frontal images (PA/AP): {len(meta):,}")

    # Join labels (study-level) with metadata (image-level)
    df = meta.merge(labels, on=["subject_id", "study_id"], how="inner")
    print(f"  joined with CheXpert labels: {len(df):,} rows")

    # CheXpert U-Ignore binarisation per label
    keep_cols = [c for c in MIMIC_PROBE_LABELS if c in df.columns]
    if len(keep_cols) != 14:
        missing = [c for c in MIMIC_PROBE_LABELS if c not in df.columns]
        print(f"  WARNING: {len(missing)} label cols missing: {missing}")
    for c in keep_cols:
        x = df[c].to_numpy()
        df[c] = np.where(x == 1.0, 1.0, np.where(x == 0.0, 0.0, np.nan))

    # Keep rows with at least one non-NaN binarised label
    has_any = df[keep_cols].notna().any(axis=1)
    df = df[has_any].reset_index(drop=True)
    print(f"  binarised + ≥1 non-NaN label: {len(df):,} rows")

    # Build image_path column (relative to the image folder)
    # MIMIC-CXR-JPG: files/p10/p10000032/s50414267/<dicom_id>.jpg
    def _img_path(row):
        sub = str(int(row["subject_id"]))
        std = str(int(row["study_id"]))
        return f"p{sub[:2]}/p{sub}/s{std}/{row['dicom_id']}.jpg"
    df["image_path"] = df.apply(_img_path, axis=1)
    # patient_id required by base dataset (mirrors EmoryCXRDataset usage)
    df["patient_id"] = df["subject_id"].astype(str)

    if args.cohort_size > len(df):
        print(f"  WARNING: requested cohort_size={args.cohort_size} > available "
              f"{len(df)}; capping at {len(df)}")
        cohort = df.reset_index(drop=True)
    else:
        rng = np.random.default_rng(args.cohort_seed)
        cohort_idx = rng.choice(len(df), size=args.cohort_size, replace=False)
        cohort_idx.sort()
        cohort = df.iloc[cohort_idx].reset_index(drop=True)
    print(f"  cohort (seed={args.cohort_seed}): {len(cohort):,} rows, "
          f"unique patients: {cohort['patient_id'].nunique()}")

    rng2 = np.random.default_rng(args.cohort_seed + 1)
    n_test = int(round(len(cohort) * args.test_frac))
    test_idx = rng2.choice(len(cohort), size=n_test, replace=False)
    test_mask = np.zeros(len(cohort), dtype=bool)
    test_mask[test_idx] = True
    test_df = cohort[test_mask].reset_index(drop=True)
    train_df = cohort[~test_mask].reset_index(drop=True)
    print(f"  train={len(train_df):,}  test={len(test_df):,}")
    return train_df, test_df


def make_mimic_dataset(df: pd.DataFrame, transform, cache_dir: str):
    from radharmony.dataset import MIMICCXRJPGDataset

    MIMICCXRJPGDataset.LABEL_COLS = list(MIMIC_PROBE_LABELS)
    return MIMICCXRJPGDataset(
        base_image_dir=paths(MIMIC_KEY)["base_image_dir"],
        csv_path=None,
        label_csv_path=None,
        transform=transform,
        output_cls=True,
        cache_dir=cache_dir,
        harmonized_df=df,
    )


def main() -> None:
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"MIMIC-CXR-JPG Linear Probe | backbone={args.backbone} | device={args.device}")
    print(f"  cohort_size={args.cohort_size} cohort_seed={args.cohort_seed} "
          f"test_frac={args.test_frac}")
    print(f"{'='*60}\n")

    run_id = args.run_id or args.backbone
    cache_dir = os.path.join(args.cache_dir, run_id)
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/4] Building backbone ({args.backbone}) ...")
    transform, encoder = make_backbone(args.backbone, args.device, args.ckpt)

    print(f"[2/4] Building MIMIC 25K cohort + 80/20 split ...")
    train_df, test_df = build_cohort_dfs(args)
    train_ds = make_mimic_dataset(train_df, transform, os.path.join(cache_dir, "train"))
    test_ds = make_mimic_dataset(test_df, transform, os.path.join(cache_dir, "test"))

    embedding_cache = os.path.join(cache_dir, "embeddings", "mimic")
    if args.force_extract:
        for suffix in ("train", "test"):
            p = f"{embedding_cache}.{suffix}.pkl"
            if os.path.exists(p):
                os.remove(p)

    print(f"[3/4] Configuring LinearProbeEvaluator ...")
    from radharmony.evaluator import LinearProbeEvaluator
    probe = LinearProbeEvaluator(
        image_encoder=encoder,
        train_dataset=train_ds,
        test_dataset=test_ds,
        n_seeds=args.n_seeds,
        n_bootstrap=args.n_bootstrap,
        n_train_samples=args.n_train_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        embedding_cache=embedding_cache,
        output_dir=output_dir,
    )

    print(f"[4/4] Running linear probe ...")
    df = probe.evaluate()
    saved = probe.save_results(df)
    print(f"\nResults saved → {saved}")
    summary = pd.read_csv(os.path.join(saved, "results_summary.csv"))
    auroc_cols = [c for c in summary.columns if "auroc" in c]
    group_cols = ["label"] + (["n_train"] if "n_train" in summary.columns else [])
    if auroc_cols:
        print(f"\n{'='*60}")
        print("AUROC summary (mean ± std over seeds + bootstrap)")
        print(f"{'='*60}")
        print(summary[group_cols + auroc_cols].to_string(index=False))


if __name__ == "__main__":
    main()
