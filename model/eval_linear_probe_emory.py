"""Linear probe evaluation on EmoryCXR v2 with a fixed 25K-image training cohort.

Mirror of ``eval_linear_probe_taix.py`` but:
  • In-domain Emory training (NOT external validation against a TAIX-trained probe;
    for that see ``eval_linear_probe_external.py``).
  • A deterministic 25 000-image subset sampled with ``seed=42`` from frontal
    Emory studies that have at least one non-NaN finding label. The SAME 25K
    rows are used for every backbone so cross-model comparisons are apples-to-apples.
  • 80/20 within-cohort train/test split, same split across backbones.

Protocol
--------
- Cohort        : Emory v2 frontal images joined with finding labels
                  (``emory_joined.parquet`` provides image_path + labels).
- Train         : 20 000 images (80%% of the 25 000 fixed cohort)
- Test          : 5 000 images (20%%)
- Probe         : per-label logistic regression via radharmony.evaluator.LinearProbeEvaluator
- Variance      : n_seeds * n_bootstrap (same defaults as the TAIX driver)
- Metrics       : per-label AUROC/AUPRC/F1 + macro average

Reuses the same backbone factories as ``eval_linear_probe_taix.py`` so
DINOv3 / CheXFound / NaViT / LeJEPA / the six contrastive backbones all work
through one --backbone flag.

Usage
-----
  Set EMORY_JOINED_PARQUET to the emory_joined.parquet file (or give --joined_parquet).

  python eval_linear_probe_emory.py --backbone raddino --device cuda:0
  python eval_linear_probe_emory.py --backbone dinov3  --device cuda:4
  python eval_linear_probe_emory.py --backbone chexfound --device cuda:5
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

# Reuse the TAIX driver's backbone dispatch — same _make_* functions and the
# same --backbone vocabulary.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from eval_linear_probe_taix import OUTPUT_ROOT, make_backbone  # noqa: E402
from dataset_paths import paths  # noqa: E402  (the taix import puts scripts/ on sys.path)

# Registry key of the Emory PNG images. The image folder comes from the data paths file.
EMORY_KEY = "emory_cxr"

DEFAULT_CACHE_DIR = os.path.join(OUTPUT_ROOT, "emory_lp_cache")
DEFAULT_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "emory_lp_results")

# Finding-label columns to probe. The 14 standard CheXpert labels (NOT the
# TAIX-transformed gpt-oss-120b labels). In emory_joined.parquet, two of these
# have a "_chexpert" suffix to disambiguate them from the gpt-oss columns:
#   atelectasis_chexpert       → renamed to "atelectasis" in build_cohort_dfs
#   pleural_effusion_chexpert  → renamed to "pleural_effusion"
# The other 12 columns have no overlap.
EMORY_PROBE_LABELS = [
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
        description="Emory CXR linear probe — fixed 25K cohort, in-domain train/test"
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
    p.add_argument("--n_seeds", type=int, default=3,
                   help="Train-subsample replicates (lower than TAIX's 5 to keep this fast).")
    p.add_argument("--n_bootstrap", type=int, default=200)
    p.add_argument("--n_train_samples", type=int, nargs="+", default=None,
                   help="Train sizes to sweep (default: full 20K train).")
    p.add_argument("--cohort_size", type=int, default=25000,
                   help="Total cohort N (default 25000). Same seed → same N rows across backbones.")
    p.add_argument("--cohort_seed", type=int, default=42,
                   help="Sampling seed for the fixed cohort. Do NOT change across backbones.")
    p.add_argument("--test_frac", type=float, default=0.20,
                   help="Fraction of the cohort held out for test. Same seed → same split.")
    p.add_argument("--joined_parquet", type=str, default=os.environ.get("EMORY_JOINED_PARQUET"),
                   help="emory_joined.parquet (image_path + labels). Default: env EMORY_JOINED_PARQUET.")
    p.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--force-extract", action="store_true",
                   help="Delete embedding cache before running (when backbone changes).")
    args = p.parse_args()
    if not args.joined_parquet:
        p.error("give --joined_parquet or set EMORY_JOINED_PARQUET to the emory_joined.parquet file")
    return args


def build_cohort_dfs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) — the fixed 25K cohort split deterministically.

    Pulled from emory_joined.parquet (which already joins image_path + labels).
    Filters to frontal==1 and rows with >=1 non-NaN probe-label. Deterministic
    sampling on `cohort_seed` so the SAME 25K rows are produced regardless of
    backbone, machine, or invocation order. patient-level grouping is preserved
    in the sense that we sample by row index but keep the patient_id column for
    downstream evaluator-side group analysis.
    """
    df = pd.read_parquet(args.joined_parquet)

    # Map the CheXpert versions of overlapping columns into the canonical
    # 14-label names. emory_joined keeps both gpt-oss and chexpert variants
    # of atelectasis / pleural_effusion; the suffix-less columns are gpt-oss.
    # We want CheXpert here.
    for chxp_col, canon in (("atelectasis_chexpert", "atelectasis"),
                            ("pleural_effusion_chexpert", "pleural_effusion")):
        if chxp_col in df.columns:
            df[canon] = df[chxp_col]

    # Frontal only
    if "frontal" in df.columns:
        df = df[df["frontal"] == 1].copy()

    # Keep only rows where at least one probe label is non-NaN
    keep_cols = [c for c in EMORY_PROBE_LABELS if c in df.columns]
    if not keep_cols:
        raise RuntimeError(
            f"None of {EMORY_PROBE_LABELS} present in emory_joined.parquet "
            f"columns ({list(df.columns)})"
        )

    # CheXpert binarisation — STANDARD convention (1=present, 0=absent,
    # -1=uncertain→NaN, NaN→NaN). My earlier "inverted" hypothesis was wrong:
    # the real root cause was a bug in radharmony.evaluator.classification.
    # linear_probe._fit_predict_per_label which int-cast NaN labels into
    # garbage, creating a 3-class LR problem where predict_proba[:, 1]
    # returned P(absent). Fix landed in linear_probe.py; binarisation here
    # reverted to the standard convention.
    for c in keep_cols:
        x = df[c].to_numpy()
        df[c] = np.where(x == 1.0, 1.0, np.where(x == 0.0, 0.0, np.nan))

    has_any = df[keep_cols].notna().any(axis=1)
    df = df[has_any].reset_index(drop=True)
    print(f"  emory_joined frontal+labeled+binarized rows (CheXpert U-Ignore): {len(df):,}")

    if args.cohort_size > len(df):
        raise RuntimeError(
            f"requested cohort_size={args.cohort_size} > available rows={len(df)}"
        )

    # Deterministic cohort sample
    rng = np.random.default_rng(args.cohort_seed)
    cohort_idx = rng.choice(len(df), size=args.cohort_size, replace=False)
    cohort_idx.sort()
    cohort = df.iloc[cohort_idx].reset_index(drop=True)
    print(f"  cohort rows (seed={args.cohort_seed}): {len(cohort):,}  "
          f"unique patients: {cohort['patient_id'].nunique() if 'patient_id' in cohort.columns else '?'}")

    # 80/20 split, same seed
    n_test = int(round(len(cohort) * args.test_frac))
    test_idx = rng.choice(len(cohort), size=n_test, replace=False)
    test_mask = np.zeros(len(cohort), dtype=bool)
    test_mask[test_idx] = True
    test_df = cohort[test_mask].reset_index(drop=True)
    train_df = cohort[~test_mask].reset_index(drop=True)
    print(f"  train={len(train_df):,}  test={len(test_df):,}")
    return train_df, test_df


def make_emory_dataset(df: pd.DataFrame, transform, cache_dir: str):
    """Build an EmoryCXRDataset over an already-harmonized df.

    EmoryCXRDataset accepts ``harmonized_df=`` and skips its own harmonization,
    using the provided df rows directly. The df must have ``image_path`` and the
    probe-label columns (which it does — emory_joined.parquet provides both).
    """
    from radharmony.dataset import EmoryCXRDataset

    # EmoryCXRDataset.LABEL_COLS defaults to the CheXpert-14 panel; we want
    # to probe the gpt-oss-120b Emory labels (atelectasis ±L/R, pleural_effusion
    # ±L/R, pulmonary_opacities ±L/R, heart_size, pulmonary_congestion). The
    # cls output comes from these columns, so override the class attribute.
    EmoryCXRDataset.LABEL_COLS = list(EMORY_PROBE_LABELS)

    # EmoryCXRDataset wants base_image_dir to resolve image_path relatively.
    return EmoryCXRDataset(
        base_image_dir=paths(EMORY_KEY)["base_image_dir"],
        label_csv_path=None,    # labels come from harmonized_df, not a CSV reread
        report_csv_path=None,
        transform=transform,
        output_cls=True,
        output_report=False,
        cache_dir=cache_dir,
        harmonized_df=df,
    )


def main() -> None:
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"Emory Linear Probe  |  backbone={args.backbone}  |  device={args.device}")
    print(f"  cohort_size={args.cohort_size}  cohort_seed={args.cohort_seed}  "
          f"test_frac={args.test_frac}")
    print(f"{'='*60}\n")

    if args.run_id is not None:
        run_id = args.run_id
    elif args.ckpt is not None:
        run_id = os.path.splitext(os.path.basename(args.ckpt))[0]
    else:
        run_id = args.backbone

    cache_dir = os.path.join(args.cache_dir, run_id)
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/4] Building backbone ({args.backbone}) ...")
    transform, encoder = make_backbone(args.backbone, args.device, args.ckpt)

    print(f"[2/4] Building Emory 25K cohort + 80/20 split ...")
    train_df, test_df = build_cohort_dfs(args)
    train_ds = make_emory_dataset(train_df, transform, os.path.join(cache_dir, "train"))
    test_ds = make_emory_dataset(test_df, transform, os.path.join(cache_dir, "test"))

    embedding_cache = os.path.join(cache_dir, "embeddings", "emory")
    if args.force_extract:
        for suffix in ("train", "test"):
            p = f"{embedding_cache}.{suffix}.pkl"
            if os.path.exists(p):
                os.remove(p)
                print(f"  [force-extract] deleted {p}")

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
    print(f"      seeds={args.n_seeds}  bootstrap={args.n_bootstrap}  "
          f"n_train={args.n_train_samples or 'full'}")

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
