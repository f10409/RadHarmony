"""External validation of TAIX-Ray linear probes on Emory CXR and MIMIC-CXR.

Protocol
--------
- Probe TRAINED on the full TAIX-Ray train split (loaded from existing pkl cache).
- Probe TESTED on an external dataset (Emory or MIMIC) using phase-grounded,
  TAIX-harmonized labels.
- Label binarization: harmonized severity scores > 0 → 1 (present), = 0 → 0
  (absent), NaN → excluded from that label's evaluation.

Labels evaluated (the 8 TAIX probe labels):
  atelectasis_left, atelectasis_right, heart_size,
  pleural_effusion_left, pleural_effusion_right, pulmonary_congestion,
  pulmonary_opacities_left, pulmonary_opacities_right

Datasets
--------
  emory  — EmoryCXR v2 (zarr, 320×320 uint8)
             Requires: zarr  →  pip install zarr
             Labels   : --emory_labels (emory_joined.parquet)
             Images   : <--emory_zarr_dir>/{patient}/{study}/{img}.zarr
             Filter   : frontal == 1

  mimic  — MIMIC-CXR v2 (DICOM)
             Labels   : --mimic_labels (mimic_taix_labels.parquet)
             Image CSV: <--mimic_dicom_dir>/temp_mimic.csv
                        (--mimic_dicom_dir is the mimic-cxr/2.1.0 DICOM folder)
             Images   : same directory as the CSV
             Filter   : DICOM ViewPosition in {PA, AP}

Each path argument can also come from an environment variable (see --help).
Caches and results go under RADHARMONY_OUTPUT_DIR (default: ./outputs).

Usage
-----
  python eval_linear_probe_external.py \\
      --backbone chexagent --target emory --device cuda:0

  python eval_linear_probe_external.py \\
      --backbone navit \\
      --ckpt /path/to/mimic-cls-ij1-vic-s-scratch-dp01.ckpt \\
      --target mimic --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAIX_LABEL_COLS = [
    "atelectasis_left",
    "atelectasis_right",
    "heart_size",
    "pleural_effusion_left",
    "pleural_effusion_right",
    "pulmonary_congestion",
    "pulmonary_opacities_left",
    "pulmonary_opacities_right",
]

# Caches and results go under RADHARMONY_OUTPUT_DIR (default: ./outputs).
OUTPUT_ROOT = os.environ.get("RADHARMONY_OUTPUT_DIR") or "./outputs"
TAIX_CACHE_ROOT = os.path.join(OUTPUT_ROOT, "taix_lp_cache")
DEFAULT_CACHE_DIR = os.path.join(OUTPUT_ROOT, "taix_lp_cache_external")
DEFAULT_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "taix_lp_results_external")

# File name of the MIMIC image list in --mimic_dicom_dir.
MIMIC_DICOM_CSV_NAME = "temp_mimic.csv"

# Inputs of each --target: (argument, environment variable, what it is).
TARGET_INPUTS = {
    "emory": [
        ("emory_labels", "EMORY_JOINED_PARQUET", "the emory_joined.parquet file"),
        ("emory_zarr_dir", "EMORY_ZARR_DIR", "the folder of the Emory zarr images"),
    ],
    "mimic": [
        ("mimic_labels", "MIMIC_TAIX_LABELS", "the mimic_taix_labels.parquet file"),
        ("mimic_dicom_dir", "MIMIC_DICOM_DIR", f"the MIMIC-CXR DICOM folder with {MIMIC_DICOM_CSV_NAME}"),
    ],
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="External validation: TAIX-trained probe → Emory or MIMIC"
    )
    p.add_argument(
        "--backbone",
        choices=["chexagent", "medsiglip", "biomed_clip", "lejepa", "navit"],
        required=True,
    )
    p.add_argument("--ckpt", type=str, default=None,
                   help="Checkpoint path (required for lejepa / navit)")
    p.add_argument(
        "--target",
        choices=["emory", "mimic"],
        required=True,
        help="External dataset to evaluate on",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--n_bootstrap", type=int, default=200)
    p.add_argument(
        "--max_samples", type=int, default=None,
        help="Cap external dataset size (for quick testing)",
    )
    p.add_argument(
        "--force_extract", action="store_true",
        help="Re-extract external embeddings even if cache exists",
    )
    p.add_argument("--taix_cache_dir", type=str, default=TAIX_CACHE_ROOT,
                   help="Root of the TAIX embedding cache (contains {run_id}/embeddings/taix.train.pkl)")
    p.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR,
                   help="Where to cache external embeddings")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--run_id", type=str, default=None,
                   help="Override the run identifier used for cache/result paths (default: checkpoint stem)")
    for inputs in TARGET_INPUTS.values():
        for name, env, what in inputs:
            p.add_argument(f"--{name}", type=str, default=os.environ.get(env),
                           help=f"{what} (default: env {env})")
    args = p.parse_args()
    for name, env, what in TARGET_INPUTS[args.target]:
        if not getattr(args, name):
            p.error(f"--target {args.target} needs --{name} or env {env} ({what})")
    return args


# ---------------------------------------------------------------------------
# Backbone factories for external (pre-loaded tensor) inputs
# ---------------------------------------------------------------------------


def make_external_backbone(backbone: str, device: str, ckpt: str | None = None):
    """Return (encoder, preprocess_fn) for external validation.

    preprocess_fn: list of (1, H, W) float32 tensors [0, 1] → (B, C, H', W') batch tensor
    Batched so HF processors can be called once per batch (29x speedup vs per-image).

    No MONAI LoadImaged — images are already loaded by the dataset.
    """
    import torch.nn.functional as F

    _model_dir = os.path.dirname(os.path.abspath(__file__))
    if _model_dir not in sys.path:
        sys.path.insert(0, _model_dir)

    from radharmony.evaluator import ImageEncoderWrapper
    from radharmony.evaluator.backbones._utils import to_3channel, normalize_to_uint8, to_pil_rgb

    if backbone == "navit":
        # NaViT wrapper is proprietary and lives OUTSIDE this repo (never tracked).
        _navit_dir = os.environ.get("NAVIT_PRIVATE_DIR")
        if not _navit_dir:
            raise SystemExit("Set NAVIT_PRIVATE_DIR to the folder of navit_reference_wrapper.py.")
        if _navit_dir not in sys.path:
            sys.path.insert(0, _navit_dir)
        from navit_reference_wrapper import NaViTReferenceEncoder
        enc = NaViTReferenceEncoder.from_checkpoint(ckpt, device=device).eval()
        encoder = ImageEncoderWrapper.from_custom(enc).to(device)

        def preprocess_fn(xs: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack([
                F.interpolate(x.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False)[0]
                for x in xs
            ])

        return encoder, preprocess_fn

    if backbone == "lejepa":
        from modules.lejepa_encoder import LeJEPAEncoder
        enc = LeJEPAEncoder.from_checkpoint(ckpt, device=device).eval()
        encoder = ImageEncoderWrapper.from_custom(enc).to(device)

        def preprocess_fn(xs: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack([
                F.interpolate(x.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False)[0]
                for x in xs
            ])

        return encoder, preprocess_fn

    if backbone == "chexagent":
        from transformers import AutoModel, AutoProcessor
        hub = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"
        model = AutoModel.from_pretrained(hub).to(device).eval()
        processor = AutoProcessor.from_pretrained(hub)

        def preprocess_fn(xs: list[torch.Tensor]) -> torch.Tensor:
            return processor(
                images=[normalize_to_uint8(to_3channel(x)) for x in xs],
                return_tensors="pt",
            )["pixel_values"]

        def _call(m, x):
            return m.vision_model(pixel_values=x).pooler_output

        encoder = ImageEncoderWrapper.from_custom(model, model_call=_call).to(device)
        return encoder, preprocess_fn

    if backbone == "medsiglip":
        from transformers import AutoModel, AutoProcessor
        hub = "google/medsiglip-448"
        model = AutoModel.from_pretrained(hub).to(device).eval()
        processor = AutoProcessor.from_pretrained(hub)

        def preprocess_fn(xs: list[torch.Tensor]) -> torch.Tensor:
            return processor(
                images=[normalize_to_uint8(to_3channel(x)) for x in xs],
                return_tensors="pt",
            )["pixel_values"]

        def _call(m, x):
            return m.vision_model(pixel_values=x).pooler_output

        encoder = ImageEncoderWrapper.from_custom(model, model_call=_call).to(device)
        return encoder, preprocess_fn

    if backbone == "biomed_clip":
        import open_clip
        hub = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        biomed_model, _, preprocess_val = open_clip.create_model_and_transforms(hub)
        biomed_model = biomed_model.to(device).eval()

        def preprocess_fn(xs: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack([preprocess_val(to_pil_rgb(x)) for x in xs])

        encoder = ImageEncoderWrapper.from_custom(
            biomed_model, model_call=lambda m, x: m.encode_image(x),
        ).to(device)
        return encoder, preprocess_fn

    raise ValueError(f"Unknown backbone: {backbone!r}")


# ---------------------------------------------------------------------------
# External datasets
# ---------------------------------------------------------------------------


def _binarize_labels(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """Convert severity-grade or binary columns to {0, 1, NaN} float array."""
    arr = df[cols].to_numpy(dtype=float)
    # Severity > 0 → 1 (present); 0 → 0 (absent); NaN stays NaN
    arr = np.where(arr > 0, 1.0, np.where(arr == 0, 0.0, np.nan))
    return arr


class EmoryExternalDataset(Dataset):
    """Emory CXR frontal images with phase-grounded TAIX labels.

    Reads 320×320 uint8 zarr arrays and applies the provided MONAI transform.
    Requires the ``zarr`` package (pip install zarr).
    """

    def __init__(
        self,
        labels_parquet: str,
        zarr_root: str,
        label_cols: Sequence[str] = TAIX_LABEL_COLS,
        max_samples: int | None = None,
    ):
        try:
            import zarr as _zarr  # noqa: F401
        except ImportError:
            raise ImportError(
                "zarr is required for Emory images. Install with: pip install zarr"
            )

        df = pd.read_parquet(labels_parquet)
        # Keep frontal only; drop rows where ALL labels are NaN
        df = df[df["frontal"] == 1].copy()
        label_arr = _binarize_labels(df, label_cols)
        has_any = np.any(~np.isnan(label_arr), axis=1)
        df = df[has_any].reset_index(drop=True)
        label_arr = label_arr[has_any]

        if max_samples is not None:
            df = df.iloc[:max_samples].reset_index(drop=True)
            label_arr = label_arr[:max_samples]

        # Build full zarr paths (image_path is *.png; replace extension)
        zarr_paths = (
            df["image_path"]
            .str.replace(r"\.png$", ".zarr", regex=True)
            .map(lambda p: os.path.join(zarr_root, p))
        )

        self.zarr_paths = zarr_paths.tolist()
        self.labels = label_arr.astype(np.float32)

        print(f"  Emory: {len(self.zarr_paths):,} frontal images with ≥1 label")

    def __len__(self) -> int:
        return len(self.zarr_paths)

    def __getitem__(self, idx: int) -> dict:
        import zarr

        arr = np.asarray(zarr.open(self.zarr_paths[idx], mode="r")[:])  # (H, W) uint8
        img = torch.tensor(arr, dtype=torch.float32).unsqueeze(0) / 255.0  # (1, H, W)
        return {"img": img, "cls": torch.from_numpy(self.labels[idx].copy())}


class MIMICExternalDataset(Dataset):
    """MIMIC-CXR frontal images with phase-grounded TAIX labels.

    Reads DICOM files via pydicom and filters to PA/AP views.
    """

    def __init__(
        self,
        dicom_csv: str,
        labels_parquet: str,
        img_root: str,
        label_cols: Sequence[str] = TAIX_LABEL_COLS,
        max_samples: int | None = None,
    ):
        import pydicom  # noqa: F401

        # Load image manifest and labels
        dicom_df = pd.read_csv(dicom_csv)
        labels_df = pd.read_parquet(labels_parquet)

        # Join on study_id
        merged = dicom_df.merge(labels_df, on="study_id", how="inner")

        # Filter frontal by reading ViewPosition from DICOM headers (cached)
        print("  MIMIC: filtering frontal views (reading DICOM headers)…")
        frontal_mask = self._get_frontal_mask(merged, img_root)
        merged = merged[frontal_mask].reset_index(drop=True)

        # Binarize labels; drop rows with no labels
        label_arr = _binarize_labels(merged, label_cols)
        has_any = np.any(~np.isnan(label_arr), axis=1)
        merged = merged[has_any].reset_index(drop=True)
        label_arr = label_arr[has_any]

        # One image per study: prefer PA over AP (keep first after sort)
        # Already one row per DICOM, so no dedup needed here

        if max_samples is not None:
            merged = merged.iloc[:max_samples].reset_index(drop=True)
            label_arr = label_arr[:max_samples]

        self.img_paths = [
            os.path.join(img_root, p) for p in merged["path"].tolist()
        ]
        self.labels = label_arr.astype(np.float32)

        print(f"  MIMIC: {len(self.img_paths):,} frontal images with ≥1 label")

    @staticmethod
    def _get_frontal_mask(df: pd.DataFrame, img_root: str) -> list[bool]:
        import pydicom

        frontal_views = {"PA", "AP"}
        mask = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  reading DICOM headers", leave=False):
            try:
                ds = pydicom.dcmread(
                    os.path.join(img_root, row["path"]),
                    stop_before_pixels=True,
                )
                vp = str(getattr(ds, "ViewPosition", "")).upper().strip()
                mask.append(vp in frontal_views)
            except Exception:
                mask.append(False)
        return mask

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> dict:
        import pydicom

        ds = pydicom.dcmread(self.img_paths[idx])
        arr = ds.pixel_array.astype(np.float32)
        photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        if "MONOCHROME1" in photometric:
            arr = 1.0 - arr
        img = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W) float32 [0, 1]
        return {"img": img, "cls": torch.from_numpy(self.labels[idx])}


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


def _raw_collate(batch):
    """Collate that leaves 'img' as a list (variable spatial dims) and stacks 'cls'."""
    return {
        "img": [item["img"] for item in batch],
        "cls": torch.stack([item["cls"] for item in batch]),
    }


def extract_and_cache(
    encoder,
    dataset: Dataset,
    preprocess_fn,
    device: str,
    batch_size: int,
    num_workers: int,
    cache_path: str,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract embeddings from dataset using batched preprocess_fn.

    preprocess_fn: list of (1, H, W) float32 tensors → (B, C, H', W') batch tensor
    Called once per batch; HF processor batching gives ~29x speedup over per-image.
    """
    if not force and os.path.exists(cache_path):
        print(f"  Loading external embeddings from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        return payload["feats"], payload["labels"]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_raw_collate,
    )

    encoder.eval()
    encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)

    feats_list, labels_list = [], []
    autocast_ctx = torch.autocast(
        device_type="cuda" if device.startswith("cuda") else "cpu",
        dtype=torch.bfloat16,
    )
    with torch.no_grad(), autocast_ctx:
        for batch in tqdm(loader, desc="extracting embeddings", unit="batch"):
            raw_list = batch["img"]  # list of (1, H, W) float32 tensors
            imgs = preprocess_fn(raw_list)   # (B, C, H', W')
            out = encoder(imgs.to(device))
            feats_list.append(out.float().cpu().numpy())
            labels_list.append(batch["cls"].float().numpy())

    feats = np.concatenate(feats_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"feats": feats, "labels": labels}, f)
    print(f"  Cached external embeddings → {cache_path}")

    return feats, labels


# ---------------------------------------------------------------------------
# Probe evaluation (train on TAIX, test on external)
# ---------------------------------------------------------------------------


def _fit_predict(X_tr, y_tr, X_te, y_te, label_cols):
    from sklearn.linear_model import LogisticRegression

    probs = np.full((X_te.shape[0], len(label_cols)), np.nan, dtype=np.float32)
    for k in range(len(label_cols)):
        tr_valid = ~np.isnan(y_tr[:, k])
        te_valid = ~np.isnan(y_te[:, k])
        y_k = y_tr[tr_valid, k].astype(int)
        if len(np.unique(y_k)) < 2 or te_valid.sum() == 0:
            continue
        clf = LogisticRegression(
            class_weight="balanced", C=1.0, max_iter=10000, solver="lbfgs"
        )
        clf.fit(X_tr[tr_valid], y_k)
        p = np.full(X_te.shape[0], np.nan, dtype=np.float32)
        p[te_valid] = clf.predict_proba(X_te[te_valid])[:, 1]
        probs[:, k] = p
    return probs


def _compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    valid = ~np.isnan(y_score) & ~np.isnan(y_true)
    if valid.sum() < 2 or len(np.unique(y_true[valid])) < 2:
        return np.nan
    return roc_auc_score(y_true[valid].astype(int), y_score[valid])


def run_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_cols: list[str],
    n_seeds: int,
    n_bootstrap: int,
    base_seed: int = 0,
) -> pd.DataFrame:
    rows = []
    for seed_i in range(n_seeds):
        rng = np.random.default_rng(base_seed + seed_i)
        probs = _fit_predict(X_train, y_train, X_test, y_test, label_cols)

        # Point estimates
        for k, name in enumerate(label_cols):
            auroc = _compute_auroc(y_test[:, k], probs[:, k])
            rows.append({
                "label": name, "seed": base_seed + seed_i,
                "bootstrap": -1, "auroc": auroc,
            })

        # Bootstrap CIs
        n_te = X_test.shape[0]
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_te, size=n_te)
            for k, name in enumerate(label_cols):
                auroc = _compute_auroc(y_test[idx, k], probs[idx, k])
                rows.append({
                    "label": name, "seed": base_seed + seed_i,
                    "bootstrap": b, "auroc": auroc,
                })

    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    point = df[df["bootstrap"] == -1].groupby("label")["auroc"]
    boot = df[df["bootstrap"] >= 0].groupby("label")["auroc"]
    summary = pd.DataFrame({
        "auroc_mean": point.mean(),
        "auroc_std": point.std(),
        "auroc_ci_lo": boot.quantile(0.025),
        "auroc_ci_hi": boot.quantile(0.975),
    }).reset_index()
    macro = summary["auroc_mean"].mean()
    macro_lo = summary["auroc_ci_lo"].mean()
    macro_hi = summary["auroc_ci_hi"].mean()
    macro_row = pd.DataFrame([{
        "label": "macro_average",
        "auroc_mean": macro,
        "auroc_std": np.nan,
        "auroc_ci_lo": macro_lo,
        "auroc_ci_hi": macro_hi,
    }])
    return pd.concat([summary, macro_row], ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    run_id = (
        args.run_id
        if args.run_id is not None
        else os.path.splitext(os.path.basename(args.ckpt))[0]
        if args.ckpt is not None
        else args.backbone
    )

    print(f"\n{'='*60}")
    print(f"External Linear Probe  |  backbone={args.backbone}  |  target={args.target}")
    print(f"  run_id: {run_id}  |  device={args.device}")
    print(f"{'='*60}\n")

    # --- 1. Load TAIX train embeddings ----------------------------------------
    taix_train_pkl = os.path.join(
        args.taix_cache_dir, run_id, "embeddings", "taix.train.pkl"
    )
    if not os.path.exists(taix_train_pkl):
        # Fallback: old-style backbone-name path (pre checkpoint-stem renaming)
        taix_train_pkl_fallback = os.path.join(
            args.taix_cache_dir, args.backbone, "embeddings", "taix.train.pkl"
        )
        if os.path.exists(taix_train_pkl_fallback):
            taix_train_pkl = taix_train_pkl_fallback
            print(f"  [fallback] using backbone-name cache: {taix_train_pkl}")
        else:
            raise FileNotFoundError(
                f"TAIX train cache not found at:\n  {taix_train_pkl}\n"
                f"  {taix_train_pkl_fallback}\n"
                f"Run eval_linear_probe_taix.py --backbone {args.backbone} first."
            )

    print(f"[1/4] Loading TAIX train embeddings from {taix_train_pkl} ...")
    with open(taix_train_pkl, "rb") as f:
        train_payload = pickle.load(f)
    X_train = train_payload["feats"]
    y_train_raw = train_payload["labels"]
    train_label_names = train_payload["labels_names"]

    # Align TAIX train labels to TAIX_LABEL_COLS order
    col_idx = [train_label_names.index(c) for c in TAIX_LABEL_COLS]
    X_train = X_train
    y_train = y_train_raw[:, col_idx]
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")

    # --- 2. Build backbone & preprocess_fn ------------------------------------
    print(f"[2/4] Building backbone ({args.backbone}) ...")
    encoder, preprocess_fn = make_external_backbone(args.backbone, args.device, args.ckpt)

    # --- 3. Build external dataset --------------------------------------------
    print(f"[3/4] Building {args.target} dataset ...")
    if args.target == "emory":
        ds = EmoryExternalDataset(
            labels_parquet=args.emory_labels,
            zarr_root=args.emory_zarr_dir,
            max_samples=args.max_samples,
        )
    else:
        ds = MIMICExternalDataset(
            dicom_csv=os.path.join(args.mimic_dicom_dir, MIMIC_DICOM_CSV_NAME),
            labels_parquet=args.mimic_labels,
            img_root=args.mimic_dicom_dir,
            max_samples=args.max_samples,
        )

    # --- 4. Extract / load external embeddings --------------------------------
    _cap_suffix = f"_{args.max_samples}" if args.max_samples is not None else ""
    ext_cache_dir = os.path.join(args.cache_dir, run_id, "embeddings")
    ext_cache_path = os.path.join(ext_cache_dir, f"{args.target}{_cap_suffix}.pkl")

    print(f"[4/4] Extracting {args.target} embeddings ...")
    X_test, y_test_raw = extract_and_cache(
        encoder, ds, preprocess_fn,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_path=ext_cache_path,
        force=args.force_extract,
    )
    # y_test_raw is already binarized (from dataset's label array)
    y_test = y_test_raw
    print(f"  X_test: {X_test.shape}  y_test: {y_test.shape}")

    # --- 5. Run probe ---------------------------------------------------------
    print(f"\nRunning probe  seeds={args.n_seeds}  bootstrap={args.n_bootstrap} ...")
    df_raw = run_probe(
        X_train, y_train, X_test, y_test,
        label_cols=TAIX_LABEL_COLS,
        n_seeds=args.n_seeds,
        n_bootstrap=args.n_bootstrap,
    )

    # --- 6. Save & display ---------------------------------------------------
    out_dir = os.path.join(args.output_dir, run_id, f"{args.target}{_cap_suffix}")
    os.makedirs(out_dir, exist_ok=True)
    df_raw.to_csv(os.path.join(out_dir, "results.csv"), index=False)

    summary = summarise(df_raw)
    summary.to_csv(os.path.join(out_dir, "results_summary.csv"), index=False)

    print(f"\nResults saved → {out_dir}")
    print(f"\n{'='*60}")
    print(f"AUROC — {args.target} (trained on TAIX-Ray, tested externally)")
    print(f"{'='*60}")
    auroc_cols = ["label", "auroc_mean", "auroc_ci_lo", "auroc_ci_hi"]
    print(summary[auroc_cols].to_string(index=False))


if __name__ == "__main__":
    main()
