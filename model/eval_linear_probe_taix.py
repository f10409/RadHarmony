"""Linear probe evaluation on TAIX-Ray using the predefined train/test split.

Supports four frozen backbones:
  chexagent   — StanfordAIMI/XraySigLIP (via transformers)
  medsiglip   — google/medsiglip-448 (via transformers)
  biomed_clip — microsoft/BiomedCLIP (via OpenCLIP)
  lejepa      — LeJEPA2D self-supervised model (requires --ckpt)

Protocol
--------
- Train split  : TAIX predefined "train" (137,593 images)
- Test split   : TAIX predefined "test" (42,928 images)
- Probe        : per-label logistic regression (L2, balanced class weights)
- Variance     : n_seeds independent train subsamples × n_bootstrap test resamples
- Metrics      : AUROC, AUPRC, F1 per label + macro average reported by
                 radharmony.evaluator.LinearProbeEvaluator

Outputs (per backbone)
----------------------
  <output_dir>/<backbone>/results.csv
  <output_dir>/<backbone>/results_summary.csv

Usage
-----
  # CheXagent / MedSigLIP / BiomedCLIP — no checkpoint needed
  python eval_linear_probe_taix.py --backbone chexagent --device cuda:0

  # LeJEPA2D — checkpoint required
  python eval_linear_probe_taix.py \\
      --backbone lejepa \\
      --ckpt /path/to/mimic-cxp-from-imagenet-dp01-bs160-e10_last.ckpt \\
      --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import torch
from monai.utils.enums import TraceKeys

# Allow MONAI's TraceKeys through PyTorch 2.6+ weights_only checkpoint loading.
# add_safe_globals only exists on torch >= 2.4; older envs (e.g. venv B's
# pinned torch 2.3.1 for MedImageInsights) don't need it.
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([TraceKeys])

# On some hosts, torch-2.12+cu130 ships a broken cuDNN sublibrary that fails conv2d
# (CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED). Opt-in disable via env on affected
# hosts — native conv is slower but correct. (NaViT path uses flash_attn/cu126
# and no conv, so it's unaffected either way.)
if os.environ.get("DISABLE_CUDNN") == "1":
    torch.backends.cudnn.enabled = False

# The data paths file reader is in <repo>/scripts.
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from dataset_paths import paths  # noqa: E402

# Registry key of the TAIX-Ray 512 images. The folder and the CSV come from the data paths file.
TAIX_KEY = "taix_ray_512"

# Caches and results go under RADHARMONY_OUTPUT_DIR (default: ./outputs).
OUTPUT_ROOT = os.environ.get("RADHARMONY_OUTPUT_DIR") or "./outputs"
DEFAULT_CACHE_DIR = os.path.join(OUTPUT_ROOT, "taix_lp_cache")
DEFAULT_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "taix_lp_results")


def require_env(name: str, what: str) -> str:
    """Return the value of environment variable ``name``. Stop with a clear message when it is not set."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Set {name} to {what}.")
    return value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TAIX-Ray linear probe — predefined train/test split, multiple backbones"
    )
    p.add_argument(
        "--backbone",
        choices=["chexagent", "medsiglip", "biomed_clip", "lejepa", "navit",
                 "raddino", "medimageinsights", "eva_x", "dinov3", "chexfound",
                 "siglip2", "ark_plus", "medical_mae"],
        required=True,
        help="Backbone to evaluate",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to LeJEPA2D .ckpt file (required for --backbone lejepa)",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--n_seeds",
        type=int,
        default=5,
        help="Number of train-subsample replicates for variance estimation",
    )
    p.add_argument(
        "--n_bootstrap",
        type=int,
        default=200,
        help="Bootstrap resamples per seed for 95%% CI on test metrics",
    )
    p.add_argument(
        "--n_train_samples",
        type=int,
        nargs="+",
        default=None,
        help="Train sizes to sweep (default: full train set)",
    )
    p.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--run_id", type=str, default=None,
                   help="Override the run identifier used for cache/result paths (default: checkpoint stem)")
    p.add_argument(
        "--force-extract",
        action="store_true",
        help="Delete existing embedding cache before running (use when the backbone has changed)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Backbone factories
# ---------------------------------------------------------------------------


def _make_navit(ckpt_path: str, device: str):
    """NaViT-base with CLS pooling and proper RoPE, img_size=256, patch_size=16, 4 registers."""
    from radharmony.dataset import RadiologyTransform2D
    from radharmony.evaluator import ImageEncoderWrapper

    # NaViT wrapper is proprietary and lives OUTSIDE this repo (never tracked).
    _navit_dir = require_env("NAVIT_PRIVATE_DIR", "the folder of navit_reference_wrapper.py")
    if _navit_dir not in sys.path:
        sys.path.insert(0, _navit_dir)

    from navit_reference_wrapper import NaViTReferenceEncoder

    # Infer architecture from the checkpoint stem (e.g. base-10ep-256-p16,
    # large-10ep-512-p32). Old generic ckpt names (no size/res/patch tokens)
    # fall back to base / 256 / p16 — the prior NaViT convention. Env vars
    # NAVIT_SIZE / NAVIT_PATCH / NAVIT_IMG_SIZE override the parse.
    import re as _re
    _stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    _size = os.environ.get("NAVIT_SIZE") or ("large" if _re.search(r"(^|[-_])large([-_]|$)", _stem) else "base")
    _mp = _re.search(r"[-_]p(\d+)([-_]|$)", _stem)
    _patch = int(os.environ.get("NAVIT_PATCH") or (_mp.group(1) if _mp else 16))
    _mr = _re.search(r"[-_](256|320|384|448|512|768|1024)([-_]|$)", _stem)
    _img = int(os.environ.get("NAVIT_IMG_SIZE") or (_mr.group(1) if _mr else 256))

    print(f"  Loading NaViT reference (RoPE + CLS pool) from checkpoint: {ckpt_path}")
    print(f"  inferred: size={_size}  patch_size={_patch}  img_size={_img}")
    encoder_module = NaViTReferenceEncoder.from_checkpoint(
        ckpt_path, device=device, size=_size, patch_size=_patch
    )
    encoder_module.eval()

    print(f"  embed_dim: {encoder_module.embed_dim}  pool: cls (RoPE)")

    transform = RadiologyTransform2D(
        img_size=_img,
        output_keys={"img", "cls"},
    ).get_transform()

    encoder = ImageEncoderWrapper.from_custom(encoder_module).to(device)
    return transform, encoder


def _make_lejepa(ckpt_path: str, device: str):
    from radharmony.dataset import RadiologyTransform2D
    from radharmony.evaluator import ImageEncoderWrapper

    _model_dir = os.path.dirname(os.path.abspath(__file__))
    if _model_dir not in sys.path:
        sys.path.insert(0, _model_dir)

    from modules.lejepa_encoder import LeJEPAEncoder

    print(f"  Loading LeJEPAEncoder from checkpoint: {ckpt_path}")
    encoder_module = LeJEPAEncoder.from_checkpoint(ckpt_path, device=device)
    encoder_module.eval()

    # img_size is fixed at 256 (patch_embed.projs.256 in checkpoint)
    img_size = 256
    print(f"  img_size: {img_size}  embed_dim: {encoder_module.embed_dim}")

    transform = RadiologyTransform2D(
        img_size=img_size,
        output_keys={"img", "cls"},
    ).get_transform()

    encoder = ImageEncoderWrapper.from_custom(encoder_module).to(device)
    return transform, encoder


def _make_chexagent(device: str):
    from radharmony.evaluator.backbones import make_chexagent

    transform, encoder, _, _ = make_chexagent(device=device)
    return transform, encoder


def _make_medsiglip(device: str):
    from radharmony.evaluator.backbones import make_medsiglip

    transform, encoder, _, _ = make_medsiglip(device=device)
    return transform, encoder


def _make_biomed_clip(device: str):
    from radharmony.evaluator.backbones import make_biomed_clip

    transform, encoder, _, _ = make_biomed_clip(device=device)
    return transform, encoder


def _make_raddino(device: str):
    from radharmony.evaluator.backbones import make_raddino

    transform, encoder = make_raddino(device=device)
    return transform, encoder


def _make_medimageinsights(device: str):
    from radharmony.evaluator.backbones import make_medimageinsights

    transform, encoder, _, _ = make_medimageinsights(device=device)
    return transform, encoder


def _make_eva_x(device: str):
    from radharmony.evaluator.backbones import make_eva_x

    transform, encoder = make_eva_x(device=device)
    return transform, encoder


def _make_dinov3(device: str):
    from radharmony.evaluator.backbones import make_dinov3

    transform, encoder = make_dinov3(device=device)
    return transform, encoder


def _make_chexfound(device: str):
    from radharmony.evaluator.backbones import make_chexfound

    # CHEXFOUND_DIR is the CheXFound clone. teacher_checkpoint.pth is in <chexfound_dir>/weights/.
    # With no CHEXFOUND_DIR, make_chexfound uses its default folder in third_party_models/.
    chexfound_dir = os.environ.get("CHEXFOUND_DIR")
    transform, encoder = make_chexfound(chexfound_dir=chexfound_dir, device=device)
    return transform, encoder


def _make_siglip2(device: str):
    from radharmony.evaluator.backbones import make_siglip2

    transform, encoder, _, _ = make_siglip2(device=device)
    return transform, encoder


def _make_ark_plus(device: str):
    from radharmony.evaluator.backbones import make_ark_plus

    # With no ARK_CKPT or ARK_TIMM_DIR, make_ark_plus uses its default files in third_party_models/Ark/.
    ckpt = os.environ.get("ARK_CKPT")
    legacy = os.environ.get("ARK_TIMM_DIR")
    transform, encoder = make_ark_plus(checkpoint_path=ckpt, legacy_timm_dir=legacy, device=device)
    return transform, encoder


def _make_medical_mae(device: str):
    from radharmony.evaluator.backbones import make_medical_mae

    # With no MEDMAE_* variable, make_medical_mae uses its default files in third_party_models/medical_mae/.
    ckpt = os.environ.get("MEDMAE_CKPT")
    mae_dir = os.environ.get("MEDMAE_DIR")
    legacy = os.environ.get("MEDMAE_TIMM_DIR")
    transform, encoder = make_medical_mae(
        variant="base", checkpoint_path=ckpt, mae_dir=mae_dir, legacy_timm_dir=legacy, device=device
    )
    return transform, encoder


def make_backbone(backbone: str, device: str, ckpt: str | None = None):
    if backbone == "navit":
        if ckpt is None:
            raise ValueError("--ckpt is required for --backbone navit")
        return _make_navit(ckpt, device)
    elif backbone == "lejepa":
        if ckpt is None:
            raise ValueError("--ckpt is required for --backbone lejepa")
        return _make_lejepa(ckpt, device)
    elif backbone == "chexagent":
        return _make_chexagent(device)
    elif backbone == "medsiglip":
        return _make_medsiglip(device)
    elif backbone == "biomed_clip":
        return _make_biomed_clip(device)
    elif backbone == "raddino":
        return _make_raddino(device)
    elif backbone == "medimageinsights":
        return _make_medimageinsights(device)
    elif backbone == "eva_x":
        return _make_eva_x(device)
    elif backbone == "dinov3":
        return _make_dinov3(device)
    elif backbone == "chexfound":
        return _make_chexfound(device)
    elif backbone == "siglip2":
        return _make_siglip2(device)
    elif backbone == "ark_plus":
        return _make_ark_plus(device)
    elif backbone == "medical_mae":
        return _make_medical_mae(device)
    else:
        raise ValueError(f"Unknown backbone: {backbone!r}")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def build_split_datasets(transform, cache_dir: str):
    """Return (train_ds, test_ds) using TAIX-Ray predefined split."""
    from radharmony.dataset.taix_ray import TAIXRay512Dataset

    taix = paths(TAIX_KEY)
    img_dir = taix["base_image_dir"]

    # Harmonize once against the full CSV to get the split column.
    full_ds = TAIXRay512Dataset(
        base_image_dir=img_dir,
        csv_path=taix["csv_path"],
        transform=transform,
        output_cls=True,
        label_mode="binary",
        cache_dir=os.path.join(cache_dir, "full"),
    )
    df = full_ds.get_harmonized_df()

    if "split" not in df.columns:
        raise ValueError(
            "Harmonized DataFrame has no 'split' column. "
            "Make sure annotation.csv contains a 'Split' column."
        )

    counts = df["split"].value_counts().to_dict()
    print(
        f"  TAIX-Ray: {len(df):,} total  "
        f"train={counts.get('train',0):,}  "
        f"val={counts.get('val',0):,}  "
        f"test={counts.get('test',0):,}"
    )

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    train_ds = TAIXRay512Dataset(
        base_image_dir=img_dir,
        harmonized_df=train_df,
        transform=transform,
        output_cls=True,
        label_mode="binary",
        cache_dir=os.path.join(cache_dir, "train"),
    )
    test_ds = TAIXRay512Dataset(
        base_image_dir=img_dir,
        harmonized_df=test_df,
        transform=transform,
        output_cls=True,
        label_mode="binary",
        cache_dir=os.path.join(cache_dir, "test"),
    )
    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"TAIX-Ray Linear Probe  |  backbone={args.backbone}  |  device={args.device}")
    if args.ckpt is not None:
        _display_run_id = args.run_id or os.path.splitext(os.path.basename(args.ckpt))[0]
        print(f"  run_id: {_display_run_id}")
    print(f"{'='*60}\n")

    # For checkpoint-based backbones use the checkpoint stem as the run identifier
    # so results are traceable to the exact model file (e.g. mimic-cls-ij1-vic-s-scratch-dp01).
    if args.run_id is not None:
        run_id = args.run_id
    elif args.ckpt is not None:
        run_id = os.path.splitext(os.path.basename(args.ckpt))[0]
    else:
        run_id = args.backbone

    cache_dir = os.path.join(args.cache_dir, run_id)
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"[1/4] Building backbone ({args.backbone}) ...")
    transform, encoder = make_backbone(args.backbone, args.device, args.ckpt)

    print(f"[2/4] Loading TAIX-Ray train/test splits ...")
    train_ds, test_ds = build_split_datasets(transform, cache_dir)

    embedding_cache = os.path.join(cache_dir, "embeddings", "taix")

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

    # Pretty-print AUROC summary
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
