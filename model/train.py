"""LeJEPA 2D pretraining on chest X-ray datasets via RadHarmony."""

from __future__ import annotations

import argparse
import os

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from monai.utils.enums import TraceKeys
from torch.utils.data import ConcatDataset, DataLoader

# Allow MONAI's TraceKeys enum through PyTorch 2.6+ weights_only checkpoint loading
torch.serialization.add_safe_globals([TraceKeys])

from radharmony.dataset import (
    CheXpertDataset,
    ChestXray14Dataset,
    MIMICCXRDataset,
    RadiologyTransform2D,
)

from datasets import LeJEPAWrapper, multicrop_collate_fn, subsample_dataset
from modules import LeJEPA2D
from transforms import EvalMultiCropTransform, MultiCropTransform


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LeJEPA 2D pretraining")

    # Data paths
    g = p.add_argument_group("data")
    g.add_argument("--chexpert_dir", type=str, default=None)
    g.add_argument("--chexpert_csv", type=str, default=None)
    g.add_argument("--mimic_dir", type=str, default=None)
    g.add_argument("--mimic_csv", type=str, default=None)
    g.add_argument(
        "--mimic_label_csv",
        type=str,
        default=None,
        help="Path to mimic-cxr-2.0.0-chexpert.csv for CheXpert labels",
    )
    g.add_argument("--chestxray14_dir", type=str, default=None)
    g.add_argument("--chestxray14_csv", type=str, default=None)
    g.add_argument("--cache_dir", type=str, default="./cache")
    g.add_argument(
        "--probe_label",
        type=str,
        default="pneumothorax",
        help="Target label for the online probe",
    )

    # Scaling study
    g = p.add_argument_group("scaling")
    g.add_argument("--sample_fraction", type=float, default=1.0)
    g.add_argument("--sample_count", type=int, default=None)
    g.add_argument("--seed", type=int, default=42)

    # Model
    g = p.add_argument_group("model")
    g.add_argument(
        "--model_name", type=str, default="naflexvit_base_patch16_gap.e300_s576_in1k"
    )
    g.add_argument("--img_size", type=int, default=256)
    g.add_argument("--proj_dim", type=int, default=128)
    g.add_argument("--lam", type=float, default=0.025)
    g.add_argument("--lam_warmup", type=int, default=6000)
    g.add_argument("--global_views", type=int, default=2)
    g.add_argument("--local_views", type=int, default=8)
    g.add_argument("--gradient_checkpointing", action="store_true")

    # Training
    g = p.add_argument_group("training")
    g.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Max epochs (ignored if --max_steps is set)",
    )
    g.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Fixed step budget; overrides --epochs for scaling studies",
    )
    g.add_argument("--batch_size", type=int, default=32)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--weight_decay", type=float, default=0.05)
    g.add_argument("--num_workers", type=int, default=8)
    g.add_argument("--devices", type=str, default="0")
    g.add_argument("--precision", type=str, default="bf16-mixed")
    g.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate before optimizer step",
    )

    # Checkpointing
    g = p.add_argument_group("checkpointing")
    g.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Resume full training state from checkpoint",
    )
    g.add_argument(
        "--init_ckpt",
        type=str,
        default=None,
        help="Load model weights only (fresh optimizer, scheduler, epoch counter) — for resolution adaptation",
    )
    g.add_argument("--output_dir", type=str, default="./weights")

    # WandB
    g = p.add_argument_group("wandb")
    g.add_argument("--project", type=str, default="LeJEPA-2D")
    g.add_argument("--entity", type=str, default=None)
    g.add_argument("--test_name", type=str, default="lejepa2d")

    return p.parse_args()


# -----------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------


def _make_base_transform(img_size: int):
    """RadHarmony transform: load → normalize → resize → pad → output img+cls."""
    return RadiologyTransform2D(
        img_size=img_size, output_keys={"img", "cls"}
    ).get_transform()


def _build_datasets(args) -> tuple[list, list, list[str]]:
    """Instantiate RadHarmony datasets and return train/val PersistentDatasets."""
    transform = _make_base_transform(args.img_size)
    train_sets: list = []
    val_sets: list = []
    names: list[str] = []

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

    for name, DatasetCls, kwargs in dataset_configs:
        ds = DatasetCls(**kwargs)
        target_col = next(
            (c for c in ds.LABEL_COLS if c.lower() == args.probe_label.lower()),
            None,
        )
        if target_col is None:
            print(f"  {name}: skipping — '{args.probe_label}' not in LABEL_COLS")
            continue
        ds.LABEL_COLS = [target_col]
        df = ds.get_harmonized_df()
        if target_col not in df.columns:
            print(
                f"  {name}: skipping — '{target_col}' column missing from harmonized DataFrame (label file not found?)"
            )
            continue
        ds.verify_images()
        train_rh, val_rh = ds.get_datasets(n_splits=10, num_cores=4)
        train_sets.append(train_rh)
        val_sets.append(val_rh)
        names.append(name)
        print(f"  {name}: train={len(train_rh):,}  val={len(val_rh):,}")

    return train_sets, val_sets, names


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse GPU list
    devices = [int(d) for d in args.devices.split(",")]

    print("Building datasets …")
    train_sets, val_sets, names = _build_datasets(args)
    if not train_sets:
        raise SystemExit("No dataset directories provided. Use --chexpert_dir, etc.")

    # Wrap with multi-crop transforms
    train_crop = MultiCropTransform(
        num_global_views=args.global_views,
        num_local_views=args.local_views,
        img_size=args.img_size,
    )
    val_crop = EvalMultiCropTransform(img_size=args.img_size)

    wrapped_train = [LeJEPAWrapper(rh, train_crop) for rh in train_sets]
    wrapped_val = [LeJEPAWrapper(rh, val_crop) for rh in val_sets]

    combined_train = (
        ConcatDataset(wrapped_train) if len(wrapped_train) > 1 else wrapped_train[0]
    )
    combined_val = (
        ConcatDataset(wrapped_val) if len(wrapped_val) > 1 else wrapped_val[0]
    )

    # Scaling study — subsample training set
    if args.sample_count is not None:
        combined_train = subsample_dataset(
            combined_train, count=args.sample_count, seed=args.seed
        )
        print(
            f"Subsampled to {len(combined_train):,} training samples (count={args.sample_count})"
        )
    elif args.sample_fraction < 1.0:
        combined_train = subsample_dataset(
            combined_train, fraction=args.sample_fraction, seed=args.seed
        )
        print(
            f"Subsampled to {len(combined_train):,} training samples (fraction={args.sample_fraction})"
        )

    print(f"Total: train={len(combined_train):,}  val={len(combined_val):,}")

    train_loader = DataLoader(
        combined_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=multicrop_collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        combined_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=multicrop_collate_fn,
        pin_memory=True,
        persistent_workers=False,
    )

    # Model
    model = LeJEPA2D(
        model_name=args.model_name,
        img_size=args.img_size,
        proj_dim=args.proj_dim,
        lam=args.lam,
        lam_warmup_steps=args.lam_warmup,
        num_global_views=args.global_views,
        num_local_views=args.local_views,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Load weights only (for resolution adaptation / fine-tuning)
    if args.init_ckpt:
        print(f"Loading model weights from {args.init_ckpt} (fresh training state)")
        ckpt = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename=f"{args.test_name}/epoch_{{epoch}}_loss{{val/pred_loss:.4f}}",
        auto_insert_metric_name=False,
        monitor="val/pred_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    # Logger
    wandb_logger = WandbLogger(
        name=args.test_name,
        project=args.project,
        entity=args.entity,
        save_dir=args.output_dir,
        config=vars(args),
    )

    # Trainer
    strategy = "ddp" if len(devices) > 1 else "auto"
    trainer_kwargs = dict(
        devices=devices,
        accelerator="gpu",
        strategy=strategy,
        precision=args.precision,
        callbacks=[checkpoint_cb],
        logger=wandb_logger,
        log_every_n_steps=10,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    if args.max_steps > 0:
        trainer_kwargs["max_steps"] = args.max_steps
    else:
        trainer_kwargs["max_epochs"] = args.epochs

    trainer = L.Trainer(**trainer_kwargs)

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)


if __name__ == "__main__":
    main()
