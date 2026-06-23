"""Linear probe over frozen dense backbone features.

Trains a 1×1 ``nn.Conv2d`` head with bilinear upsampling on the frozen patch
features produced by a segmentation-mode backbone. The head is the only
trainable component, matching the "linear probe" protocol used for
classification.

- **Loss**: ``nn.CrossEntropyLoss`` with optional per-class weights.
- **Split modes**: k-fold (``GroupKFold`` on ``patient_id``) or fixed-split
  with ``n_seeds`` × ``n_bootstrap`` variance estimation over the test set.
- **Output**: per-(class, fold|seed, [bootstrap]) metric rows;
  ``label='macro_average'`` rows aggregate foreground classes.
- **Predictions**: when ``save_predictions=True`` and ``output_dir`` is set,
  per-image GT / Pred / Prob PNGs are written under
  ``<output_dir>/{GT,Pred,Prob}/<run-id>/``.
- **Saving**: ``store_final_model=True`` (or calling ``fit()``) trains a
  deployment head on all data and stores it as ``final_head_``; ``save_head``
  / ``load_head`` persist just the trainable head (the frozen backbone is
  rebuilt from its factory), and ``predict`` runs a head over any dataset,
  returning per-image masks (and, optionally, the input image for overlays).
"""

from __future__ import annotations

import copy
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .._registry import register_evaluator
from ..metrics.segmentation import (
    METRIC_KEYS,
    compute_metrics as compute_seg_metrics,
    macro_average,
)
from .base import BaseSegEvaluator


class _LinearProbeSegHead(nn.Module):
    """1×1 Conv2d on dense features + bilinear upsample to the mask size."""

    def __init__(self, in_channels: int, num_classes: int, out_size: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.out_size = out_size

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        logits = self.conv(feats)
        return F.interpolate(
            logits,
            size=(self.out_size, self.out_size),
            mode="bilinear",
            align_corners=False,
        )


@register_evaluator("linear_probe_seg")
class LinearProbeSegEvaluator(BaseSegEvaluator):
    """Frozen-feature segmentation linear probe (1×1 Conv2d head)."""

    def __init__(
        self,
        image_encoder,
        *,
        n_folds: int = 5,
        n_train_samples: list[int] | None = None,
        epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        class_weights: list[float] | None = None,
        save_predictions: bool = False,
        store_final_model: bool = False,
        early_stop_metric: str = "dice",
        early_stop_patience: int = 5,
        val_fraction: float = 0.1,
        **base_kwargs,
    ):
        super().__init__(image_encoder, **base_kwargs)
        self.n_folds = n_folds
        self.n_train_samples = n_train_samples
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.class_weights = class_weights
        self.save_predictions = save_predictions
        self.early_stop_metric = early_stop_metric
        self.early_stop_patience = early_stop_patience
        self.val_fraction = val_fraction
        # Deployment head trained on all data (set by fit / store_final_model).
        self.store_final_model = store_final_model
        self.final_head_: nn.Module | None = None
        self.final_feat_dim_: int | None = None
        self.final_out_size_: int | None = None

    # ── Construction helpers ──────────────────────────────────────────────────
    def _make_head(self, feat_dim: int, out_size: int) -> nn.Module:
        return _LinearProbeSegHead(feat_dim, self.num_classes, out_size).to(self.device)

    def _loss_fn(self) -> nn.Module:
        if self.class_weights is not None:
            w = torch.tensor(self.class_weights, dtype=torch.float32, device=self.device)
            return nn.CrossEntropyLoss(weight=w)
        return nn.CrossEntropyLoss()

    def _mask_to_target(self, mask: torch.Tensor) -> torch.Tensor:
        """Convert a batch mask to class-index targets ``[B, H, W]`` (int64).

        Accepts ``[B, 1, H, W]`` / ``[B, H, W]`` single-channel masks
        (any value range — {0,1}, {0,255}, etc.) or ``[B, C, H, W]`` one-hot.
        Single-channel masks are rescaled to ``[0, num_classes-1]`` by
        dividing by the per-batch max — so {0, 255} binary PNGs and
        proper {0, 1, ..., K-1} indices both land on the right targets.
        """
        if mask.ndim == 4 and mask.shape[1] > 1:
            return mask.argmax(dim=1).long()
        if mask.ndim == 4 and mask.shape[1] == 1:
            m = mask.squeeze(1).float()
        elif mask.ndim == 3:
            m = mask.float()
        else:
            raise ValueError(
                f"Unexpected mask shape {tuple(mask.shape)}; "
                f"expected [B,1,H,W], [B,C,H,W], or [B,H,W]."
            )
        m_max = float(m.max().item())
        if m_max > 0:
            m = m / m_max * (self.num_classes - 1)
        return m.round().long().clamp(0, self.num_classes - 1)

    # ── Train / predict loops ────────────────────────────────────────────────
    def _train_one(
        self, head: nn.Module, train_loader, val_loader=None
    ) -> nn.Module:
        criterion = self._loss_fn()
        optim = torch.optim.AdamW(
            head.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.epochs)
        self._freeze_encoder()

        best_state = None
        best_metric = -float("inf")
        patience = 0

        pbar = tqdm(range(self.epochs), desc="training", unit="epoch")
        for _ in pbar:
            head.train()
            running, n_batches = 0.0, 0
            for batch in train_loader:
                imgs = batch["img"].to(self.device, non_blocking=True)
                target = self._mask_to_target(batch["mask"]).to(
                    self.device, non_blocking=True
                )
                with torch.no_grad(), self._autocast_cm():
                    feats = self.image_encoder(imgs)
                logits = head(feats.float())
                loss = criterion(logits, target)
                optim.zero_grad()
                loss.backward()
                optim.step()
                running += loss.item()
                n_batches += 1
            sched.step()

            postfix = {"loss": f"{running / max(n_batches, 1):.4f}"}
            if val_loader is not None:
                val_metric = self._val_metric(head, val_loader)
                postfix[self.early_stop_metric] = f"{val_metric:.4f}"
                if val_metric > best_metric:
                    best_metric = val_metric
                    best_state = copy.deepcopy(head.state_dict())
                    patience = 0
                else:
                    patience += 1
                    if patience >= self.early_stop_patience:
                        pbar.set_postfix(**postfix)
                        break
            pbar.set_postfix(**postfix)

        if best_state is not None:
            head.load_state_dict(best_state)
        return head

    @torch.no_grad()
    def _val_metric(self, head: nn.Module, loader) -> float:
        """Macro foreground ``early_stop_metric`` across the val loader.

        Background (class 0) is excluded so the signal tracks the actual
        segmentation target — otherwise a head that predicts all-background
        scores near-perfect on most CXR mask datasets.
        """
        head.eval()
        self._freeze_encoder()
        per_image_panels: list[dict[int, dict]] = []
        for batch in loader:
            imgs = batch["img"].to(self.device, non_blocking=True)
            with self._autocast_cm():
                feats = self.image_encoder(imgs)
            logits = head(feats.float())
            preds = logits.argmax(dim=1).cpu().numpy()
            y = self._mask_to_target(batch["mask"]).cpu().numpy()
            for i in range(preds.shape[0]):
                per_image_panels.append(
                    compute_seg_metrics(y[i], preds[i], self.num_classes)
                )
        if not per_image_panels:
            return -float("inf")
        fg = range(1, self.num_classes) if self.num_classes > 1 else range(self.num_classes)
        vals: list[float] = []
        for panel in per_image_panels:
            class_vals = [panel[c].get(self.early_stop_metric, np.nan) for c in fg]
            class_vals = [v for v in class_vals if not np.isnan(v)]
            if class_vals:
                vals.append(float(np.mean(class_vals)))
        return float(np.mean(vals)) if vals else -float("inf")

    @torch.no_grad()
    def _predict(
        self,
        head: nn.Module,
        loader,
        image_ids: list[str],
    ) -> list[dict]:
        """Return per-image dicts ``{image_id, y_true, y_pred, prob}``."""
        head.eval()
        self._freeze_encoder()
        out: list[dict] = []
        cursor = 0
        for batch in loader:
            imgs = batch["img"].to(self.device, non_blocking=True)
            with self._autocast_cm():
                feats = self.image_encoder(imgs)
            logits = head(feats.float())
            probs = torch.softmax(logits, dim=1).float().cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            y = self._mask_to_target(batch["mask"]).cpu().numpy()
            for i in range(preds.shape[0]):
                out.append({
                    "image_id": image_ids[cursor + i],
                    "y_true": y[i],
                    "y_pred": preds[i],
                    "prob": probs[i],
                })
            cursor += preds.shape[0]
        return out

    # ── Scoring helpers ──────────────────────────────────────────────────────
    def _per_image_panels(self, preds: list[dict]) -> list[dict[int, dict]]:
        return [
            compute_seg_metrics(p["y_true"], p["y_pred"], self.num_classes)
            for p in preds
        ]

    def _score_rows(
        self, per_image: list[dict[int, dict]], tmpl: dict
    ) -> list[dict]:
        rows = []
        for c in range(self.num_classes):
            row = dict(tmpl, label=f"class_{c}")
            for m in METRIC_KEYS:
                vals = [pi[c][m] for pi in per_image]
                row[m] = float(np.nanmean(vals)) if vals else np.nan
            rows.append(row)
        return rows

    def _bootstrap_rows(
        self, per_image: list[dict[int, dict]], tmpl: dict, seed: int
    ) -> list[dict]:
        if self.n_bootstrap <= 0 or not per_image:
            return []
        rng = np.random.default_rng(seed)
        n = len(per_image)
        rows: list[dict] = []
        for b in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            tmpl_b = dict(tmpl, bootstrap=b)
            for c in range(self.num_classes):
                row = dict(tmpl_b, label=f"class_{c}")
                for m in METRIC_KEYS:
                    vals = [per_image[i][c][m] for i in idx]
                    row[m] = float(np.nanmean(vals)) if vals else np.nan
                rows.append(row)
        return rows

    # ── Optional PNG dumps ───────────────────────────────────────────────────
    def _save_predictions(self, preds: list[dict], run_id: str) -> None:
        if not self.save_predictions or self.output_dir is None:
            return
        roots = {k: os.path.join(self.output_dir, k, run_id) for k in ("GT", "Pred", "Prob")}
        for d in roots.values():
            os.makedirs(d, exist_ok=True)
        scale = 255 // max(self.num_classes - 1, 1)
        for p in preds:
            fn = p["image_id"]
            Image.fromarray((p["y_true"] * scale).astype(np.uint8)).save(
                os.path.join(roots["GT"], f"{fn}.png")
            )
            Image.fromarray((p["y_pred"] * scale).astype(np.uint8)).save(
                os.path.join(roots["Pred"], f"{fn}.png")
            )
            # Foreground prob: class 1 for binary; max non-bg prob otherwise.
            if self.num_classes == 2:
                fg = p["prob"][1]
            else:
                fg = p["prob"][1:].max(axis=0)
            Image.fromarray((fg * 255).astype(np.uint8)).save(
                os.path.join(roots["Prob"], f"{fn}.png")
            )

    # ── Split helpers ────────────────────────────────────────────────────────
    def _iter_n_train(self, n_available: int) -> list[int]:
        if self.n_train_samples is None:
            return [-1]
        return [min(n, n_available) for n in self.n_train_samples]

    @staticmethod
    def _subsample(n_available: int, n_train: int, rng) -> np.ndarray:
        if n_train == -1 or n_train >= n_available:
            return np.arange(n_available)
        return rng.choice(n_available, size=n_train, replace=False)

    def _carve_inner_val(
        self, sub: np.ndarray, rng, patient_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Carve a patient-grouped inner val slice from a train index array.

        Returns ``(inner_train_idx, inner_val_idx)``. When ``val_fraction``
        is zero, the subsample is too small, or all rows share one patient,
        ``inner_val_idx`` is ``None`` and the full ``sub`` is returned as
        train.
        """
        if self.val_fraction <= 0.0 or len(sub) < 2:
            return sub, None
        sub_pids = patient_ids[sub]
        if len(np.unique(sub_pids)) < 2:
            return sub, None
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=self.val_fraction,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        inner_train_pos, inner_val_pos = next(
            gss.split(np.arange(len(sub)), groups=sub_pids)
        )
        return sub[inner_train_pos], sub[inner_val_pos]

    @staticmethod
    def _patient_ids(ds) -> np.ndarray:
        df = ds._get_harmonized_df().dropna(subset=["image_path"])
        return df["patient_id"].astype(str).values

    @staticmethod
    def _image_ids(ds) -> list[str]:
        df = ds._get_harmonized_df().dropna(subset=["image_path"])
        return [
            os.path.splitext(os.path.basename(p))[0]
            for p in df["image_path"].astype(str).values
        ]

    # ── Final-model fit, persistence, and prediction ─────────────────────────
    def _head_config(self) -> dict:
        """Subclass-specific head hyper-params, recorded in checkpoints."""
        return {}

    def _fit_final_head(self, dataset) -> nn.Module:
        """Train one head on all of ``dataset`` (minus an inner-val slice for
        early stopping) and store it as ``self.final_head_``.

        This is the *deployment* head — trained on every available image
        rather than a fold's train split — so it is what ``save_head`` writes.
        """
        torch_ds = dataset.get_datasets(n_splits=None, num_cores=self.num_workers)
        feat_dim, out_size = self._probe_shapes(self._make_loader(torch_ds, shuffle=False))
        patient_ids = self._patient_ids(dataset)
        rng = np.random.default_rng(self.base_seed)
        inner_train_idx, inner_val_idx = self._carve_inner_val(
            np.arange(len(torch_ds)), rng, patient_ids
        )
        train_loader = self._make_loader(
            Subset(torch_ds, inner_train_idx.tolist()), shuffle=True, drop_last=True
        )
        val_loader = (
            self._make_loader(Subset(torch_ds, inner_val_idx.tolist()), shuffle=False)
            if inner_val_idx is not None
            else None
        )
        head = self._make_head(feat_dim, out_size)
        self._train_one(head, train_loader, val_loader)
        self.final_head_ = head
        self.final_feat_dim_ = feat_dim
        self.final_out_size_ = out_size
        return head

    def fit(self, dataset=None) -> nn.Module:
        """Train and store a deployment head without running the k-fold benchmark.

        Trains on ``dataset`` (defaults to the evaluator's ``dataset`` /
        ``train_dataset``), sets ``self.final_head_``, and returns it — so a
        single head can be saved (:meth:`save_head`) and reused
        (:meth:`predict`) without the full cross-validated ``evaluate()`` run.
        """
        ds = dataset or self.dataset or self.train_dataset
        if ds is None:
            raise ValueError(
                "fit() needs a dataset — pass one, or construct the evaluator "
                "with dataset=/train_dataset=."
            )
        return self._fit_final_head(ds)

    def inner_val_indices(self, dataset=None) -> np.ndarray:
        """Dataset indices of the early-stopping inner-val slice ``fit()`` holds out.

        Deterministic given ``base_seed`` and the dataset's patient grouping, so
        it reproduces the exact slice :meth:`_fit_final_head` carved — without
        retraining, and even after :meth:`load_head` (which never recomputes it).
        These images are never in the gradient-training set — only used to pick
        the early-stopping epoch — so they are the cheapest honest target for a
        qualitative prediction viz (mild optimism remains, since they drove model
        selection). Returns an empty array when ``val_fraction`` disables the slice.
        """
        ds = dataset or self.dataset or self.train_dataset
        if ds is None:
            raise ValueError(
                "inner_val_indices() needs a dataset — pass one, or construct the "
                "evaluator with dataset=/train_dataset=."
            )
        patient_ids = self._patient_ids(ds)
        rng = np.random.default_rng(self.base_seed)
        _, inner_val_idx = self._carve_inner_val(
            np.arange(len(patient_ids)), rng, patient_ids
        )
        return inner_val_idx if inner_val_idx is not None else np.empty(0, dtype=int)

    def save_head(self, path: str | None = None) -> str:
        """Persist ``self.final_head_`` (state dict + shape/config metadata).

        ``path`` defaults to ``<output_dir>/final_head.pt``. Only the trainable
        head is saved; the frozen backbone is rebuilt from its factory at load
        time, so the checkpoint is small. Returns the path written.
        """
        if self.final_head_ is None:
            raise RuntimeError(
                "No head to save — call fit() or evaluate() with "
                "store_final_model=True first."
            )
        if path is None:
            if self.output_dir is None:
                raise ValueError("save_head needs a path, or set output_dir on the evaluator.")
            path = os.path.join(self.output_dir, "final_head.pt")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "state_dict": self.final_head_.state_dict(),
                "feat_dim": self.final_feat_dim_,
                "out_size": self.final_out_size_,
                "num_classes": self.num_classes,
                "evaluator": type(self).__name__,
                "head_config": self._head_config(),
            },
            path,
        )
        return path

    def load_head(self, path: str) -> nn.Module:
        """Rebuild a head from a :meth:`save_head` checkpoint and load its weights.

        Construct the evaluator with a matching configuration (same
        ``num_classes`` and head hyper-params); the checkpoint's shape metadata
        drives ``_make_head``. Sets and returns ``self.final_head_``.
        """
        ckpt = torch.load(path, map_location=self.device)
        if ckpt.get("num_classes") != self.num_classes:
            warnings.warn(
                f"checkpoint num_classes={ckpt.get('num_classes')} != evaluator "
                f"num_classes={self.num_classes}; load may fail.",
                stacklevel=2,
            )
        if ckpt.get("head_config", {}) != self._head_config():
            warnings.warn(
                f"checkpoint head_config={ckpt.get('head_config')} != evaluator "
                f"head_config={self._head_config()}; construct the evaluator with "
                "matching head args.",
                stacklevel=2,
            )
        head = self._make_head(int(ckpt["feat_dim"]), int(ckpt["out_size"]))
        head.load_state_dict(ckpt["state_dict"])
        head.eval()
        self.final_head_ = head
        self.final_feat_dim_ = int(ckpt["feat_dim"])
        self.final_out_size_ = int(ckpt["out_size"])
        return head

    @torch.no_grad()
    def predict(
        self,
        dataset,
        *,
        head: nn.Module | None = None,
        indices=None,
        return_images: bool = False,
    ) -> list[dict]:
        """Run a trained head over ``dataset`` and return per-image results.

        Each dict has ``image_id``, ``y_true`` ``[H, W]`` (class indices),
        ``y_pred`` ``[H, W]``, and ``prob`` ``[C, H, W]``. With
        ``return_images=True`` an ``image`` key holds a grayscale ``[H, W]``
        view of the input — min-max normalized and resized to the mask grid —
        ready for overlay plots. ``indices`` restricts to a subset of the
        dataset (e.g. a held-out fold or a few examples); ``head`` defaults to
        ``self.final_head_``.
        """
        head = head if head is not None else self.final_head_
        if head is None:
            raise RuntimeError(
                "predict() needs a head — pass head=, or call fit()/load_head() first."
            )
        head.eval()
        self._freeze_encoder()
        torch_ds = dataset.get_datasets(n_splits=None, num_cores=self.num_workers)
        image_ids = self._image_ids(dataset)
        if indices is not None:
            indices = [int(i) for i in indices]
            torch_ds = Subset(torch_ds, indices)
            image_ids = [image_ids[i] for i in indices]
        loader = self._make_loader(torch_ds, shuffle=False)
        out: list[dict] = []
        cursor = 0
        for batch in loader:
            imgs = batch["img"].to(self.device, non_blocking=True)
            with self._autocast_cm():
                feats = self.image_encoder(imgs)
            logits = head(feats.float())
            prob = torch.softmax(logits, dim=1).float().cpu().numpy()
            pred = logits.argmax(dim=1).cpu().numpy()
            y = self._mask_to_target(batch["mask"]).cpu().numpy()
            disp = None
            if return_images:
                d = F.interpolate(
                    imgs.float(), size=logits.shape[-2:], mode="bilinear", align_corners=False
                )
                disp = d.mean(dim=1).cpu().numpy()
            for i in range(pred.shape[0]):
                rec = {
                    "image_id": image_ids[cursor + i],
                    "y_true": y[i],
                    "y_pred": pred[i],
                    "prob": prob[i],
                }
                if return_images:
                    g = disp[i]
                    rec["image"] = (g - g.min()) / (g.max() - g.min() + 1e-8)
                out.append(rec)
            cursor += pred.shape[0]
        return out

    # ── Evaluation entry points ──────────────────────────────────────────────
    def evaluate(self) -> pd.DataFrame:
        if self.split_mode == "kfold":
            return self._evaluate_kfold()
        return self._evaluate_fixed()

    def _evaluate_kfold(self) -> pd.DataFrame:
        torch_ds = self.dataset.get_datasets(n_splits=None, num_cores=self.num_workers)
        n_total = len(torch_ds)
        patient_ids = self._patient_ids(self.dataset)
        all_image_ids = self._image_ids(self.dataset)

        probe_loader = self._make_loader(torch_ds, shuffle=False)
        feat_dim, out_size = self._probe_shapes(probe_loader)

        rows: list[dict] = []
        rng = np.random.default_rng(self.base_seed)
        gkf = GroupKFold(n_splits=self.n_folds)
        for fold_i, (train_idx, test_idx) in enumerate(
            gkf.split(np.arange(n_total), groups=patient_ids)
        ):
            for n_train in self._iter_n_train(len(train_idx)):
                sub = train_idx[self._subsample(len(train_idx), n_train, rng)]
                inner_train_idx, inner_val_idx = self._carve_inner_val(
                    sub, rng, patient_ids
                )
                train_loader = self._make_loader(
                    Subset(torch_ds, inner_train_idx.tolist()),
                    shuffle=True,
                    drop_last=True,
                )
                val_loader = (
                    self._make_loader(
                        Subset(torch_ds, inner_val_idx.tolist()), shuffle=False
                    )
                    if inner_val_idx is not None
                    else None
                )
                test_loader = self._make_loader(
                    Subset(torch_ds, test_idx.tolist()), shuffle=False
                )

                head = self._make_head(feat_dim, out_size)
                self._train_one(head, train_loader, val_loader)

                test_image_ids = [all_image_ids[i] for i in test_idx.tolist()]
                preds = self._predict(head, test_loader, test_image_ids)
                per_image = self._per_image_panels(preds)

                tmpl = {
                    "n_train": n_train,
                    "fold": fold_i,
                    "seed": -1,
                    "bootstrap": -1,
                }
                rows.extend(self._score_rows(per_image, tmpl))
                self._save_predictions(preds, run_id=f"fold{fold_i}_n{n_train}")

        df = pd.DataFrame(rows)
        if self.store_final_model:
            self._fit_final_head(self.dataset)
        return macro_average(df)

    def _evaluate_fixed(self) -> pd.DataFrame:
        train_torch_ds = self.train_dataset.get_datasets(
            n_splits=None, num_cores=self.num_workers
        )
        test_torch_ds = self.test_dataset.get_datasets(
            n_splits=None, num_cores=self.num_workers
        )
        n_train_total = len(train_torch_ds)
        all_test_image_ids = self._image_ids(self.test_dataset)
        train_patient_ids = self._patient_ids(self.train_dataset)

        probe_loader = self._make_loader(train_torch_ds, shuffle=False)
        feat_dim, out_size = self._probe_shapes(probe_loader)

        rows: list[dict] = []
        for seed_i in range(self.n_seeds):
            seed = self.base_seed + seed_i
            rng = np.random.default_rng(seed)
            for n_train in self._iter_n_train(n_train_total):
                sub = self._subsample(n_train_total, n_train, rng)
                inner_train_idx, inner_val_idx = self._carve_inner_val(
                    sub, rng, train_patient_ids
                )
                train_loader = self._make_loader(
                    Subset(train_torch_ds, inner_train_idx.tolist()),
                    shuffle=True,
                    drop_last=True,
                )
                val_loader = (
                    self._make_loader(
                        Subset(train_torch_ds, inner_val_idx.tolist()), shuffle=False
                    )
                    if inner_val_idx is not None
                    else None
                )
                test_loader = self._make_loader(test_torch_ds, shuffle=False)

                head = self._make_head(feat_dim, out_size)
                self._train_one(head, train_loader, val_loader)

                preds = self._predict(head, test_loader, all_test_image_ids)
                per_image = self._per_image_panels(preds)

                tmpl = {
                    "n_train": n_train,
                    "fold": -1,
                    "seed": seed,
                    "bootstrap": -1,
                }
                rows.extend(self._score_rows(per_image, tmpl))
                rows.extend(
                    self._bootstrap_rows(
                        per_image,
                        tmpl,
                        self.bootstrap_seed + seed_i * 10_000 + max(n_train, 0),
                    )
                )
                self._save_predictions(preds, run_id=f"seed{seed}_n{n_train}")

        df = pd.DataFrame(rows)
        if self.store_final_model:
            self._fit_final_head(self.train_dataset)
        return macro_average(df)
