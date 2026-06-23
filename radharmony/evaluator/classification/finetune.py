"""Minimal full fine-tune classification evaluator.

Trains a classification head (optionally + the encoder) with AdamW +
cosine schedule under BCE loss. Deliberately minimal — heavy training
workflows (DDP, WandB, mixed-precision, MONAI ignite) stay outside of
radharmony. For large-scale experiments, use the reference scripts.

When ``freeze_backbone=True`` the encoder is never in the gradient tape
and we reduce to training a head over cached embeddings — much faster
and exactly equivalent.

Saving / reuse: ``store_final_model=True`` (or calling ``fit()``) trains a
deployment model on all data and stores it as ``final_encoder_`` /
``final_head_``; ``save_model`` / ``load_model`` persist the trainable
weights (the head always, plus the encoder when ``freeze_backbone=False``),
and ``predict`` runs the model over any dataset, returning per-image label
probabilities.
"""

from __future__ import annotations

import copy
import os
import warnings

import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Subset

from .._registry import register_evaluator
from ..metrics.classification import compute_metrics, macro_average, METRIC_KEYS
from .base import BaseClsEvaluator


@register_evaluator("finetune")
class FinetuneEvaluator(BaseClsEvaluator):
    """Minimal full fine-tune evaluator — single GPU, BCE / cross-entropy, AdamW + cosine LR."""

    def __init__(
        self,
        image_encoder,
        *,
        n_folds: int = 5,
        n_train_samples: list[int] | None = None,
        epochs: int = 20,
        lr: float = 1e-4,
        backbone_lr_scale: float = 0.1,
        weight_decay: float = 0.05,
        freeze_backbone: bool = False,
        head: nn.Module | None = None,
        loss: str = "bce",
        early_stop_metric: str = "auroc",
        early_stop_patience: int = 3,
        val_fraction: float = 0.1,
        store_final_model: bool = False,
        **base_kwargs,
    ):
        super().__init__(image_encoder, **base_kwargs)
        self.n_folds = n_folds
        self.n_train_samples = n_train_samples
        self.epochs = epochs
        self.lr = lr
        self.backbone_lr_scale = backbone_lr_scale
        self.weight_decay = weight_decay
        self.freeze_backbone = freeze_backbone
        self.head = head
        self.loss_name = loss
        self.early_stop_metric = early_stop_metric
        self.early_stop_patience = early_stop_patience
        self.val_fraction = val_fraction
        # Deployment model trained on all data (set by fit / store_final_model).
        self.store_final_model = store_final_model
        self.final_encoder_: nn.Module | None = None
        self.final_head_: nn.Module | None = None
        self.final_feat_dim_: int | None = None
        self.final_num_labels_: int | None = None
        self.final_labels_: list[str] | None = None

    def _make_head(self, feat_dim: int, num_labels: int) -> nn.Module:
        if self.head is not None:
            return copy.deepcopy(self.head).to(self.device)
        return nn.Linear(feat_dim, num_labels).to(self.device)

    def _loss_fn(self):
        if self.loss_name == "bce":
            return nn.BCEWithLogitsLoss()
        if self.loss_name == "ce":
            return nn.CrossEntropyLoss()
        raise ValueError(f"Unknown loss {self.loss_name!r}; expected 'bce' or 'ce'.")

    def _probe_feat_dim(self, sample_loader) -> int:
        """Run one batch through the encoder to discover D."""
        self._freeze_encoder()
        batch = next(iter(sample_loader))
        imgs = batch["img"].to(self.device)
        with torch.no_grad(), self._autocast_cm():
            feats = self.image_encoder(imgs)
        return int(feats.shape[1])

    def _train_one(
        self,
        encoder: nn.Module,
        head: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        num_labels: int,
    ) -> nn.Module:
        criterion = self._loss_fn()
        if self.freeze_backbone:
            param_groups = [{"params": list(head.parameters())}]
        else:
            param_groups = [
                {"params": list(head.parameters()), "lr": self.lr},
                {"params": list(encoder.parameters()), "lr": self.lr * self.backbone_lr_scale},
            ]
        optim = torch.optim.AdamW(param_groups, lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.epochs)

        best_state = None
        best_metric = -float("inf")
        patience = 0

        pbar = tqdm(range(self.epochs), desc="training", unit="epoch")
        for _ in pbar:
            if self.freeze_backbone:
                encoder.eval()
            else:
                encoder.train()
            head.train()
            for batch in train_loader:
                imgs = batch["img"].to(self.device, non_blocking=True)
                cls = batch["cls"]
                if cls.ndim == 1:
                    cls = cls.view(-1, num_labels)
                cls = cls.float().to(self.device, non_blocking=True)
                optim.zero_grad()
                if self.freeze_backbone:
                    with torch.no_grad(), self._autocast_cm():
                        feats = encoder(imgs)
                else:
                    with self._autocast_cm():
                        feats = encoder(imgs)
                logits = head(feats.float())
                loss = criterion(logits, cls)
                loss.backward()
                optim.step()
            sched.step()

            if val_loader is None:
                continue
            # Early stopping on val
            val_metric, val_loss = self._val_metric(encoder, head, val_loader, num_labels)
            pbar.set_postfix(
                val_loss=f"{val_loss:.4f}",
                **{self.early_stop_metric: f"{val_metric:.4f}"},
            )
            if val_metric > best_metric:
                best_metric = val_metric
                best_state = {
                    "encoder": copy.deepcopy(encoder.state_dict()),
                    "head": copy.deepcopy(head.state_dict()),
                }
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stop_patience:
                    break

        if best_state is not None:
            encoder.load_state_dict(best_state["encoder"])
            head.load_state_dict(best_state["head"])
        return head

    @torch.no_grad()
    def _val_metric(self, encoder, head, loader, num_labels) -> tuple[float, float]:
        """Return ``(early_stop_metric, val_loss)`` averaged over the val set."""
        criterion = self._loss_fn()
        encoder.eval()
        head.eval()
        probs_all, y_all, loss_sum, n_batches = [], [], 0.0, 0
        for batch in loader:
            imgs = batch["img"].to(self.device, non_blocking=True)
            cls = batch["cls"]
            if cls.ndim == 1:
                cls = cls.view(-1, num_labels)
            cls_gpu = cls.float().to(self.device, non_blocking=True)
            with self._autocast_cm():
                feats = encoder(imgs)
            logits = head(feats.float())
            loss_sum += criterion(logits, cls_gpu).item()
            n_batches += 1
            probs_all.append(torch.sigmoid(logits).float().cpu().numpy())
            y_all.append(cls.float().numpy())
        y = np.concatenate(y_all)
        p = np.concatenate(probs_all)
        vals = []
        for k in range(num_labels):
            if len(np.unique(y[:, k])) < 2:
                continue
            m = compute_metrics(y[:, k], p[:, k], self.threshold_strategy)
            if not np.isnan(m.get(self.early_stop_metric, np.nan)):
                vals.append(m[self.early_stop_metric])
        metric = float(np.mean(vals)) if vals else -float("inf")
        val_loss = loss_sum / n_batches if n_batches > 0 else float("nan")
        return metric, val_loss

    @torch.no_grad()
    def _predict(self, encoder, head, loader, num_labels) -> tuple[np.ndarray, np.ndarray]:
        encoder.eval()
        head.eval()
        probs_all, y_all = [], []
        for batch in loader:
            imgs = batch["img"].to(self.device, non_blocking=True)
            cls = batch["cls"]
            if cls.ndim == 1:
                cls = cls.view(-1, num_labels)
            with self._autocast_cm():
                feats = encoder(imgs)
            logits = head(feats.float())
            probs_all.append(torch.sigmoid(logits).float().cpu().numpy())
            y_all.append(cls.float().numpy())
        return np.concatenate(y_all), np.concatenate(probs_all)

    def _score_rows(self, probs, y_test, tmpl) -> list[dict]:
        rows = []
        for k, name in enumerate(self.active_labels):
            row = dict(tmpl, label=name)
            row.update(compute_metrics(y_test[:, k], probs[:, k], self.threshold_strategy))
            rows.append(row)
        return rows

    def _bootstrap_rows(self, probs, y_test, tmpl, seed) -> list[dict]:
        if self.n_bootstrap <= 0:
            return []
        rng = np.random.default_rng(seed)
        n = y_test.shape[0]
        rows = []
        for b in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            t = dict(tmpl, bootstrap=b)
            for k, name in enumerate(self.active_labels):
                row = dict(t, label=name)
                row.update(
                    compute_metrics(y_test[idx, k], probs[idx, k], self.threshold_strategy)
                )
                rows.append(row)
        return rows

    def _iter_n_train(self, n_available: int):
        if self.n_train_samples is None:
            return [-1]
        return [min(n, n_available) for n in self.n_train_samples]

    def _subsample(self, n_available, n_train, rng) -> np.ndarray:
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

    def _resolve_labels(self, ds) -> tuple[list[str], np.ndarray]:
        """Trigger label resolution (sorts LABEL_COLS) and return indices.

        Stores the resolved label index on ``self._label_idx`` so downstream
        helpers (e.g. ``_SliceLabelsLoader``) can slice batch["cls"] without
        having re-run extract_embeddings.
        """
        _ = ds.get_datasets(n_splits=None, num_cores=self.num_workers)
        all_labels = list(ds.LABEL_COLS)
        idx = self._resolve_label_indices(all_labels)
        self._label_idx = idx
        return all_labels, idx

    def _make_loader(self, torch_ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            torch_ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _sliced_labels(self, batch_cls: torch.Tensor, num_all: int) -> torch.Tensor:
        """Slice the full label tensor down to self.active_labels columns."""
        if batch_cls.ndim == 1:
            batch_cls = batch_cls.view(-1, num_all)
        return batch_cls[:, self._label_idx]

    def _build_encoder(self) -> nn.Module:
        """A fresh trainable encoder: the shared frozen one, or a deep copy.

        With ``freeze_backbone=True`` the encoder weights never change, so we
        reuse ``self.image_encoder`` (frozen). Otherwise each run trains its own
        copy so folds / seeds don't leak fine-tuned weights into one another.

        ``requires_grad`` is set explicitly on both branches — ``_probe_feat_dim``
        calls ``_freeze_encoder()`` on the shared encoder, and a naïve
        ``deepcopy`` would inherit that frozen state, silently degenerating
        full fine-tuning into head-only training.
        """
        encoder = (
            self.image_encoder
            if self.freeze_backbone
            else copy.deepcopy(self.image_encoder)
        ).to(self.device)
        for p in encoder.parameters():
            p.requires_grad_(not self.freeze_backbone)
        return encoder

    @staticmethod
    def _image_ids(ds) -> list[str]:
        df = ds._get_harmonized_df().dropna(subset=["image_path"])
        return [
            os.path.splitext(os.path.basename(p))[0]
            for p in df["image_path"].astype(str).values
        ]

    # ── Final-model fit, persistence, and prediction ─────────────────────────
    def _fit_final_model(self, dataset) -> nn.Module:
        """Train one encoder + head on all of ``dataset`` (minus an inner-val
        slice for early stopping) and store them as ``self.final_encoder_`` /
        ``self.final_head_``.

        This is the *deployment* model — trained on every available image
        rather than a fold's train split — so it is what ``save_model`` writes.
        """
        all_labels, _ = self._resolve_labels(dataset)
        num_active = len(self.active_labels)
        torch_ds = dataset.get_datasets(n_splits=None, num_cores=self.num_workers)
        df = dataset._get_harmonized_df().dropna(subset=["image_path"])
        patient_ids = df["patient_id"].astype(str).values

        feat_dim = self._probe_feat_dim(self._make_loader(torch_ds, shuffle=False))
        rng = np.random.default_rng(self.base_seed)
        inner_train_idx, inner_val_idx = self._carve_inner_val(
            np.arange(len(torch_ds)), rng, patient_ids
        )
        train_loader = self._make_loader(
            Subset(torch_ds, inner_train_idx.tolist()), shuffle=True
        )
        val_loader = (
            self._make_loader(Subset(torch_ds, inner_val_idx.tolist()), shuffle=False)
            if inner_val_idx is not None
            else None
        )
        wrapped_train = _SliceLabelsLoader(train_loader, self._label_idx, len(all_labels))
        wrapped_val = (
            _SliceLabelsLoader(val_loader, self._label_idx, len(all_labels))
            if val_loader is not None
            else None
        )

        encoder = self._build_encoder()
        head = self._make_head(feat_dim, num_active)
        self._train_one(encoder, head, wrapped_train, wrapped_val, num_active)

        self.final_encoder_ = encoder
        self.final_head_ = head
        self.final_feat_dim_ = feat_dim
        self.final_num_labels_ = num_active
        self.final_labels_ = list(self.active_labels)
        return head

    def fit(self, dataset=None) -> nn.Module:
        """Train and store a deployment model without running the benchmark.

        Trains on ``dataset`` (defaults to the evaluator's ``dataset`` /
        ``train_dataset``), sets ``self.final_encoder_`` / ``self.final_head_``,
        and returns the head — so a single model can be saved
        (:meth:`save_model`) and reused (:meth:`predict`) without the full
        cross-validated ``evaluate()`` run.
        """
        ds = dataset or self.dataset or self.train_dataset
        if ds is None:
            raise ValueError(
                "fit() needs a dataset — pass one, or construct the evaluator "
                "with dataset=/train_dataset=."
            )
        return self._fit_final_model(ds)

    def save_model(self, path: str | None = None) -> str:
        """Persist the trained head (and encoder, if fine-tuned) + metadata.

        ``path`` defaults to ``<output_dir>/final_model.pt``. The head state
        dict is always saved; the encoder state dict is saved only when
        ``freeze_backbone=False`` (a frozen encoder is rebuilt unchanged from
        its factory at load time). Returns the path written.
        """
        if self.final_head_ is None:
            raise RuntimeError(
                "No model to save — call fit() or evaluate() with "
                "store_final_model=True first."
            )
        if path is None:
            if self.output_dir is None:
                raise ValueError("save_model needs a path, or set output_dir on the evaluator.")
            path = os.path.join(self.output_dir, "final_model.pt")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        ckpt = {
            "head_state_dict": self.final_head_.state_dict(),
            "feat_dim": self.final_feat_dim_,
            "num_labels": self.final_num_labels_,
            "labels": self.final_labels_,
            "freeze_backbone": self.freeze_backbone,
            "evaluator": type(self).__name__,
        }
        if not self.freeze_backbone and self.final_encoder_ is not None:
            ckpt["encoder_state_dict"] = self.final_encoder_.state_dict()
        torch.save(ckpt, path)
        return path

    def load_model(self, path: str) -> nn.Module:
        """Rebuild encoder + head from a :meth:`save_model` checkpoint.

        Construct the evaluator with a matching configuration (same
        ``image_encoder`` factory, ``head=``, and ``freeze_backbone=``); the
        checkpoint's shape metadata drives ``_make_head``. Sets
        ``self.final_encoder_`` / ``self.final_head_`` and returns the head.
        """
        ckpt = torch.load(path, map_location=self.device)
        if ckpt.get("freeze_backbone") != self.freeze_backbone:
            warnings.warn(
                f"checkpoint freeze_backbone={ckpt.get('freeze_backbone')} != "
                f"evaluator freeze_backbone={self.freeze_backbone}; encoder weights "
                "may not match.",
                stacklevel=2,
            )
        head = self._make_head(int(ckpt["feat_dim"]), int(ckpt["num_labels"]))
        head.load_state_dict(ckpt["head_state_dict"])
        head.eval()
        encoder = self._build_encoder()
        if "encoder_state_dict" in ckpt:
            encoder.load_state_dict(ckpt["encoder_state_dict"])
        encoder.eval()

        self.final_encoder_ = encoder
        self.final_head_ = head
        self.final_feat_dim_ = int(ckpt["feat_dim"])
        self.final_num_labels_ = int(ckpt["num_labels"])
        self.final_labels_ = ckpt.get("labels")
        return head

    @torch.no_grad()
    def predict(
        self,
        dataset,
        *,
        encoder: nn.Module | None = None,
        head: nn.Module | None = None,
        indices=None,
    ) -> list[dict]:
        """Run a trained model over ``dataset`` and return per-image results.

        Each dict has ``image_id``, ``y_true`` ``[L]`` (0/1 labels), and
        ``prob`` ``[L]`` (sigmoid probabilities), where the ``L`` columns are
        ``self.final_labels_`` (set at fit/load time). ``indices`` restricts to
        a subset of the dataset; ``encoder`` / ``head`` default to
        ``self.final_encoder_`` / ``self.final_head_``. The dataset's label
        columns are sliced to match the trained label set, so predict on a
        dataset whose ``LABEL_COLS`` cover the same labels.
        """
        head = head if head is not None else self.final_head_
        encoder = encoder if encoder is not None else self.final_encoder_
        if head is None or encoder is None:
            raise RuntimeError(
                "predict() needs a trained model — pass encoder=/head=, or call "
                "fit()/load_model() first."
            )
        all_labels, _ = self._resolve_labels(dataset)
        num_active = len(self.active_labels)
        if self.final_num_labels_ is not None and num_active != self.final_num_labels_:
            raise ValueError(
                f"dataset resolves to {num_active} active labels but the trained "
                f"model expects {self.final_num_labels_}; pass labels= so they match."
            )
        encoder.eval()
        head.eval()
        torch_ds = dataset.get_datasets(n_splits=None, num_cores=self.num_workers)
        image_ids = self._image_ids(dataset)
        if indices is not None:
            indices = [int(i) for i in indices]
            torch_ds = Subset(torch_ds, indices)
            image_ids = [image_ids[i] for i in indices]
        loader = _SliceLabelsLoader(
            self._make_loader(torch_ds, shuffle=False), self._label_idx, len(all_labels)
        )
        out: list[dict] = []
        cursor = 0
        for batch in loader:
            imgs = batch["img"].to(self.device, non_blocking=True)
            cls = batch["cls"]
            if cls.ndim == 1:
                cls = cls.view(-1, num_active)
            with self._autocast_cm():
                feats = encoder(imgs)
            probs = torch.sigmoid(head(feats.float())).float().cpu().numpy()
            y = cls.float().numpy()
            for i in range(probs.shape[0]):
                out.append({
                    "image_id": image_ids[cursor + i],
                    "y_true": y[i],
                    "prob": probs[i],
                })
            cursor += probs.shape[0]
        return out

    # ── Evaluation entry points ──────────────────────────────────────────────
    def evaluate(self) -> pd.DataFrame:
        if self.split_mode == "kfold":
            return self._evaluate_kfold()
        return self._evaluate_fixed()

    def _evaluate_kfold(self) -> pd.DataFrame:
        all_labels, _ = self._resolve_labels(self.dataset)
        num_active = len(self.active_labels)
        torch_ds = self.dataset.get_datasets(n_splits=None, num_cores=self.num_workers)
        n_total = len(torch_ds)

        _df = self.dataset._get_harmonized_df().dropna(subset=["image_path"])
        patient_ids = _df["patient_id"].astype(str).values

        probe_loader = self._make_loader(torch_ds, shuffle=False)
        feat_dim = self._probe_feat_dim(probe_loader)

        rows = []
        rng = np.random.default_rng(self.base_seed)
        gkf = GroupKFold(n_splits=self.n_folds)

        for fold_i, (train_idx, test_idx) in enumerate(gkf.split(np.arange(n_total), groups=patient_ids)):
            for n_train in self._iter_n_train(len(train_idx)):
                sub = train_idx[self._subsample(len(train_idx), n_train, rng)]
                inner_train_idx, inner_val_idx = self._carve_inner_val(
                    sub, rng, patient_ids
                )
                train_loader = self._make_loader(
                    Subset(torch_ds, inner_train_idx.tolist()), shuffle=True
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

                encoder = self._build_encoder()
                head = self._make_head(feat_dim, num_active)

                # Wrap the dataloaders so batch["cls"] is sliced to active labels.
                wrapped_train = _SliceLabelsLoader(train_loader, self._label_idx, len(all_labels))
                wrapped_val = (
                    _SliceLabelsLoader(val_loader, self._label_idx, len(all_labels))
                    if val_loader is not None
                    else None
                )
                wrapped_test = _SliceLabelsLoader(test_loader, self._label_idx, len(all_labels))

                self._train_one(encoder, head, wrapped_train, wrapped_val, num_active)
                y_true, probs = self._predict(encoder, head, wrapped_test, num_active)

                tmpl = {"n_train": n_train, "fold": fold_i, "seed": -1, "bootstrap": -1}
                rows.extend(self._score_rows(probs, y_true, tmpl))

        df = pd.DataFrame(rows)
        if self.store_final_model:
            self._fit_final_model(self.dataset)
        return macro_average(df)

    def _evaluate_fixed(self) -> pd.DataFrame:
        all_train, _ = self._resolve_labels(self.train_dataset)
        # Test dataset — resolve separately but expect identical label order.
        test_torch_ds = self.test_dataset.get_datasets(n_splits=None, num_cores=self.num_workers)
        train_torch_ds = self.train_dataset.get_datasets(n_splits=None, num_cores=self.num_workers)
        num_active = len(self.active_labels)
        n_train_total = len(train_torch_ds)

        _train_df = self.train_dataset._get_harmonized_df().dropna(subset=["image_path"])
        train_patient_ids = _train_df["patient_id"].astype(str).values

        probe_loader = self._make_loader(train_torch_ds, shuffle=False)
        feat_dim = self._probe_feat_dim(probe_loader)

        rows = []
        for seed_i in range(self.n_seeds):
            seed = self.base_seed + seed_i
            rng = np.random.default_rng(seed)
            for n_train in self._iter_n_train(n_train_total):
                sub = self._subsample(n_train_total, n_train, rng)
                inner_train_idx, inner_val_idx = self._carve_inner_val(
                    sub, rng, train_patient_ids
                )
                inner_train_loader = self._make_loader(
                    Subset(train_torch_ds, inner_train_idx.tolist()), shuffle=True
                )
                inner_val_loader = (
                    self._make_loader(
                        Subset(train_torch_ds, inner_val_idx.tolist()), shuffle=False
                    )
                    if inner_val_idx is not None
                    else None
                )
                test_loader = self._make_loader(test_torch_ds, shuffle=False)

                encoder = self._build_encoder()
                head = self._make_head(feat_dim, num_active)

                wrapped_train = _SliceLabelsLoader(
                    inner_train_loader, self._label_idx, len(all_train)
                )
                wrapped_val = (
                    _SliceLabelsLoader(inner_val_loader, self._label_idx, len(all_train))
                    if inner_val_loader is not None
                    else None
                )
                wrapped_test = _SliceLabelsLoader(
                    test_loader, self._label_idx, len(all_train)
                )

                self._train_one(encoder, head, wrapped_train, wrapped_val, num_active)
                y_true, probs = self._predict(encoder, head, wrapped_test, num_active)

                tmpl = {"n_train": n_train, "fold": -1, "seed": seed, "bootstrap": -1}
                rows.extend(self._score_rows(probs, y_true, tmpl))
                rows.extend(
                    self._bootstrap_rows(
                        probs, y_true, tmpl, self.bootstrap_seed + seed_i * 10_000 + max(n_train, 0)
                    )
                )

        df = pd.DataFrame(rows)
        if self.store_final_model:
            self._fit_final_model(self.train_dataset)
        return macro_average(df)


class _SliceLabelsLoader:
    """Wraps a DataLoader to slice ``batch['cls']`` down to the active label columns."""

    def __init__(self, loader, label_idx, num_all_labels):
        self.loader = loader
        self.label_idx = label_idx
        self.num_all_labels = num_all_labels

    def __iter__(self):
        for batch in self.loader:
            cls = batch["cls"]
            if cls.ndim == 1:
                cls = cls.view(-1, self.num_all_labels)
            batch = {**batch, "cls": cls[:, self.label_idx]}
            yield batch

    def __len__(self):
        return len(self.loader)
