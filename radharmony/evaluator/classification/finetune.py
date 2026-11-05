"""Minimal full fine-tune classification evaluator.

Trains a classification head (optionally + the encoder) with AdamW +
cosine schedule, with the BCE loss (default) or the CE loss (``loss="ce"``).
It runs under autocast with ``autocast_dtype``.
It is small on purpose. Heavy training workflows (DDP, WandB, MONAI ignite)
stay outside of radharmony. For large-scale experiments, use the reference scripts.

With ``freeze_backbone=True``, only the head trains. The encoder still
runs on every batch (under ``torch.no_grad()``). The features are not cached.

Saving and reuse: ``store_final_model=True`` (or calling ``fit()``) trains a
deployment model on all data and stores it as ``final_encoder_`` and
``final_head_``. ``save_model`` and ``load_model`` save and load the
trainable weights: the head always, and the encoder when
``freeze_backbone=False``. ``predict`` runs the model over any dataset and
returns per-image label probabilities.
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
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset

from .._registry import register_evaluator
from ..metrics.classification import compute_metrics, macro_average
from ..sampling import carve_val, group_ids, image_ids, seed_torch, RUN_FINAL, RUN_FIXED, RUN_KFOLD
from ._probe_utils import bootstrap_rows, label_order, pick_thresholds, score_rows
from .base import BaseClsEvaluator


@register_evaluator("finetune")
class FinetuneEvaluator(BaseClsEvaluator):
    """Minimal full fine-tune evaluator: single GPU, BCE / cross-entropy, AdamW + cosine LR."""

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
        # Fine-tune moves the encoder to the device and sets requires_grad on its weights.
        if not isinstance(image_encoder, nn.Module):
            raise TypeError(f"FinetuneEvaluator needs an nn.Module encoder, not {type(image_encoder).__name__}. "
                            "Use the encoder of a backbone recipe, or wrap your function in an nn.Module.")
        # A frozen backbone makes this a frozen-feature probe, so it L2-normalizes
        # the features by default, as the other probes do (D-17).
        if freeze_backbone:
            base_kwargs.setdefault("l2_normalize", True)
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
        if self.loss_name == "ce" and num_labels < 2:
            # A softmax over 1 logit is always 1, so the loss is 0 and nothing trains (D-47, V-E-27).
            raise ValueError("loss='ce' needs 2 or more label columns. Use loss='bce' for 1 label column.")
        if self.head is not None:
            return copy.deepcopy(self.head).to(self.device)
        return nn.Linear(feat_dim, num_labels).to(self.device)

    def _loss_fn(self):
        if self.loss_name == "bce":
            return nn.BCEWithLogitsLoss()
        if self.loss_name == "ce":
            return nn.CrossEntropyLoss()
        raise ValueError(f"Unknown loss {self.loss_name!r}; expected 'bce' or 'ce'.")

    def _head_input(self, feats: torch.Tensor) -> torch.Tensor:
        """Return the features for the head, L2-normalized when ``l2_normalize`` is set (D-17)."""
        feats = feats.float()
        return F.normalize(feats, dim=1) if self.l2_normalize else feats

    def _probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Turn logits into label scores, as the loss does (V-E-19).

        ``loss="ce"`` trains a softmax over the labels, so each row sums to 1.
        ``loss="bce"`` trains one sigmoid per label.
        """
        if self.loss_name == "ce":
            return torch.softmax(logits, dim=1)
        return torch.sigmoid(logits)

    def _masked_loss(self, logits: torch.Tensor, cls: torch.Tensor) -> torch.Tensor | None:
        """Return the loss over the labels that are 0 or 1. A NaN label adds no loss (D-18).

        BCE: the mean over the usable rows of each label, then the mean over the
        labels that have a usable row. With no NaN, this equals the plain BCE mean. CE: a row with a
        NaN label is dropped. Returns ``None`` when no label in the batch is usable.
        """
        valid = (cls == 0) | (cls == 1)
        if self.loss_name == "ce":
            keep = valid.all(dim=1)
            return self._loss_fn()(logits[keep], cls[keep]) if keep.any() else None
        if self.loss_name != "bce":
            self._loss_fn()  # raises ValueError for an unknown loss name
        if not valid.any():
            return None
        per = F.binary_cross_entropy_with_logits(
            logits, torch.where(valid, cls, torch.zeros_like(cls)), reduction="none"
        )
        n_valid = valid.sum(dim=0)
        per_label = (per * valid).sum(dim=0) / n_valid.clamp(min=1)
        return per_label.sum() / (n_valid > 0).sum()

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
                logits = head(self._head_input(feats))
                loss = self._masked_loss(logits, cls)
                if loss is None:
                    continue
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
            # -inf means no signal, for example no positive in the val part.
            # Such an epoch does not count for patience (V-E-16). If no epoch
            # has a signal, all epochs run and the last weights stay.
            if not np.isfinite(val_metric):
                continue
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
            logits = head(self._head_input(feats))
            loss = self._masked_loss(logits, cls_gpu)
            if loss is not None:
                loss_sum += loss.item()
                n_batches += 1
            probs_all.append(self._probs(logits).float().cpu().numpy())
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
            logits = head(self._head_input(feats))
            probs_all.append(self._probs(logits).float().cpu().numpy())
            y_all.append(cls.float().numpy())
        return np.concatenate(y_all), np.concatenate(probs_all)

    def _val_thresholds(self, encoder, head, val_loader, num_labels) -> np.ndarray:
        """Pick one threshold per label on the inner validation part (D-29)."""
        if val_loader is None:
            return pick_thresholds(None, None, self.threshold_strategy, num_labels)
        y_val, p_val = self._predict(encoder, head, val_loader, num_labels)
        return pick_thresholds(y_val, p_val, self.threshold_strategy, num_labels)

    def _iter_n_train(self, n_available: int):
        if self.n_train_samples is None:
            return [-1]
        # Keep the requested n (also above n_available): one summary row per point (V-E-21).
        return list(self.n_train_samples)

    def _subsample(self, n_available, n_train, rng) -> np.ndarray:
        if n_train == -1 or n_train >= n_available:
            return np.arange(n_available)
        return rng.choice(n_available, size=n_train, replace=False)

    def _carve_inner_val(
        self, sub: np.ndarray, rng, patient_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Carve a patient-grouped inner val slice from a train index array.

        Returns ``(inner_train_idx, inner_val_idx)``. The full ``sub`` is the
        train set and ``inner_val_idx`` is ``None`` in 3 cases. Either
        ``val_fraction`` is zero, or the subsample is too small, or all rows
        share one patient. The inner val part also picks the threshold (D-29).
        """
        return carve_val(sub, rng, patient_ids, self.val_fraction)

    def _resolve_labels(self, ds) -> tuple[list[str], np.ndarray]:
        """Trigger label resolution (sorts LABEL_COLS) and return indices.

        Stores the resolved label index on ``self._label_idx`` so downstream
        helpers (for example ``_SliceLabelsLoader``) can slice batch["cls"] without
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

        ``requires_grad`` is set on both branches. ``_probe_feat_dim`` calls
        ``_freeze_encoder()`` on the shared encoder. A plain ``deepcopy`` would
        keep that frozen state and silently turn full fine-tuning into
        head-only training.
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
        return image_ids(ds)

    # ── Final-model fit, persistence, and prediction ─────────────────────────
    def _fit_final_model(self, dataset) -> nn.Module:
        """Train one encoder + head on all of ``dataset`` (minus an inner-val
        slice for early stopping) and store them as ``self.final_encoder_`` /
        ``self.final_head_``.

        This is the *deployment* model. It trains on every available image,
        not on a fold's train split, and ``save_model`` writes it.
        """
        all_labels, _ = self._resolve_labels(dataset)
        num_active = len(self.active_labels)
        torch_ds = dataset.get_datasets(n_splits=None, num_cores=self.num_workers)
        patient_ids = group_ids(dataset)

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

        seed_torch(self.base_seed, RUN_FINAL)
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
        and returns the head. You can then save one model
        (:meth:`save_model`) and reuse it (:meth:`predict`) without the full
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
        dict is always saved. The encoder state dict is saved only when
        ``freeze_backbone=False``, because a frozen encoder is rebuilt
        unchanged from its factory at load time. Returns the path written.
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
            "l2_normalize": self.l2_normalize,
            "loss": self.loss_name,
            "evaluator": type(self).__name__,
        }
        if not self.freeze_backbone and self.final_encoder_ is not None:
            ckpt["encoder_state_dict"] = self.final_encoder_.state_dict()
        torch.save(ckpt, path)
        return path

    def load_model(self, path: str) -> nn.Module:
        """Rebuild encoder + head from a :meth:`save_model` checkpoint.

        Construct the evaluator with the same ``image_encoder`` factory,
        ``head=`` and ``freeze_backbone=``. It warns when ``freeze_backbone``
        differs from the checkpoint. It takes ``l2_normalize`` and ``loss``
        from the checkpoint, with a warning when they differ.
        The checkpoint shape metadata drives ``_make_head``. Sets
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
        # The head needs the features and the score function that it was trained
        # with. A checkpoint from before l2_normalize was used has raw features.
        for attr, key, old in (("l2_normalize", "l2_normalize", False), ("loss_name", "loss", self.loss_name)):
            value = ckpt.get(key, old)
            if value != getattr(self, attr):
                warnings.warn(
                    f"checkpoint {key}={value!r} is not the evaluator {key}={getattr(self, attr)!r}. "
                    "The evaluator now uses the checkpoint value.",
                    stacklevel=2,
                )
                setattr(self, attr, value)
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

        Each dict has ``image_id``, ``y_true`` ``[L]`` (0/1 labels) and
        ``prob`` ``[L]``. ``prob`` is a sigmoid per label, or a softmax over the
        labels with ``loss="ce"``. The ``L`` columns are ``self.final_labels_``
        (set by fit or load). ``indices`` limits the run to a subset of the
        dataset. ``encoder`` and ``head`` default to ``self.final_encoder_`` and
        ``self.final_head_``. The method takes the dataset label columns by the
        names in ``self.final_labels_`` (V-E-30). It raises ``ValueError`` when
        the dataset ``LABEL_COLS`` do not have one of these names. With no
        ``final_labels_`` (a head passed in), it uses ``labels=``.
        """
        head = head if head is not None else self.final_head_
        encoder = encoder if encoder is not None else self.final_encoder_
        if head is None or encoder is None:
            raise RuntimeError(
                "predict() needs a trained model — pass encoder=/head=, or call "
                "fit()/load_model() first."
            )
        torch_ds = dataset.get_datasets(n_splits=None, num_cores=self.num_workers)  # sorts LABEL_COLS
        if self.final_labels_ is not None:
            all_labels = list(dataset.LABEL_COLS)
            missing = [lb for lb in self.final_labels_ if lb not in all_labels]
            if missing:
                raise ValueError(
                    f"The model was trained on {self.final_labels_}, but the dataset "
                    f"LABEL_COLS {all_labels} do not have {missing}."
                )
            label_idx = np.asarray([all_labels.index(lb) for lb in self.final_labels_], dtype=int)
        else:
            all_labels, label_idx = self._resolve_labels(dataset)
            if self.final_num_labels_ is not None and len(label_idx) != self.final_num_labels_:
                raise ValueError(
                    f"dataset resolves to {len(label_idx)} active labels but the trained "
                    f"model expects {self.final_num_labels_}; pass labels= so they match."
                )
        num_active = len(label_idx)
        encoder.eval()
        head.eval()
        image_ids = self._image_ids(dataset)
        if indices is not None:
            indices = [int(i) for i in indices]
            torch_ds = Subset(torch_ds, indices)
            image_ids = [image_ids[i] for i in indices]
        loader = _SliceLabelsLoader(
            self._make_loader(torch_ds, shuffle=False), label_idx, len(all_labels)
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
            probs = self._probs(head(self._head_input(feats))).float().cpu().numpy()
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

        patient_ids = group_ids(self.dataset)

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

                seed_torch(self.base_seed, RUN_KFOLD, fold_i, n_train)
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
                thr = self._val_thresholds(encoder, head, wrapped_val, num_active)
                y_true, probs = self._predict(encoder, head, wrapped_test, num_active)

                tmpl = {"n_train": n_train, "fold": fold_i, "seed": -1, "bootstrap": -1}
                rows.extend(score_rows(
                    probs, y_true, self.active_labels, thr, self.threshold_strategy, tmpl
                ))

        df = pd.DataFrame(rows)
        if self.store_final_model:
            self._fit_final_model(self.dataset)
        return macro_average(df)

    def _evaluate_fixed(self) -> pd.DataFrame:
        all_train, _ = self._resolve_labels(self.train_dataset)
        train_idx, train_labels = self._label_idx, self.active_labels
        # get_datasets() sorts LABEL_COLS. Then pair the test label columns with
        # the train labels by name (V-E-18).
        test_torch_ds = self.test_dataset.get_datasets(n_splits=None, num_cores=self.num_workers)
        all_test = list(self.test_dataset.LABEL_COLS)
        test_active = self._resolve_label_indices(all_test)
        test_idx = test_active[label_order(train_labels, self.active_labels)]
        self._label_idx, self._active_labels = train_idx, train_labels
        train_torch_ds = self.train_dataset.get_datasets(n_splits=None, num_cores=self.num_workers)
        num_active = len(self.active_labels)
        n_train_total = len(train_torch_ds)

        train_patient_ids = group_ids(self.train_dataset)
        test_pids = group_ids(self.test_dataset)

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

                seed_torch(seed, RUN_FIXED, n_train)
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
                wrapped_test = _SliceLabelsLoader(test_loader, test_idx, len(all_test))

                self._train_one(encoder, head, wrapped_train, wrapped_val, num_active)
                thr = self._val_thresholds(encoder, head, wrapped_val, num_active)
                y_true, probs = self._predict(encoder, head, wrapped_test, num_active)

                tmpl = {"n_train": n_train, "fold": -1, "seed": seed, "bootstrap": -1}
                rows.extend(score_rows(
                    probs, y_true, self.active_labels, thr, self.threshold_strategy, tmpl
                ))
                rows.extend(bootstrap_rows(
                    probs, y_true, self.active_labels, thr, self.threshold_strategy, tmpl,
                    self.n_bootstrap, self.bootstrap_seed + seed_i * 10_000 + max(n_train, 0),
                    groups=test_pids,
                ))

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
