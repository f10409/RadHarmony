"""Base class for vision classification evaluators.

Adds the frozen image encoder, per-split embedding cache, and the shared
``extract_embeddings`` routine on top of :class:`BaseEvaluator`.
"""

from __future__ import annotations

import os
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..base import BaseEvaluator


class BaseClsEvaluator(BaseEvaluator):
    """Base for the four classification evaluators (linear probe, kNN, zero-shot, finetune).

    The encoder contract is intentionally minimal: ``image_encoder`` is any
    ``nn.Module`` / callable with ``forward(imgs) -> Tensor[B, D]``. The
    evaluator does not inspect the encoder's type or internals; the user is
    responsible for wrapping HuggingFace/timm/CLIP models into this shape.
    """

    def __init__(
        self,
        image_encoder,
        *,
        dataset=None,
        train_dataset=None,
        test_dataset=None,
        labels: list[str] | None = None,
        device: str = "cuda",
        batch_size: int = 64,
        num_workers: int = 4,
        autocast_dtype: torch.dtype | None = torch.bfloat16,
        embedding_cache: str | None = None,
        l2_normalize: bool = False,
        output_dir: str | None = None,
        n_seeds: int = 1,
        base_seed: int = 0,
        n_bootstrap: int = 0,
        bootstrap_seed: int = 0,
        threshold_strategy: str = "youden",
    ):
        super().__init__(
            dataset=dataset,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            output_dir=output_dir,
            n_seeds=n_seeds,
            base_seed=base_seed,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            threshold_strategy=threshold_strategy,
        )
        self.image_encoder = image_encoder
        self.labels = labels
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.autocast_dtype = autocast_dtype
        self.embedding_cache = embedding_cache
        self.l2_normalize = l2_normalize

        # Captured during the first extract_embeddings call — respects the
        # post-sort order at radharmony/dataset/base.py:546.
        self._active_labels: list[str] | None = None
        self._label_idx: np.ndarray | None = None
        # One patient_id string per sample, in DataLoader order. Set by
        # extract_embeddings and used by _evaluate_kfold for GroupKFold.
        self._patient_ids: np.ndarray | None = None

    def _resolve_label_indices(self, all_labels: list[str]) -> np.ndarray:
        """Return indices into *all_labels* corresponding to self.labels (case-insensitive)."""
        if self.labels is None:
            self._active_labels = list(all_labels)
            return np.arange(len(all_labels))

        lower = [s.lower() for s in all_labels]
        idxs = []
        resolved = []
        for name in self.labels:
            try:
                i = lower.index(name.lower())
            except ValueError:
                raise KeyError(
                    f"Label {name!r} not found in dataset LABEL_COLS {all_labels}"
                )
            idxs.append(i)
            resolved.append(all_labels[i])
        self._active_labels = resolved
        return np.asarray(idxs, dtype=int)

    def _freeze_encoder(self):
        if isinstance(self.image_encoder, torch.nn.Module):
            self.image_encoder.eval()
            self.image_encoder.to(self.device)
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)

    def extract_embeddings(self, ds) -> tuple[np.ndarray, np.ndarray]:
        """Extract pooled embeddings + label matrix for a RadHarmony dataset.

        Returns ``(feats [N, D], labels [N, L])`` as numpy arrays, with
        ``L = len(self.labels)`` (or all labels if ``self.labels is None``).
        """
        torch_ds = ds.get_datasets(n_splits=None, num_cores=self.num_workers)
        # LABEL_COLS is sorted in place at radharmony/dataset/base.py:546.
        all_labels = list(ds.LABEL_COLS)

        # Build patient_ids aligned with the DataLoader order.  get_data_dict
        # drops rows with a null image_path (and null path-valued extra cols);
        # for the evaluator use case (output_cls only) image_path is the sole
        # path column, so one dropna call reproduces the exact row set.
        _df = ds._get_harmonized_df().dropna(subset=["image_path"])
        self._patient_ids = _df["patient_id"].astype(str).values
        idx = self._resolve_label_indices(all_labels)

        loader = DataLoader(
            torch_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        self._freeze_encoder()
        cm = (
            torch.autocast(device_type=self._autocast_device, dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else nullcontext()
        )

        feats_list, labels_list = [], []
        with torch.no_grad(), cm:
            for batch in tqdm(loader, desc="extracting embeddings", unit="batch"):
                imgs = batch["img"].to(self.device, non_blocking=True)
                feats = self.image_encoder(imgs)
                if isinstance(feats, torch.Tensor):
                    feats = feats.float().detach().cpu().numpy()
                else:
                    raise TypeError(
                        f"image_encoder must return a Tensor [B, D]; got {type(feats).__name__}. "
                        f"Wrap your model in a callable that extracts the pooled feature."
                    )
                cls = batch["cls"]
                if cls.ndim == 1:
                    cls = cls.view(-1, len(all_labels))
                feats_list.append(feats)
                labels_list.append(cls.float().numpy())

        feats_arr = np.concatenate(feats_list, axis=0)
        labels_arr = np.concatenate(labels_list, axis=0)[:, idx]
        self._label_idx = idx
        return feats_arr, labels_arr

    @property
    def _autocast_device(self) -> str:
        if self.device.startswith("cuda"):
            return "cuda"
        return "cpu"

    def _autocast_cm(self):
        """Autocast context manager honoring ``self.autocast_dtype``.

        Bridges the common dtype mismatch where a dataset emits bfloat16
        tensors (``RadiologyEncoderTransform`` casts the image to bf16 at the
        end of the pipeline) but encoder weights are fp32 — autocast handles
        per-op casts transparently. Mirrors ``BaseSegEvaluator._autocast_cm``.
        """
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(
            device_type=self._autocast_device, dtype=self.autocast_dtype
        )

    def _load_or_extract(
        self, ds, cache_suffix: str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract embeddings, or load from ``embedding_cache`` if present.

        Caches store **raw** features; L2 normalization (when enabled via
        ``l2_normalize=True``) is applied after load so the same cache file
        is reusable across evaluators with different normalize settings.
        """
        cache_path = None
        if self.embedding_cache is not None:
            if cache_suffix:
                cache_path = f"{self.embedding_cache}.{cache_suffix}.pkl"
            else:
                cache_path = self.embedding_cache
                if not cache_path.endswith(".pkl"):
                    cache_path = cache_path + ".pkl"

        if cache_path is not None and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            self._active_labels = payload["labels_names"]
            self._label_idx = payload.get("label_idx")
            self._patient_ids = payload.get("patient_ids")
            if self._patient_ids is None:
                # Stale cache written before patient_ids were added — recompute.
                _df = ds._get_harmonized_df().dropna(subset=["image_path"])
                self._patient_ids = _df["patient_id"].astype(str).values
            return self._maybe_normalize(payload["feats"]), payload["labels"]

        feats, labels = self.extract_embeddings(ds)

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "feats": feats,
                        "labels": labels,
                        "labels_names": self._active_labels,
                        "label_idx": self._label_idx,
                        "patient_ids": self._patient_ids,
                    },
                    f,
                )
        return self._maybe_normalize(feats), labels

    def _maybe_normalize(self, feats: np.ndarray) -> np.ndarray:
        if not self.l2_normalize:
            return feats
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return feats / norms

    @property
    def active_labels(self) -> list[str]:
        if self._active_labels is None:
            raise RuntimeError(
                "active_labels is only available after extract_embeddings() has run."
            )
        return self._active_labels
