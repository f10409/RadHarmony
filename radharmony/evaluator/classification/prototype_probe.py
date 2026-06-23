"""Prototype (nearest-centroid) classifier over frozen image encoder features.

For each label, compute the mean embedding of positive and negative training
examples (L2-normalised), then score each test embedding as:

    cosine_sim(x, centroid_pos) − cosine_sim(x, centroid_neg)

No optimisation is required — the two centroids per label are computed in a
single pass over training embeddings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .._registry import register_evaluator
from ..metrics.classification import compute_metrics, macro_average, METRIC_KEYS
from .base import BaseClsEvaluator


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms == 0.0, 1.0, norms)


@register_evaluator("prototype_probe")
class PrototypeProbeEvaluator(BaseClsEvaluator):
    """Per-label nearest-centroid classifier on frozen encoder features."""

    def __init__(
        self,
        image_encoder,
        *,
        n_folds: int = 5,
        n_train_samples: list[int] | None = None,
        store_final_model: bool = False,
        **base_kwargs,
    ):
        super().__init__(image_encoder, **base_kwargs)
        self.n_folds = n_folds
        self.n_train_samples = n_train_samples
        self.store_final_model = store_final_model
        self.final_models_: dict | None = None

    def _centroid_scores(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """Return ``(N_test, L)`` score matrix (NaN where a label has no positives or no negatives)."""
        X_train_n = _l2_normalize(X_train)
        X_test_n = _l2_normalize(X_test)
        n_labels = y_train.shape[1]
        scores = np.full((X_test.shape[0], n_labels), np.nan, dtype=np.float32)
        for k in range(n_labels):
            pos_mask = y_train[:, k] == 1
            neg_mask = ~pos_mask
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue
            c_pos = X_train_n[pos_mask].mean(axis=0)
            c_pos /= max(float(np.linalg.norm(c_pos)), 1e-8)
            c_neg = X_train_n[neg_mask].mean(axis=0)
            c_neg /= max(float(np.linalg.norm(c_neg)), 1e-8)
            scores[:, k] = X_test_n @ c_pos - X_test_n @ c_neg
        return scores

    def _score_rows(self, scores, y_test, row_template) -> list[dict]:
        rows = []
        for k, name in enumerate(self.active_labels):
            row = dict(row_template, label=name)
            if np.isnan(scores[:, k]).any():
                row.update({m: np.nan for m in METRIC_KEYS})
            else:
                row.update(compute_metrics(y_test[:, k], scores[:, k], self.threshold_strategy))
            rows.append(row)
        return rows

    def _bootstrap_rows(self, scores, y_test, row_template, seed) -> list[dict]:
        if self.n_bootstrap <= 0:
            return []
        rng = np.random.default_rng(seed)
        n = y_test.shape[0]
        rows = []
        for b in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            tmpl = dict(row_template, bootstrap=b)
            for k, name in enumerate(self.active_labels):
                row = dict(tmpl, label=name)
                if np.isnan(scores[:, k]).any():
                    row.update({m: np.nan for m in METRIC_KEYS})
                else:
                    row.update(
                        compute_metrics(y_test[idx, k], scores[idx, k], self.threshold_strategy)
                    )
                rows.append(row)
        return rows

    def _fit_final_models(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Compute centroids on all training data; populate ``self.final_models_``.

        ``final_models_`` maps each label name to a ``(c_pos, c_neg)`` tuple of
        L2-normalized centroid vectors, or ``None`` if the label has no positives
        or no negatives in the training set.
        """
        X_n = _l2_normalize(X_train)
        self.final_models_ = {}
        for k, name in enumerate(self.active_labels):
            pos_mask = y_train[:, k] == 1
            neg_mask = ~pos_mask
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                self.final_models_[name] = None
                continue
            c_pos = X_n[pos_mask].mean(axis=0)
            c_pos /= max(float(np.linalg.norm(c_pos)), 1e-8)
            c_neg = X_n[neg_mask].mean(axis=0)
            c_neg /= max(float(np.linalg.norm(c_neg)), 1e-8)
            self.final_models_[name] = (c_pos, c_neg)

    def _iter_n_train(self, n_available: int):
        if self.n_train_samples is None:
            return [-1]
        return [min(n, n_available) for n in self.n_train_samples]

    def _subsample(self, n_available: int, n_train: int, rng) -> np.ndarray:
        if n_train == -1 or n_train >= n_available:
            return np.arange(n_available)
        return rng.choice(n_available, size=n_train, replace=False)

    def evaluate(self) -> pd.DataFrame:
        if self.split_mode == "kfold":
            return self._evaluate_kfold()
        return self._evaluate_fixed()

    def _evaluate_kfold(self) -> pd.DataFrame:
        feats, labels = self._load_or_extract(self.dataset, cache_suffix=None)
        gkf = GroupKFold(n_splits=self.n_folds)
        rows = []
        rng = np.random.default_rng(self.base_seed)
        for fold_i, (train_idx, val_idx) in enumerate(gkf.split(feats, groups=self._patient_ids)):
            X_tr_all, y_tr_all = feats[train_idx], labels[train_idx]
            X_val, y_val = feats[val_idx], labels[val_idx]
            for n_train in self._iter_n_train(len(train_idx)):
                sub = self._subsample(len(train_idx), n_train, rng)
                scores = self._centroid_scores(X_tr_all[sub], y_tr_all[sub], X_val)
                tmpl = {"n_train": n_train, "fold": fold_i, "seed": -1, "bootstrap": -1}
                rows.extend(self._score_rows(scores, y_val, tmpl))
        if self.store_final_model:
            self._fit_final_models(feats, labels)
        return macro_average(pd.DataFrame(rows))

    def _evaluate_fixed(self) -> pd.DataFrame:
        X_train, y_train = self._load_or_extract(self.train_dataset, cache_suffix="train")
        X_test, y_test = self._load_or_extract(self.test_dataset, cache_suffix="test")
        rows = []
        for seed_i in range(self.n_seeds):
            seed = self.base_seed + seed_i
            rng = np.random.default_rng(seed)
            for n_train in self._iter_n_train(X_train.shape[0]):
                sub = self._subsample(X_train.shape[0], n_train, rng)
                scores = self._centroid_scores(X_train[sub], y_train[sub], X_test)
                tmpl = {"n_train": n_train, "fold": -1, "seed": seed, "bootstrap": -1}
                rows.extend(self._score_rows(scores, y_test, tmpl))
                rows.extend(
                    self._bootstrap_rows(
                        scores, y_test, tmpl,
                        self.bootstrap_seed + seed_i * 10_000 + max(n_train, 0),
                    )
                )
        if self.store_final_model:
            self._fit_final_models(X_train, y_train)
        return macro_average(pd.DataFrame(rows))
