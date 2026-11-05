"""Prototype (nearest-centroid) classifier over frozen image encoder features.

For each label, compute the mean embedding of positive and negative training
examples, then score each test embedding as:

    cosine_sim(x, centroid_pos) − cosine_sim(x, centroid_neg)

No optimization is needed. The two centroids per label come from a
single pass over the training embeddings.

The threshold metrics use one threshold per label from a validation part of
the train patients (``val_fraction``, D-29). The probe does not fit on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .._registry import register_evaluator
from ..metrics.classification import macro_average
from ._probe_utils import bootstrap_rows, fit_score, label_mask, load_fixed_split, row_patient_ids, score_rows
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
        val_fraction: float = 0.1,
        **base_kwargs,
    ):
        # L2-normalize the features by default, as the other probes do. The
        # centroids are then means of unit vectors. l2_normalize=False makes
        # them means of the raw features (D-17).
        base_kwargs.setdefault("l2_normalize", True)
        super().__init__(image_encoder, **base_kwargs)
        self.n_folds = n_folds
        self.n_train_samples = n_train_samples
        self.store_final_model = store_final_model
        # Part of the train patients that picks the threshold, not the fit (D-29).
        self.val_fraction = val_fraction
        self.final_models_: dict | None = None

    def _centroid_scores(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """Return ``(N_test, L)`` score matrix (NaN where a label has no positives or no negatives).

        The centroids come from the features as given (L2-normalized when
        ``l2_normalize`` is set). The score is a cosine, so each test row is
        normalized.
        """
        X_test_n = _l2_normalize(X_test)
        n_labels = y_train.shape[1]
        scores = np.full((X_test.shape[0], n_labels), np.nan, dtype=np.float32)
        for k in range(n_labels):
            centroids = self._centroids(X_train, y_train[:, k])
            if centroids is not None:
                c_pos, c_neg = centroids
                scores[:, k] = X_test_n @ c_pos - X_test_n @ c_neg
        return scores

    @staticmethod
    def _centroids(X: np.ndarray, y_k: np.ndarray):
        """Return the unit-norm ``(c_pos, c_neg)`` of one label, or ``None`` if a class is missing.

        Only rows with label 0 or 1 count. A NaN row is not a negative (D-18).
        """
        valid = label_mask(y_k)
        pos_mask = valid & (y_k == 1)
        neg_mask = valid & (y_k == 0)
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return None
        c_pos = X[pos_mask].mean(axis=0)
        c_pos /= max(float(np.linalg.norm(c_pos)), 1e-8)
        c_neg = X[neg_mask].mean(axis=0)
        c_neg /= max(float(np.linalg.norm(c_neg)), 1e-8)
        return c_pos, c_neg

    def _fit_final_models(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Compute centroids on all training data and set ``self.final_models_``.

        ``final_models_`` maps each label name to a ``(c_pos, c_neg)`` tuple of
        L2-normalized centroid vectors. It maps to ``None`` if the label has no
        positives or no negatives in the training set.
        """
        self.final_models_ = {
            name: self._centroids(X_train, y_train[:, k])
            for k, name in enumerate(self.active_labels)
        }

    def _iter_n_train(self, n_available: int):
        if self.n_train_samples is None:
            return [-1]
        # Keep the requested n (also above n_available): one summary row per point (V-E-21).
        return list(self.n_train_samples)

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
        patient_ids = row_patient_ids(self, self.dataset, len(feats))
        gkf = GroupKFold(n_splits=self.n_folds)
        rows = []
        rng = np.random.default_rng(self.base_seed)
        for fold_i, (train_idx, test_idx) in enumerate(gkf.split(feats, groups=patient_ids)):
            for n_train in self._iter_n_train(len(train_idx)):
                sub = train_idx[self._subsample(len(train_idx), n_train, rng)]
                scores, thr = fit_score(
                    self._centroid_scores, feats, labels, sub, feats[test_idx],
                    patient_ids, rng, self.val_fraction, self.threshold_strategy,
                )
                tmpl = {"n_train": n_train, "fold": fold_i, "seed": -1, "bootstrap": -1}
                rows.extend(score_rows(
                    scores, labels[test_idx], self.active_labels, thr, self.threshold_strategy, tmpl
                ))
        if self.store_final_model:
            self._fit_final_models(feats, labels)
        return macro_average(pd.DataFrame(rows))

    def _evaluate_fixed(self) -> pd.DataFrame:
        X_train, y_train, train_pids, X_test, y_test, test_pids = load_fixed_split(self)
        rows = []
        for seed_i in range(self.n_seeds):
            seed = self.base_seed + seed_i
            rng = np.random.default_rng(seed)
            for n_train in self._iter_n_train(X_train.shape[0]):
                sub = self._subsample(X_train.shape[0], n_train, rng)
                scores, thr = fit_score(
                    self._centroid_scores, X_train, y_train, sub, X_test,
                    train_pids, rng, self.val_fraction, self.threshold_strategy,
                )
                tmpl = {"n_train": n_train, "fold": -1, "seed": seed, "bootstrap": -1}
                rows.extend(score_rows(
                    scores, y_test, self.active_labels, thr, self.threshold_strategy, tmpl
                ))
                rows.extend(bootstrap_rows(
                    scores, y_test, self.active_labels, thr, self.threshold_strategy, tmpl,
                    self.n_bootstrap, self.bootstrap_seed + seed_i * 10_000 + max(n_train, 0),
                    groups=test_pids,
                ))
        if self.store_final_model:
            self._fit_final_models(X_train, y_train)
        return macro_average(pd.DataFrame(rows))
