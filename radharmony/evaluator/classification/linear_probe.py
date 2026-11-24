"""Linear probe over frozen image encoder features.

Fits a per-label :class:`sklearn.linear_model.LogisticRegression` on top
of frozen backbone embeddings. Supports both k-fold and fixed-split modes
with an ``n_train`` sweep and (fixed-split only) ``n_seeds`` × ``n_bootstrap``
variance estimation.

The threshold metrics use one threshold per label from a validation part of
the train patients (``val_fraction``, D-29). The probe does not fit on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from .._registry import register_evaluator
from ..metrics.classification import macro_average
from ._probe_utils import bootstrap_rows, fit_score, label_mask, load_fixed_split, row_patient_ids, score_rows
from .base import BaseClsEvaluator


DEFAULT_PROBE_KWARGS = {
    "class_weight": "balanced",
    "C": 1.0,
    "max_iter": 10000,
    "solver": "lbfgs",
}


@register_evaluator("linear_probe")
class LinearProbeEvaluator(BaseClsEvaluator):
    """Per-label logistic regression on frozen encoder features."""

    def __init__(
        self,
        image_encoder,
        *,
        n_folds: int = 5,
        n_train_samples: list[int] | None = None,
        probe_cls: type = LogisticRegression,
        probe_kwargs: dict | None = None,
        store_final_model: bool = False,
        val_fraction: float = 0.1,
        **base_kwargs,
    ):
        # L2-normalize features by default. The probe then uses the same
        # unit-norm features as the knn, svm and prototype probes, so the
        # comparison tests the classifier, not feature scaling. The
        # cache still stores raw features (normalization is applied after load).
        base_kwargs.setdefault("l2_normalize", True)
        super().__init__(image_encoder, **base_kwargs)
        self.n_folds = n_folds
        self.n_train_samples = n_train_samples
        self.probe_cls = probe_cls
        self.probe_kwargs = {**DEFAULT_PROBE_KWARGS, **(probe_kwargs or {})}
        self.store_final_model = store_final_model
        # Part of the train patients that picks the threshold, not the fit (D-29).
        self.val_fraction = val_fraction
        self.final_models_: dict | None = None

    def _fit_label(self, X_train: np.ndarray, y_k: np.ndarray):
        """Fit one probe on the rows where label ``y_k`` is 0 or 1.

        A NaN label (CheXpert U-Ignore) is dropped, not cast to int. A cast
        gives integer garbage (for example -2147483648), so sklearn sees 3
        classes and ``predict_proba[:, 1]`` is ``P(class=0)``. That gave
        AUROC = 1 - true AUROC on the Emory and MIMIC runs (2026-05-21).

        Returns the fitted probe, or ``None`` if a class is missing.
        """
        mask = label_mask(y_k)
        y = y_k[mask].astype(int)
        if len(np.unique(y)) < 2:
            return None
        clf = self.probe_cls(**self.probe_kwargs)
        clf.fit(X_train[mask], y)
        return clf

    def _fit_predict_per_label(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """Fit one probe per label column, return probability matrix (N_test, L).

        A label with a missing class gets NaN scores.
        """
        n_labels = y_train.shape[1]
        probs = np.full((X_test.shape[0], n_labels), np.nan, dtype=np.float32)
        for k in range(n_labels):
            clf = self._fit_label(X_train, y_train[:, k])
            if clf is not None:
                probs[:, k] = clf.predict_proba(X_test)[:, 1]
        return probs

    def _fit_final_models(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit one probe per label on all training data and set ``self.final_models_``."""
        self.final_models_ = {
            name: self._fit_label(X_train, y_train[:, k])
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
                probs, thr = fit_score(
                    self._fit_predict_per_label, feats, labels, sub, feats[test_idx],
                    patient_ids, rng, self.val_fraction, self.threshold_strategy,
                )
                tmpl = {
                    "n_train": n_train,
                    "fold": fold_i,
                    "seed": -1,
                    "bootstrap": -1,
                }
                rows.extend(score_rows(
                    probs, labels[test_idx], self.active_labels, thr, self.threshold_strategy, tmpl
                ))

        df = pd.DataFrame(rows)
        if self.store_final_model:
            self._fit_final_models(feats, labels)
        return macro_average(df)

    def _evaluate_fixed(self) -> pd.DataFrame:
        X_train, y_train, train_pids, X_test, y_test, test_pids = load_fixed_split(self)

        rows = []
        for seed_i in range(self.n_seeds):
            seed = self.base_seed + seed_i
            rng = np.random.default_rng(seed)
            for n_train in self._iter_n_train(X_train.shape[0]):
                sub = self._subsample(X_train.shape[0], n_train, rng)
                probs, thr = fit_score(
                    self._fit_predict_per_label, X_train, y_train, sub, X_test,
                    train_pids, rng, self.val_fraction, self.threshold_strategy,
                )
                tmpl = {
                    "n_train": n_train,
                    "fold": -1,
                    "seed": seed,
                    "bootstrap": -1,
                }
                rows.extend(score_rows(
                    probs, y_test, self.active_labels, thr, self.threshold_strategy, tmpl
                ))
                boot_seed = self.bootstrap_seed + seed_i * 10_000 + max(n_train, 0)
                rows.extend(bootstrap_rows(
                    probs, y_test, self.active_labels, thr, self.threshold_strategy, tmpl,
                    self.n_bootstrap, boot_seed, groups=test_pids,
                ))

        df = pd.DataFrame(rows)
        if self.store_final_model:
            self._fit_final_models(X_train, y_train)
        return macro_average(df)
