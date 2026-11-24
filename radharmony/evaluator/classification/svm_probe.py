"""SVM probe over frozen image encoder features.

Fits a per-label :class:`sklearn.svm.LinearSVC` on top of frozen backbone
embeddings. The decision function (not a sigmoid probability) is the
ranking score, so AUROC and AUPRC are reliable. Threshold-based metrics use
the Youden or F1 strategy on the decision values, which is valid. The
decision values are not in [0, 1], so a fixed threshold must be on their
scale. ``threshold_strategy="fixed:0"`` is the decision boundary of the SVM.

The threshold metrics use one threshold per label from a validation part of
the train patients (``val_fraction``, D-29). The probe does not fit on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold

from .._registry import register_evaluator
from ..metrics.classification import macro_average
from ._probe_utils import bootstrap_rows, fit_score, label_mask, load_fixed_split, row_patient_ids, score_rows
from .base import BaseClsEvaluator


DEFAULT_SVM_KWARGS: dict = {
    "class_weight": "balanced",
    "C": 1.0,
    "max_iter": 10000,
    "dual": False,  # primal is much faster when n_samples > n_features (typical for CXR embedding probes)
}


@register_evaluator("svm_probe")
class SVMProbeEvaluator(BaseClsEvaluator):
    """Per-label linear SVM probe on frozen encoder features.

    Parameters
    ----------
    svm_kwargs :
        Keyword arguments forwarded to :class:`sklearn.svm.LinearSVC`.
        Merged with ``DEFAULT_SVM_KWARGS``; caller-supplied values win.
    """

    def __init__(
        self,
        image_encoder,
        *,
        n_folds: int = 5,
        n_train_samples: list[int] | None = None,
        svm_kwargs: dict | None = None,
        store_final_model: bool = False,
        val_fraction: float = 0.1,
        **base_kwargs,
    ):
        # L2-normalize features by default. The LinearSVC L2 penalty needs
        # features on a similar scale. Foundation-model embeddings have
        # outlier dims that would dominate the optimizer.
        base_kwargs.setdefault("l2_normalize", True)
        super().__init__(image_encoder, **base_kwargs)
        self.n_folds = n_folds
        self.n_train_samples = n_train_samples
        self.svm_kwargs = {**DEFAULT_SVM_KWARGS, **(svm_kwargs or {})}
        self.store_final_model = store_final_model
        # Part of the train patients that picks the threshold, not the fit (D-29).
        self.val_fraction = val_fraction
        self.final_models_: dict | None = None

    def _fit_predict_per_label(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """Fit one LinearSVC per label and return the decision matrix ``(N_test, L)``."""
        n_labels = y_train.shape[1]
        scores = np.full((X_test.shape[0], n_labels), np.nan, dtype=np.float32)
        for k in range(n_labels):
            svm = self._fit_label(X_train, y_train[:, k])
            if svm is not None:
                scores[:, k] = svm.decision_function(X_test)
        return scores

    def _fit_label(self, X_train: np.ndarray, y_k: np.ndarray):
        """Fit one LinearSVC on the rows where label ``y_k`` is 0 or 1 (NaN rows are dropped).

        Returns the fitted SVM, or ``None`` if a class is missing.
        """
        mask = label_mask(y_k)
        y = y_k[mask].astype(int)
        if len(np.unique(y)) < 2:
            return None
        svm = LinearSVC(**self.svm_kwargs)
        svm.fit(X_train[mask], y)
        return svm

    def _fit_final_models(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit one LinearSVC per label on all training data and set ``self.final_models_``."""
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
                scores, thr = fit_score(
                    self._fit_predict_per_label, feats, labels, sub, feats[test_idx],
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
                    self._fit_predict_per_label, X_train, y_train, sub, X_test,
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
