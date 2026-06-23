"""SVM probe over frozen image encoder features.

Fits a per-label :class:`sklearn.svm.LinearSVC` on top of frozen backbone
embeddings. The decision function (not a sigmoid probability) is used as the
ranking score, so AUROC and AUPRC are reliable. Threshold-based metrics use
the Youden or F1 strategy applied to the decision values and are valid;
``threshold_strategy="fixed:<float>"`` is not meaningful since decision values
are not in [0, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold

from .._registry import register_evaluator
from ..metrics.classification import compute_metrics, macro_average, METRIC_KEYS
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
        **base_kwargs,
    ):
        # L2-normalize features by default — LinearSVC's L2 penalty assumes
        # features are on a similar scale; foundation-model embeddings have
        # outlier dims that dominate the optimizer otherwise.
        base_kwargs.setdefault("l2_normalize", True)
        super().__init__(image_encoder, **base_kwargs)
        self.n_folds = n_folds
        self.n_train_samples = n_train_samples
        self.svm_kwargs = {**DEFAULT_SVM_KWARGS, **(svm_kwargs or {})}
        self.store_final_model = store_final_model
        self.final_models_: dict | None = None

    def _fit_predict_per_label(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """Fit one LinearSVC per label; return decision function matrix ``(N_test, L)``."""
        n_labels = y_train.shape[1]
        scores = np.full((X_test.shape[0], n_labels), np.nan, dtype=np.float32)
        for k in range(n_labels):
            y_k = y_train[:, k].astype(int)
            if len(np.unique(y_k)) < 2:
                continue
            svm = LinearSVC(**self.svm_kwargs)
            svm.fit(X_train, y_k)
            scores[:, k] = svm.decision_function(X_test)
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
        """Fit one LinearSVC per label on all training data; populate ``self.final_models_``."""
        self.final_models_ = {}
        for k, name in enumerate(self.active_labels):
            y_k = y_train[:, k].astype(int)
            if len(np.unique(y_k)) < 2:
                self.final_models_[name] = None
                continue
            svm = LinearSVC(**self.svm_kwargs)
            svm.fit(X_train, y_k)
            self.final_models_[name] = svm

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
                scores = self._fit_predict_per_label(X_tr_all[sub], y_tr_all[sub], X_val)
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
                scores = self._fit_predict_per_label(X_train[sub], y_train[sub], X_test)
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
