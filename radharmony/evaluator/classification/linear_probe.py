"""Linear probe over frozen image encoder features.

Fits a per-label :class:`sklearn.linear_model.LogisticRegression` on top
of frozen backbone embeddings. Supports both k-fold and fixed-split modes
with an ``n_train`` sweep and (fixed-split only) ``n_seeds`` × ``n_bootstrap``
variance estimation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from .._registry import register_evaluator
from ..metrics.classification import compute_metrics, macro_average, METRIC_KEYS
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
        **base_kwargs,
    ):
        # L2-normalize features by default so the probe operates on the same
        # unit-norm representation as the knn/svm/prototype probes — the
        # comparison then isolates the classifier, not feature scaling. The
        # cache still stores raw features (normalization is applied after load).
        base_kwargs.setdefault("l2_normalize", True)
        super().__init__(image_encoder, **base_kwargs)
        self.n_folds = n_folds
        self.n_train_samples = n_train_samples
        self.probe_cls = probe_cls
        self.probe_kwargs = {**DEFAULT_PROBE_KWARGS, **(probe_kwargs or {})}
        self.store_final_model = store_final_model
        self.final_models_: dict | None = None

    def _fit_predict_per_label(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> np.ndarray:
        """Fit one probe per label column, return probability matrix (N_test, L).

        Per-label NaN masking: a row with NaN at label k is dropped from
        that label's training set (CheXpert U-Ignore semantics). Without
        this, the prior code did ``y_train[:, k].astype(int)`` which casts
        NaN to integer garbage (e.g. -2147483648). sklearn then sees a
        3-class problem (garbage, 0, 1), and ``predict_proba[:, 1]`` returns
        ``P(class=0)`` instead of ``P(positive)`` because the garbage value
        sorts first — producing AUROC = 1 − true_AUROC (systematic inversion
        across all labels). Confirmed 2026-05-21 on Emory + MIMIC raddino runs.
        """
        n_labels = y_train.shape[1]
        probs = np.zeros((X_test.shape[0], n_labels), dtype=np.float32)
        for k in range(n_labels):
            y_k_raw = y_train[:, k]
            mask = ~np.isnan(y_k_raw)
            if mask.sum() < 2:
                probs[:, k] = np.nan
                continue
            y_k = y_k_raw[mask].astype(int)
            if len(np.unique(y_k)) < 2:
                probs[:, k] = np.nan
                continue
            clf = self.probe_cls(**self.probe_kwargs)
            clf.fit(X_train[mask], y_k)
            probs[:, k] = clf.predict_proba(X_test)[:, 1]
        return probs

    def _score_rows(
        self,
        probs: np.ndarray,
        y_test: np.ndarray,
        row_template: dict,
    ) -> list[dict]:
        """Compute per-label metric rows for a single (fold|seed, n_train) point."""
        rows = []
        for k, name in enumerate(self.active_labels):
            row = dict(row_template)
            row["label"] = name
            if np.isnan(probs[:, k]).any():
                row.update({m: np.nan for m in METRIC_KEYS})
            else:
                row.update(
                    compute_metrics(y_test[:, k], probs[:, k], self.threshold_strategy)
                )
            rows.append(row)
        return rows

    def _bootstrap_rows(
        self,
        probs: np.ndarray,
        y_test: np.ndarray,
        row_template: dict,
        seed: int,
    ) -> list[dict]:
        if self.n_bootstrap <= 0:
            return []
        # n_train can be -1 (full-train sentinel), making the composed seed
        # negative; modern numpy rejects negative seeds. Coerce to uint32.
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        n = y_test.shape[0]
        rows = []
        for b in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            tmpl = dict(row_template)
            tmpl["bootstrap"] = b
            for k, name in enumerate(self.active_labels):
                row = dict(tmpl)
                row["label"] = name
                if np.isnan(probs[:, k]).any():
                    row.update({m: np.nan for m in METRIC_KEYS})
                else:
                    row.update(
                        compute_metrics(
                            y_test[idx, k], probs[idx, k], self.threshold_strategy
                        )
                    )
                rows.append(row)
        return rows

    def _fit_final_models(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit one probe per label on all training data; populate ``self.final_models_``."""
        self.final_models_ = {}
        for k, name in enumerate(self.active_labels):
            y_k = y_train[:, k].astype(int)
            if len(np.unique(y_k)) < 2:
                self.final_models_[name] = None
                continue
            clf = self.probe_cls(**self.probe_kwargs)
            clf.fit(X_train, y_k)
            self.final_models_[name] = clf

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
                X_tr, y_tr = X_tr_all[sub], y_tr_all[sub]
                probs = self._fit_predict_per_label(X_tr, y_tr, X_val, y_val)
                tmpl = {
                    "n_train": n_train,
                    "fold": fold_i,
                    "seed": -1,
                    "bootstrap": -1,
                }
                rows.extend(self._score_rows(probs, y_val, tmpl))

        df = pd.DataFrame(rows)
        if self.store_final_model:
            self._fit_final_models(feats, labels)
        return macro_average(df)

    def _evaluate_fixed(self) -> pd.DataFrame:
        X_train, y_train = self._load_or_extract(
            self.train_dataset, cache_suffix="train"
        )
        # The test set label set must match the train set — use the same resolved indices.
        # Re-running extract on test uses the same self.labels, so the ordering matches.
        X_test, y_test = self._load_or_extract(self.test_dataset, cache_suffix="test")

        rows = []
        for seed_i in range(self.n_seeds):
            seed = self.base_seed + seed_i
            rng = np.random.default_rng(seed)
            for n_train in self._iter_n_train(X_train.shape[0]):
                sub = self._subsample(X_train.shape[0], n_train, rng)
                X_tr = X_train[sub]
                y_tr = y_train[sub]
                probs = self._fit_predict_per_label(X_tr, y_tr, X_test, y_test)
                tmpl = {
                    "n_train": n_train,
                    "fold": -1,
                    "seed": seed,
                    "bootstrap": -1,
                }
                rows.extend(self._score_rows(probs, y_test, tmpl))

                boot_seed = self.bootstrap_seed + seed_i * 10_000 + max(n_train, 0)
                rows.extend(self._bootstrap_rows(probs, y_test, tmpl, boot_seed))

        df = pd.DataFrame(rows)
        if self.store_final_model:
            self._fit_final_models(X_train, y_train)
        return macro_average(df)
