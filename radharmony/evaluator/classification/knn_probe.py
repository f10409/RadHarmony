"""k-NN probe over frozen image encoder features.

For each test embedding, find its *k* nearest train neighbors (cosine by
default) and score each label as the mean of the neighbors' labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

from .._registry import register_evaluator
from ..metrics.classification import compute_metrics, macro_average, METRIC_KEYS
from .base import BaseClsEvaluator


@register_evaluator("knn_probe")
class KNNProbeEvaluator(BaseClsEvaluator):
    """Per-label soft-vote k-NN probe on frozen features.

    Features are L2-normalized by default via the base-class
    ``l2_normalize`` flag (so the default ``metric="cosine"`` operates on
    unit vectors). Pass ``l2_normalize=False`` to feed raw embeddings into
    the chosen ``metric``.
    """

    def __init__(
        self,
        image_encoder,
        *,
        k: int = 20,
        metric: str = "cosine",
        n_folds: int = 5,
        n_train_samples: list[int] | None = None,
        store_final_model: bool = False,
        **base_kwargs,
    ):
        # L2-normalize via the base-class boundary so the cache stores raw
        # features and the normalize behavior matches the SVM probe.
        base_kwargs.setdefault("l2_normalize", True)
        super().__init__(image_encoder, **base_kwargs)
        self.k = k
        self.metric = metric
        self.n_folds = n_folds
        self.n_train_samples = n_train_samples
        self.store_final_model = store_final_model
        self.final_models_: dict | None = None

    def _knn_scores(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    ) -> np.ndarray:
        k = min(self.k, X_train.shape[0])
        nn = NearestNeighbors(n_neighbors=k, metric=self.metric)
        nn.fit(X_train)
        _, idx = nn.kneighbors(X_test)
        # idx: (N_test, k) — take mean of neighbor labels per label column
        # y_train: (N_train, L) → neighbor_labels: (N_test, k, L)
        neighbor_labels = y_train[idx]
        return neighbor_labels.mean(axis=1)

    def _score_rows(self, scores, y_test, row_template) -> list[dict]:
        rows = []
        for k, name in enumerate(self.active_labels):
            row = dict(row_template)
            row["label"] = name
            row.update(
                compute_metrics(y_test[:, k], scores[:, k], self.threshold_strategy)
            )
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
            tmpl = dict(row_template)
            tmpl["bootstrap"] = b
            for k, name in enumerate(self.active_labels):
                row = dict(tmpl)
                row["label"] = name
                row.update(
                    compute_metrics(
                        y_test[idx, k], scores[idx, k], self.threshold_strategy
                    )
                )
                rows.append(row)
        return rows

    def _fit_final_models(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the kNN index on all training data; populate ``self.final_models_``.

        ``final_models_`` is a dict with two keys:
        ``"nn"`` — the fitted :class:`~sklearn.neighbors.NearestNeighbors` index, and
        ``"y_train"`` — the full ``(N, L)`` label matrix for soft-vote scoring.
        """
        k = min(self.k, X_train.shape[0])
        nn = NearestNeighbors(n_neighbors=k, metric=self.metric)
        nn.fit(X_train)
        self.final_models_ = {"nn": nn, "y_train": y_train}

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
                scores = self._knn_scores(X_tr, y_tr, X_val)
                tmpl = {"n_train": n_train, "fold": fold_i, "seed": -1, "bootstrap": -1}
                rows.extend(self._score_rows(scores, y_val, tmpl))

        df = pd.DataFrame(rows)
        if self.store_final_model:
            self._fit_final_models(feats, labels)
        return macro_average(df)

    def _evaluate_fixed(self) -> pd.DataFrame:
        X_train, y_train = self._load_or_extract(self.train_dataset, cache_suffix="train")
        X_test, y_test = self._load_or_extract(self.test_dataset, cache_suffix="test")

        rows = []
        for seed_i in range(self.n_seeds):
            seed = self.base_seed + seed_i
            rng = np.random.default_rng(seed)
            for n_train in self._iter_n_train(X_train.shape[0]):
                sub = self._subsample(X_train.shape[0], n_train, rng)
                X_tr = X_train[sub]
                y_tr = y_train[sub]
                scores = self._knn_scores(X_tr, y_tr, X_test)
                tmpl = {"n_train": n_train, "fold": -1, "seed": seed, "bootstrap": -1}
                rows.extend(self._score_rows(scores, y_test, tmpl))
                rows.extend(
                    self._bootstrap_rows(
                        scores, y_test, tmpl, self.bootstrap_seed + seed_i * 10_000 + max(n_train, 0)
                    )
                )

        df = pd.DataFrame(rows)
        if self.store_final_model:
            self._fit_final_models(X_train, y_train)
        return macro_average(df)
