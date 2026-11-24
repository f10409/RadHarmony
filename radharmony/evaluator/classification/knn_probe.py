"""k-NN probe over frozen image encoder features.

For each test embedding, find its *k* nearest train neighbors (cosine by
default) and score each label as the mean of the neighbors' labels.

The threshold metrics use one threshold per label from a validation part of
the train patients (``val_fraction``, D-29). The probe does not fit on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

from .._registry import register_evaluator
from ..metrics.classification import macro_average
from ._probe_utils import bootstrap_rows, fit_score, label_mask, load_fixed_split, row_patient_ids, score_rows
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
        val_fraction: float = 0.1,
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
        # Part of the train patients that picks the threshold, not the fit (D-29).
        self.val_fraction = val_fraction
        self.final_models_: dict | None = None

    @staticmethod
    def _label_groups(y_train: np.ndarray):
        """Group the label columns that have 0/1 labels on the same rows.

        Yields ``(cols, rows)``: a list of label indices and their bool row mask.
        """
        valid = label_mask(y_train)
        groups: dict[bytes, list[int]] = {}
        for k in range(valid.shape[1]):
            groups.setdefault(valid[:, k].tobytes(), []).append(k)
        for cols in groups.values():
            yield cols, valid[:, cols[0]]

    def _fit_index(self, X: np.ndarray) -> NearestNeighbors:
        nn = NearestNeighbors(n_neighbors=min(self.k, X.shape[0]), metric=self.metric)
        nn.fit(X)
        return nn

    def _knn_scores(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    ) -> np.ndarray:
        """Return ``(N_test, L)`` scores: the mean label of the k nearest train rows.

        Each label searches only the rows where it is 0 or 1, so a NaN label
        is not a neighbor (D-18). A label with no such row gets NaN scores.
        """
        scores = np.full((X_test.shape[0], y_train.shape[1]), np.nan, dtype=np.float32)
        for cols, rows in self._label_groups(y_train):
            if not rows.any():
                continue
            _, idx = self._fit_index(X_train[rows]).kneighbors(X_test)
            # y_train[rows][:, cols][idx] has shape (N_test, k, len(cols)).
            scores[:, cols] = y_train[rows][:, cols][idx].mean(axis=1)
        return scores

    def _fit_final_models(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the kNN index on all training data and set ``self.final_models_``.

        ``final_models_`` maps each label name to a dict with two keys.
        ``"nn"`` is a fitted :class:`~sklearn.neighbors.NearestNeighbors` index
        over the rows where the label is 0 or 1. ``"y_train"`` holds the label
        of those rows. The score of new features ``x`` is
        ``y_train[nn.kneighbors(x)[1]].mean(axis=1)``. A label with no such
        row maps to ``None``.
        """
        self.final_models_ = {}
        for cols, rows in self._label_groups(y_train):
            nn = self._fit_index(X_train[rows]) if rows.any() else None
            for k in cols:
                self.final_models_[self.active_labels[k]] = (
                    None if nn is None else {"nn": nn, "y_train": y_train[rows, k]}
                )

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
                    self._knn_scores, feats, labels, sub, feats[test_idx],
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
                    self._knn_scores, X_train, y_train, sub, X_test,
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
