"""Shared steps of the classification probes.

These helpers are not base code: each probe calls them from its own loop.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..metrics.classification import METRIC_KEYS, compute_metrics, pick_threshold
from ..sampling import bootstrap_indices, carve_val, group_ids


def label_mask(y: np.ndarray) -> np.ndarray:
    """Return True where a label is 0 or 1.

    A -1 (uncertain), a NaN (not mentioned or not graded, D-05) and any other
    value give False. A probe fits each label only on the rows where this mask is True
    (D-18). ``compute_metrics`` uses the same rule for the test labels.
    """
    return np.isin(np.asarray(y, dtype=np.float64), (0.0, 1.0))


def label_order(train_names, test_names) -> list[int]:
    """Return the test column of each train label, matched by name (V-E-18).

    The match ignores case, as ``labels=`` does. Raises ``ValueError`` when the
    two label sets differ, because a probe for one label must not be scored on
    another label.

    Example: train ``["a", "b"]`` and test ``["b", "a"]`` give ``[1, 0]``.
    """
    lower = [str(n).lower() for n in test_names]
    if sorted(str(n).lower() for n in train_names) != sorted(lower):
        raise ValueError(
            f"The train and test datasets have different labels: train "
            f"{list(train_names)}, test {list(test_names)}. Pass labels= with "
            "the labels that both datasets have."
        )
    return [lower.index(str(n).lower()) for n in train_names]


def row_patient_ids(ev, ds, n_rows: int) -> np.ndarray:
    """Return one patient id per feature row of ``ds``.

    The base class takes the ids from the rows with an image path. When
    ``get_data_dict`` drops more rows (for example an empty ``mask_path``),
    those ids do not match the features. Then the ids come from the rows
    that the dataset gives (V-E-9).
    """
    pids = ev._patient_ids
    if pids is not None and len(pids) == n_rows:
        return pids
    pids = group_ids(ds)
    if len(pids) != n_rows:
        raise ValueError(f"{n_rows} feature rows, but {len(pids)} patient ids.")
    return pids


def load_fixed_split(ev):
    """Load the train and test features of a fixed-split probe ``ev``.

    The test label columns are put in the order of the train labels, by name
    (V-E-18). ``ev.active_labels`` are the train labels after the call.

    Returns ``(X_train, y_train, train_pids, X_test, y_test, test_pids)``.
    """
    X_train, y_train = ev._load_or_extract(ev.train_dataset, cache_suffix="train")
    train_labels = ev.active_labels
    train_pids = row_patient_ids(ev, ev.train_dataset, len(X_train))
    X_test, y_test = ev._load_or_extract(ev.test_dataset, cache_suffix="test")
    y_test = y_test[:, label_order(train_labels, ev.active_labels)]
    test_pids = row_patient_ids(ev, ev.test_dataset, len(X_test))
    ev._active_labels = train_labels
    return X_train, y_train, train_pids, X_test, y_test, test_pids


def pick_thresholds(y_val, s_val, strategy: str, n_labels: int) -> np.ndarray:
    """Pick one threshold per label on validation data (D-29).

    ``y_val`` and ``s_val`` are ``(N_val, L)``, or ``None`` when there is no
    validation part. Then every threshold is NaN, except for
    ``"fixed:<float>"``. See :func:`pick_threshold` for the other NaN cases.
    """
    if y_val is None:
        return np.full(n_labels, pick_threshold(None, None, strategy))
    return np.array([pick_threshold(y_val[:, k], s_val[:, k], strategy) for k in range(n_labels)])


def fit_score(
    fit_predict: Callable,
    X: np.ndarray,
    y: np.ndarray,
    sub: np.ndarray,
    X_test: np.ndarray,
    patient_ids: np.ndarray,
    rng,
    val_fraction: float,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a frozen-feature probe, then score the test rows (D-29).

    The train rows ``sub`` lose a validation part (``val_fraction`` of their
    patients). The probe fits on the other rows. Its scores on the validation
    part give one threshold per label.

    Args:
        fit_predict: ``(X_fit, y_fit, X_eval) -> scores (N_eval, L)``.
        X, y: train features and labels. ``sub`` and ``patient_ids`` index them.
        X_test: test features.
        rng: numpy Generator for the validation split.

    Returns:
        ``(test_scores, thresholds)``.
    """
    fit_idx, val_idx = carve_val(sub, rng, patient_ids, val_fraction)
    if val_idx is None:
        scores = fit_predict(X[fit_idx], y[fit_idx], X_test)
        return scores, pick_thresholds(None, None, strategy, y.shape[1])
    scores = fit_predict(X[fit_idx], y[fit_idx], np.concatenate([X[val_idx], X_test]))
    n_val = len(val_idx)
    thresholds = pick_thresholds(y[val_idx], scores[:n_val], strategy, y.shape[1])
    return scores[n_val:], thresholds


def score_rows(scores, y_test, names, thresholds, strategy, tmpl, rows=None) -> list[dict]:
    """Return one metric row per label.

    Args:
        scores, y_test: ``(N_test, L)`` scores and labels.
        names: label names, in column order.
        thresholds: one threshold per label, from validation data.
        strategy: the ``threshold_strategy`` (used only when a threshold is None).
        tmpl: the variance columns (``n_train``, ``fold``, ``seed``, ``bootstrap``).
        rows: test row indices to score (for a bootstrap sample). Default: all.

    A label with a NaN score gets NaN metrics.
    """
    idx = slice(None) if rows is None else rows
    out = []
    for k, name in enumerate(names):
        row = dict(tmpl, label=name)
        if np.isnan(scores[:, k]).any():
            row.update({m: np.nan for m in METRIC_KEYS})
        else:
            row.update(compute_metrics(y_test[idx, k], scores[idx, k], strategy, threshold=thresholds[k]))
        out.append(row)
    return out


def bootstrap_rows(
    scores, y_test, names, thresholds, strategy, tmpl, n_bootstrap, seed, groups=None
) -> list[dict]:
    """Return the metric rows of ``n_bootstrap`` bootstrap samples of the test set.

    Each sample draws test patients (``groups``, one id per test row) and keeps
    all their images (D-32). Every sample uses the same thresholds (D-29). The
    other arguments are the same as in :func:`score_rows`.
    """
    out = []
    for b, idx in enumerate(bootstrap_indices(len(y_test), n_bootstrap, seed, groups)):
        out.extend(score_rows(scores, y_test, names, thresholds, strategy, dict(tmpl, bootstrap=b), idx))
    return out
