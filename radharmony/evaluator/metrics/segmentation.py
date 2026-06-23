"""Segmentation metric panel for radharmony evaluators.

Per-class panel computed by binarizing both ``y_true`` and ``y_pred``
against each class index:

- ``dice``        — 2 * |A ∩ B| / (|A| + |B|)
- ``iou``         — |A ∩ B| / |A ∪ B|  (Jaccard)
- ``pixel_acc``   — (TP + TN) / N
- ``tpr``         — sensitivity / recall
- ``tnr``         — specificity
- ``ppv``         — precision
- ``npv``         — negative predictive value

The macro-average row excludes ``class_0`` (background) by default, matching
standard segmentation reporting where background is uninformative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


METRIC_KEYS = ["dice", "iou", "pixel_acc", "tpr", "tnr", "ppv", "npv"]


def _nan_metrics() -> dict:
    return {k: np.nan for k in METRIC_KEYS}


def _binary_panel(y_true_b: np.ndarray, y_pred_b: np.ndarray) -> dict:
    yt = y_true_b.astype(bool).ravel()
    yp = y_pred_b.astype(bool).ravel()
    tp = int(np.logical_and(yp, yt).sum())
    fp = int(np.logical_and(yp, ~yt).sum())
    fn = int(np.logical_and(~yp, yt).sum())
    tn = int(np.logical_and(~yp, ~yt).sum())
    pos_t = tp + fn
    pos_p = tp + fp
    n = tp + tn + fp + fn
    union = tp + fp + fn
    return {
        "dice": (2 * tp) / (pos_p + pos_t) if (pos_p + pos_t) > 0 else np.nan,
        "iou": tp / union if union > 0 else np.nan,
        "pixel_acc": (tp + tn) / n if n > 0 else np.nan,
        "tpr": tp / pos_t if pos_t > 0 else np.nan,
        "tnr": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "ppv": tp / pos_p if pos_p > 0 else np.nan,
        "npv": tn / (tn + fn) if (tn + fn) > 0 else np.nan,
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> dict[int, dict]:
    """Compute the per-class metric panel for one image.

    ``y_true`` / ``y_pred`` are integer arrays of identical shape containing
    class indices in ``[0, num_classes)``. Returns ``{class_idx: panel}``.
    """
    out: dict[int, dict] = {}
    for c in range(num_classes):
        out[c] = _binary_panel(y_true == c, y_pred == c)
    return out


def macro_average(df: pd.DataFrame, exclude_background: bool = True) -> pd.DataFrame:
    """Append ``label='macro_average'`` rows per variance group.

    Groups by every variance axis present in *df* among
    ``{fold, seed, bootstrap, n_train}`` and takes ``np.nanmean`` across
    per-class rows. By default the ``class_0`` (background) row is
    excluded from the macro.
    """
    if df.empty or "label" not in df.columns:
        return df

    base = df[df["label"] != "macro_average"]
    if exclude_background:
        base = base[base["label"] != "class_0"]
    group_cols = [c for c in ["fold", "seed", "bootstrap", "n_train"] if c in df.columns]

    reserved = {"label", "fold", "seed", "bootstrap", "n_train"}
    metric_cols = [
        c for c in base.columns
        if c not in reserved and pd.api.types.is_numeric_dtype(base[c])
    ]

    macro_rows: list[dict] = []
    if group_cols:
        grouped = base.groupby(group_cols, dropna=False)
    else:
        grouped = [((), base)]

    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row: dict = {"label": "macro_average"}
        row.update(dict(zip(group_cols, key)))
        for m in metric_cols:
            row[m] = float(np.nanmean(sub[m].to_numpy()))
        macro_rows.append(row)

    if not macro_rows:
        return df
    return pd.concat([df, pd.DataFrame(macro_rows)], ignore_index=True)
