"""Classification metric panel for radharmony evaluators.

Per-label panel:

- Threshold-free: ``auroc``, ``auprc``.
- Threshold-based: ``f1``, ``accuracy``, ``balanced_accuracy``, ``tpr`` (sensitivity),
  ``tnr`` (specificity), ``ppv`` (precision), ``npv``, ``mcc``, plus the chosen
  ``threshold``.

Threshold selection via ``threshold_strategy``:

- ``"youden"`` (default): argmax(TPR - FPR) on the ROC curve.
- ``"f1"``: argmax F1 on the PR curve.
- ``"fixed:<float>"``: numeric threshold, e.g. ``"fixed:0.5"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    matthews_corrcoef,
    balanced_accuracy_score,
    accuracy_score,
)


METRIC_KEYS = [
    "auroc",
    "auprc",
    "f1",
    "accuracy",
    "balanced_accuracy",
    "tpr",
    "tnr",
    "ppv",
    "npv",
    "mcc",
    "threshold",
]


def _nan_metrics() -> dict:
    return {k: np.nan for k in METRIC_KEYS}


def _pick_threshold(y_true: np.ndarray, y_score: np.ndarray, strategy: str) -> float:
    if strategy.startswith("fixed:"):
        return float(strategy.split(":", 1)[1])

    if strategy == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_score)
        # roc_curve prepends an "inf" threshold at index 0; skip it if present
        j = tpr - fpr
        return float(thr[int(np.argmax(j))])

    if strategy == "f1":
        prec, rec, thr = precision_recall_curve(y_true, y_score)
        # thr has length len(prec)-1 — align by dropping the last prec/rec
        f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
        if len(thr) == 0:
            return 0.5
        return float(thr[int(np.argmax(f1))])

    raise ValueError(
        f"Unknown threshold_strategy: {strategy!r}. "
        f"Expected 'youden', 'f1', or 'fixed:<float>'."
    )


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold_strategy: str = "youden",
) -> dict:
    """Compute the full metric panel for one binary-label column.

    Returns NaN-filled dict if the label is degenerate (<2 positives or <2 negatives).
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    # Multi-label datasets (e.g. Emory gpt-oss-120b labels) routinely have NaN
    # for ungraded conditions per row. Mask them out per-label before scoring.
    # Two failure modes: (a) NaN survives all the way → roc_auc_score raises
    # "Input y_true contains NaN"; (b) NaN gets int-cast somewhere upstream →
    # turns into integer garbage like -2147483648, which then trips sklearn's
    # multi-class detector ("multi_class must be in ('ovo','ovr')"). Reject
    # anything that isn't exactly 0 or 1 (or 0.0/1.0).
    y_true_f = y_true.astype(np.float64, copy=False)
    finite = ~np.isnan(y_true_f) & np.isin(y_true_f, [0.0, 1.0])
    y_true = y_true_f[finite].astype(int)
    y_score = y_score[finite]

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos < 2 or n_neg < 2:
        return _nan_metrics()

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    thr = _pick_threshold(y_true, y_score, threshold_strategy)
    y_pred = (y_score >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    f1 = (
        2 * ppv * tpr / (ppv + tpr)
        if (ppv is not np.nan and tpr is not np.nan and (ppv + tpr) > 0)
        else np.nan
    )

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1": float(f1) if not np.isnan(f1) else np.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "tpr": float(tpr) if not np.isnan(tpr) else np.nan,
        "tnr": float(tnr) if not np.isnan(tnr) else np.nan,
        "ppv": float(ppv) if not np.isnan(ppv) else np.nan,
        "npv": float(npv) if not np.isnan(npv) else np.nan,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "threshold": float(thr),
    }


def macro_average(df: pd.DataFrame) -> pd.DataFrame:
    """Append ``label='macro_average'`` rows per variance group.

    Groups by every variance axis present in *df* among
    {``fold``, ``seed``, ``bootstrap``, ``n_train``} and takes ``np.nanmean``
    across per-label rows (excluding any pre-existing ``macro_average`` rows).
    """
    if df.empty or "label" not in df.columns:
        return df

    base = df[df["label"] != "macro_average"]
    group_cols = [c for c in ["fold", "seed", "bootstrap", "n_train"] if c in df.columns]

    reserved = {"label", "fold", "seed", "bootstrap", "n_train"}
    metric_cols = [
        c for c in base.columns
        if c not in reserved and pd.api.types.is_numeric_dtype(base[c])
    ]

    macro_rows = []
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
