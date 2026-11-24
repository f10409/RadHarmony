"""Checks on the source IDs of a harmonized table (D-01)."""
from __future__ import annotations

import warnings

import pandas as pd


def warn_shared_study_ids(df: pd.DataFrame, dataset: str, source_col: str) -> list[str]:
    """Warn when one ``study_id`` belongs to more than one ``patient_id`` (D-01).

    The harmonizer keeps the source study ID. So code that groups by study must use
    ``(patient_id, study_id)``.

    Args:
        df: Table with ``patient_id`` and ``study_id``.
        dataset: Dataset name for the message, for example ``"BRAX"``.
        source_col: Source column of ``study_id``, for the message.

    Returns:
        The shared study IDs, sorted. An empty list gives no warning.
    """
    n = df.groupby("study_id")["patient_id"].nunique()
    shared = sorted(n[n > 1].index)
    if shared:
        warnings.warn(
            f"{dataset}: {len(shared)} study_id ({source_col}) values belong to more than "
            f"one patient: {', '.join(shared)}. Group by patient_id and study_id together.",
            UserWarning,
            stacklevel=3,
        )
    return shared
