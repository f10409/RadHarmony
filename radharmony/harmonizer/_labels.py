"""The CheXpert-style label rule (D-05): 1, 0, -1 (uncertain) and a blank (not mentioned)."""

import warnings

import pandas as pd

#: The values of ``not_mentioned``.
NOT_MENTIONED = ("negative", "nan")


def chexpert_codes(df: pd.DataFrame, cols, not_mentioned: str = "negative",
                   drop_uncertain: bool = True) -> pd.DataFrame:
    """Apply the CheXpert-style label rule of D-05 to the label columns ``cols`` of ``df``.

    -1 (uncertain) stays -1, and ``drop_uncertain=True`` drops the rows with -1 in any of
    ``cols``. A blank (not mentioned) becomes 0 with ``not_mentioned="negative"`` (the
    default, as most CheXpert papers do), and stays NaN with ``"nan"``. A row with every
    label blank has no label at all, so it stays NaN in both cases.

    Args:
        df: the table. It is not changed.
        cols: the label columns. The function skips the ones that ``df`` does not have.
        not_mentioned: ``"negative"`` or ``"nan"``.
        drop_uncertain: drop the rows with -1.

    Returns:
        The new table, with the index of ``df``.

    Raises:
        ValueError: for another value of ``not_mentioned``.
    """
    if not_mentioned not in NOT_MENTIONED:
        raise ValueError(f"not_mentioned must be one of {NOT_MENTIONED}, not {not_mentioned!r} (D-05).")
    df = df.copy()
    cols = [c for c in cols if c in df.columns]
    if not_mentioned == "negative":
        # A row with no label at all is unknown, not negative (D-05, V-H1-41).
        labeled = df[cols].notna().any(axis=1)
        df.loc[labeled, cols] = df.loc[labeled, cols].fillna(0.0)
    if drop_uncertain:
        df = df[~(df[cols] == -1).any(axis=1)]
    return df


def warn_images_without_labels(df: pd.DataFrame, label_df: pd.DataFrame, name: str, label_path) -> None:
    """Warn about the images whose study has no row in the label table (D-25, J-13).

    A labeled table keeps only the labeled rows, so the label merge leaves these
    images out. Example: in MIMIC-CXR 2.0.0, 15 train images in 8 studies have no
    row in ``mimic-cxr-2.0.0-chexpert.csv`` (V-H1-18, V-H1-32).

    Args:
        df: the image table, with a ``study_id`` column.
        label_df: the label table, with a ``study_id`` column.
        name: the dataset name at the start of the message.
        label_path: the label file, for the message.
    """
    missing = ~df["study_id"].astype(str).isin(set(label_df["study_id"].astype(str)))
    if missing.any():
        warnings.warn(
            f"{name}: {int(missing.sum())} images in {df.loc[missing, 'study_id'].nunique()} "
            f"studies have no row in {label_path}, so the table leaves them out (D-25).",
            UserWarning,
            stacklevel=3,
        )
