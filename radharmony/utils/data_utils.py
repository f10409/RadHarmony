import itertools
import warnings

import numpy as np
import pandas as pd
from functools import partial
from concurrent.futures import ProcessPoolExecutor


#: Output keys that aggregate *multiple* source columns into a per-row list.
#: ``cls`` for one-hot/binary labels; ``reg`` for continuous regression targets.
_LIST_OUTPUT_KEYS: frozenset = frozenset({"cls", "reg"})


def _is_empty(values: pd.Series) -> pd.Series:
    """True where a value is missing: NaN, None or a blank string."""
    return values.isna() | values.astype(str).str.strip().eq("")


def _required_cols(df, cols) -> list[str]:
    """Return the requested columns whose empty value drops the row (D-33).

    These are the file path columns (ending with "_path", for example
    ``mask_path``) and ``report``. A box column stays: no box is a valid sample.
    """
    return [
        df_col for out_key, df_col in (cols or {}).items()
        if out_key not in _LIST_OUTPUT_KEYS
        and df_col in df.columns
        and (df_col.endswith("_path") or out_key == "report")
    ]


def kept_rows(df, img_col, cols=None):
    """Return the rows of ``df`` that ``get_data_dict`` turns into samples.

    The evaluators use it too, so that their patient and image ids follow the
    same rows as the dataset (V-E-9).

    Args:
        df: the harmonized table.
        img_col: the image path column.
        cols: dict mapping output key to df column(s), as in ``get_data_dict``.
    """
    # Drop rows with a null or blank image path. They have no loadable file (V-P-23).
    df = df[~_is_empty(df[img_col])]

    # Drop rows with no value for a requested mask or report (D-33). Then an
    # empty path never reaches MONAI's LoadImageD, and a report is never NaN.
    for df_col in _required_cols(df, cols):
        df = df[~_is_empty(df[df_col])]
    return df


def get_data_dict_part(df_part, base_path, img_col, cols=None):
    """
    Optimized worker function using vectorization.

    cols: dict mapping output key → df column(s).
          "cls" / "reg" → list of column names (aggregated into a list per row).
          Any other key → single column name (passed through as string).
          Omit a key entirely to exclude it from the output.
    """
    df_temp = kept_rows(df_part.copy(), img_col, cols)

    if df_temp.empty:
        return []

    # Vectorized string concatenation, if base_path is None or empty, use
    # paths as-is (they must be absolute in that case).
    if base_path:
        if not base_path.endswith("/"):
            base_path += "/"
        df_temp["img"] = base_path + df_temp[img_col].astype(str)
    else:
        df_temp["img"] = df_temp[img_col].astype(str)

    cols_to_keep = ["img"]

    for out_key, df_col in (cols or {}).items():
        if out_key in _LIST_OUTPUT_KEYS:
            present = [c for c in df_col if c in df_temp.columns]
            if present:
                df_temp[out_key] = df_temp.apply(
                    lambda x, ps=present: [x[c] for c in ps], axis=1
                )
                cols_to_keep.append(out_key)
        elif df_col in df_temp.columns:
            # Pass list-valued columns (for example bbox) through as-is. Cast the
            # rest (file paths) to str so MONAI can read them.
            first_val = df_temp[df_col].dropna().iloc[0] if not df_temp[df_col].dropna().empty else None
            if isinstance(first_val, list):
                df_temp[out_key] = df_temp[df_col]
            else:
                df_temp[out_key] = df_temp[df_col].astype(str)
            cols_to_keep.append(out_key)

    return df_temp[cols_to_keep].to_dict("records")


def get_data_dict(df, base_path, img_path_col, cols=None, num_cores=2):
    """
    Main function that accepts configuration as arguments and parallelizes processing.

    cols: dict mapping output key → df column(s). See get_data_dict_part for details.
    """
    # Warn about requested columns missing from the DataFrame so the user
    # knows *before* training if a key is absent from data dicts.
    for out_key, df_col in (cols or {}).items():
        if out_key in _LIST_OUTPUT_KEYS:
            missing = [c for c in df_col if c not in df.columns]
            if missing:
                warnings.warn(
                    f"output_{out_key}=True but {len(missing)} "
                    f"{'label' if out_key == 'cls' else 'regression'} column(s) "
                    f"are missing from the DataFrame: {missing}. "
                    f"The '{out_key}' vector may be incomplete.",
                    UserWarning,
                    stacklevel=2,
                )
        elif df_col not in df.columns:
            warnings.warn(
                f"output_{out_key}=True but column '{df_col}' is missing "
                "from the harmonized DataFrame. "
                f"The '{out_key}' key will be absent from data dicts.",
                UserWarning,
                stacklevel=2,
            )

    # Warn here, in the main process: a warning in a worker process is lost.
    with_img = df[~_is_empty(df[img_path_col])]
    for df_col in _required_cols(with_img, cols):
        n = int(_is_empty(with_img[df_col]).sum())
        if n:
            warnings.warn(
                f"{n} of {len(with_img)} rows have no '{df_col}', so the dataset drops them (D-33).",
                UserWarning,
                stacklevel=2,
            )

    # Freeze the configuration arguments for the worker function
    func = partial(
        get_data_dict_part,
        base_path=base_path,
        img_col=img_path_col,
        cols=cols,
    )

    # If single core, run directly (good for debugging). The evaluators pass their
    # DataLoader num_workers here, and 0 is a normal value for it, so 0 means 1 core.
    if num_cores <= 1:
        return func(df)

    # Split dataframe for parallel processing, use iloc to guarantee DataFrame slices
    indices = np.array_split(np.arange(len(df)), num_cores)
    parts = [df.iloc[idx] for idx in indices if len(idx) > 0]

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        data_dicts = list(executor.map(func, parts))

    return list(itertools.chain(*data_dicts))


def split_data(df, group_column, n_splits, random_state=56):
    """Split ``df`` into ``(train, val)`` by group. Val gets about ``1 / n_splits`` of the groups.

    Raises:
        ValueError: when train or val has no group (D-10), for example ``n_splits=1``.
    """
    frac = 1 / n_splits
    unique_groups = df[group_column].drop_duplicates()
    val_groups = unique_groups.sample(frac=frac, random_state=random_state)

    is_val = df[group_column].isin(val_groups)

    df_val = df[is_val].reset_index(drop=True)
    df_train = df[~is_val].reset_index(drop=True)
    if df_train.empty or df_val.empty:
        raise ValueError(
            f"n_splits={n_splits} with {len(unique_groups)} patients gives an empty "
            f"{'train' if df_train.empty else 'validation'} split (D-10)."
        )

    return df_train, df_val


def kfold_splits(df, group_column, n_splits, random_state=56):
    """Yield (train_df, val_df) for each of n_splits folds.

    Groups (patients) are shuffled once then divided into n_splits equal
    buckets.  Each fold uses one bucket as val and the rest as train.

    Args:
        df: DataFrame to split.
        group_column: Column whose unique values define groups (for example patient_id).
        n_splits: Number of folds.
        random_state: Shuffle seed.

    Yields:
        Tuple of (train_df, val_df) for each fold.

    Raises:
        ValueError: when a fold has no group in train or in val (D-10).
    """
    unique_groups = (
        df[group_column].drop_duplicates().sample(frac=1, random_state=random_state)
    )
    if not 2 <= n_splits <= len(unique_groups):
        raise ValueError(
            f"n_splits={n_splits} with {len(unique_groups)} patients gives an empty fold (D-10). "
            "Use 2 to the number of patients."
        )
    buckets = np.array_split(unique_groups.values, n_splits)

    for val_groups in buckets:
        is_val = df[group_column].isin(val_groups)
        yield df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)
