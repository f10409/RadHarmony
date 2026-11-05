"""The harmonizer contract (L-008, L-014): the checks that every harmonizer must pass.

``contract_failures(h, df, tmp_path)`` runs the checks on a harmonizer ``h``
and the result ``df`` of ``h.harmonize()``. CI runs it on fake data
(``tests/test_harmonizer_contract.py``). The full run runs it on real data
(``tests/test_dataset_stats.py``).
"""
from __future__ import annotations

import copy
import json
import math
import numbers
import os

import pandas as pd

from radharmony.harmonizer.base import BaseHarmonizer

def _snake(name: str) -> str:
    return name.lower().replace(" ", "_")


def _boxes(cell) -> list:
    """The boxes in one ``bbox`` cell: a list of boxes, one flat box or a JSON string."""
    if isinstance(cell, str):
        cell = json.loads(cell) if cell.strip() else []
    if cell is None or (not isinstance(cell, (list, tuple)) and pd.isna(cell)) or len(cell) == 0:
        return []
    return list(cell) if isinstance(cell[0], (list, tuple)) else [cell]


def _labels(cell) -> list:
    """The labels in one ``bbox_labels`` cell: a list or a JSON string."""
    if isinstance(cell, str):
        cell = json.loads(cell) if cell.strip() else []
    return list(cell) if isinstance(cell, (list, tuple)) else []


def _sorted_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["study_id", "image_path"], kind="stable").reset_index(drop=True)


def contract_failures(h, df: pd.DataFrame, tmp_path, reharmonize: bool = True,
                      dataset_cls=None) -> dict[str, str]:
    """Run the L-008 and L-014 checks. Return ``{check: problem}`` for each check that fails.

    Args:
        h: the harmonizer, after ``df = h.harmonize()``. The checks do not change it.
        df: the result of ``h.harmonize()``.
        tmp_path: a folder for the saved file.
        reharmonize: also harmonize the loaded harmonizer again. A full harmonize
            of a large dataset is slow, so the run on real data turns it off.
        dataset_cls: the dataset class of the registry key. When given, it must load
            the saved harmonizer with ``harmonizer_path=`` (L-014). Only the run on real
            data knows the key, so CI does not run this check.
    """
    out = {}
    # D-24: harmonized_df equals the last harmonize() result.
    try:
        pd.testing.assert_frame_equal(h.harmonized_df, df)
    except AssertionError as e:
        out["harmonized_df"] = str(e).splitlines()[0]

    # D-08: the same table after save and load.
    path = str(tmp_path / "contract.pkl")
    h.save(path)
    loaded = type(h).load_from_saved(path)
    try:
        pd.testing.assert_frame_equal(loaded.harmonized_df, df)
    except AssertionError as e:
        out["save_load"] = str(e).splitlines()[0]
    loaded_labels = list(getattr(loaded, "LABEL_COLS", []))
    # D-08 (L-014): the dataset class loads its own saved harmonizer. It is not empty, and its
    # image paths are in the table. It can drop rows or change label values on purpose (for
    # example uncertain labels), so the values are not compared. Its label columns in the table
    # are the harmonizer's. A test split has none in the table.
    if dataset_cls is not None:
        try:
            ds = dataset_cls(harmonizer_path=path, cache_dir=None)
            got = ds.get_harmonized_df()
            extra = set(got["image_path"]) - set(df["image_path"])
            if extra or (len(df) and not len(got)):
                out["harmonizer_path"] = f"{len(got)} rows, {len(extra)} not in the table"
            if hasattr(h, "get_label_cols"):
                names = {n for c in h.get_label_cols() for n in (c, _snake(c))}
                want = sorted(c for c in df.columns if c in names)
                have = sorted(c for c in ds.LABEL_COLS if c in df.columns)
                if have != want:
                    wrong = sorted(set(have) ^ set(want))
                    out["harmonizer_path_labels"] = f"{len(wrong)} label columns differ, for example {wrong[:3]}"
        except Exception as e:  # noqa: BLE001  (any error is a contract failure)
            out["harmonizer_path"] = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
    # D-24 and D-08: harmonize() replaces the table that load_from_saved() put there,
    # and the loaded arguments give the same table.
    if reharmonize:
        try:
            stale = type(h).load_from_saved(path)
            stale.set_harmonized_df(df.iloc[:0])
            again = stale.harmonize()
            pd.testing.assert_frame_equal(stale.harmonized_df, again)
            pd.testing.assert_frame_equal(again, df)
        except Exception as e:  # noqa: BLE001  (any error is a contract failure)
            out["harmonize_after_load"] = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"

    # D-05: LABEL_COLS and the table label columns are sorted by snake_case name.
    if hasattr(h, "get_label_cols"):
        labels = [_snake(c) for c in h.LABEL_COLS]
        in_table = [c for c in df.columns if c in set(labels)]
        if labels != sorted(labels) or in_table != sorted(in_table):
            out["label_order"] = f"not sorted: {labels[:4]}..."
        # D-08: LABEL_COLS has the same items in the same order after load.
        if loaded_labels != list(h.LABEL_COLS):
            out["label_order_after_load"] = f"{list(h.LABEL_COLS)[:4]} -> {loaded_labels[:4]}"

    # D-02: study_ids= gives the rows of a filter after harmonize(). The VQA harmonizers have no study_ids=.
    if isinstance(h, BaseHarmonizer) and len(df):
        sid = df["study_id"].iloc[0]
        try:
            got = copy.deepcopy(h).harmonize(study_ids=[sid])
            pd.testing.assert_frame_equal(_sorted_rows(got), _sorted_rows(df[df["study_id"] == sid]))
        except Exception as e:  # noqa: BLE001  (any error is a contract failure)
            out["study_ids"] = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"

    # D-02 (L-014): a study id that no row has gives 0 rows with the same columns. A split
    # class gets such an id when the study is in another split.
    if isinstance(h, BaseHarmonizer):
        try:
            none = copy.deepcopy(h).harmonize(study_ids=["no_such_study"])
            if len(none) or list(none.columns) != list(df.columns):
                out["no_study"] = f"{len(none)} rows, columns {list(none.columns)[:4]}"
        except Exception as e:  # noqa: BLE001  (any error is a contract failure)
            out["no_study"] = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"

    # D-04: image_path is relative to base_image_dir, so the table works on any computer.
    if "image_path" in df.columns:
        bad = [p for p in df["image_path"].dropna() if os.path.isabs(str(p))]
        if bad:
            out["image_path_relative"] = f"{len(bad)} absolute image paths, for example {bad[0]!r}"

    # D-07: every box value is in [0, 1] (NaN and inf fail).
    if "bbox" in df.columns:
        bad = [b for cell in df["bbox"] for b in _boxes(cell)
               if not all(isinstance(v, numbers.Real) and math.isfinite(v) and 0 <= v <= 1 for v in b)]
        if bad:
            out["boxes"] = f"{len(bad)} boxes not in [0, 1], for example {bad[0]}"

    # D-07 (Q-17): each box label is snake case, as the label columns (D-05). PadChest-GR
    # joins the labels of one sentence with " + ", so each part is checked.
    if "bbox_labels" in df.columns:
        bad = [lab for cell in df["bbox_labels"] for lab in _labels(cell)
               if any(part != _snake(part) for part in str(lab).split(" + "))]
        if bad:
            out["box_labels"] = f"{len(bad)} box labels not in snake case, for example {bad[0]!r}"
    return out
