"""The base merges raise on a repeated key, so a table keeps one row per image (D-03, V-H2-17).

No data needed. Decided by Frank Li on 2026-10-29: raise an error, not collect the lines.
"""
from __future__ import annotations

import pandas as pd
import pytest

from radharmony.harmonizer.base import BaseHarmonizer


class _Fake(BaseHarmonizer):
    """3 images, a label CSV and a box CSV, joined on ``image_id``."""

    LABEL_COLS = ["effusion"]
    LABEL_JOIN_COLS = ["image_id"]
    BBOX_JOIN_COLS = ["image_id"]
    BBOX_SOURCE_COL = "box"

    def _build_patient_id(self):
        self.df["patient_id"] = self.df["image_id"]

    def _build_study_id(self):
        self.df["study_id"] = self.df["image_id"]

    def _build_image_path(self):
        self.df["image_path"] = self.df["image_id"] + ".png"


def _write(tmp_path, labels, boxes):
    pd.DataFrame({"image_id": ["a", "b", "c"]}).to_csv(tmp_path / "images.csv", index=False)
    pd.DataFrame(labels).to_csv(tmp_path / "labels.csv", index=False)
    pd.DataFrame(boxes).to_csv(tmp_path / "boxes.csv", index=False)
    return _Fake(csv_path=str(tmp_path / "images.csv"), label_csv_path=str(tmp_path / "labels.csv"),
                 bbox_csv_path=str(tmp_path / "boxes.csv"))


def test_a_repeated_box_key_raises(tmp_path):
    """Image ``a`` has 2 lines in the box CSV. The error names the key and the hook."""
    h = _write(tmp_path, {"image_id": ["a", "b", "c"], "effusion": [1, 0, 0]},
               {"image_id": ["a", "a", "b", "c"], "box": ["x", "y", "z", "w"]})
    pattern = r"box file has more than one line for \{'image_id': 'a'\}.*_preprocess_bbox_df"
    with pytest.raises(ValueError, match=pattern):
        h.harmonize()


def test_a_repeated_label_key_raises(tmp_path):
    h = _write(tmp_path, {"image_id": ["a", "a", "b", "c"], "effusion": [1, 0, 0, 0]},
               {"image_id": ["a", "b", "c"], "box": ["x", "y", "z"]})
    pattern = r"label file has more than one line for \{'image_id': 'a'\}.*_preprocess_label_df"
    with pytest.raises(ValueError, match=pattern):
        h.harmonize()


def test_one_line_per_key_keeps_one_row_per_image(tmp_path):
    h = _write(tmp_path, {"image_id": ["a", "b", "c"], "effusion": [1, 0, 0]},
               {"image_id": ["a", "b", "c"], "box": ["x", "y", "z"]})
    df = h.harmonize()
    assert sorted(df["image_path"]) == ["a.png", "b.png", "c.png"]
    assert sorted(df["bbox"]) == ["x", "y", "z"]


class _FakeReport(_Fake):
    """The images of ``_Fake``, with one report file for each study."""

    LABEL_COLS = []
    BBOX_SOURCE_COL = None
    REPORT_JOIN_COLS = ["study_id"]
    REPORT_PATH_COL = "path"


def _report_harmonizer(tmp_path, studies):
    """Write 3 images and a report CSV with one line for each item of ``studies``."""
    pd.DataFrame({"image_id": ["a", "b", "c"]}).to_csv(tmp_path / "images.csv", index=False)
    paths = [f"r{i}.txt" for i in range(len(studies))]
    for p in paths:
        (tmp_path / p).write_text("text " + p)
    pd.DataFrame({"study_id": studies, "path": paths}).to_csv(tmp_path / "reports.csv", index=False)
    return _FakeReport(csv_path=str(tmp_path / "images.csv"), report_csv_path=str(tmp_path / "reports.csv"),
                       report_base_dir=str(tmp_path))


def test_a_repeated_report_key_raises(tmp_path):
    """V-H2-31 (D-03): study ``a`` has 2 report lines. Before, image ``a`` got 2 rows (4 rows from 3 images)."""
    h = _report_harmonizer(tmp_path, ["a", "a", "b", "c"])
    pattern = r"report file has more than one line for \{'study_id': 'a'\}.*_preprocess_report_df"
    with pytest.raises(ValueError, match=pattern):
        h.harmonize()


def test_one_report_line_per_key_keeps_one_row_per_image(tmp_path):
    """V-H2-31: with one report line per study, each image gets its own report and no row is added."""
    df = _report_harmonizer(tmp_path, ["a", "b", "c"]).harmonize()
    assert df[["image_path", "report"]].values.tolist() == [
        ["a.png", "text r0.txt"], ["b.png", "text r1.txt"], ["c.png", "text r2.txt"]]
