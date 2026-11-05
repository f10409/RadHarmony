"""VinDr-PCXR harmonizer tests on a tiny fake source folder in ``tmp_path``."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from test_harmonizer_boxes import _write_dicom


def test_an_unreadable_dicom_of_a_boxed_image_warns(tmp_path):
    """R-code-19: a boxed image whose DICOM file cannot be read loses its box, and a warning says so.

    ``train_2`` has a box but its file is not a DICOM file. ``train_3`` has a box and no file.
    ``train_1`` has only a "No finding" row (no coordinates), so it does not count.
    """
    from radharmony.harmonizer.vindr_pcxr.vindr_pcxr import _LABEL_MAP, VinDrPCXRHarmonizer

    nan = float("nan")
    for split in ("train", "test"):
        ids = [f"{split}_1", f"{split}_2", f"{split}_3"]
        _write_dicom(tmp_path / split / f"{split}_1.dicom", rows=2000, cols=1000)
        if split == "train":
            (tmp_path / split / "train_2.dicom").write_bytes(b"not a dicom file")
        else:
            _write_dicom(tmp_path / split / "test_2.dicom", rows=2000, cols=1000)
            _write_dicom(tmp_path / split / "test_3.dicom", rows=2000, cols=1000)
        pd.DataFrame({"image_id": ids, **{c: [0, 0, 0] for c in _LABEL_MAP}}
                     ).to_csv(tmp_path / f"image_labels_{split}.csv", index=False)
        pd.DataFrame({"image_id": ids, "class_name": ["No finding", "Consolidation", "Cardiomegaly"],
                      "x_min": [nan, 100, 100], "y_min": [nan, 400, 400], "x_max": [nan, 300, 300],
                      "y_max": [nan, 800, 800]}).to_csv(tmp_path / f"annotations_{split}.csv", index=False)
    with pytest.warns(UserWarning, match=r"VinDr-PCXR train: 2 images with boxes have no readable DICOM"):
        df = VinDrPCXRHarmonizer(base_dir=str(tmp_path)).harmonize().set_index("study_id")
    assert pd.isna(df.loc["train_2", "bbox"]) and pd.isna(df.loc["train_3", "bbox"])
    assert json.loads(df.loc["test_3", "bbox_labels"]) == ["cardiomegaly"]
