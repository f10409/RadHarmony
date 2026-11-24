"""VinDr-CXR box tests on a tiny fake source folder in ``tmp_path``."""
from __future__ import annotations

import pandas as pd
import pytest

from test_harmonizer_boxes import _write_dicom


def _source(tmp_path, with_rad_id):
    """img0 has a Nodule/Mass box of 2 readers (train) or 1 agreed box (test)."""
    rad = {"rad_id": ["R1", "R2"]} if with_rad_id else {}
    pd.DataFrame({"image_id": ["img0", "img0"] if with_rad_id else ["img0"], "Nodule/Mass": 1,
                  **{k: v for k, v in rad.items()}}).to_csv(tmp_path / "labels.csv", index=False)
    rows = [("img0", 10, 10, 30, 30), ("img0", 11, 10, 31, 30)] if with_rad_id else [("img0", 10, 10, 30, 30)]
    pd.DataFrame(rows, columns=["image_id", "x_min", "y_min", "x_max", "y_max"]).assign(
        class_name="Nodule/Mass", **({"rad_id": ["R1", "R2"]} if with_rad_id else {})
    ).to_csv(tmp_path / "annotations.csv", index=False)
    _write_dicom(tmp_path / "img0.dicom", 100, 200)


@pytest.mark.parametrize("split", ["train", "test"])
def test_boxes_with_no_image_folder_warn(split, tmp_path):
    """V-H3-35: a box needs the image size, so a box CSV with no base_image_dir gives a warning.

    Before, every bbox was empty and nothing warned. With the folder, img0 keeps its box.
    """
    from radharmony.harmonizer.vindr_cxr.vindr_cxr_test import VinDrCXRTestHarmonizer
    from radharmony.harmonizer.vindr_cxr.vindr_cxr_train import VinDrCXRTrainHarmonizer

    cls = VinDrCXRTrainHarmonizer if split == "train" else VinDrCXRTestHarmonizer
    _source(tmp_path, with_rad_id=split == "train")
    kw = dict(csv_path=str(tmp_path / "labels.csv"), bbox_csv_path=str(tmp_path / "annotations.csv"))
    with pytest.warns(UserWarning, match=r"base_image_dir is not.*1 images get no box"):
        df = cls(**kw).harmonize()
    assert list(df.iloc[0]["bbox"]) == []
    df = cls(base_image_dir=str(tmp_path), **kw).harmonize()
    assert len(df.iloc[0]["bbox"]) == 1
