"""RSNA Pneumonia (Kaggle) harmonizer tests on a tiny fake source folder in ``tmp_path``.

The expected values come from the fake DICOM headers, not from the harmonizer.
"""
from __future__ import annotations

import pandas as pd
import pytest

from test_harmonizer_boxes import _write_dicom


def _write_images(folder, views):
    """Write one header-only DICOM file ``p<i>.dcm`` for each ViewPosition in ``views``."""
    for i, view in enumerate(views):
        tags = {"PatientSex": "F", "PatientAge": "40"}
        if view is not None:
            tags["ViewPosition"] = view
        _write_dicom(folder / f"p{i}.dcm", rows=1024, cols=1024, **tags)


def test_train_view_position_comes_from_the_dicom_tag(tmp_path):
    """Q-18 (D-01): the release has no view column, so view_position is the DICOM tag ViewPosition.

    A file with no tag gives NaN.
    """
    from radharmony.harmonizer import RSNAPneumoniaKaggleHarmonizer

    _write_images(tmp_path / "img", ["PA", "AP", None])
    pd.DataFrame({"patientId": ["p0", "p1", "p2"], "class": ["Normal", "Lung Opacity", "Normal"]}
                 ).to_csv(tmp_path / "class.csv", index=False)
    df = RSNAPneumoniaKaggleHarmonizer(csv_path=str(tmp_path / "class.csv"),
                                       base_image_dir=str(tmp_path / "img")).harmonize()
    assert df.set_index("patient_id")["view_position"].fillna("NaN").to_dict() == {
        "p0": "PA", "p1": "AP", "p2": "NaN"}


def test_test_view_position_comes_from_the_dicom_tag(tmp_path):
    """Q-18: the test split reads the tag too."""
    from radharmony.harmonizer import RSNAPneumoniaKaggleTestHarmonizer

    _write_images(tmp_path, ["AP", "PA"])
    df = RSNAPneumoniaKaggleTestHarmonizer(base_image_dir=str(tmp_path)).harmonize()
    assert df["view_position"].tolist() == ["AP", "PA"]


def test_test_folder_with_no_dicom_names_the_folder(tmp_path):
    """V-H3-28 (D-35): a folder with no .dcm file raises FileNotFoundError with the folder.

    Example: the parent folder of ``stage_2_test_images/`` (V-H3-5). Before, it raised KeyError.
    """
    from radharmony.harmonizer import RSNAPneumoniaKaggleTestHarmonizer

    (tmp_path / "stage_2_test_images").mkdir()
    with pytest.raises(FileNotFoundError, match=r"No \.dcm file in .*stage_2_test_images"):
        RSNAPneumoniaKaggleTestHarmonizer(base_image_dir=str(tmp_path)).harmonize()
