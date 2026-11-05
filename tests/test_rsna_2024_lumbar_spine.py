"""RSNA 2024 lumbar spine tests: lost points warn (D-07), and the dataset class options.

The tests build a tiny fake source folder in ``tmp_path``. The expected values
come from the fake source files, not from the harmonizer.
"""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

from test_harmonizer_boxes import _write_slice


def _write_source(tmp_path, points):
    """Write train.csv, the series CSV and the point CSV for study 1, series 10.

    ``points`` is a list of InstanceNumber values. Each gets one point at (4, 2).
    """
    from radharmony.harmonizer.rsna_2024_lumbar_spine.rsna_2024_lumbar_spine import _BASE_TARGETS

    pd.DataFrame({"study_id": [1], **{t: ["Normal/Mild"] for t in _BASE_TARGETS}}
                 ).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"study_id": [1], "series_id": [10], "series_description": ["Sagittal T1"]}
                 ).to_csv(tmp_path / "series.csv", index=False)
    pd.DataFrame({"study_id": 1, "series_id": 10, "instance_number": points,
                  "condition": "Spinal Canal Stenosis", "level": "L1/L2", "x": 4.0, "y": 2.0}
                 ).to_csv(tmp_path / "coords.csv", index=False)


def _harmonize(tmp_path):
    from radharmony.harmonizer import RSNA2024LumbarSpineTrainHarmonizer

    return RSNA2024LumbarSpineTrainHarmonizer(
        csv_path=str(tmp_path / "train.csv"), series_description_csv_path=str(tmp_path / "series.csv"),
        coord_csv_path=str(tmp_path / "coords.csv"), base_image_dir=str(tmp_path / "img"),
        num_workers=1).harmonize()


def test_points_on_a_second_sub_series_warn_with_the_count(tmp_path):
    """V-H3-33: the 3-D loader reads only the first GDCM sub-series of a folder.

    Slices 1 to 3 have one SeriesDate and slices 4 and 5 another, so GDCM gives 2
    sub-series (as series 1973833645/1328374636). Of the 3 points, 2 are on slices
    4 and 5. So the table has 1 box, and the harmonizer warns about the 2 lost points.
    """
    from pydicom.uid import generate_uid

    series, study = generate_uid(), generate_uid()
    for i in range(1, 6):
        _write_slice(tmp_path / "img" / "1" / "10" / f"{i}.dcm", i, (-4.0 * i, 0.0, 100.0), 100 * i,
                     series, study)
    import pydicom
    for i in (4, 5):
        p = tmp_path / "img" / "1" / "10" / f"{i}.dcm"
        ds = pydicom.dcmread(p)
        ds.SeriesDate = "20200102"
        ds.save_as(p)
    _write_source(tmp_path, [2, 4, 5])
    with pytest.warns(UserWarning, match=r"2 of 3 points .*not on the loaded volume"):
        df = _harmonize(tmp_path)
    assert len(df.iloc[0]["bbox"]) == 1
    assert df.iloc[0]["bbox_labels"] == ["spinal_canal_stenosis_l1_l2"]


def test_a_series_that_cannot_be_read_warns(tmp_path):
    """R-code-20: a series folder with no readable DICOM gives no box, and a warning names it."""
    folder = tmp_path / "img" / "1" / "10"
    folder.mkdir(parents=True)
    (folder / "1.dcm").write_bytes(b"not a dicom file")
    _write_source(tmp_path, [1, 1])
    with pytest.warns(UserWarning, match=r"2 of 2 points .*1 series could not be read.*1/10"):
        df = _harmonize(tmp_path)
    assert df.iloc[0]["bbox"] == []


def test_all_points_on_the_volume_give_no_warning(tmp_path):
    """V-H3-33 (control): a series with one sub-series keeps every point, so nothing warns."""
    from pydicom.uid import generate_uid

    series, study = generate_uid(), generate_uid()
    for i in range(1, 4):
        _write_slice(tmp_path / "img" / "1" / "10" / f"{i}.dcm", i, (-4.0 * i, 0.0, 100.0), 100 * i,
                     series, study)
    _write_source(tmp_path, [1, 3])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        df = _harmonize(tmp_path)
    assert len(df.iloc[0]["bbox"]) == 2


# --------------------------------------------------------------------------
# Dataset classes
# --------------------------------------------------------------------------

def _table():
    """A harmonized table with one series of each type (as the harmonizer writes it)."""
    return pd.DataFrame({
        "patient_id": ["1", "1", "1"], "study_id": ["1", "1", "1"], "series_id": ["10", "11", "12"],
        "image_path": ["1/10", "1/11", "1/12"],
        "view_position": ["Axial T2", "Sagittal T1", "Sagittal T2/STIR"],
    })


@pytest.mark.parametrize("cls_name", ["RSNA2024LumbarSpineTrainDataset", "RSNA2024LumbarSpineTestDataset"])
def test_series_filter_applies_to_a_given_table(cls_name, tmp_path):
    """V-D-32: ``series_filter`` keeps only that series type, also with ``harmonized_df=``."""
    import radharmony.dataset as rd

    ds = getattr(rd, cls_name)(base_image_dir=str(tmp_path), harmonized_df=_table(), cache_dir=None,
                               series_filter="Axial T2")
    assert ds.get_harmonized_df()["view_position"].tolist() == ["Axial T2"]


def test_output_bbox_with_no_point_csv_leaves_out_the_box_keys(tmp_path):
    """V-D-40: with no ``bbox`` column, ``output_bbox=True`` warns, and ``ds[0]`` has no box keys (D-11).

    Before, the warning promised empty boxes, but ``ds[0]`` raised KeyError 'bbox'.
    """
    import numpy as np
    import SimpleITK as sitk

    from radharmony.dataset import RSNA2024LumbarSpineDataset

    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((4, 8, 8), dtype=np.int16)), str(tmp_path / "v.nii.gz"))
    df = pd.DataFrame({"patient_id": ["1"], "study_id": ["1"], "image_path": ["v.nii.gz"]})
    with pytest.warns(UserWarning, match="no 'bbox' column"):
        ds = RSNA2024LumbarSpineDataset(base_image_dir=str(tmp_path), output_bbox=True, cache_dir=None,
                                        harmonized_df=df).get_datasets()
    assert sorted(ds[0]) == ["img"]
