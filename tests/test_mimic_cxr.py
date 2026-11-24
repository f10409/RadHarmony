"""Tests for the MIMIC-CXR (DICOM) harmonizer on a small fake folder.

Run with::

    pytest tests/test_mimic_cxr.py -v
"""
from __future__ import annotations

import pandas as pd
import pytest

from radharmony.harmonizer import MIMICCXRHarmonizer


def test_mimic_cxr_dicom_warns_about_images_without_labels(tmp_path):
    """V-H1-32: the DICOM class warns as the JPG class does (V-H1-18) when a study has no label row (D-25)."""
    pd.DataFrame({"subject_id": [10000001, 10000001, 10000002],
                  "study_id": [50000001, 50000001, 50000002],
                  "dicom_id": ["d1", "d2", "d3"],
                  "path": ["files/p10/p10000001/s50000001/d1.dcm", "files/p10/p10000001/s50000001/d2.dcm",
                           "files/p10/p10000002/s50000002/d3.dcm"]},
                 ).to_csv(tmp_path / "cxr-record-list.csv", index=False)
    pd.DataFrame({"subject_id": [10000001], "study_id": [50000001], "Atelectasis": [1.0]},
                 ).to_csv(tmp_path / "chexpert.csv", index=False)
    h = MIMICCXRHarmonizer(csv_path=str(tmp_path / "cxr-record-list.csv"),
                           label_csv_path=str(tmp_path / "chexpert.csv"))
    with pytest.warns(UserWarning, match="MIMIC-CXR: 1 images in 1 studies have no row"):
        df = h.harmonize(drop_uncertain=False)
    assert sorted(df["image_path"]) == ["p10/p10000001/s50000001/d1.dcm", "p10/p10000001/s50000001/d2.dcm"]


def _dicom_with_view(path, view: str) -> None:
    """Write a small DICOM file with the tag ``ViewPosition``."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.ViewPosition = view
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)


def test_no_dicom_base_dir_gives_no_view_position(tmp_path):
    """D-01: the view comes only from dicom_base_dir. The DICOM files in base_image_dir are not read for it."""
    from radharmony.dataset import MIMICCXRDataset

    files = tmp_path / "files"
    _dicom_with_view(files / "p10" / "p10000001" / "s50000001" / "d1.dcm", "PA")
    pd.DataFrame({"subject_id": [10000001], "study_id": [50000001], "dicom_id": ["d1"],
                  "path": ["files/p10/p10000001/s50000001/d1.dcm"]}).to_csv(tmp_path / "cxr-record-list.csv",
                                                                            index=False)
    without = MIMICCXRDataset(base_image_dir=str(files), cache_dir=None)._get_harmonized_df()
    assert "view_position" not in without.columns
    with_dir = MIMICCXRDataset(base_image_dir=str(files), dicom_base_dir=str(files), cache_dir=None)
    assert with_dir._get_harmonized_df()["view_position"].tolist() == ["PA"]
