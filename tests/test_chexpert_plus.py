"""Tests for the CheXpert Plus harmonizer on a small fake folder.

Run with::

    pytest tests/test_chexpert_plus.py -v
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

from radharmony.harmonizer.chexpert.chexpert_plus import CheXpertPlusHarmonizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dataset_paths import is_complete, load  # noqa: E402


def _fake(tmp_path, n_patients=3, on_disk=(0, 1)):
    """Write a CSV with one image per patient, and folders for ``on_disk`` only."""
    rows = []
    for i in range(n_patients):
        stem = f"train/patient{i:05d}/study1/view1_frontal"
        rows.append({"path_to_image": stem + ".jpg", "path_to_dcm": stem + ".dcm",
                     "frontal_lateral": "Frontal", "ap_pa": "AP", "split": "train",
                     "report": "No acute findings.", "age": 50, "sex": "Female",
                     "race": "White", "ethnicity": "Non-Hispanic"})
    csv = tmp_path / "df_chexpert_plus_240401.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    base = tmp_path / "Uncompressed"
    for i in on_disk:
        (base / "train" / f"patient{i:05d}" / "study1").mkdir(parents=True)
    return csv, base


def test_missing_patient_folders_warn(tmp_path):
    """V-H1-7, V-D-17: rows with no patient folder give a warning."""
    csv, base = _fake(tmp_path)
    h = CheXpertPlusHarmonizer(csv_path=str(csv), base_image_dir=str(base))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = h.harmonize()
    assert len(df) == 3  # the harmonizer keeps every row (D-02)
    msgs = [str(x.message) for x in w if "verify_images" in str(x.message)]
    assert msgs and "1 of 3 source rows" in msgs[0], [str(x.message) for x in w]


def test_no_warning_when_every_folder_exists(tmp_path):
    csv, base = _fake(tmp_path, on_disk=(0, 1, 2))
    h = CheXpertPlusHarmonizer(csv_path=str(csv), base_image_dir=str(base))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        h.harmonize()
    assert not [x for x in w if "verify_images" in str(x.message)]


def test_a_split_class_counts_the_missing_folders_of_its_own_split(tmp_path):
    """V-H1-45: the warning counts the rows of the split table, not the rows of both splits."""
    from radharmony.harmonizer import CheXpertPlusTrainHarmonizer, CheXpertPlusValidHarmonizer

    # 3 train rows with no folder. 2 valid rows: patient90000 has a folder, patient90001 has none.
    rows = []
    for split, pid in [("train", 0), ("train", 1), ("train", 2), ("valid", 90000), ("valid", 90001)]:
        stem = f"{split}/patient{pid:05d}/study1/view1_frontal"
        rows.append({"path_to_image": stem + ".jpg", "path_to_dcm": stem + ".dcm", "frontal_lateral": "Frontal",
                     "ap_pa": "AP", "split": split, "report": "x", "age": 50, "sex": "Female"})
    csv = tmp_path / "df_chexpert_plus_240401.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    base = tmp_path / "Uncompressed"
    (base / "valid" / "patient90000" / "study1").mkdir(parents=True)
    (base / "train").mkdir()
    for cls, expected in [(CheXpertPlusValidHarmonizer, "1 of 2 source rows"),
                          (CheXpertPlusTrainHarmonizer, "3 of 3 source rows")]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cls(csv_path=str(csv), base_image_dir=str(base)).harmonize()
        msgs = [str(x.message) for x in w if "verify_images" in str(x.message)]
        assert len(msgs) == 1 and expected in msgs[0], (cls.__name__, msgs)


@pytest.mark.skipif(not is_complete(), reason="the data paths file is not complete (a public clone)")
def test_every_chexpert_plus_key_reads_the_labels_of_the_impression():
    """D-48 (Q-7): the label file of each CheXpert Plus key is impression_fixed.json, not report_fixed.json."""
    keys = {k: v for k, v in load().items() if k.startswith("chexpert_plus")}
    assert set(keys) == {"chexpert_plus", "chexpert_plus_train", "chexpert_plus_valid"}
    for key, e in keys.items():
        assert Path(e["paths"]["label_json_path"]).name == "impression_fixed.json", key
