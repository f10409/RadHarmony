"""PadChest harmonizer: source IDs (D-01)."""

import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dataset_paths import is_complete, paths  # noqa: E402


def _fake_csv(path):
    """3 images. Study S1 has 2 patients (P1, P2), as one real PadChest StudyID does."""
    pd.DataFrame({
        "ImageID": ["a.png", "b.png", "c.png"],
        "ImageDir": [0, 0, 1],
        "StudyID": ["S1", "S1", "S2"],
        "PatientID": ["P1", "P2", "P3"],
        "Projection": ["PA", "PA", "L"],
        "Labels": ["['normal']", "['cardiomegaly']", "['normal']"],
        "Report": ["a", "b", "c"],
    }).to_csv(path, index=False)


def test_a_study_id_of_2_patients_warns(tmp_path):
    """V-H2-32 (D-01): study_id keeps the source StudyID, and the harmonizer warns when 2 patients share it."""
    from radharmony.harmonizer import PadChestHarmonizer

    csv = tmp_path / "padchest.csv"
    _fake_csv(csv)
    with pytest.warns(UserWarning, match=r"PadChest: 1 study_id \(StudyID\).*: S1\."):
        df = PadChestHarmonizer(csv_path=str(csv)).harmonize()
    assert df.loc[df["study_id"] == "S1", "patient_id"].tolist() == ["P1", "P2"]


def test_no_warning_when_each_study_has_one_patient(tmp_path):
    """D-01: a table where each study has one patient gives no warning about study_id."""
    from radharmony.harmonizer import PadChestHarmonizer

    csv = tmp_path / "padchest.csv"
    _fake_csv(csv)
    pd.read_csv(csv).assign(PatientID=["P1", "P1", "P3"]).to_csv(csv, index=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        PadChestHarmonizer(csv_path=str(csv)).harmonize()
    assert not [x for x in w if "study_id" in str(x.message)]


def test_real_padchest_names_each_shared_study_id():
    """V-H2-32 (D-01): the warning names each StudyID that the raw CSV gives to 2 or more patients."""
    from radharmony.harmonizer import PadChestHarmonizer

    p = paths("padchest", missing_ok=not is_complete())
    if not os.path.isfile(p.get("csv_path", "")):
        pytest.skip("PadChest is not mounted")
    raw = pd.read_csv(p["csv_path"], usecols=["StudyID", "PatientID"])
    n = raw.astype(str).groupby("StudyID")["PatientID"].nunique()
    shared = sorted(n[n > 1].index)
    assert shared  # the 2019 release has one: 216840111366964012558082906712010004190248051
    h = PadChestHarmonizer(csv_path=p["csv_path"])
    h.df = raw.astype(str)
    h._build_patient_id()
    with pytest.warns(UserWarning, match=f"PadChest: {len(shared)} study_id") as w:
        h._build_study_id()
    assert all(s in str(w[0].message) for s in shared)
