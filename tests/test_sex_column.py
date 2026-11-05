"""D-26: one ``sex`` column with the values "M", "F" or NaN.

Each harmonizer with sex declares ``SEX_SOURCE_COL``. One base step maps the source
values to ``sex``. The expected values come from D-26 in
``docs/wiki/design.md`` and from the raw source formats (D-23), not from the
code under test.
"""
import importlib
import inspect
import json
import pkgutil

import numpy as np
import pandas as pd
import pytest

import radharmony.harmonizer as H
from radharmony.harmonizer.base import BaseHarmonizer


class _Toy(BaseHarmonizer):
    SEX_SOURCE_COL = "raw_sex"


def _table(values) -> pd.DataFrame:
    ids = [str(i) for i in range(len(values))]
    return pd.DataFrame({"patient_id": ids, "study_id": ids,
                         "image_path": [i + ".png" for i in ids], "raw_sex": values})


def _all_harmonizer_classes():
    """Every BaseHarmonizer subclass in the radharmony.harmonizer package."""
    found = set()
    for mod in pkgutil.walk_packages(H.__path__, H.__name__ + "."):
        try:
            module = importlib.import_module(mod.name)
        except Exception:  # a module that needs an optional extra
            continue
        for obj in vars(module).values():
            if inspect.isclass(obj) and issubclass(obj, BaseHarmonizer):
                found.add(obj)
    return sorted(found, key=lambda c: c.__qualname__)


# --------------------------------------------------------------------------
# the base step
# --------------------------------------------------------------------------
def test_base_step_maps_the_source_values():
    """M, F and the words male and female map to M and F. Unknown or other gives NaN."""
    raw = ["M", "F", "Male", "female", " m ", "O", "Unknown", "", None, np.nan]
    out = _Toy(csv_path=None)._select_harmonized_columns(_table(raw))
    assert out["sex"].iloc[:5].tolist() == ["M", "F", "M", "F", "M"]
    assert out["sex"].iloc[5:].isna().all()


def test_base_step_warns_on_a_value_it_does_not_know():
    """A typo must not vanish into NaN with no message."""
    with pytest.warns(UserWarning, match="raw_sex"):
        out = _Toy(csv_path=None)._select_harmonized_columns(_table(["M", "femal"]))
    assert out["sex"].isna().tolist() == [False, True]


def test_no_source_gives_no_sex_column():
    """A harmonizer with no SEX_SOURCE_COL gets no sex column."""
    out = BaseHarmonizer(csv_path=None)._select_harmonized_columns(_table(["M"]))
    assert "sex" not in out.columns


# --------------------------------------------------------------------------
# every harmonizer
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", [
    # D-26 gaps: patient_sex, sex_female only, source words, or no sex column.
    "ShenzhenCXRHarmonizer", "ReXGradientHarmonizer", "EmoryCXRHarmonizer", "BRAXHarmonizer",
    "MRRATEHarmonizer", "RSNABoneAgeHarmonizer", "CheXpertHarmonizer", "CheXpertPlusHarmonizer",
    "ChestXray14Harmonizer", "CTRATEHarmonizer", "RSNAPneumoniaKaggleHarmonizer",
    "SIIMACRPTXHarmonizer", "SIIMCOVID19Harmonizer",
    # Had a sex column already.
    "TAIXRayHarmonizer", "MontgomeryCXRHarmonizer", "PadChestGRHarmonizer",
    # The source has sex too: PatientSex_DICOM in the CSV, PatientSex in the DICOM headers.
    "PadChestHarmonizer", "VinDrCXRTrainHarmonizer", "VinDrCXRTestHarmonizer",
])
def test_harmonizer_declares_a_sex_source(name):
    """Each harmonizer whose source has the sex names its source column."""
    cls = {c.__qualname__: c for c in _all_harmonizer_classes()}[name]
    assert cls.SEX_SOURCE_COL


def test_no_harmonizer_outputs_its_own_sex_column():
    """The base step owns ``sex``. A copy in EXTRA_OUTPUT_COLS would give 2 columns."""
    bad = [c.__qualname__ for c in _all_harmonizer_classes() if "sex" in list(c.EXTRA_OUTPUT_COLS)]
    assert not bad


# --------------------------------------------------------------------------
# datasets that parse or convert the source
# --------------------------------------------------------------------------
def _write_text_set(root, prefix, texts):
    """Write a Montgomery or Shenzhen folder: CXR_png/<stem>.png and ClinicalReadings/<stem>.txt."""
    (root / "CXR_png").mkdir()
    (root / "ClinicalReadings").mkdir()
    for i, text in enumerate(texts):
        stem = f"{prefix}_{i + 1:04d}_0"
        (root / "CXR_png" / f"{stem}.png").touch()
        (root / "ClinicalReadings" / f"{stem}.txt").write_text(text)
    return str(root / "CXR_png")


def test_montgomery_other_sex_is_nan_and_not_in_the_report(tmp_path):
    """V-H2-13: the raw line "Patient's Sex: O" (MCUCXR_0080) is sex NaN, not report text."""
    base = _write_text_set(tmp_path, "MCUCXR", [
        "Patient's Sex: O \nPatient's Age: 005Y\nnormal\n",
        "Patient's Sex: F \nPatient's Age: 027Y\nnormal\n",
    ])
    df = H.MontgomeryCXRHarmonizer(base_dir=base).harmonize()
    assert df["sex"].isna().tolist() == [True, False]
    assert df["sex"].iloc[1] == "F"
    assert df["report"].tolist() == ["normal", "normal"]
    assert df["age_years"].tolist() == [5.0, 27.0]


def test_shenzhen_parses_every_first_line_form(tmp_path):
    """V-H3-11: the forms of the real first lines, with their sex and age.

    "male 42" has no unit, so its age is NaN (D-06: the code never guesses the unit).
    """
    lines = ["femal 32yrs", "male35yrs", "male 16month", "female 64days", "male 42",
             "Male , 30yrs", "female41yrs", "male 1yr"]
    base = _write_text_set(tmp_path, "CHNCXR", [line + "\r\nnormal" for line in lines])
    df = H.ShenzhenCXRHarmonizer(base_dir=base).harmonize()
    assert df["sex"].tolist() == ["F", "M", "M", "F", "M", "M", "F", "M"]
    ages = df["age_years"].tolist()
    assert ages[:2] == [32.0, 35.0]
    assert ages[2] == pytest.approx(16 / 12)
    assert ages[3] == pytest.approx(64 / 365.25)
    assert np.isnan(ages[4])
    assert ages[5:] == [30.0, 41.0, 1.0]


def test_rexgradient_other_sex_is_nan(tmp_path):
    """V-H2-13: ReXGradient PatientSex "O" gives sex NaN."""
    def entry(i, sex):
        return {"ImagePath": [f"../deid_png/p{i}/s{i}/a.png"], "ImageViewPosition": ["PA"],
                "PatientID": f"p{i}", "StudyInstanceUid": f"s{i}", "AccessionNumber": f"a{i}",
                "PatientSex": sex, "PatientAge": "045Y"}
    meta = tmp_path / "train_metadata_view_position.json"
    meta.write_text(json.dumps({"k0": entry(0, "O"), "k1": entry(1, "M"), "k2": entry(2, "F")}))
    df = H.ReXGradientHarmonizer(csv_path=str(meta)).harmonize()
    assert df["sex"].isna().tolist() == [True, False, False]
    assert df["sex"].iloc[1:].tolist() == ["M", "F"]


def test_bone_age_sex_comes_from_the_male_flag(tmp_path):
    """train.csv has ``male`` (True or False). D-26 keeps ``sex_female`` too."""
    images = tmp_path / "boneage-training-dataset"
    images.mkdir()
    for i in (1, 2):
        (images / f"{i}.png").touch()
    pd.DataFrame({"id": [1, 2], "boneage": [100, 120], "male": [True, False]}).to_csv(
        tmp_path / "train.csv", index=False)
    df = H.RSNABoneAgeHarmonizer(base_image_dir=str(images)).harmonize()
    assert df["sex"].tolist() == ["M", "F"]
    assert df["sex_female"].tolist() == [0, 1]


def test_chexpert_sex_comes_from_the_csv(tmp_path):
    """D-26 gap: CheXpert without demographics_path had no sex. train.csv has ``Sex``."""
    paths = [f"CheXpert-v1.0/train/patient0000{i}/study1/view1_frontal.jpg" for i in (1, 2, 3)]
    labels = {c: [0.0] * 3 for c in H.CheXpertHarmonizer.LABEL_COLS}
    pd.DataFrame({"Path": paths, "Sex": ["Male", "Female", "Unknown"], "Age": [50, 60, 70],
                  "Frontal/Lateral": ["Frontal"] * 3, "AP/PA": ["AP"] * 3, **labels}).to_csv(
        tmp_path / "train.csv", index=False)
    df = H.CheXpertHarmonizer(csv_path=str(tmp_path / "train.csv")).harmonize()
    assert df["sex"].iloc[:2].tolist() == ["M", "F"]
    assert pd.isna(df["sex"].iloc[2])


def test_ct_rate_sex_and_age_come_from_the_metadata_csv(tmp_path):
    """D-26 and D-06 gaps: train_metadata.csv has PatientSex and PatientAge (DICOM age strings)."""
    names = ["train_1_a_1.nii.gz", "train_2_a_1.nii.gz", "train_3_a_1.nii.gz"]
    labels = {c: [0] * 3 for c in H.CTRATEHarmonizer.LABEL_COLS}
    pd.DataFrame({"VolumeName": names, **labels}).to_csv(tmp_path / "labels.csv", index=False)
    pd.DataFrame({"VolumeName": names, "PatientSex": ["M", "F", np.nan],
                  "PatientAge": ["049Y", "018M", np.nan],
                  "ImageOrientationPatient": ["[1, 0, 0, 0, 1, 0]"] * 3}).to_csv(
        tmp_path / "meta.csv", index=False)
    df = H.CTRATEHarmonizer(csv_path=str(tmp_path / "labels.csv"),
                            view_position_csv_path=str(tmp_path / "meta.csv")).harmonize()
    assert df["sex"].iloc[:2].tolist() == ["M", "F"]
    assert pd.isna(df["sex"].iloc[2])
    assert df["age_years"].iloc[:2].tolist() == [49.0, 1.5]
    assert pd.isna(df["age_years"].iloc[2])


def test_dataset_with_harmonizer_does_not_use_sex_as_a_label(tmp_path):
    """D-05: the label columns are the labels of the dataset, not ``sex`` or ``age_years``.

    Before D-08 (2026-11-02), with ``harmonizer=``, the dataset took every column outside a fixed
    list in ``radharmony/dataset/base.py`` as a label. ``sex`` is a string, so ``output_cls=True``
    then failed (V-H3-18 had the same cause).
    """
    from radharmony.dataset import CheXpertTrainDataset

    paths = [f"CheXpert-v1.0/train/patient0000{i}/study1/view1_frontal.jpg" for i in (1, 2)]
    labels = {c: [0.0] * 2 for c in H.CheXpertHarmonizer.LABEL_COLS}
    pd.DataFrame({"Path": paths, "Sex": ["Male", "Female"], "Age": [50, 60],
                  "Frontal/Lateral": ["Frontal"] * 2, "AP/PA": ["AP"] * 2, **labels}).to_csv(
        tmp_path / "train.csv", index=False)
    h = H.CheXpertHarmonizer(csv_path=str(tmp_path / "train.csv"))
    h.harmonize()
    ds = CheXpertTrainDataset(base_image_dir=str(tmp_path), harmonizer=h, output_cls=True,
                              cache_dir=None)
    ds.get_harmonized_df()
    assert "sex" not in ds.LABEL_COLS
