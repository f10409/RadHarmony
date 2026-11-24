"""D-06 and D-26: sex and age that are only in the DICOM headers.

The harmonizers read the tags ``PatientSex`` and ``PatientAge``. A missing
or empty tag gives NaN. The tests write small DICOM files with known tags, so
the expected values come from the files, not from the code (D-23).
"""
import numpy as np
import pandas as pd
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from radharmony.harmonizer import (
    RSNAPneumoniaKaggleHarmonizer,
    RSNAPneumoniaKaggleTestHarmonizer,
    SIIMACRPTXHarmonizer,
    SIIMACRPTXTestHarmonizer,
    SIIMCOVID19Harmonizer,
    SIIMCOVID19TestHarmonizer,
    VinDrCXRTestHarmonizer,
    VinDrCXRTrainHarmonizer,
)
from radharmony.harmonizer._dicom_tags import add_patient_tags

# (PatientSex, PatientAge) per file. None: the tag is not in the header.
TAGS = [("M", "58"), ("F", ""), (None, None), ("O", "7")]
SEX = ["M", "F", np.nan, np.nan]


def _dicom(path, sex, age, **more):
    """Write a small DICOM file with the given tags."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    if sex is not None:
        ds.PatientSex = sex
    if age is not None:
        ds.PatientAge = age
    for key, value in more.items():
        setattr(ds, key, value)
    ds.save_as(str(path), enforce_file_format=True)


def _same(got, expected):
    """Compare two lists where NaN equals NaN."""
    def norm(values):
        return [None if pd.isna(v) else v for v in values]
    return norm(got) == norm(expected)


def test_add_patient_tags_keeps_the_text_and_gives_nan_for_missing(tmp_path):
    """An empty tag, a missing tag and a file that is not DICOM give NaN."""
    for i, (sex, age) in enumerate(TAGS):
        _dicom(tmp_path / f"{i}.dcm", sex, age)
    (tmp_path / "broken.dcm").write_text("not a DICOM file")
    df = pd.DataFrame({"image_path": [f"{i}.dcm" for i in range(4)] + ["broken.dcm"]})
    with pytest.warns(UserWarning, match="1 of 5 DICOM files could not be read"):
        out = add_patient_tags(df, str(tmp_path), num_workers=1)
    assert _same(out["PatientSex"].tolist(), ["M", "F", np.nan, "O", np.nan])
    assert _same(out["PatientAge"].tolist(), ["58", np.nan, np.nan, "7", np.nan])


def test_rsna_pneumonia_kaggle_train(tmp_path):
    """PatientAge is a plain number of years. The sex O gives NaN."""
    images = tmp_path / "stage_2_train_images"
    for i, (sex, age) in enumerate(TAGS):
        _dicom(images / f"p{i}.dcm", sex, age)
    pd.DataFrame({"patientId": [f"p{i}" for i in range(4)], "class": ["Normal"] * 4}).to_csv(
        tmp_path / "class.csv", index=False)
    df = RSNAPneumoniaKaggleHarmonizer(csv_path=str(tmp_path / "class.csv"),
                                       base_image_dir=str(images)).harmonize()
    assert _same(df["sex"].tolist(), SEX)
    assert _same(df["age_years"].tolist(), [58.0, np.nan, np.nan, 7.0])


def test_rsna_pneumonia_kaggle_test(tmp_path):
    """The test class lists the files and reads the same tags."""
    for i, (sex, age) in enumerate(TAGS):
        _dicom(tmp_path / f"p{i}.dcm", sex, age)
    df = RSNAPneumoniaKaggleTestHarmonizer(base_image_dir=str(tmp_path)).harmonize()
    assert _same(df["sex"].tolist(), SEX)
    assert _same(df["age_years"].tolist(), [58.0, np.nan, np.nan, 7.0])


def test_siim_acr_train_and_test(tmp_path):
    """The train and test classes read the tags of the files in the tree."""
    for i, (sex, age) in enumerate(TAGS):
        _dicom(tmp_path / f"study{i}" / "series" / f"img{i}.dcm", sex, age)
    rle = [" 1 2", " -1", " -1", " -1"]
    pd.DataFrame({"ImageId": [f"img{i}" for i in range(4)], " EncodedPixels": rle}).to_csv(
        tmp_path / "train-rle.csv", index=False)
    train = SIIMACRPTXHarmonizer(csv_path=str(tmp_path / "train-rle.csv"),
                                 dicom_dir=str(tmp_path)).harmonize()
    test = SIIMACRPTXTestHarmonizer(dicom_dir=str(tmp_path)).harmonize()
    for df in (train, test.sort_values("image_path", ignore_index=True)):
        assert _same(df["sex"].tolist(), SEX)
        assert _same(df["age_years"].tolist(), [58.0, np.nan, np.nan, 7.0])


def test_siim_covid19_sex_but_no_age(tmp_path):
    """The SIIM-COVID19 headers have no PatientAge, so the table has no age_years."""
    images = tmp_path / "train"
    for i, (sex, _age) in enumerate(TAGS):
        _dicom(images / f"st{i}" / "se" / f"sop{i}.dcm", sex, None, PatientID=f"pat{i}")
    pd.DataFrame({"id": [f"st{i}_study" for i in range(4)], "Negative for Pneumonia": [1] * 4,
                  "Typical Appearance": [0] * 4, "Indeterminate Appearance": [0] * 4,
                  "Atypical Appearance": [0] * 4}).to_csv(tmp_path / "study.csv", index=False)
    pd.DataFrame({"id": [f"sop{i}_image" for i in range(4)], "boxes": [np.nan] * 4,
                  "label": ["none 1 0 0 1 1"] * 4,
                  "StudyInstanceUID": [f"st{i}" for i in range(4)]}).to_csv(
        tmp_path / "image.csv", index=False)
    train = SIIMCOVID19Harmonizer(csv_path=str(tmp_path / "study.csv"),
                                  image_csv_path=str(tmp_path / "image.csv"),
                                  base_image_dir=str(images), num_workers=1).harmonize()
    test = SIIMCOVID19TestHarmonizer(base_image_dir=str(images)).harmonize()
    for df in (train, test.sort_values("image_path", ignore_index=True)):
        assert _same(df["sex"].tolist(), SEX)
        assert "age_years" not in df.columns


@pytest.mark.parametrize("cls", [VinDrCXRTrainHarmonizer, VinDrCXRTestHarmonizer])
def test_vindr_cxr_reads_dicom_age_strings(tmp_path, cls):
    """VinDr-CXR headers use DICOM age strings ("045Y"). A bad one ("Y") gives NaN."""
    tags = [("M", "045Y"), ("F", "Y"), (None, None), ("O", "018M")]
    for i, (sex, age) in enumerate(tags):
        _dicom(tmp_path / f"img{i}.dicom", sex, age)
    rows = {"image_id": [f"img{i}" for i in range(4)], "No finding": [1] * 4}
    if cls is VinDrCXRTrainHarmonizer:
        rows["rad_id"] = ["R1"] * 4
    pd.DataFrame(rows).to_csv(tmp_path / "labels.csv", index=False)
    df = cls(csv_path=str(tmp_path / "labels.csv"), base_image_dir=str(tmp_path)).harmonize()
    df = df.sort_values("image_path", ignore_index=True)
    assert _same(df["sex"].tolist(), SEX)
    assert _same(df["age_years"].tolist(), [45.0, np.nan, np.nan, 1.5])


@pytest.mark.parametrize("cls", [VinDrCXRTrainHarmonizer, VinDrCXRTestHarmonizer])
def test_vindr_cxr_age_0_is_unknown(tmp_path, cls):
    """D-06: in VinDr-CXR, a PatientAge of 0 ("000Y" or "000D") means unknown, so it gives NaN."""
    ages = ["000Y", "000D", "002Y", "001M"]
    for i, age in enumerate(ages):
        _dicom(tmp_path / f"img{i}.dicom", "M", age)
    rows = {"image_id": [f"img{i}" for i in range(4)], "No finding": [1] * 4}
    if cls is VinDrCXRTrainHarmonizer:
        rows["rad_id"] = ["R1"] * 4
    pd.DataFrame(rows).to_csv(tmp_path / "labels.csv", index=False)
    df = cls(csv_path=str(tmp_path / "labels.csv"), base_image_dir=str(tmp_path)).harmonize()
    df = df.sort_values("image_path", ignore_index=True)
    assert _same(df["age_years"].tolist(), [np.nan, np.nan, 2.0, 1 / 12])


def test_vindr_pcxr_reads_age_and_sex_from_the_dicom_tags(tmp_path):
    """V-H3-31 (D-06, D-26): VinDr-PCXR has age and sex only in the DICOM headers. Child ages use M and D."""
    from radharmony.harmonizer import VinDrPCXRHarmonizer

    tags = [("M", "005Y"), ("F", "006M"), (None, None), ("O", "030D")]
    for split in ("train", "test"):
        (tmp_path / split).mkdir()
    for i, (sex, age) in enumerate(tags):
        _dicom(tmp_path / "train" / f"img{i}.dicom", sex, age, Rows=8, Columns=8)
    pd.DataFrame({"image_id": [f"img{i}" for i in range(4)], "No finding": [1] * 4}).to_csv(
        tmp_path / "image_labels_train.csv", index=False)
    pd.DataFrame({"image_id": [], "No finding": []}).to_csv(tmp_path / "image_labels_test.csv", index=False)
    df = VinDrPCXRHarmonizer(base_dir=str(tmp_path)).harmonize().sort_values("image_path", ignore_index=True)
    assert _same(df["sex"].tolist(), SEX)
    assert _same(df["age_years"].tolist(), [5.0, 0.5, np.nan, 30 / 365.25])


def test_rsna_abdominal_trauma_reads_the_demographics_file(tmp_path):
    """V-H3-32 (D-06, D-26): train_demographics_2024.csv gives age and sex. "90+" gives 90, "Unknown" gives NaN."""
    from radharmony.harmonizer import RSNAAbdominalTrauma2023Harmonizer

    ids = [1, 2, 3, 4]
    pd.DataFrame({"patient_id": ids, **{c: [0] * 4 for c in RSNAAbdominalTrauma2023Harmonizer.LABEL_COLS}}).to_csv(
        tmp_path / "train_2024.csv", index=False)
    pd.DataFrame({"patient_id": ids, "series_id": [10, 20, 30, 40], "aortic_hu": [100.0] * 4,
                  "incomplete_organ": [0] * 4}).to_csv(tmp_path / "train_series_meta.csv", index=False)
    pd.DataFrame({"patient_id": ids, "Age": ["45", "90+", "Unknown", "18"], "Sex": ["M", "F", "Unknown", "F"]}).to_csv(
        tmp_path / "train_demographics_2024.csv", index=False)
    df = RSNAAbdominalTrauma2023Harmonizer(csv_path=str(tmp_path / "train_2024.csv"),
                                           series_meta_csv_path=str(tmp_path / "train_series_meta.csv")).harmonize()
    df = df.sort_values("patient_id", ignore_index=True)
    assert _same(df["age_years"].tolist(), [45.0, 90.0, np.nan, 18.0])
    assert _same(df["sex"].tolist(), ["M", "F", np.nan, "F"])

    (tmp_path / "train_demographics_2024.csv").unlink()
    with pytest.warns(UserWarning, match="train_demographics_2024.csv"):
        df = RSNAAbdominalTrauma2023Harmonizer(csv_path=str(tmp_path / "train_2024.csv"),
                                               series_meta_csv_path=str(tmp_path / "train_series_meta.csv")).harmonize()
    assert "age_years" not in df.columns and "sex" not in df.columns and len(df) == 4


def test_save_and_load_keep_an_empty_label_list(tmp_path):
    """D-08: LABEL_COLS has the same items after save and load.

    The RSNA test split has no labels. Its table has ``sex`` and ``age_years``,
    and these are not labels after a load either.
    """
    for i, (sex, age) in enumerate(TAGS):
        _dicom(tmp_path / f"p{i}.dcm", sex, age)
    h = RSNAPneumoniaKaggleTestHarmonizer(base_image_dir=str(tmp_path))
    h.harmonize()
    h.save(str(tmp_path / "h.pkl"))
    loaded = RSNAPneumoniaKaggleTestHarmonizer.load_from_saved(str(tmp_path / "h.pkl"))
    assert loaded.get_label_cols() == []
