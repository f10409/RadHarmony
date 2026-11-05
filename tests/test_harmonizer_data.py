"""Data fixes in single harmonizers.

Each test pins one finding. The expected values come from the source files
or from the dataset docs, not from the code under test (D-23). The tests on
NAS data skip when the data is not readable.

Run with::

    pytest tests/test_harmonizer_data.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dataset_paths import MissingPathsError, is_complete, path_state, paths  # noqa: E402


def _readable(key: str) -> dict:
    """Return the paths of ``key``, or skip the test when the data is not readable."""
    try:
        p = paths(key)
    except ValueError as e:
        pytest.skip(str(e))
    except MissingPathsError as e:  # a clone with no complete paths file (a dev key must exist)
        if is_complete():
            raise
        pytest.skip(str(e))
    blocked = [v for v in p.values() if path_state(v) != "ok"]
    if blocked:
        pytest.skip(f"{key}: data not readable: {blocked[0]}")
    return p


# ---------------------------------------------------------------------------
# V-H1-20: ChestX-ray14 box labels use the label column names
# ---------------------------------------------------------------------------


def _cxr14_bbox_fixture(tmp_path):
    """Write a 2-image ChestX-ray14 folder. The BBox list says "Infiltrate"."""
    (tmp_path / "images_001" / "images").mkdir(parents=True)
    for name in ("00000001_000.png", "00000002_000.png"):
        Image.fromarray(np.zeros((100, 100), dtype=np.uint8)).save(
            tmp_path / "images_001" / "images" / name)
    pd.DataFrame({
        "Image Index": ["00000001_000.png", "00000002_000.png"],
        "Finding Labels": ["Infiltration", "Atelectasis|Mass"],
        "Follow-up #": [0, 0],
        "Patient ID": [1, 2],
        "View Position": ["PA", "AP"],
    }).to_csv(tmp_path / "Data_Entry_2017.csv", index=False)
    # The real file has a broken header: "Bbox [x", "y", "w", "h]" and 3 empty columns.
    (tmp_path / "BBox_List_2017.csv").write_text(
        "Image Index,Finding Label,Bbox [x,y,w,h],,,\n"
        "00000001_000.png,Infiltrate,10,20,30,40,,,\n"
        "00000002_000.png,Atelectasis,5,5,10,10,,,\n"
        "00000002_000.png,Mass,50,50,20,20,,,\n")
    return tmp_path


def test_cxr14_bbox_labels_match_label_columns(tmp_path):
    """V-H1-20: the BBox list says "Infiltrate", the label column is ``infiltration``."""
    from radharmony.harmonizer import ChestXray14BboxHarmonizer

    root = _cxr14_bbox_fixture(tmp_path)
    df = ChestXray14BboxHarmonizer(
        csv_path=str(root / "Data_Entry_2017.csv"),
        bbox_csv_path=str(root / "BBox_List_2017.csv"),
        base_image_dir=str(root),
    ).harmonize()
    row = df.set_index("patient_id").loc["1"]
    assert row["bbox_labels"] == ["infiltration"]
    assert row["infiltration"] == 1
    for labels in df["bbox_labels"]:
        assert set(labels) <= set(df.columns), labels


def test_cxr14_bbox_labels_match_label_columns_on_nas():
    """V-H1-20 on the real files: each box label is a label column."""
    from radharmony.harmonizer import ChestXray14BboxHarmonizer

    df = ChestXray14BboxHarmonizer(**_readable("chestxray14_bbox")).harmonize()
    names = {lab for labels in df["bbox_labels"] for lab in labels}
    assert names <= set(df.columns), names - set(df.columns)


# ---------------------------------------------------------------------------
# V-H2-9: MR-RATE report_section="impression" gives one "report" column
# ---------------------------------------------------------------------------


def _mrrate_fixture(tmp_path):
    """Write a 2-study MR-RATE meta tree (3 series), with full, findings and impression text."""
    m = tmp_path / "meta"
    for d in ("metadata", "reports", "pathology_labels"):
        (m / d).mkdir(parents=True)
    pd.DataFrame({"patient_uid": ["P1", "P1", "P2"], "study_uid": ["S1", "S1", "S2"],
                  "series_id": ["t1w-raw-sag", "flair-raw-ax", "t1w-raw-ax"],
                  "classified_modality": ["T1w", "FLAIR", "T1w"],
                  "is_center_modality": [True, False, True]},
                 ).to_csv(m / "metadata" / "batch00_metadata.csv", index=False)
    pd.DataFrame({"patient_uid": ["P1", "P2"], "study_uid": ["S1", "S2"],
                  "batch_id": ["batch00"] * 2, "split": ["train", "test"]},
                 ).to_csv(m / "splits.csv", index=False)
    pd.DataFrame({"study_uid": ["S1", "S2"], "report": ["full S1", "full S2"],
                  "findings": ["find S1", "find S2"], "impression": ["imp S1", "imp S2"]},
                 ).to_csv(m / "reports" / "batch00_reports.csv", index=False)
    pd.DataFrame({"study_uid": ["S1", "S2"], "Glioma": [1, 0]},
                 ).to_csv(m / "pathology_labels" / "mrrate_labels.csv", index=False)
    return tmp_path


@pytest.mark.parametrize("section,text", [("full", "full S1"), ("findings", "find S1"),
                                          ("impression", "imp S1")])
def test_mrrate_report_section_gives_one_report_column(tmp_path, section, text):
    """V-H2-9: each ``report_section`` gives one ``report`` column with that text."""
    from radharmony.harmonizer.mrrate import MRRATEHarmonizer

    df = MRRATEHarmonizer(base_dir=str(_mrrate_fixture(tmp_path)),
                          report_section=section).harmonize()
    assert list(df.columns).count("report") == 1
    assert set(df.loc[df["study_id"] == "S1", "report"]) == {text}
    assert set(df.loc[df["study_id"] == "S1", "impression"]) == {"imp S1"}


# ---------------------------------------------------------------------------
# V-H2-7: Open-I MeSH terms match whole words, not parts of words
# ---------------------------------------------------------------------------

# MeSH term -> the labels it must set. Tuberculosis is a disease, not a tube,
# and "Abnormal" is not "Normal".
_MESH_CASES = [
    (["Tuberculosis"], set()),
    (["Tuberculosis/healed"], set()),
    (["Abnormal"], set()),
    (["Normal"], {"no_finding"}),
    (["Tube, Inserted"], {"support_devices"}),
    (["Tubes"], {"support_devices"}),
    (["Catheter"], {"support_devices"}),
    (["Catheters, Indwelling/right"], {"support_devices"}),
    (["Pulmonary Atelectasis/base/left"], {"atelectasis"}),
    (["Lung/hypoinflation"], set()),  # Q-15: low lung volume is not atelectasis
    (["Pleural Effusion/bilateral"], {"pleural_effusion"}),
    # V-H2-42: the device terms that OpenI uses. A pacemaker study has "Implanted Medical Device".
    (["Implanted Medical Device"], {"support_devices"}),
    (["Medical Device/left"], {"support_devices"}),
    (["Stents"], {"support_devices"}),
    (["Breast Implants"], set()),
    (["Medical Device/breast/bilateral"], set()),  # a breast implant
]


@pytest.mark.parametrize("terms,positive", _MESH_CASES, ids=lambda v: str(v))
def test_openi_mesh_terms_match_whole_words(terms, positive):
    """V-H2-7: "tube" does not match "Tuberculosis", "normal" does not match "Abnormal"."""
    from radharmony.harmonizer.openi_cxr.openi_cxr import _mesh_to_labels

    assert {k for k, v in _mesh_to_labels(terms).items() if v} == positive


def test_openi_tuberculosis_report_has_no_support_device(tmp_path):
    """V-H2-7 through harmonize(): a healed TB report gives support_devices 0."""
    from radharmony.harmonizer.openi_cxr import OpenICXRHarmonizer

    (tmp_path / "ecgen-radiology").mkdir()
    (tmp_path / "ecgen-radiology" / "1.xml").write_text(
        '<eCitation><uId id="CXR1"/><pmcId id="1"/><MedlineCitation><Article><Abstract>'
        '<AbstractText Label="FINDINGS">Calcified granuloma.</AbstractText>'
        '<AbstractText Label="IMPRESSION">Old TB.</AbstractText>'
        '</Abstract></Article></MedlineCitation><MeSH><major>Tuberculosis/healed</major></MeSH>'
        '<parentImage id="CXR1_1_IM-0001-3001"><figureId>F1</figureId></parentImage></eCitation>')
    df = OpenICXRHarmonizer(base_dir=str(tmp_path)).harmonize()
    assert df["support_devices"].tolist() == [0]
    assert df["no_finding"].tolist() == [0]


def test_openi_no_indexing_gives_nan_in_every_label():
    """V-H2-41 (D-05): the MeSH term "No Indexing" is no label at all, so every label is NaN, not 0."""
    from radharmony.harmonizer.openi_cxr.openi_cxr import _mesh_to_labels

    assert all(pd.isna(v) for v in _mesh_to_labels(["No Indexing"]).values())
    assert all(pd.isna(v) for v in _mesh_to_labels([]).values())  # no term at all
    assert _mesh_to_labels(["No Indexing", "Cardiomegaly"])["cardiomegaly"] == 1


@pytest.mark.parametrize("figure, tag, view", [
    ("frontal", "LL", "lateral"), ("frontal", "RL", "lateral"), ("frontal", "LATERAL", "lateral"),
    ("lateral", "AP", "AP"), ("frontal", "AP", "AP"),
    ("frontal", "PA", "PA"), ("lateral", "PA", "lateral"),  # lateral images also carry the PA tag
    ("frontal", "", "frontal"), ("lateral", "", "lateral"), ("other", "PA", "other"),
])
def test_openi_view_takes_the_dicom_tag_when_it_names_ap_or_a_lateral(figure, tag, view):
    """V-H2-40: the figure number gives the wrong view for some images. The DICOM tag wins when it says AP or a
    lateral view. A PA tag counts only on figure F1, because many lateral images also have PA."""
    from radharmony.harmonizer.openi_cxr.openi_cxr import _view_from_tag

    assert _view_from_tag(figure, tag) == view


def test_openi_view_reads_the_dicom_tag(tmp_path):
    """V-H2-40 through harmonize(): figure F2 with the DICOM tag AP is an AP image, not LATERAL."""
    from test_harmonizer_boxes import _write_dicom

    from radharmony.harmonizer.openi_cxr import OpenICXRHarmonizer

    (tmp_path / "ecgen-radiology").mkdir()
    (tmp_path / "ecgen-radiology" / "1.xml").write_text(
        '<eCitation><uId id="CXR1"/><pmcId id="1"/><MeSH><major>Normal</major></MeSH>'
        '<parentImage id="CXR1_1_IM-0001-3001"><figureId>F1</figureId></parentImage>'
        '<parentImage id="CXR1_1_IM-0001-4001"><figureId>F2</figureId></parentImage></eCitation>')
    _write_dicom(tmp_path / "1" / "1_IM-0001-4001.dcm", 8, 8, ViewPosition="AP")  # F1 has no DICOM file
    df = OpenICXRHarmonizer(base_dir=str(tmp_path)).harmonize()
    assert df["view_position"].tolist() == ["FRONTAL", "AP"]


# ---------------------------------------------------------------------------
# V-H1-28: CT-RATE gives one view_position value for one orientation
# ---------------------------------------------------------------------------


def test_ct_rate_view_position_is_the_nearest_plane(tmp_path):
    """V-H1-28 (D-01): view_position is the nearest plane, and image_orientation keeps the vector.

    The slice normal is row x column of the DICOM ImageOrientationPatient. Its largest axis gives
    the plane. z (feet to head) is axial, y (front to back) is coronal and x (right to left) is sagittal.
    """
    from radharmony.harmonizer import CTRATEHarmonizer

    names = [f"train_{i}_a_1.nii.gz" for i in range(1, 6)]
    pd.DataFrame({"VolumeName": names, "Atelectasis": [0, 1, 0, 0, 1]}).to_csv(
        tmp_path / "labels.csv", index=False)
    oblique = "[0.9524330922, -0.3010073798, 0.0476000237, 0.3012412698, 0.9535450522, 0.002351755993]"
    pd.DataFrame({"VolumeName": names, "ImageOrientationPatient": [
        "[1, 0, 0, 0, 1, 0]",
        "[1.000000, 0.000000, 0.000000, 0.000000, 1.000000, -0.000000]",
        oblique,  # normal about (0.0, 0.0, 1.0): a tilted axial volume
        "[1, 0, 0, 0, 0, -1]",
        "[0, 1, 0, 0, 0, -1]",
    ]}).to_csv(tmp_path / "metadata.csv", index=False)
    df = CTRATEHarmonizer(csv_path=str(tmp_path / "labels.csv"),
                          view_position_csv_path=str(tmp_path / "metadata.csv")).harmonize()
    row = df.set_index("patient_id")
    assert row["view_position"].tolist() == ["axial", "axial", "axial", "coronal", "sagittal"]
    # The 2 source spellings of one orientation give one vector (the most common spelling).
    assert row.loc["train_1", "image_orientation"] == row.loc["train_2", "image_orientation"] == "[1, 0, 0, 0, 1, 0]"
    assert row.loc["train_3", "image_orientation"] == oblique  # an oblique vector keeps every digit


def test_ct_rate_orientation_count_on_nas():
    """V-H1-28 (D-01) on the real table: one vector for each distinct orientation, and the plane counts."""
    import json

    from radharmony.harmonizer import CTRATETrainHarmonizer

    p = _readable("ct_rate_train")
    raw = pd.read_csv(p["view_position_csv_path"], usecols=["VolumeName", "ImageOrientationPatient"])
    raw = raw.drop_duplicates("VolumeName")
    # +0.0 turns -0.0 into 0.0, so "-0.000000" and "0" are one number.
    vectors = raw["ImageOrientationPatient"].map(lambda s: tuple(float(x) + 0.0 for x in json.loads(s)))
    normal = vectors.map(lambda v: np.abs(np.cross(v[:3], v[3:])).argmax())
    expected = normal.map({0: "sagittal", 1: "coronal", 2: "axial"}).value_counts().to_dict()
    df = CTRATETrainHarmonizer(**p).harmonize()
    assert df["image_orientation"].nunique() == vectors.nunique()
    assert df["view_position"].value_counts().to_dict() == expected


# ---------------------------------------------------------------------------
# V-H1-24: Chest ImaGenome without base_image_dir raises a clear error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["ChestImaGenomeGoldHarmonizer", "ChestImaGenomeSilverHarmonizer"])
def test_chest_imagenome_needs_base_image_dir(tmp_path, name):
    """V-H1-24: the wiki marks ``base_image_dir`` as required (the box sizes come from the DICOM headers)."""
    import radharmony.harmonizer as H

    h = getattr(H, name)(annotation_dir=str(tmp_path))
    with pytest.raises(ValueError, match="base_image_dir"):
        h.harmonize()


# ---------------------------------------------------------------------------
# V-H1-21, V-H1-27: MedPMC tag table
# ---------------------------------------------------------------------------


def _medpmc_index(tmp_path):
    """Write a MedPMC index with one single-panel and one multi-panel figure."""
    (tmp_path / "webdataset" / "metadata").mkdir(parents=True)
    pd.DataFrame({"source_type": ["singlepanel", "multipanel"], "pmcid": ["PMC1", "PMC2"],
                  "image_id": ["s1", "m1"], "parent_image_id": [None, None],
                  "subfigure_index": [None, None], "figure_label": ["1", "2"],
                  "shard": ["a", "b"], "key": ["k1", "k2"], "image_ext": [".jpg", ".jpg"]},
                 ).to_parquet(tmp_path / "webdataset" / "metadata" / "medpmc_index.parquet")
    return tmp_path


def test_medpmc_multipanel_row_uses_contains_specialty(tmp_path):
    """V-H1-21: a multi-panel figure whose caption mentions radiology is kept (code comment)."""
    from radharmony.harmonizer import MedPMCHarmonizer

    root = _medpmc_index(tmp_path)
    (root / "tags").mkdir()
    pd.DataFrame({"image_id": ["s1", "m1"], "caption": ["CT of chest", "Panel A: chest CT; B: MRI"],
                  "specialty": ["radiology", None], "modality": ["ct", None],
                  "tag_confidence": ["high", None], "tag_source": ["caption", None],
                  "tag_method": ["lex", None], "tag_scope": [None, "multipanel"],
                  "contains_specialty": [None, "radiology"]},
                 ).to_parquet(root / "tags" / "medpmc_tags.parquet")
    df = MedPMCHarmonizer(base_dir=str(root), radiology_only=True, include_multipanel=True).harmonize()
    assert sorted(df["question_id"]) == ["m1", "s1"]
    assert df.set_index("question_id").loc["m1", "contains_specialty"] == "radiology"


def test_medpmc_without_tag_table_warns_that_answer_is_empty(tmp_path):
    """V-H1-27: the index has no caption, so without a tag table every answer is None. It warns."""
    from radharmony.harmonizer import MedPMCHarmonizer

    root = _medpmc_index(tmp_path)
    with pytest.warns(UserWarning, match="caption"):
        df = MedPMCHarmonizer(base_dir=str(root)).harmonize()
    assert df["answer"].isna().all()


# ---------------------------------------------------------------------------
# V-H2-8: MS-CXR-T image paths are relative to the MIMIC-CXR-JPG files/ folder (D-04)
# ---------------------------------------------------------------------------


def test_ms_cxr_t_image_path_has_no_files_prefix(tmp_path):
    """V-H2-8, D-04: ``image_path`` is relative to ``files/``, like the other MIMIC-based tables."""
    from radharmony.harmonizer import MSCXRTHarmonizer

    row = {"dicom_id": "p10/p10000001/s50000002/aaa", "previous_dicom_id": "p10/p10000001/s50000001/bbb",
           "subject_id": 10000001, "study_id": 50000002, "previous_study_id": 50000001}
    for f in ("consolidation", "edema", "pleural_effusion", "pneumonia", "pneumothorax"):
        row[f + "_progression"], row[f + "_label_quality"] = "stable", "one_expert"
    pd.DataFrame([row]).to_csv(tmp_path / "ms_cxr_t.csv", index=False)
    df = MSCXRTHarmonizer(csv_path=str(tmp_path / "ms_cxr_t.csv")).harmonize()
    assert df["image_path"].tolist() == ["p10/p10000001/s50000002/aaa.jpg"]
    assert df["previous_image_path"].tolist() == ["p10/p10000001/s50000001/bbb.jpg"]


@pytest.mark.skipif(not is_complete(), reason="the data paths file is not complete (a public clone)")
def test_ms_cxr_t_base_image_dir_is_the_files_folder():
    """V-H2-8, D-04: ``ms_cxr_t`` uses the same ``files/`` folder as ``ms_cxr`` and MIMIC-CXR-JPG."""
    from dataset_paths import entry

    base = entry("ms_cxr_t")["paths"]["base_image_dir"]
    assert base == entry("ms_cxr")["paths"]["base_image_dir"]
    assert base == entry("mimic_cxr_jpg")["paths"]["base_image_dir"]
    assert base.rstrip("/").endswith("/files")


# ---------------------------------------------------------------------------
# V-H1-18: MIMIC-CXR-JPG says how many images have no label row
# ---------------------------------------------------------------------------


def test_mimic_cxr_jpg_warns_about_images_without_labels(tmp_path):
    """V-H1-18: images of a study with no row in the label CSV are left out (D-25), with a warning."""
    from radharmony.harmonizer import MIMICCXRJPGHarmonizer

    pd.DataFrame({"dicom_id": ["d1", "d2", "d3"], "subject_id": [10000001, 10000001, 10000002],
                  "study_id": [50000001, 50000001, 50000002], "ViewPosition": ["PA", "LL", "AP"]},
                 ).to_csv(tmp_path / "meta.csv", index=False)
    pd.DataFrame({"subject_id": [10000001], "study_id": [50000001], "Atelectasis": [1.0]},
                 ).to_csv(tmp_path / "chexpert.csv", index=False)
    h = MIMICCXRJPGHarmonizer(csv_path=str(tmp_path / "meta.csv"),
                              label_csv_path=str(tmp_path / "chexpert.csv"))
    with pytest.warns(UserWarning, match="1 images in 1 studies"):
        df = h.harmonize(drop_uncertain=False)
    assert sorted(df["image_path"]) == ["p10/p10000001/s50000001/d1.jpg",
                                        "p10/p10000001/s50000001/d2.jpg"]


def test_mimic_cxr_jpg_images_without_labels_on_nas():
    """V-H1-18 on the real files: 15 images in 8 studies have no row in the CheXpert label CSV."""
    from radharmony.harmonizer import MIMICCXRJPGHarmonizer

    p = _readable("mimic_cxr_jpg")
    meta = pd.read_csv(p["csv_path"], usecols=["study_id"])
    labels = pd.read_csv(p["label_csv_path"], usecols=["study_id"])
    missing = meta[~meta["study_id"].isin(labels["study_id"])]
    with pytest.warns(UserWarning, match=f"{len(missing)} images in {missing['study_id'].nunique()} studies"):
        df = MIMICCXRJPGHarmonizer(**p).harmonize(drop_uncertain=False)
    assert len(df) == len(meta) - len(missing)


# ---------------------------------------------------------------------------
# V-H2-15: Montgomery images of one patient share a patient_id
# ---------------------------------------------------------------------------


def _montgomery_fixture(tmp_path, readings: dict[str, str]):
    """Write a Montgomery folder: one PNG and one clinical reading for each stem."""
    img = tmp_path / "CXR_png"
    cr = tmp_path / "ClinicalReadings"
    img.mkdir()
    cr.mkdir()
    for stem, text in readings.items():
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(img / f"{stem}.png")
        (cr / f"{stem}.txt").write_text(f"Patient's Sex: M\nPatient's Age: 040Y\n{text}\n")
    return img


def test_montgomery_same_patient_shares_patient_id(tmp_path):
    """V-H2-15: a reading that says "same pt as" links the images to one patient."""
    from radharmony.harmonizer import MontgomeryCXRHarmonizer

    img = _montgomery_fixture(tmp_path, {
        "MCUCXR_0113_1": "cavitary TB",
        "MCUCXR_0117_1": "stable CXR since 6M (same pt as MCUCXR_0113_1)",
        "MCUCXR_0162_1": "TB",
        "MCUCXR_0166_1": "NO REPORT (same pt as MCUCXR_0162_1)",
        "MCUCXR_0170_1": "similar to prior (same pt as MCUCXR_0166_1, MCUCXR_0162_1)",
        "MCUCXR_0200_0": "normal",
    })
    df = MontgomeryCXRHarmonizer(str(img)).harmonize().set_index("image_path")
    pid = df["patient_id"]
    assert pid["MCUCXR_0117_1.png"] == pid["MCUCXR_0113_1.png"]
    assert pid["MCUCXR_0170_1.png"] == pid["MCUCXR_0166_1.png"] == pid["MCUCXR_0162_1.png"]
    assert pid.nunique() == 3
    assert df["study_id"].nunique() == 6  # each image is its own study


def test_montgomery_placeholder_is_no_report(tmp_path):
    """Q-16 (D-33): "NO REPORT (same pt as ...)" is no report (NaN). The patient link still reads it."""
    from radharmony.harmonizer import MontgomeryCXRHarmonizer

    img = _montgomery_fixture(tmp_path, {
        "MCUCXR_0162_1": "TB",
        "MCUCXR_0166_1": "NO REPORT (same pt as MCUCXR_0162_1)",
    })
    df = MontgomeryCXRHarmonizer(str(img)).harmonize().set_index("image_path")
    assert pd.isna(df.loc["MCUCXR_0166_1.png", "report"])
    assert df.loc["MCUCXR_0162_1.png", "report"] == "TB"
    assert df.loc["MCUCXR_0166_1.png", "patient_id"] == df.loc["MCUCXR_0162_1.png", "patient_id"]


def test_montgomery_patients_on_nas():
    """V-H2-15 on the real files: 3 readings link 5 images to 2 patients, so 138 images have 135 patients."""
    import re

    from radharmony.harmonizer import MontgomeryCXRHarmonizer

    p = _readable("montgomery_cxr")
    cr = Path(p["base_image_dir"]).parent / "ClinicalReadings"
    links = {f.stem: re.findall(r"MCUCXR_\d{4}_[01]", f.read_text(errors="replace").split("same pt as", 1)[1])
             for f in cr.glob("*.txt") if "same pt as" in f.read_text(errors="replace")}
    assert links == {"MCUCXR_0117_1": ["MCUCXR_0113_1"], "MCUCXR_0166_1": ["MCUCXR_0162_1"],
                     "MCUCXR_0170_1": ["MCUCXR_0166_1", "MCUCXR_0162_1"]}
    df = MontgomeryCXRHarmonizer(p["base_image_dir"]).harmonize()
    assert len(df) == 138
    assert df["patient_id"].nunique() == 135
    assert df["study_id"].nunique() == 138


# ---------------------------------------------------------------------------
# V-H1-9: BRAX warns about a study_id (AccessionNumber) of 2 patients
# ---------------------------------------------------------------------------


def test_brax_warns_about_shared_accession(tmp_path):
    """V-H1-9: an AccessionNumber of 2 patients gives a warning that names it. The data stays."""
    from radharmony.harmonizer import BRAXHarmonizer

    pd.DataFrame({
        "DicomPath": ["Anonymized_DICOMs/id_a/Study_1.1/Series_1/image-1.dcm",
                      "Anonymized_DICOMs/id_b/Study_1.2/Series_2/image-2.dcm",
                      "Anonymized_DICOMs/id_b/Study_2.1/Series_3/image-3.dcm"],
        "PngPath": ["images/a.png", "images/b.png", "images/c.png"],
        "PatientID": ["id_a", "id_b", "id_b"],
        "AccessionNumber": [111, 111, 222],
        "ViewPosition": ["PA", "PA", "AP"],
    }).to_csv(tmp_path / "master.csv", index=False)
    with pytest.warns(UserWarning, match=r"1 study_id .*111"):
        df = BRAXHarmonizer(csv_path=str(tmp_path / "master.csv")).harmonize()
    assert df["study_id"].tolist() == ["111", "111", "222"]


def test_brax_age_group_85_or_more_gives_85(tmp_path):
    """V-H1-8 (D-06): BRAX ages are 5-year groups. age_years is the low edge, so "85 or more" gives 85."""
    from radharmony.harmonizer import BRAXHarmonizer

    pd.DataFrame({
        "DicomPath": [f"Anonymized_DICOMs/id_{i}/Study_{i}.1/Series_1/image-{i}.dcm" for i in range(3)],
        "PngPath": [f"images/{i}.png" for i in range(3)],
        "PatientID": ["id_0", "id_1", "id_2"],
        "AccessionNumber": [1, 2, 3],
        "ViewPosition": ["PA", "PA", "PA"],
        "PatientAge": ["0", "80", "85 or more"],
    }).to_csv(tmp_path / "master.csv", index=False)
    df = BRAXHarmonizer(csv_path=str(tmp_path / "master.csv")).harmonize()
    assert df["age_years"].tolist() == [0.0, 80.0, 85.0]
    assert df["patient_age"].tolist() == ["0", "80", "85 or more"]  # the source value stays


def test_brax_shared_accessions_on_nas():
    """V-H1-9 on the real file: the warning names the accessions of 2 patients (pandas count)."""
    from radharmony.harmonizer import BRAXHarmonizer

    p = _readable("brax")
    raw = pd.read_csv(p["csv_path"], usecols=["PatientID", "AccessionNumber"])
    n = raw.groupby("AccessionNumber")["PatientID"].nunique()
    shared = sorted(str(a) for a in n[n > 1].index)
    assert len(shared) == 2
    with pytest.warns(UserWarning, match=rf"{len(shared)} study_id .*{shared[0]}.*{shared[1]}"):
        BRAXHarmonizer(**p).harmonize()


# ---------------------------------------------------------------------------
# V-H1-12: Emory warns about accessions with 2 different label or report rows
# ---------------------------------------------------------------------------


def test_emory_prefers_the_real_row_of_a_study(tmp_path):
    """V-H1-12 (D-03): a blank report template and a "No Finding only" label row lose to another row.

    Study 1: a blank template and a real report, and a "No Finding only" label row after a real one.
    Study 2: the "No Finding only" label row comes first. Study 3: 2 real reports and 2 real label rows,
    so the first row stays, with a warning. Study 4: only a blank template, so no report (NaN).
    """
    from radharmony.harmonizer import EmoryCXRHarmonizer

    acc = [f"90000000000000{i}" for i in range(1, 5)]
    pd.DataFrame({"empi_anon": [1, 2, 3, 4], "AccessionNumber_anon": acc,
                  "ImagePath": ["1/a.png", "2/b.png", "3/c.png", "4/d.png"], "ViewPosition": ["PA"] * 4,
                  "Sex": ["M", "F", "M", "F"], "Age": [50, 60, 70, 80]}).to_csv(tmp_path / "meta.csv", index=False)
    pd.DataFrame({"AccessionNumber_anon": [acc[0], acc[0], acc[1], acc[1], acc[2], acc[2]],
                  "No Finding": [None, 1.0, 1.0, None, None, None],
                  "Atelectasis": [1.0, None, None, None, 1.0, 0.0],
                  "Cardiomegaly": [None, None, None, 1.0, None, None]}).to_csv(tmp_path / "labels.csv", index=False)
    pd.DataFrame({"AccessionNumber_anon": [acc[0], acc[0], acc[2], acc[2], acc[3]],
                  "deid_report": ["REPORT\r\r\n\n___ ", "XR CHEST: no acute finding.", "Old text.", "New text.",
                                  "REPORT"]}).to_csv(tmp_path / "reports.csv", index=False)
    h = EmoryCXRHarmonizer(csv_path=str(tmp_path / "meta.csv"), label_csv_path=str(tmp_path / "labels.csv"),
                           report_csv_path=str(tmp_path / "reports.csv"))
    with pytest.warns(UserWarning) as rec:
        df = h.harmonize()
    msgs = [str(w.message) for w in rec if "Emory CXR" in str(w.message)]
    assert any("1 accessions" in m and "label" in m for m in msgs), msgs  # study 3 only
    assert any("1 accessions" in m and "report" in m for m in msgs), msgs  # study 3 only
    row = df.set_index("study_id")
    assert row.loc[acc[0], "report"] == "XR CHEST: no acute finding."
    assert row.loc[acc[0], "atelectasis"] == 1.0 and pd.isna(row.loc[acc[0], "no_finding"])
    assert row.loc[acc[1], "cardiomegaly"] == 1.0 and pd.isna(row.loc[acc[1], "no_finding"])
    assert row.loc[acc[2], "report"] == "Old text." and row.loc[acc[2], "atelectasis"] == 1.0
    assert pd.isna(row.loc[acc[3], "report"])


def test_emory_conflicting_duplicate_rows_on_nas():
    """V-H1-12 (D-03) on the real files: the warning counts equal a pandas count of the raw CSVs.

    The count leaves out a blank report template and a "No Finding only" label row when the study has
    another row.
    """
    import re
    import warnings

    from radharmony.harmonizer import EmoryCXRHarmonizer
    from radharmony.harmonizer.emory_cxr.emory_cxr import _FINDING_LABEL_COLS

    p = _readable("emory_cxr")
    acc = "AccessionNumber_anon"
    in_meta = set(pd.read_csv(p["csv_path"], usecols=[acc], dtype=str)[acc].str.strip())

    def conflicts(t, drop):
        t = t[~(drop & t[acc].isin(set(t.loc[~drop, acc])))].drop_duplicates()
        return len(set(t.loc[t[acc].duplicated(), acc]) & in_meta)

    lab = pd.read_csv(p["label_csv_path"])
    lab[acc] = lab[acc].astype(str).str.strip()
    cols = [c for c in _FINDING_LABEL_COLS if c in lab.columns]
    lab = lab[[acc] + cols]
    nf_only = (lab[cols].notna().sum(axis=1) == 1) & (lab["No Finding"] == 1)
    rep = pd.read_csv(p["report_csv_path"], dtype=str)[[acc, "deid_report"]]
    rep[acc] = rep[acc].str.strip()
    blank = rep["deid_report"].map(lambda t: pd.isna(t) or not re.sub(r"[\s_]+", "", re.sub(r"(?i)^\s*report", "", t)))
    rep.loc[blank, "deid_report"] = None
    expected = {"label": conflicts(lab, nf_only), "report": conflicts(rep, blank)}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        EmoryCXRHarmonizer(**p).harmonize()
    msgs = " | ".join(str(w.message) for w in rec if "Emory CXR" in str(w.message))
    assert f"Emory CXR: {expected['label']} accessions have 2 or more different rows in the label CSV" in msgs
    assert f"Emory CXR: {expected['report']} accessions have 2 or more different rows in the report CSV" in msgs


@pytest.mark.parametrize("value", ["[0, 0, 0, 0, 0, 0]", "[1, 0, 0, 0, 1]", "5", None, "not a list"])
def test_ct_rate_bad_orientation_gives_no_plane(value):
    """D-01: a vector that is not 6 numbers, or has no normal, gives NaN, not a plane."""
    from radharmony.harmonizer.ct_rate.ct_rate import _nearest_plane
    assert pd.isna(_nearest_plane(value))


def test_rsna_2024_lumbar_blank_severity_is_nan(tmp_path):
    """Q-17 (D-05): a blank severity gives NaN in its 3 one-hot columns, not 0."""
    from radharmony.harmonizer import RSNA2024LumbarSpineTrainHarmonizer
    from radharmony.harmonizer.rsna_2024_lumbar_spine.rsna_2024_lumbar_spine import _BASE_TARGETS

    blank, severe = _BASE_TARGETS[:2]
    labels = {t: ["Normal/Mild"] for t in _BASE_TARGETS}
    labels[blank], labels[severe] = [None], ["Severe"]
    pd.DataFrame({"study_id": [1], **labels}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"study_id": [1], "series_id": [10], "series_description": ["Sagittal T1"]}
                 ).to_csv(tmp_path / "series.csv", index=False)
    row = RSNA2024LumbarSpineTrainHarmonizer(
        csv_path=str(tmp_path / "train.csv"), series_description_csv_path=str(tmp_path / "series.csv"),
        num_workers=1).harmonize().iloc[0]
    grades = ("normal_mild", "moderate", "severe")
    assert all(pd.isna(row[f"{blank}_{g}"]) for g in grades)
    assert [row[f"{severe}_{g}"] for g in grades] == [0, 0, 1]


def test_padchest_empty_label_list_is_nan(tmp_path):
    """Q-15 (D-05): an empty label list says nothing, so all labels are NaN, not 0 (negative)."""
    from radharmony.harmonizer import PadChestHarmonizer

    pd.DataFrame({
        "ImageID": ["a.png", "b.png"], "ImageDir": [0, 0], "PatientID": ["p1", "p2"], "StudyID": ["s1", "s2"],
        "Labels": ["[]", "['cardiomegaly']"], "Projection": ["PA", "PA"], "PatientSex_DICOM": ["M", "F"],
        "PatientBirth": [1950.0, 1960.0], "StudyDate_DICOM": [20140915, 20140915],
    }).to_csv(tmp_path / "padchest.csv", index=False)
    h = PadChestHarmonizer(csv_path=str(tmp_path / "padchest.csv"))
    df = h.harmonize().set_index("patient_id")
    labels = [c.lower().replace(" ", "_") for c in h.LABEL_COLS]
    assert df.loc["p1", labels].isna().all()
    assert df.loc["p2", "cardiomegaly"] == 1 and df.loc["p2", "normal"] == 0


# --------------------------------------------------------------------------
# Q-4 (D-05): the CheXpert-style label codes
# --------------------------------------------------------------------------
@pytest.mark.parametrize("not_mentioned, drop_uncertain, expected", [
    ("negative", True, [[1.0, 0.0], [0.0, 0.0]]),             # the default: a blank is 0, the -1 row goes
    ("negative", False, [[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]]),
    ("nan", True, [[1.0, 0.0], [np.nan, 0.0]]),
    ("nan", False, [[1.0, 0.0], [np.nan, 0.0], [-1.0, np.nan]]),
])
def test_chexpert_codes_follow_d05(not_mentioned, drop_uncertain, expected):
    """-1 (uncertain) stays -1. A blank (not mentioned) is 0 by default, or NaN with not_mentioned="nan"."""
    from radharmony.harmonizer._labels import chexpert_codes

    df = pd.DataFrame({"a": [1.0, np.nan, -1.0], "b": [0.0, 0.0, np.nan], "id": ["x", "y", "z"]})
    out = chexpert_codes(df, ["a", "b", "not_here"], not_mentioned=not_mentioned, drop_uncertain=drop_uncertain)
    np.testing.assert_array_equal(out[["a", "b"]].to_numpy(), np.array(expected))
    assert df["a"].isna().sum() == 1  # the input stays as it was


def test_chexpert_codes_keep_nan_for_a_row_with_no_label():
    """V-H1-41 (D-05): a row with every label blank has no label at all, so it stays NaN, not 0."""
    from radharmony.harmonizer._labels import chexpert_codes

    df = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, np.nan]})
    out = chexpert_codes(df, ["a", "b"])
    np.testing.assert_array_equal(out.to_numpy(), np.array([[1.0, 0.0], [np.nan, np.nan]]))


def test_chexpert_codes_reject_another_value():
    from radharmony.harmonizer._labels import chexpert_codes

    with pytest.raises(ValueError, match="not_mentioned"):
        chexpert_codes(pd.DataFrame({"a": [1.0]}), ["a"], not_mentioned="zero")


@pytest.mark.parametrize("drop_uncertain", [True, False])
def test_mimic_jpg_keeps_uncertain_as_minus_one(tmp_path, drop_uncertain):
    """Q-4 (D-05): MIMIC-CXR-JPG keeps -1 (uncertain), and drop_uncertain drops its row. Before, -1 became NaN."""
    from radharmony.harmonizer import MIMICCXRJPGHarmonizer

    meta = pd.DataFrame({"dicom_id": ["d1", "d2", "d3"], "study_id": [11, 12, 13], "subject_id": [1, 2, 3],
                         "ViewPosition": ["PA"] * 3})
    meta.to_csv(tmp_path / "meta.csv", index=False)
    # Row 3 has a label (Cardiomegaly), so its blank Edema is "not mentioned", not "no label at all".
    labels = meta[["subject_id", "study_id"]].assign(Edema=[1.0, -1.0, np.nan], Cardiomegaly=[0.0, 0.0, 1.0])
    labels.to_csv(tmp_path / "labels.csv", index=False)
    h = MIMICCXRJPGHarmonizer(csv_path=str(tmp_path / "meta.csv"), label_csv_path=str(tmp_path / "labels.csv"))
    got = h.harmonize(drop_uncertain=drop_uncertain).set_index("patient_id")["edema"].to_dict()
    assert got == ({"1": 1.0, "3": 0.0} if drop_uncertain else {"1": 1.0, "2": -1.0, "3": 0.0})


def test_chexpert_not_mentioned_nan_keeps_the_blank(tmp_path):
    """Q-4 (D-05): CheXpert with not_mentioned="nan" keeps a blank label as NaN, and -1 as -1."""
    from radharmony.harmonizer import CheXpertHarmonizer

    labels = list(CheXpertHarmonizer.LABEL_COLS)
    rows = [{"Path": f"CheXpert-v1.0/train/patient{i}/study1/view1_frontal.jpg", "Sex": "Male", "Age": 50,
             "Frontal/Lateral": "Frontal", "AP/PA": "AP", **{c: np.nan for c in labels}} for i in range(3)]
    rows[0]["Edema"], rows[1]["Edema"] = 1.0, -1.0
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    h = CheXpertHarmonizer(csv_path=str(tmp_path / "train.csv"))
    got = h.harmonize(drop_uncertain=False, not_mentioned="nan").set_index("patient_id")["edema"]
    assert got["patient0"] == 1.0 and got["patient1"] == -1.0 and pd.isna(got["patient2"])

    from radharmony.dataset import CheXpertDataset

    ds = CheXpertDataset(base_image_dir=str(tmp_path), csv_path=str(tmp_path / "train.csv"), cache_dir=None,
                         drop_uncertain=False, not_mentioned="nan")
    assert ds.get_harmonized_df()["edema"].isna().sum() == 1  # the dataset class passes the argument on


_EDEMA = [1.0, -1.0, None]  # positive, uncertain, blank (not mentioned)


def _q4_chexpert(d):
    from radharmony.harmonizer import CheXpertHarmonizer

    rows = [{"Path": f"CheXpert-v1.0/train/p{i}/study1/view1_frontal.jpg", "Sex": "Male", "Age": 50,
             "Frontal/Lateral": "Frontal", "AP/PA": "AP", **{c: None for c in CheXpertHarmonizer.LABEL_COLS},
             "Edema": e} for i, e in enumerate(_EDEMA)]
    pd.DataFrame(rows).to_csv(d / "train.csv", index=False)
    return CheXpertHarmonizer(csv_path=str(d / "train.csv")), ["p0", "p1", "p2"]


def _q4_chexpert_plus(d):
    import json

    from test_chexpert_plus import _fake

    from radharmony.harmonizer.chexpert.chexpert_plus import CheXpertPlusHarmonizer

    csv, base = _fake(d, n_patients=3, on_disk=(0, 1, 2))
    paths = pd.read_csv(csv)["path_to_image"]
    (d / "labels.jsonl").write_text("\n".join(json.dumps({"path_to_image": p, "Edema": e})
                                              for p, e in zip(paths, _EDEMA, strict=True)))
    h = CheXpertPlusHarmonizer(csv_path=str(csv), label_json_path=str(d / "labels.jsonl"), base_image_dir=str(base))
    return h, ["patient00000", "patient00001", "patient00002"]


def _q4_mimic_tables(d, labeled_test=False):
    meta = pd.DataFrame({"dicom_id": ["d1", "d2", "d3"], "study_id": [11, 12, 13], "subject_id": [1, 2, 3],
                         "ViewPosition": ["PA"] * 3, "path": [f"files/p10/p{i}/s1{i}/d{i}.jpg" for i in (1, 2, 3)]})
    meta.to_csv(d / "meta.csv", index=False)
    labels = meta[["study_id"] if labeled_test else ["subject_id", "study_id"]].assign(Edema=_EDEMA)
    labels.to_csv(d / "labels.csv", index=False)
    return str(d / "meta.csv"), str(d / "labels.csv")


def _q4_mimic_jpg(d):
    from radharmony.harmonizer import MIMICCXRJPGHarmonizer

    meta, labels = _q4_mimic_tables(d)
    return MIMICCXRJPGHarmonizer(csv_path=meta, label_csv_path=labels), ["1", "2", "3"]


def _q4_mimic_jpg_test(d):
    from radharmony.harmonizer.mimic_cxr.mimic_cxr_jpg_test import MIMICCXRJPGTestHarmonizer

    meta, labels = _q4_mimic_tables(d, labeled_test=True)
    return MIMICCXRJPGTestHarmonizer(csv_path=meta, label_csv_path=labels), ["1", "2", "3"]


def _q4_mimic_dicom(d):
    from radharmony.harmonizer import MIMICCXRHarmonizer

    meta, labels = _q4_mimic_tables(d)
    return MIMICCXRHarmonizer(csv_path=meta, label_csv_path=labels), ["1", "2", "3"]


@pytest.mark.parametrize("build", [_q4_chexpert, _q4_chexpert_plus, _q4_mimic_jpg, _q4_mimic_jpg_test,
                                   _q4_mimic_dicom], ids=lambda f: f.__name__[4:])
def test_every_chexpert_style_harmonizer_follows_d05(build, tmp_path):
    """Q-4 (D-05): 1 stays 1, -1 stays -1, and a blank stays NaN with not_mentioned="nan"."""
    h, ids = build(tmp_path)
    got = h.harmonize(drop_uncertain=False, not_mentioned="nan").set_index("patient_id")["edema"]
    assert got[ids[0]] == 1.0 and got[ids[1]] == -1.0 and pd.isna(got[ids[2]])
    (tmp_path / "again").mkdir()
    h2, _ = build(tmp_path / "again")
    with pytest.raises(ValueError, match="not_mentioned"):
        h2.harmonize(not_mentioned="zero")


@pytest.mark.parametrize("build", [_q4_chexpert, _q4_chexpert_plus, _q4_mimic_jpg, _q4_mimic_jpg_test,
                                   _q4_mimic_dicom], ids=lambda f: f.__name__[4:])
def test_a_row_with_no_label_at_all_is_nan_by_default(build, tmp_path):
    """V-H1-41 (D-05): with the default call, a source row with every label blank gets NaN, not 0."""
    h, ids = build(tmp_path)
    got = h.harmonize().set_index("patient_id")
    labels = [c.lower().replace(" ", "_") for c in h.LABEL_COLS]
    assert got.loc[ids[2], [c for c in labels if c in got.columns]].isna().all()  # every label is blank
    assert got.loc[ids[0], "edema"] == 1.0


# ---------------------------------------------------------------------------
# MedSeg COVID CT slices (V-H1-42, Q-11, Q-12 of the 2026-11-02 review)
# ---------------------------------------------------------------------------

#: The dataset1 slice map in the release format: only the first row of a case has the age and sex. The
#: "Not used" row lists SIRM cases, not slices.
_MEDSEG_CLINICAL_CSV = (
    "SIRM Case,Slice position in .nii.gz,Patient age,Patient Sex (Male = 0 Female = 1),,,\n"
    "7,1,45,1,,,\n"
    "7,0,,,,,\n"
    ",,,,,,\n"
    "Not used,48,35,26,20,15,12\n"
)


def _medseg_release(d, subset="dataset1", clinical_csv=_MEDSEG_CLINICAL_CSV):
    """Write a tiny MedSeg release. Each slice k has HU values in a gradient along y (rows) and x (columns)."""
    import SimpleITK as sitk

    y, x = np.mgrid[0:8, 0:6]
    vol = np.stack([(y * 100 - 600 + 30 * x + 10 * k) for k in range(2)]).astype(np.int16)  # (z, y, x)
    msk = np.zeros_like(vol, np.uint8)
    msk[:, 1, 2] = 1
    msk[1, 5, 4] = 2
    if subset == "dataset1":
        files = {"tr_im.nii.gz": vol, "tr_mask.nii.gz": msk}
        if clinical_csv is not None:
            (d / "Test-Images-Clinical-Details.csv").write_text(clinical_csv)
    else:
        (d / "rp_im").mkdir()
        (d / "rp_msk").mkdir()
        files = {"rp_im/1.nii.gz": vol, "rp_msk/1.nii.gz": msk}
    for name, arr in files.items():
        sitk.WriteImage(sitk.GetImageFromArray(arr), str(d / name))
    return vol


def test_medseg_slices_use_the_usual_axial_view(tmp_path):
    """Q-11 (D-12): a slice PNG has its rows along y, as ``arr[k]`` of the volume. Before, it was the transpose."""
    from radharmony.harmonizer import MedSegCOVIDCTHarmonizer

    vol = _medseg_release(tmp_path)
    df = MedSegCOVIDCTHarmonizer(base_dir=str(tmp_path), slice_output_dir=str(tmp_path / "s")).harmonize()
    png = np.array(Image.open(tmp_path / "s" / "dataset1_hu-1000_400" / "images" / df["image_path"][1]))
    expected = (np.clip((vol[1] + 1000.0) / 1400.0, 0, 1) * 255).round().astype(np.uint8)  # lung window by hand
    np.testing.assert_array_equal(png, expected)
    mask = np.array(Image.open(df["mask_path"][1]))
    assert mask.shape == (8, 6) and mask[1, 2] == 255 and mask[5, 4] == 255 and mask.sum() == 2 * 255


def test_medseg_dataset1_patient_comes_from_the_slice_map(tmp_path):
    """D-10, D-06, D-26: ``Test-Images-Clinical-Details.csv`` maps each ``tr_im`` slice to a SIRM case.

    Both slices of the fake release are case 7, so they are 1 patient. The age and sex are only on the
    first row of the case, and they apply to all its slices. Sex 1 is female.
    """
    from radharmony.harmonizer import MedSegCOVIDCTHarmonizer

    _medseg_release(tmp_path)
    df = MedSegCOVIDCTHarmonizer(base_dir=str(tmp_path), slice_output_dir=str(tmp_path / "s")).harmonize()
    assert df["patient_id"].tolist() == ["7", "7"]
    assert df["age_years"].tolist() == [45.0, 45.0]
    assert df["sex"].tolist() == ["F", "F"]


def test_medseg_dataset1_without_the_slice_map_raises(tmp_path):
    """D-10: with no slice map, each slice would be its own patient, and a split could share patients."""
    from radharmony.harmonizer import MedSegCOVIDCTHarmonizer

    _medseg_release(tmp_path, clinical_csv=None)
    with pytest.raises(FileNotFoundError, match="Test-Images-Clinical-Details.csv"):
        MedSegCOVIDCTHarmonizer(base_dir=str(tmp_path), slice_output_dir=str(tmp_path / "s")).harmonize()


def test_medseg_new_hu_window_writes_new_slices(tmp_path):
    """V-H1-42: the window is in the folder name, so a new window does not use the PNGs of the old one."""
    from radharmony.harmonizer import MedSegCOVIDCTHarmonizer

    _medseg_release(tmp_path)
    out = []
    for window in ((-1000.0, 400.0), (-160.0, 240.0)):
        h = MedSegCOVIDCTHarmonizer(base_dir=str(tmp_path), slice_output_dir=str(tmp_path / "s"), hu_window=window)
        df = h.harmonize()
        out.append(Path(df["mask_path"][0]).parent.parent)
    assert out[0] != out[1] and out[1].name == "dataset1_hu-160_240"
    lung = np.array(Image.open(out[0] / "images" / "dataset1_tr_0000.png"))
    soft = np.array(Image.open(out[1] / "images" / "dataset1_tr_0000.png"))
    assert not np.array_equal(lung, soft)


def test_medseg_dataset2_pleural_effusion_is_nan(tmp_path):
    """Q-12 (D-05): the dataset2 masks have no pleural effusion class, so the label is NaN, not 0."""
    from radharmony.harmonizer import MedSegCOVIDCTHarmonizer

    _medseg_release(tmp_path, subset="dataset2")
    df = MedSegCOVIDCTHarmonizer(base_dir=str(tmp_path), slice_output_dir=str(tmp_path / "s"),
                                 subset="dataset2").harmonize()
    assert df["pleural_effusion"].isna().all()
    assert df["ground_glass"].tolist() == [1, 1] and df["consolidation"].tolist() == [0, 1]


def test_medseg_dataset2_sample_has_nan_pleural_effusion_in_cls(tmp_path):
    """Q-12: ``ds[0]`` of dataset2 with ``output_cls`` loads, and ``cls`` has NaN for pleural_effusion.

    ``cls`` is alphabetical: any_lesion, consolidation, ground_glass, pleural_effusion. Slice 1 of the fake
    release has ground glass and consolidation.
    """
    import torch

    from radharmony.dataset import MedSegCOVIDCTDataset

    _medseg_release(tmp_path, subset="dataset2")
    ds = MedSegCOVIDCTDataset(base_image_dir=str(tmp_path), slice_output_dir=str(tmp_path / "s"),
                              subset="dataset2", output_cls=True, cache_dir=None).get_datasets()
    cls = ds[1]["cls"].float()
    assert cls[:3].tolist() == [1.0, 1.0, 1.0] and torch.isnan(cls[3])


def test_medseg_dataset_reads_the_folder_of_its_harmonizer(tmp_path):
    """V-H1-42: a dataset built from a harmonizer object reads the slice folder of that harmonizer's window."""
    from radharmony.dataset import MedSegCOVIDCTDataset
    from radharmony.harmonizer import MedSegCOVIDCTHarmonizer

    _medseg_release(tmp_path)
    h = MedSegCOVIDCTHarmonizer(base_dir=str(tmp_path), slice_output_dir=str(tmp_path / "s"), hu_window=(-160, 240))
    h.harmonize()
    ds = MedSegCOVIDCTDataset(harmonizer=h, cache_dir=None)
    assert ds.base_image_dir == str(tmp_path / "s" / "dataset1_hu-160_240" / "images")
    assert tuple(ds.get_datasets()[0]["img"].shape) == (1, 224, 224)


def test_medseg_slice_folder_keeps_close_windows_apart():
    """V-H1-42: 2 windows that differ in the 5th decimal get 2 folders."""
    from radharmony.harmonizer.medseg_covid_ct.medseg_covid_ct import slice_folder

    assert slice_folder("a", (-1000.123456, 400)) != slice_folder("a", (-1000.12, 400))
    assert slice_folder("a", (-1000.0, 400.0)) == "a_hu-1000_400"
