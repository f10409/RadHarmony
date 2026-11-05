"""Official split classes keep only their own rows (D-09, D-25). No data needed.

Each test writes a few fake source files with the real file names and
columns, then checks the rows of each split class. The real counts are in
``configs/datasets/*.json`` (``expected_stats``), checked by
``tests/test_dataset_stats.py`` where the data is readable.

Run with::

    .venv/bin/python -m pytest tests/test_official_splits.py -v
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from fixtures.stub_preprocessor import StubPreprocessor
from radharmony.dataset import (
    RANZCRClipDataset,
    RANZCRClipTestDataset,
    RANZCRClipTrainDataset,
    VinDrPCXRDataset,
    VinDrPCXRTestDataset,
    VinDrPCXRTrainDataset,
)
from radharmony.harmonizer import (
    ChexInstructVQATestHarmonizer,
    ChexInstructVQATrainHarmonizer,
    ChexInstructVQAValHarmonizer,
    MIMICCXRJPGTestHarmonizer,
    MIMICCXRTestHarmonizer,
    MIMICCXRTrainHarmonizer,
    MIMICCXRValidateHarmonizer,
    MIMICCXRJPGTrainHarmonizer,
    MIMICCXRJPGValidateHarmonizer,
)

MIMIC_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
                "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion",
                "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]


@pytest.fixture
def mimic_dir(tmp_path):
    """Fake MIMIC-CXR-JPG folder: patient 1 train, 2 and 3 validate, 4 test.

    Patient 3 has a validate study in the labeled test set, as 3 real
    patients do.
    """
    rows = [  # dicom_id, study_id, subject_id, split
        ("d1", 11, 1, "train"), ("d2", 21, 2, "validate"), ("d3", 31, 3, "validate"),
        ("d4", 41, 4, "test"), ("d5", 42, 4, "test"),
    ]
    split = pd.DataFrame(rows, columns=["dicom_id", "study_id", "subject_id", "split"])
    split.to_csv(tmp_path / "mimic-cxr-2.0.0-split.csv", index=False)
    meta = split.drop(columns="split").assign(ViewPosition="PA")
    meta.to_csv(tmp_path / "mimic-cxr-2.0.0-metadata.csv", index=False)
    labels = split[["subject_id", "study_id"]].assign(**{c: 0.0 for c in MIMIC_LABELS})
    labels.to_csv(tmp_path / "mimic-cxr-2.0.0-chexpert.csv", index=False)
    test = pd.DataFrame({"study_id": [31, 41, 42]}).assign(
        **{c: 0.0 for c in MIMIC_LABELS if c != "Lung Opacity"}, **{"Airspace Opacity": 1.0})
    test.to_csv(tmp_path / "mimic-cxr-2.1.0-test-set-labeled.csv", index=False)
    return tmp_path


def _mimic_jpg(cls, d, **kw):
    df = cls(csv_path=str(d / "mimic-cxr-2.0.0-metadata.csv"),
             label_csv_path=str(d / "mimic-cxr-2.0.0-chexpert.csv"), **kw).harmonize()
    return sorted(df["patient_id"].unique())


def test_mimic_jpg_split_classes_follow_the_split_file(mimic_dir):
    assert _mimic_jpg(MIMICCXRJPGTrainHarmonizer, mimic_dir) == ["1"]
    # D-34: patient 3 is also in the labeled test set. Validate keeps it, as the split file says.
    assert _mimic_jpg(MIMICCXRJPGValidateHarmonizer, mimic_dir) == ["2", "3"]
    test = MIMICCXRJPGTestHarmonizer(
        csv_path=str(mimic_dir / "mimic-cxr-2.0.0-metadata.csv"),
        label_csv_path=str(mimic_dir / "mimic-cxr-2.1.0-test-set-labeled.csv")).harmonize()
    assert sorted(test["patient_id"].unique()) == ["3", "4"]


def test_mimic_jpg_official_test_split_has_the_chexpert_labels(mimic_dir):
    """D-09: mimic_cxr_jpg_test_chexpert gives the test split of the split file, with the labeler labels.

    Patient 4 is the only test patient. Its labeler label for Lung Opacity is 0, and the 2.1.0 label
    (Airspace Opacity) is 1, so the label tells the 2 test classes apart.
    """
    from radharmony.harmonizer import MIMICCXRJPGTestCheXpertHarmonizer
    df = MIMICCXRJPGTestCheXpertHarmonizer(csv_path=str(mimic_dir / "mimic-cxr-2.0.0-metadata.csv"),
                                           label_csv_path=str(mimic_dir / "mimic-cxr-2.0.0-chexpert.csv")).harmonize()
    assert sorted(df["patient_id"].unique()) == ["4"]
    assert (df["lung_opacity"] == 0).all()


def test_mimic_jpg_split_file_is_required(mimic_dir):
    (mimic_dir / "mimic-cxr-2.0.0-split.csv").unlink()
    with pytest.raises(FileNotFoundError, match="split_csv_path"):
        _mimic_jpg(MIMICCXRJPGTrainHarmonizer, mimic_dir)


def test_mimic_jpg_validate_does_not_need_the_labeled_test_file(mimic_dir):
    """D-34: validate gives its split as published, so it does not read the labeled test set."""
    (mimic_dir / "mimic-cxr-2.1.0-test-set-labeled.csv").unlink()
    assert _mimic_jpg(MIMICCXRJPGValidateHarmonizer, mimic_dir) == ["2", "3"]


def test_mimic_jpg_split_save_and_load(mimic_dir, tmp_path_factory):
    h = MIMICCXRJPGValidateHarmonizer(csv_path=str(mimic_dir / "mimic-cxr-2.0.0-metadata.csv"),
                                      label_csv_path=str(mimic_dir / "mimic-cxr-2.0.0-chexpert.csv"))
    df = h.harmonize()
    path = tmp_path_factory.mktemp("saved") / "h.pkl"
    h.save(str(path))
    pd.testing.assert_frame_equal(MIMICCXRJPGValidateHarmonizer.load_from_saved(str(path)).harmonized_df, df)


def test_mimic_dicom_split_classes_follow_the_split_file(mimic_dir):
    """The DICOM classes keep the split file as it is (patient 3 stays in validate)."""
    rec = pd.read_csv(mimic_dir / "mimic-cxr-2.0.0-metadata.csv")
    rec["path"] = "files/p1/x/" + rec["dicom_id"] + ".dcm"
    rec[["subject_id", "study_id", "dicom_id", "path"]].to_csv(mimic_dir / "cxr-record-list.csv", index=False)
    got = {}
    for cls in (MIMICCXRTrainHarmonizer, MIMICCXRValidateHarmonizer, MIMICCXRTestHarmonizer):
        df = cls(csv_path=str(mimic_dir / "cxr-record-list.csv"),
                 label_csv_path=str(mimic_dir / "mimic-cxr-2.0.0-chexpert.csv")).harmonize()
        got[cls.SPLIT] = sorted(df["patient_id"].unique())
    assert got == {"train": ["1"], "validate": ["2", "3"], "test": ["4"]}


RANZCR_RAW = ["CVC - Abnormal", "CVC - Borderline", "CVC - Normal", "ETT - Abnormal", "ETT - Borderline",
              "ETT - Normal", "NGT - Abnormal", "NGT - Borderline", "NGT - Incompletely Imaged", "NGT - Normal",
              "Swan Ganz Catheter Present"]


@pytest.fixture
def ranzcr_dir(tmp_path):
    """Fake Kaggle folder: 2 labeled train images, 1 unlabeled test image."""
    train = pd.DataFrame({"StudyInstanceUID": ["a", "b"], "PatientID": ["p1", "p2"]})
    train = train.assign(**{c: 0 for c in RANZCR_RAW})
    train.to_csv(tmp_path / "train.csv", index=False)
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "t1.jpg").write_bytes(b"")
    return tmp_path


def _ids(ds):
    return sorted(ds.get_harmonized_df()["study_id"])


def test_ranzcr_split_classes(ranzcr_dir):
    """V-H2-25, V-D-20: train has only labeled rows, test has only the test images, with no labels."""
    train = RANZCRClipTrainDataset(base_image_dir=str(ranzcr_dir), output_cls=True, cache_dir=None)
    assert _ids(train) == ["a", "b"]
    train_ds, val_ds = train.get_datasets(n_splits=2)
    assert all(not pd.isna(v) for d in list(train_ds.data) + list(val_ds.data) for v in d["cls"])
    test = RANZCRClipTestDataset(base_image_dir=str(ranzcr_dir), cache_dir=None)
    df = test.get_harmonized_df()
    assert sorted(df["study_id"]) == ["t1"] and df["image_path"].tolist() == ["test/t1.jpg"]
    assert not set(RANZCRClipDataset.LABEL_COLS) & set(df.columns)
    with pytest.warns(UserWarning, match="no labels"):
        RANZCRClipTestDataset(base_image_dir=str(ranzcr_dir), output_cls=True, cache_dir=None)


def test_ranzcr_old_class_keeps_test_rows_and_fails_clearly(ranzcr_dir):
    """The old key keeps include_test_split=True. A test row with output_cls=True fails with a clear error."""
    old = RANZCRClipDataset(base_image_dir=str(ranzcr_dir), output_cls=True, cache_dir=None)
    assert _ids(old) == ["a", "b", "t1"]
    ds = old.get_datasets()
    i = next(i for i, d in enumerate(ds.data) if d["img"].endswith("t1.jpg"))
    with pytest.raises(RuntimeError) as err:
        ds[i]
    assert "test image with no labels" in str(err.value.__cause__)


def test_chexinstruct_split_classes_keep_one_split(tmp_path):
    """V-H1-4: each class keeps the questions of its own split field."""
    def rec(i, split):
        return {"unique_id": i, "study_id": f"[mimic-cxr] [5{i}]", "data_source": "MIMIC-CXR-VQA",
                "task_name": "Open-Ended VQA", "image_path": f"data/mimic-cxr/files/p1/p1{i}/s5{i}/d.jpg",
                "text": "q", "target_text": "a", "qa_pair": None, "split": split}
    path = tmp_path / "chexinstruct_vqa.json"
    path.write_text(json.dumps([rec(1, "train"), rec(2, "train"), rec(3, "val"), rec(4, "test")]))
    got = {cls.SPLIT: sorted(cls(json_path=str(path)).harmonize()["question_id"])
           for cls in (ChexInstructVQATrainHarmonizer, ChexInstructVQAValHarmonizer, ChexInstructVQATestHarmonizer)}
    assert got == {"train": ["1", "2"], "val": ["3"], "test": ["4"]}


def _chexinstruct_two_sources(tmp_path):
    """A compiled JSON with one question of each source."""
    def rec(i, source):
        return {"unique_id": i, "study_id": f"[mimic-cxr] [5{i}]", "data_source": source,
                "task_name": "Open-Ended VQA", "image_path": f"data/mimic-cxr/files/p1/p1{i}/s5{i}/d.jpg",
                "text": "q", "target_text": "a", "qa_pair": None, "split": "train"}
    path = tmp_path / "chexinstruct_vqa.json"
    path.write_text(json.dumps([rec(1, "MIMIC-CXR-VQA"), rec(2, "MIMIC-Diff-VQA")]))
    return path


def test_chexinstruct_image_path_is_relative(tmp_path):
    """Q-11 (D-04): image_path is relative to the folder of its source, and base_image_dir is the shared folder."""
    from radharmony.harmonizer import ChexInstructVQAHarmonizer

    h = ChexInstructVQAHarmonizer(json_path=str(_chexinstruct_two_sources(tmp_path)), base_image_dir="/data/files")
    assert h.harmonize()["image_path"].tolist() == ["p1/p11/s51/d.jpg", "p1/p12/s52/d.jpg"]
    assert h.base_image_dir == "/data/files"


def test_chexinstruct_dataset_joins_the_folder_of_each_source(tmp_path):
    """Q-11 (D-04): the dataset joins the shared folder, or the folder of each source."""
    from radharmony.dataset import ChexInstructVQADataset

    path = str(_chexinstruct_two_sources(tmp_path))
    one = ChexInstructVQADataset(base_image_dir="/data/files", json_path=path, cache_dir=None).get_datasets()
    assert [d["img"] for d in one.data] == ["/data/files/p1/p11/s51/d.jpg", "/data/files/p1/p12/s52/d.jpg"]
    two = ChexInstructVQADataset(source_image_dirs={"MIMIC-CXR-VQA": "/a", "MIMIC-Diff-VQA": "/b"}, json_path=path,
                                 cache_dir=None).get_datasets()
    assert [d["img"] for d in two.data] == ["/a/p1/p11/s51/d.jpg", "/b/p1/p12/s52/d.jpg"]


def test_chexinstruct_dataset_loads_an_old_table_and_a_harmonizer_object(tmp_path):
    """Q-11 (D-04): an old table with absolute paths, and a harmonizer object, give the right sample paths."""
    from radharmony.dataset import ChexInstructVQADataset
    from radharmony.harmonizer import ChexInstructVQAHarmonizer

    path = str(_chexinstruct_two_sources(tmp_path))
    want = ["/data/files/p1/p11/s51/d.jpg", "/data/files/p1/p12/s52/d.jpg"]
    h = ChexInstructVQAHarmonizer(json_path=path, base_image_dir="/data/files")
    old_table = h.harmonize().assign(image_path=want)  # the old code wrote absolute paths
    ds = ChexInstructVQADataset(base_image_dir="/data/files", harmonized_df=old_table, cache_dir=None)
    assert [d["img"] for d in ds.get_datasets().data] == want
    h2 = ChexInstructVQAHarmonizer(json_path=path, base_image_dir="/data/files")
    h2.harmonize()
    ds = ChexInstructVQADataset(harmonizer=h2, cache_dir=None)
    assert [d["img"] for d in ds.get_datasets().data] == want


def test_chexinstruct_folder_of_each_source_with_a_repeated_id(tmp_path):
    """Q-11 (D-04): the mixed class repeats an id in each split. Each sample still gets the folder of its source."""
    import json

    from radharmony.dataset import ChexInstructVQADataset

    recs = [{"unique_id": 1, "study_id": f"[mimic-cxr] [5{i}]", "data_source": src, "task_name": "Open-Ended VQA",
             "image_path": f"data/mimic-cxr/files/p1/p1{i}/s5{i}/d.jpg", "text": "q", "target_text": "a",
             "qa_pair": None, "split": split}
            for i, src, split in ((1, "MIMIC-CXR-VQA", "train"), (2, "MIMIC-Diff-VQA", "test"))]
    (tmp_path / "chexinstruct_vqa.json").write_text(json.dumps(recs))
    ds = ChexInstructVQADataset(source_image_dirs={"MIMIC-CXR-VQA": "/a", "MIMIC-Diff-VQA": "/b"},
                                json_path=str(tmp_path / "chexinstruct_vqa.json"), cache_dir=None)
    assert sorted(d["img"] for d in ds.get_datasets().data) == ["/a/p1/p11/s51/d.jpg", "/b/p1/p12/s52/d.jpg"]


def test_chexinstruct_question_id_is_unique_in_a_split_class(tmp_path):
    """V-H1-3: the source counts unique_id again in each split. A split class has no repeated id."""
    def rec(i, study, split):
        return {"unique_id": f"[Close-Ended VQA] [MIMIC-CXR-VQA] [{i}]", "study_id": f"[mimic-cxr] [{study}]",
                "data_source": "MIMIC-CXR-VQA", "task_name": "Close-Ended VQA",
                "image_path": f"data/mimic-cxr/files/p1/p1{study}/s{study}/d.jpg",
                "text": "q", "target_text": "a", "qa_pair": None, "split": split}
    path = tmp_path / "chexinstruct_vqa.json"
    path.write_text(json.dumps([rec(0, 51, "train"), rec(1, 52, "train"), rec(0, 53, "val"), rec(0, 54, "test")]))
    for cls in (ChexInstructVQATrainHarmonizer, ChexInstructVQAValHarmonizer, ChexInstructVQATestHarmonizer):
        assert cls(json_path=str(path)).harmonize()["question_id"].is_unique


def test_vindr_pcxr_split_classes(tmp_path):
    """VinDr-PCXR is not readable here, so fake files check that each class reads only its split."""
    for split, ids in (("train", ["a", "b"]), ("test", ["c"])):
        (tmp_path / split).mkdir()
        pd.DataFrame({"image_id": ids}).to_csv(tmp_path / f"image_labels_{split}.csv", index=False)
    got = {cls.__name__: sorted(cls(base_image_dir=str(tmp_path), cache_dir=None).get_harmonized_df()["study_id"])
           for cls in (VinDrPCXRTrainDataset, VinDrPCXRTestDataset, VinDrPCXRDataset)}
    assert got == {"VinDrPCXRTrainDataset": ["a", "b"], "VinDrPCXRTestDataset": ["c"],
                   "VinDrPCXRDataset": ["a", "b", "c"]}


# Item 6: a class for one split takes no other split= / splits= value. These
# datasets are restricted, not on the NAS, or too slow for a test. So these
# tests check the arguments only.
SPLIT_ARG_HARMONIZERS = [
    ("radharmony.harmonizer", "ChestImaGenomeSilverTrainHarmonizer", "split", "train", dict(annotation_dir="/x")),
    ("radharmony.harmonizer", "ChestImaGenomeSilverValidHarmonizer", "split", "valid", dict(annotation_dir="/x")),
    ("radharmony.harmonizer", "ChestImaGenomeSilverTestHarmonizer", "split", "test", dict(annotation_dir="/x")),
    ("radharmony.harmonizer", "PadChestGRTrainHarmonizer", "split", "train", dict(base_image_dir="/x")),
    ("radharmony.harmonizer", "PadChestGRValidationHarmonizer", "split", "validation", dict(base_image_dir="/x")),
    ("radharmony.harmonizer", "PadChestGRTestHarmonizer", "split", "test", dict(base_image_dir="/x")),
    ("radharmony.harmonizer", "ROCOTrainHarmonizer", "splits", ["train"], dict(base_dir="/x")),
    ("radharmony.harmonizer", "ROCOValidationHarmonizer", "splits", ["validation"], dict(base_dir="/x")),
    ("radharmony.harmonizer", "ROCOTestHarmonizer", "splits", ["test"], dict(base_dir="/x")),
    ("radharmony.harmonizer", "ROCOv2TrainHarmonizer", "splits", ["train"], dict(base_dir="/x")),
    ("radharmony.harmonizer", "ROCOv2ValidationHarmonizer", "splits", ["validation"], dict(base_dir="/x")),
    ("radharmony.harmonizer", "ROCOv2TestHarmonizer", "splits", ["test"], dict(base_dir="/x")),
    ("radharmony.harmonizer.mrrate", "MRRATETrainHarmonizer", "splits", ["train"], dict(base_dir="/x")),
    ("radharmony.harmonizer.mrrate", "MRRATEValHarmonizer", "splits", ["val"], dict(base_dir="/x")),
    ("radharmony.harmonizer.mrrate", "MRRATETestHarmonizer", "splits", ["test"], dict(base_dir="/x")),
    ("radharmony.harmonizer", "ChestXDet10TrainHarmonizer", "split", "train", dict(csv_path="/x/train.json")),
    ("radharmony.harmonizer", "ChestXDet10TestHarmonizer", "split", "test", dict(csv_path="/x/test.json")),
]


@pytest.mark.parametrize("module,name,arg,own,kwargs", SPLIT_ARG_HARMONIZERS, ids=lambda v: v if isinstance(v, str) else "")
def test_split_arg_is_fixed(module, name, arg, own, kwargs):
    """The split harmonizer keeps its own split, and another split raises ValueError."""
    import importlib
    cls = getattr(importlib.import_module(module), name)
    h = cls(**kwargs)
    assert getattr(h, arg) == own
    other = ["train"] if own != ["train"] and own != "train" else ["test"]
    with pytest.raises(ValueError, match="gives only"):
        cls(**kwargs, **{arg: other if arg == "splits" else other[0]})


@pytest.mark.parametrize("name,own", [("ReXGradientTrainHarmonizer", "train"),
                                      ("ReXGradientValidHarmonizer", "valid"), ("ReXGradientTestHarmonizer", "test")])
def test_rexgradient_split_class_rejects_the_json_of_another_split(tmp_path, name, own):
    """V-H2-46 (D-09): a ReXGradient split class reads only ``<split>_metadata_view_position.json``."""
    import radharmony.harmonizer as H

    other = "test" if own == "train" else "train"
    path = tmp_path / f"{other}_metadata_view_position.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match=f"{own}_metadata_view_position.json"):
        getattr(H, name)(csv_path=str(path), base_image_dir=str(tmp_path)).harmonize()


def test_chestxdet10_train_class_rejects_the_test_split():
    """A class named Train gives only the train split."""
    from radharmony.dataset import ChestXDet10TestDataset, ChestXDet10TrainDataset
    train = ChestXDet10TrainDataset(base_image_dir="/x", cache_dir=None)
    assert train._harmonizer.split == "train" and train._harmonizer.csv_path == "/x/train.json"
    assert ChestXDet10TestDataset(base_image_dir="/x", cache_dir=None)._harmonizer.split == "test"
    with pytest.raises(ValueError, match="gives only the 'train' split"):
        ChestXDet10TrainDataset(base_image_dir="/x", split="test", cache_dir=None)


# Item 7: one class per split where the folder or the label file picks the split.
def test_infer_path_candidate_order_and_sub_path(tmp_path):
    """V-P-8, D-35: the first candidate wins, and a sub path matches only that sub path."""
    from radharmony.utils.infer import infer_path
    base = tmp_path / "ds" / "images"
    base.mkdir(parents=True)
    for rel in ("old_tables/meta_2023.csv", "tables/meta_2025.csv", "ann/train/labels.csv", "ann/valid/labels.csv"):
        (tmp_path / "ds" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "ds" / rel).write_text("x")
    found = infer_path(str(base), "tables/meta_2025.csv", "old_tables/meta_2023.csv")
    assert found.endswith("tables/meta_2025.csv")
    assert infer_path(str(base), "ann/valid/labels.csv").endswith("ann/valid/labels.csv")


def _ct_rate_dir(tmp_path):
    """Fake CT-RATE layout: dataset/{train_fixed,valid_fixed,tables}/ with one volume per split."""
    root = tmp_path / "dataset"
    (root / "tables").mkdir(parents=True)
    for split, meta in (("train", "train_metadata.csv"), ("valid", "validation_metadata.csv")):
        (root / f"{split}_fixed").mkdir()
        vol = f"{split}_1_a_1.nii.gz"
        pd.DataFrame({"VolumeName": [vol], "Atelectasis": [1]}).to_csv(
            root / "tables" / f"{split}_predicted_labels.csv", index=False)
        pd.DataFrame({"VolumeName": [vol], "ImageOrientationPatient": ["[1, 0, 0, 0, 1, 0]"]}).to_csv(
            root / "tables" / meta, index=False)
    return root


def test_ct_rate_valid_folder_finds_the_valid_tables(tmp_path):
    """V-P-7, V-A-3: a valid image folder with no csv_path reads the valid tables, not the train ones."""
    from radharmony.dataset import CTRATEDataset, CTRATETrainDataset, CTRATEValidDataset
    root = _ct_rate_dir(tmp_path)
    old = CTRATEDataset(base_image_dir=str(root / "valid_fixed"), cache_dir=None)
    assert old._harmonizer.csv_path.endswith("tables/valid_predicted_labels.csv")
    assert old._harmonizer.view_position_csv_path.endswith("tables/validation_metadata.csv")
    assert list(old.get_harmonized_df()["patient_id"]) == ["valid_1"]
    train = CTRATETrainDataset(base_image_dir=str(root / "train_fixed"), cache_dir=None)
    assert list(train.get_harmonized_df()["patient_id"]) == ["train_1"]
    # The valid class never reads the train table, even when you pass it.
    wrong = CTRATEValidDataset(base_image_dir=str(root / "valid_fixed"), cache_dir=None,
                               csv_path=str(root / "tables" / "train_predicted_labels.csv"))
    with pytest.raises(ValueError, match="no valid_ volume"):
        wrong.get_harmonized_df()


def test_radchestct_split_class_needs_its_own_label_file():
    from radharmony.harmonizer import RadChestCTValidHarmonizer
    h = RadChestCTValidHarmonizer(csv_path="/x/meta.csv", label_csv_path="/x/imgtrain_Abnormality_and_Location_Labels.csv")
    with pytest.raises(ValueError, match="imgvalid_"):
        h.harmonize()


def test_qata_split_class_needs_its_own_folder():
    from radharmony.harmonizer import QaTaCOV19TestHarmonizer, QaTaCOV19TrainHarmonizer
    assert "Train Set" in QaTaCOV19TrainHarmonizer(base_dir="/x/QaTa-COV19-v2/Train Set/Images/").base_dir
    with pytest.raises(ValueError, match="Test Set"):
        QaTaCOV19TestHarmonizer(base_dir="/x/QaTa-COV19-v2/Train Set/Images/")


# Item 8: CheXlocalize validation, MedSeg val_im slices, V-D-12.
def _chexlocalize_dir(tmp_path):
    (tmp_path / "test" / "patient1" / "study1").mkdir(parents=True)
    pd.DataFrame({"Path": ["test/patient1/study1/view1_frontal.jpg"], "Edema": [1.0]}).to_csv(
        tmp_path / "test_labels.csv", index=False)
    (tmp_path / "val" / "patient2" / "study1").mkdir(parents=True)
    pd.DataFrame({"Path": ["CheXpert-v1.0/valid/patient2/study1/view1_frontal.jpg"], "Sex": ["Male"],
                  "Edema": [0.0]}).to_csv(tmp_path / "val_labels.csv", index=False)
    return tmp_path


def test_chexlocalize_val_class_reads_the_val_files(tmp_path):
    from radharmony.dataset import CheXlocalizeDataset, CheXlocalizeValDataset
    root = _chexlocalize_dir(tmp_path)
    val = CheXlocalizeValDataset(base_image_dir=str(root / "val"), cache_dir=None)
    assert val.get_harmonized_df()["image_path"].tolist() == ["patient2/study1/view1_frontal.jpg"]
    test = CheXlocalizeDataset(base_image_dir=str(root / "test"), cache_dir=None)
    assert test.get_harmonized_df()["image_path"].tolist() == ["patient1/study1/view1_frontal.jpg"]


def test_a_study_of_no_row_gives_an_empty_table(mimic_dir, tmp_path):
    """V-H1-38 (D-02, L-014): a study id that no row has gives 0 rows with the same columns.

    A split class gets such an id when the study is in another split. It must not crash.
    """
    from radharmony.harmonizer import CheXlocalizeHarmonizer

    root = _chexlocalize_dir(tmp_path / "chexlocalize")
    (root / "masks.json").write_text(json.dumps({}))
    harmonizers = [
        MIMICCXRJPGTrainHarmonizer(csv_path=str(mimic_dir / "mimic-cxr-2.0.0-metadata.csv"),
                                   label_csv_path=str(mimic_dir / "mimic-cxr-2.0.0-chexpert.csv")),
        CheXlocalizeHarmonizer(csv_path=str(root / "test_labels.csv"), mask_json_path=str(root / "masks.json")),
    ]
    for h in harmonizers:
        full = h.harmonize()
        none = h.harmonize(study_ids=["no_such_study"])
        assert len(none) == 0 and list(none.columns) == list(full.columns), type(h).__name__


def test_chexlocalize_mask_dir_without_output_mask(tmp_path):
    """V-D-12: output_mask=False with mask_output_dir harmonizes (no mask step)."""
    from radharmony.dataset import CheXlocalizeDataset
    root = _chexlocalize_dir(tmp_path)
    ds = CheXlocalizeDataset(base_image_dir=str(root / "test"), mask_output_dir=str(tmp_path / "m"),
                             output_mask=False, cache_dir=None)
    assert len(ds.get_harmonized_df()) == 1


def test_medseg_val_slices_have_no_labels(tmp_path, monkeypatch):
    """D-25: the val_im slices give images only."""
    import numpy as np
    import radharmony.harmonizer.medseg_covid_ct.medseg_covid_ct_val as mod
    from radharmony.harmonizer.medseg_covid_ct import MedSegCOVIDCTValHarmonizer
    (tmp_path / "val_im.nii.gz").write_bytes(b"")
    monkeypatch.setattr(mod, "_load_volume", lambda p: np.zeros((8, 8, 3), dtype=np.float32))
    df = MedSegCOVIDCTValHarmonizer(base_dir=str(tmp_path), slice_output_dir=str(tmp_path / "s")).harmonize()
    assert len(df) == 3 and "mask_path" not in df and "any_lesion" not in df
    assert (tmp_path / "s" / "dataset1_val_hu-1000_400" / "images" / "dataset1_val_0000.png").is_file()  # V-H1-42


# Item 9: ChestX-ray14 split classes need the official split list.
def test_chestxray14_split_list_is_required(tmp_path):
    from radharmony.harmonizer import ChestXray14TestHarmonizer, ChestXray14TrainHarmonizer
    csv = tmp_path / "Data_Entry_2017.csv"
    pd.DataFrame({"Image Index": ["a.png"], "Finding Labels": ["No Finding"], "Follow-up #": [0],
                  "Patient ID": [1], "View Position": ["PA"]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="needs base_image_dir"):
        ChestXray14TrainHarmonizer(csv_path=str(csv)).harmonize()
    with pytest.raises(FileNotFoundError, match="test_list.txt"):
        ChestXray14TestHarmonizer(csv_path=str(csv), base_image_dir=str(tmp_path)).harmonize()


# Item 10: RSNA Bone Age test images (not on the NAS, so fake files).
def test_rsna_bone_age_test_images_have_no_labels(tmp_path):
    """D-25: the test class lists the PNGs and gives no bone age."""
    from radharmony.dataset import RSNABoneAgeTestDataset
    for i in ("4360", "4361"):
        (tmp_path / f"{i}.png").write_bytes(b"")
    with pytest.warns(UserWarning, match="no bone age"):
        ds = RSNABoneAgeTestDataset(base_image_dir=str(tmp_path), output_reg=True, cache_dir=None)
    df = ds.get_harmonized_df()
    assert df["image_path"].tolist() == ["4360.png", "4361.png"] and "age_months" not in df
    assert not ds.output_reg


# Fixes after review.
def test_keep_split_mixin_keeps_one_split(tmp_path):
    """The VQA-RAD split classes keep the questions of one split, with no data needed."""
    from radharmony.harmonizer import VQARadTestHarmonizer, VQARadTrainHarmonizer
    rows = [{"qid": i, "phrase_type": p, "image_name": f"img{i}.jpg", "question": "q", "answer": "a",
             "question_type": "PRES", "answer_type": "OPEN", "image_organ": "CHEST"}
            for i, p in enumerate(["freeform", "para", "test_freeform"])]
    path = tmp_path / "vqa.json"
    path.write_text(json.dumps(rows))
    assert VQARadTrainHarmonizer(json_path=str(path)).harmonize()["question_id"].tolist() == ["0", "1"]
    h = VQARadTestHarmonizer(json_path=str(path))
    df = h.harmonize()
    assert df["question_id"].tolist() == ["2"]
    pd.testing.assert_frame_equal(h.harmonized_df, df)


def _taix_ray_kwargs(tmp_path, splits):
    """Write a fake annotation.csv with one row per value of ``splits``. Return the harmonizer kwargs."""
    labels = ["HeartSize", "PulmonaryCongestion", "PleuralEffusion_Right", "PleuralEffusion_Left",
              "PulmonaryOpacities_Right", "PulmonaryOpacities_Left", "Atelectasis_Right", "Atelectasis_Left"]
    rows = [{"UID": f"u{i}", "Fold": 0, "Split": s, "PatientID": f"p{i}", "PhysicianID": 1,
             "StudyDate": 20200101, "Age": 25000, "Sex": "M", **{c: 0 for c in labels}}
            for i, s in enumerate(splits)]
    csv = tmp_path / "annotation.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return dict(csv_path=str(csv))


def _chexpert_plus_kwargs(tmp_path, splits):
    """Write a fake CheXpert Plus CSV with one row per value of ``splits``. Return the harmonizer kwargs."""
    rows = []
    for i, s in enumerate(splits):
        stem = f"{s}/patient{i:05d}/study1/view1_frontal"
        rows.append({"path_to_image": stem + ".jpg", "path_to_dcm": stem + ".dcm", "frontal_lateral": "Frontal",
                     "ap_pa": "AP", "split": s, "report": "x", "age": 50, "sex": "Female"})
    csv = tmp_path / "df_chexpert_plus_240401.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return dict(csv_path=str(csv))


def _ms_cxr_kwargs(tmp_path, splits):
    """Write a fake MS-CXR CSV with one box per value of ``splits``. Return the harmonizer kwargs."""
    rows = [{"path": f"files/p10/p1000000{i}/s5000000{i}/d{i}.jpg", "dicom_id": f"d{i}", "split": s,
             "category_name": "Pneumonia", "label_text": "x", "x": 1, "y": 2, "w": 3, "h": 4,
             "image_width": 100, "image_height": 100}
            for i, s in enumerate(splits)]
    csv = tmp_path / "MS_CXR_Local_Alignment_v1.1.0.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return dict(csv_path=str(csv))


@pytest.mark.parametrize("name,own,make", [
    ("TAIXRayValHarmonizer", "val", _taix_ray_kwargs),
    ("CheXpertPlusValidHarmonizer", "valid", _chexpert_plus_kwargs),
    ("MSCXRTestHarmonizer", "test", _ms_cxr_kwargs),
])
def test_keep_split_mixin_samples_and_copies_only_its_own_split(tmp_path, name, own, make):
    """D-30: the split class removes the other splits before the image copies.

    ``sample_n=2`` gives the 2 rows of the split.
    """
    import radharmony.harmonizer as H
    h = getattr(H, name)(**make(tmp_path, ["train"] * 6 + [own] * 2))
    pre = StubPreprocessor()
    df = h.harmonize(preprocess_output_dir=str(tmp_path / "out"), preprocessor=pre, sample_n=2)
    assert len(pre.tables) == 1 and set(pre.tables[0]["split"]) == {own}
    assert len(df) == 2 and set(df["split"]) == {own}
    pd.testing.assert_frame_equal(h.harmonized_df, df)


def test_split_tuple_equals_list():
    from radharmony.harmonizer import ROCOTrainHarmonizer
    assert ROCOTrainHarmonizer(base_dir="/x", splits=("train",)).splits == ["train"]


def test_ranzcr_label_check_keeps_the_cache(ranzcr_dir):
    """The label check is one deterministic step in a flat pipeline, so PersistentDataset caches."""
    import monai as mn
    ds = RANZCRClipTrainDataset(base_image_dir=str(ranzcr_dir), output_cls=True, cache_dir=None)
    steps = ds.get_datasets().transform.transforms
    assert not any(isinstance(s, (mn.transforms.Compose, mn.transforms.Randomizable)) for s in steps)


def test_infer_path_looks_only_in_exact_places(tmp_path):
    """D-35: base_dir/<name>, then parent/<name>. A file in another folder is not found (no walk)."""
    from radharmony.utils.infer import infer_path
    base = tmp_path / "ds" / "images"
    base.mkdir(parents=True)
    for rel in ("ds/images/a.csv", "ds/b.csv", "ds/meta/c.csv", "other/c.csv"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    assert infer_path(str(base), "a.csv") == str(base / "a.csv")
    assert infer_path(str(base), "b.csv") == str(tmp_path / "ds" / "b.csv")
    assert infer_path(str(base), "c.csv") == ""  # in a sibling and in an uncle folder only
    assert infer_path(str(base), "meta/c.csv") == str(tmp_path / "ds" / "meta" / "c.csv")


def test_infer_path_raises_for_a_user_path_that_does_not_exist(tmp_path):
    """D-35: a path that the user gives must exist. The error names it."""
    from radharmony.utils.infer import infer_path
    (tmp_path / "a.csv").write_text("x")
    with pytest.raises(FileNotFoundError, match="nope.csv"):
        infer_path(str(tmp_path), "a.csv", user_path=str(tmp_path / "nope.csv"))


@pytest.mark.parametrize("name, missing", [
    ("SIIMCOVID19TrainDataset", "train_study_level.csv"),
    ("RSNA2024LumbarSpineTrainDataset", "train.csv"),
    ("VinDrCXRTrainDataset", "image_labels_train.csv"),
    ("CheXpertTrainDataset", "train.csv"),
    ("RSNAPneumoniaDataset", "pneumonia-challenge-annotations-adjudicated-kaggle_2018.json"),
    ("ChestXray14BboxDataset", "Data_Entry_2017.csv"),
    ("MIMICCXRDataset", "cxr-record-list.csv.gz"),
    ("MIMICCXRTrainDataset", "cxr-record-list.csv.gz"),
    ("MSCXRDataset", "MS_CXR_Local_Alignment_v1.1.0.csv"),
    ("MSCXRTestDataset", "MS_CXR_Local_Alignment_v1.1.0.csv"),
    ("MSCXRTDataset", "MS_CXR_T_temporal_image_classification_v1.0.0.csv"),
    ("RANZCRClipDataset", "train.csv"),
    ("RANZCRClipTrainDataset", "train.csv"),
    ("RSNA2022CervicalSpineDataset", "train.csv"),
    ("RSNA2022CervicalSpineBboxDataset", "train.csv"),
    ("RSNA2022CervicalSpineTestDataset", "test.csv"),
    ("RSNAAbdominalTrauma2023Dataset", "train_2024.csv"),
    ("RSNAAbdominalTrauma2023TestDataset", "test_series_meta.csv"),
    ("ChexInstructVQADataset", "chexinstruct_vqa.json"),
    ("RSNAPneumoniaKaggleDataset", "stage_2_detailed_class_info.csv"),
    ("RSNA2024LumbarSpineTestDataset", "test_series_descriptions.csv"),
    ("RSNAPEDetectionTestDataset", "test.csv"),
    ("CheXpertPlusDataset", "df_chexpert_plus_240401.csv"),
])
def test_a_class_with_no_data_names_the_missing_file(tmp_path, name, missing):
    """D-35: with an empty image folder, the class raises FileNotFoundError with the file name."""
    import radharmony.dataset as D
    base = tmp_path / "ds" / "images"
    base.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match=missing.replace(".", r"\.")):
        getattr(D, name)(base_image_dir=str(base), transform=None, cache_dir=None)


@pytest.mark.parametrize("name, test_file", [
    ("SIIMCOVID19TestDataset", None),
    ("SIIMACRPTXTestDataset", None),
    ("RSNA2024LumbarSpineTestDataset", "test_series_descriptions.csv"),
    ("RANZCRClipTestDataset", None),
    ("RSNA2022CervicalSpineTestDataset", "test.csv"),
    ("RSNAAbdominalTrauma2023TestDataset", "test_series_meta.csv"),
    ("RSNAPneumoniaKaggleTestDataset", None),
    ("RSNAPEDetectionTestDataset", "test.csv"),
])
def test_a_test_split_class_does_not_need_the_train_files(tmp_path, name, test_file):
    """D-35: a test-split class reads only its test files, so a missing train CSV raises no error."""
    import radharmony.dataset as D
    base = tmp_path / "ds" / "test"
    base.mkdir(parents=True)
    if test_file:
        (base / test_file).write_text("x")
    getattr(D, name)(base_image_dir=str(base), transform=None, cache_dir=None)


@pytest.mark.parametrize("name, present, missing", [
    ("ChestXray14BboxDataset", "Data_Entry_2017.csv", "BBox_List_2017.csv"),
    ("RSNA2022CervicalSpineBboxDataset", "train.csv", "train_bounding_boxes.csv"),
    ("RSNAAbdominalTrauma2023Dataset", "train_2024.csv", "train_series_meta.csv"),
])
def test_a_class_names_its_second_missing_file(tmp_path, name, present, missing):
    """D-35: a class that always reads 2 source files names the second one when only it is missing."""
    import radharmony.dataset as D
    base = tmp_path / "ds" / "images"
    base.mkdir(parents=True)
    (base / present).write_text("x")
    with pytest.raises(FileNotFoundError, match=missing.replace(".", r"\.")):
        getattr(D, name)(base_image_dir=str(base), transform=None, cache_dir=None)


@pytest.mark.parametrize("name, files", [
    ("MontgomeryCXRDataset", r"\.png"),
    ("SIIMACRPTXTestDataset", r"\.dcm"),
    ("SIIMCOVID19TestDataset", r"<study>/<series>/\*\.dcm"),
])
def test_a_class_with_an_empty_image_folder_names_the_folder(tmp_path, name, files):
    """D-35: a class that lists its images raises FileNotFoundError for an empty folder, with the file type."""
    import radharmony.dataset as D
    base = tmp_path / "ds" / "test"
    base.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match=rf"No {files} file in .*{base.name}"):
        getattr(D, name)(base_image_dir=str(base), transform=None, cache_dir=None).get_datasets()


@pytest.mark.parametrize("name, arg, value", [
    ("RSNAAbdominalTrauma2023TestDataset", "series_meta_csv_path", "test_series_meta.csv"),
    ("RSNA2024LumbarSpineTestDataset", "series_description_csv_path", "test_series_descriptions.csv"),
    ("RSNA2022CervicalSpineTestDataset", "csv_path", "test.csv"),
    ("RSNAPEDetectionTestDataset", "csv_path", "test.csv"),
])
def test_a_test_class_names_the_argument_of_a_wrong_path(tmp_path, name, arg, value):
    """D-35: a path that the user gives must exist, and the error names the argument that the user passed."""
    import radharmony.dataset as D
    base = tmp_path / "ds" / "test"
    base.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match=f"{arg} does not exist"):
        getattr(D, name)(base_image_dir=str(base), transform=None, cache_dir=None, **{arg: str(tmp_path / value)})


@pytest.mark.parametrize("name, present, flag, missing", [
    ("MIMICCXRDataset", "cxr-record-list.csv.gz", "output_cls", "mimic-cxr-2.0.0-chexpert.csv"),
    ("MIMICCXRDataset", "cxr-record-list.csv.gz", "output_report", "cxr-study-list.csv.gz"),
    ("RSNAPneumoniaKaggleDataset", "stage_2_detailed_class_info.csv", "output_bbox", "stage_2_train_labels.csv"),
    ("CheXpertPlusDataset", "df_chexpert_plus_240401.csv", "output_cls", "impression_fixed.json"),
])
def test_an_asked_output_needs_its_file(tmp_path, name, present, flag, missing):
    """D-35: a file that a class reads for one output is required when the user asks for that output."""
    import radharmony.dataset as D
    base = tmp_path / "ds" / "images"
    base.mkdir(parents=True)
    (base / present).write_text("x")
    with pytest.raises(FileNotFoundError, match=missing.replace(".", r"\.")):
        getattr(D, name)(base_image_dir=str(base), transform=None, cache_dir=None, **{flag: True})
    getattr(D, name)(base_image_dir=str(base), transform=None, cache_dir=None)  # not asked, not needed


def test_mimic_cxr_metadata_file_is_not_the_record_list(tmp_path):
    """D-35: the DICOM class needs cxr-record-list (with a path column). The JPG metadata file is not it."""
    import radharmony.dataset as D
    base = tmp_path / "ds" / "files"
    base.mkdir(parents=True)
    (base.parent / "mimic-cxr-2.0.0-metadata.csv").write_text("x")
    with pytest.raises(FileNotFoundError, match="cxr-record-list"):
        D.MIMICCXRDataset(base_image_dir=str(base), transform=None, cache_dir=None)


def test_chexpert_plus_finds_its_files_2_folders_up_and_takes_impression_fixed(tmp_path):
    """D-35, D-48: CheXpert Plus looks in base, parent and grandparent (no walk), and the label file is impression_fixed.json."""
    import radharmony.dataset as D
    root = tmp_path / "chexpertplus"
    base = root / "PNG" / "train"
    base.mkdir(parents=True)
    (root / "df_chexpert_plus_240401.csv").write_text("x")
    (root / "impression_fixed.json").write_text("x")
    (root / "PNG" / "report_fixed.json").write_text("x")  # another label file, nearer to base
    ds = D.CheXpertPlusDataset(base_image_dir=str(base), output_cls=True, transform=None, cache_dir=None)
    assert ds._csv_path == str(root / "df_chexpert_plus_240401.csv")
    assert ds._label_json_path == str(root / "impression_fixed.json")
    deeper = base / "a"
    deeper.mkdir()
    with pytest.raises(FileNotFoundError, match="df_chexpert_plus_240401"):  # 3 folders up: no walk
        D.CheXpertPlusDataset(base_image_dir=str(deeper), transform=None, cache_dir=None)


def test_ranzcr_mask_without_the_annotation_file_names_it(tmp_path):
    """D-35: output_mask=True reads train_annotations.csv, so a missing one raises FileNotFoundError."""
    import radharmony.dataset as D
    base = tmp_path / "ds"
    base.mkdir()
    (base / "train.csv").write_text("x")
    with pytest.raises(FileNotFoundError, match=r"train_annotations\.csv"):
        D.RANZCRClipDataset(base_image_dir=str(base), output_mask=True, mask_output_dir=str(tmp_path / "m"),
                            transform=None, cache_dir=None)


def test_roco_with_no_split_folder_raises_and_a_missing_one_warns(tmp_path):
    """D-35: ROCO with no split folder raises FileNotFoundError. A missing split folder gives a warning."""
    from radharmony.harmonizer import ROCOHarmonizer
    with pytest.raises(FileNotFoundError, match="data/train/radiology"):
        ROCOHarmonizer(base_dir=str(tmp_path)).harmonize()
    folder = tmp_path / "data" / "train" / "radiology"
    (folder / "images").mkdir(parents=True)
    (folder / "captions.txt").write_text("ROCO_00001\tA chest X-ray.\n")
    with pytest.warns(UserWarning, match="data/validation/radiology"):
        ROCOHarmonizer(base_dir=str(tmp_path), radiology_only=True).harmonize()


def test_infer_path_required_names_the_file_and_the_places(tmp_path):
    """D-35: required=True raises FileNotFoundError with the file, the places and the argument."""
    from radharmony.utils.infer import infer_path
    base = tmp_path / "ds" / "images"
    base.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match=r"train\.csv.*images.*ds.*Pass csv_path="):
        infer_path(str(base), "train.csv", required=True, arg="csv_path")
    assert infer_path(str(base), "train.csv") == ""  # not required: no error


def test_chexlocalize_labeled_negative_gets_an_all_zero_mask(tmp_path):
    """Q-10 (D-33): an image with 0 for all 10 masked findings is annotated, so it gets an all-zero mask.

    An image with a positive finding and no mask in the file stays "not annotated" (no mask).
    """
    import json

    import numpy as np
    from PIL import Image
    from pycocotools import mask as mask_utils

    from radharmony.harmonizer import CheXlocalizeHarmonizer

    labels = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture",
              "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
              "Pneumothorax", "Support Devices"]
    rows = []
    for patient, positive in (("patient1", "Edema"), ("patient2", None), ("patient3", "Edema")):
        path = f"test/{patient}/study1/view1_frontal.jpg"
        (tmp_path / path).parent.mkdir(parents=True)
        Image.fromarray(np.zeros((8, 8), np.uint8)).save(tmp_path / path)
        rows.append({"Path": path, **{c: float(c == positive) for c in labels}})
    pd.DataFrame(rows).to_csv(tmp_path / "test_labels.csv", index=False)
    box = np.zeros((8, 8), np.uint8, order="F")
    box[2:5, 2:5] = 1
    rle = mask_utils.encode(box)
    rle["counts"] = rle["counts"].decode()
    (tmp_path / "masks.json").write_text(json.dumps({"patient1_study1_view1_frontal": {"Edema": rle}}))

    h = CheXlocalizeHarmonizer(csv_path=str(tmp_path / "test_labels.csv"), mask_json_path=str(tmp_path / "masks.json"),
                               base_image_dir=str(tmp_path / "test"))
    df = h.harmonize()
    h.preprocess_masks(str(tmp_path / "out"))
    got = {p.split("/")[0]: m for p, m in zip(h.df["image_path"], h.df["mask_path"], strict=True)}
    assert int(np.array(Image.open(got["patient1"])).sum()) == 9 * 255  # the drawn mask
    assert int(np.array(Image.open(got["patient2"])).sum()) == 0         # the labeled negative
    assert pd.isna(got["patient3"]) and len(df) == 3                     # a positive with no mask


def _flip():
    """A dataset transform with one random step."""
    import monai as mn
    return mn.transforms.Compose([mn.transforms.RandFlipd(keys=["img"], prob=0.5)])


def _random_split_warnings(record):
    return sorted(str(w.message).split(" split ")[0].rsplit(" ", 1)[-1]
                  for w in record if "random step" in str(w.message))


def test_taix_ray_predefined_val_and_test_warn_about_a_random_step(tmp_path):
    """D-49: with no val_transform or test_transform, the val and test splits get the random step. Each warns."""
    from radharmony.dataset.taix_ray.taix_ray import TAIXRay512Dataset

    kw = _taix_ray_kwargs(tmp_path, ["train", "val", "test"])
    ds = TAIXRay512Dataset(base_image_dir=str(tmp_path), cache_dir=None, transform=_flip(), **kw)
    with pytest.warns(UserWarning, match="random step") as record:
        ds.get_datasets_predefined()
    assert _random_split_warnings(record) == ["test", "val"]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ds.get_datasets_predefined(val_transform=lambda d: d, test_transform=lambda d: d)


def test_ranzcr_predefined_test_warns_about_a_random_step(ranzcr_dir):
    """D-49: with no test_transform, the test split gets the random step of the dataset transform, with a warning."""
    ds = RANZCRClipDataset(base_image_dir=str(ranzcr_dir), cache_dir=None, transform=_flip())
    with pytest.warns(UserWarning, match="test_transform") as record:
        ds.get_datasets_predefined()
    assert _random_split_warnings(record) == ["test"]


@pytest.mark.parametrize("cls_name,full", [(f"TAIXRay{r}{s}Dataset", f"TAIXRay{r}Dataset")
                                           for r in ("512", "") for s in ("Train", "Val", "Test")])
def test_taix_ray_split_class_names_the_class_for_the_predefined_split(tmp_path, cls_name, full):
    """A split class holds one split. Before, get_datasets_predefined() raised "Split 'train' is empty"."""
    import radharmony.dataset.taix_ray.taix_ray_split as split_mod

    kw = _taix_ray_kwargs(tmp_path, ["train", "val", "test"])
    ds = getattr(split_mod, cls_name)(base_image_dir=str(tmp_path), cache_dir=None, **kw)
    with pytest.raises(TypeError, match=rf"holds one split.*Use {full}\(\.\.\.\)\.get_datasets_predefined"):
        ds.get_datasets_predefined()


@pytest.mark.parametrize("cls", [RANZCRClipTrainDataset, RANZCRClipTestDataset])
def test_ranzcr_split_class_names_the_class_for_the_predefined_split(ranzcr_dir, cls):
    """Before, get_datasets_predefined() asked for include_test_split=True, which a split class does not take."""
    ds = cls(base_image_dir=str(ranzcr_dir), cache_dir=None)
    with pytest.raises(TypeError, match=r"Use RANZCRClipDataset\(\.\.\.\)\.get_datasets_predefined"):
        ds.get_datasets_predefined()


@pytest.mark.parametrize("cls_name, other", [("RSNABoneAgeTrainDataset", "val"), ("RSNABoneAgeValDataset", "train")])
def test_bone_age_split_class_rejects_another_split(cls_name, other, tmp_path):
    """A split class gives only its own split. Before, it dropped ``split=`` with no error."""
    import radharmony.dataset as D

    with pytest.raises(ValueError, match="gives only the"):
        getattr(D, cls_name)(base_image_dir=str(tmp_path), split=other, cache_dir=None)
