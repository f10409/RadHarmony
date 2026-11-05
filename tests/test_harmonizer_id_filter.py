"""BaseHarmonizer: harmonize(study_ids=/patient_ids=) filter and report-read scoping."""

import builtins

import pandas as pd
import pytest

from radharmony.harmonizer.base import BaseHarmonizer


class _ReportHarmonizer(BaseHarmonizer):
    """Tiny harmonizer using the base harmonize() flow with labels + reports."""

    LABEL_COLS = ["Effusion"]
    LABEL_JOIN_COLS = ["patient_id", "study_id"]
    REPORT_JOIN_COLS = ["patient_id", "study_id"]
    REPORT_PATH_COL = "path"

    def _build_patient_id(self):
        self.df["patient_id"] = self.df["subject_id"].astype(str)

    def _build_study_id(self):
        self.df["study_id"] = self.df["study"].astype(str)

    def _build_image_path(self):
        self.df["image_path"] = self.df["img"]

    def _preprocess_label_df(self, label_df):
        label_df["patient_id"] = label_df["subject_id"].astype(str)
        label_df["study_id"] = label_df["study"].astype(str)
        return label_df

    def _preprocess_report_df(self, report_df):
        report_df["patient_id"] = report_df["subject_id"].astype(str)
        report_df["study_id"] = report_df["study"].astype(str)
        return report_df


@pytest.fixture
def files(tmp_path):
    # 3 patients, 4 studies, 6 images; one extra study in the report list only.
    images = pd.DataFrame({
        "subject_id": [1, 1, 1, 2, 2, 3],
        "study": [10, 10, 11, 20, 20, 30],
        "img": [f"img{i}.jpg" for i in range(6)],
    })
    labels = pd.DataFrame({"subject_id": [1, 1, 2, 3], "study": [10, 11, 20, 30],
                           "Effusion": [1.0, 0.0, 1.0, 0.0]})
    reports = pd.DataFrame({"subject_id": [1, 1, 2, 3, 4], "study": [10, 11, 20, 30, 40]})
    reports["path"] = [f"s{s}.txt" for s in reports["study"]]
    for s in reports["study"]:
        (tmp_path / f"s{s}.txt").write_text(f"report {s}")
    for name, df in (("images", images), ("labels", labels), ("reports", reports)):
        df.to_csv(tmp_path / f"{name}.csv", index=False)
    return tmp_path


@pytest.fixture
def opened(monkeypatch):
    """Record every .txt file opened (report reads)."""
    paths, real_open = [], builtins.open

    def _open(file, *a, **k):
        if str(file).endswith(".txt"):
            paths.append(str(file).rsplit("/", 1)[-1])
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", _open)
    return paths


def _make(d, csv="images.csv"):
    return _ReportHarmonizer(csv_path=str(d / csv), label_csv_path=str(d / "labels.csv"),
                             report_csv_path=str(d / "reports.csv"), report_base_dir=str(d))


def _rows(df):
    return df.sort_values("image_path").reset_index(drop=True)


def test_no_filter_skips_reports_without_images(files, opened):
    df = _make(files).harmonize()
    assert len(df) == 6
    # study 40 is in the report list but has no image: its file is never opened
    assert sorted(opened) == ["s10.txt", "s11.txt", "s20.txt", "s30.txt"]


def test_study_ids_filter_matches_filter_after(files, opened):
    full = _make(files).harmonize()
    opened.clear()
    df = _make(files).harmonize(study_ids=["10", 30])  # str and int both work
    assert sorted(opened) == ["s10.txt", "s30.txt"]
    expected = full[full["study_id"].isin(["10", "30"])]
    pd.testing.assert_frame_equal(_rows(df), _rows(expected))


def test_patient_ids_filter(files, opened):
    df = _make(files).harmonize(patient_ids=[2])
    assert sorted(df["image_path"]) == ["img3.jpg", "img4.jpg"]
    assert df["report"].tolist() == ["report 20", "report 20"]
    assert df["effusion"].tolist() == [1.0, 1.0]
    assert opened == ["s20.txt"]


def test_small_image_csv_scopes_report_reads(files, opened):
    # No filter argument: a cohort image CSV alone limits the report reads.
    pd.read_csv(files / "images.csv").iloc[:1].to_csv(files / "cohort.csv", index=False)
    df = _make(files, "cohort.csv").harmonize()
    assert df["report"].tolist() == ["report 10"]
    assert opened == ["s10.txt"]


# --- Harmonizers with their own harmonize() --------------------------------

class _CustomHarmonizer(_ReportHarmonizer):
    """Own harmonize() that never calls the base one (as ChestX-ray14 and BRAX had before D-30)."""

    def harmonize(self, **kwargs):
        self._harmonized_df_override = None
        self.df = pd.read_csv(self.csv_path)
        self._build_patient_id()
        self._build_study_id()
        self._build_image_path()
        return self._select_harmonized_columns()


class _CustomChild(_CustomHarmonizer):
    """Calls super().harmonize() into a custom one (as ChestX-ray14 train had before D-30)."""

    def harmonize(self, *args, **kwargs):
        return super().harmonize(*args, **kwargs)


class _SuperCaller(_ReportHarmonizer):
    """Own harmonize() that reaches the base one (like MIMIC-CXR)."""

    def harmonize(self, *args, drop_uncertain=True, **kwargs):
        return super().harmonize(*args, **kwargs)


def _make_cls(cls, d):
    return cls(csv_path=str(d / "images.csv"), label_csv_path=str(d / "labels.csv"),
               report_csv_path=str(d / "reports.csv"), report_base_dir=str(d))


@pytest.mark.parametrize("cls", [_CustomHarmonizer, _CustomChild])
def test_custom_harmonize_filters_after_with_warning(files, cls):
    h = _make_cls(cls, files)
    with pytest.warns(UserWarning, match="applied after harmonizing") as rec:
        df = h.harmonize(study_ids=["20"])
    assert len(rec) == 1  # one warning, even through a harmonize() chain
    assert sorted(df["image_path"]) == ["img3.jpg", "img4.jpg"]
    assert sorted(h.harmonized_df["image_path"]) == ["img3.jpg", "img4.jpg"]
    assert h._id_filter is None


def test_custom_harmonize_without_filter_unchanged(files, recwarn):
    df = _make_cls(_CustomHarmonizer, files).harmonize()
    assert len(df) == 6
    assert not [w for w in recwarn if "harmonizing" in str(w.message)]


def test_super_caller_filters_early_without_warning(files, opened, recwarn):
    df = _make_cls(_SuperCaller, files).harmonize(study_ids=["20"], drop_uncertain=False)
    assert sorted(df["image_path"]) == ["img3.jpg", "img4.jpg"]
    assert opened == ["s20.txt"]  # filtered before the report reads
    assert not [w for w in recwarn if "harmonizing" in str(w.message)]


# ---------------------------------------------------------------------------
# D-02: MIMIC-Ext-CXR-QBA filters by study and patient before the slow qa.zip step
# ---------------------------------------------------------------------------


def _qba_folder(tmp_path):
    """3 patients, 4 studies, 1 question each. qa.zip has one qa.json for each study.

    patient_metadata.parquet puts patient 1 in train, patient 2 in validate and patient 3 in test.
    """
    import json
    import zipfile

    keys = [("p10000001", "s50000001"), ("p10000001", "s50000002"), ("p10000002", "s50000003"),
            ("p10000003", "s50000004")]
    meta = tmp_path / "metadata"
    meta.mkdir()
    pd.DataFrame({"patient_id": [p for p, _ in keys], "study_id": [s for _, s in keys],
                  "question_id": ["q1"] * 4, "question.question_type": ["abnormality"] * 4,
                  "question.quality.rating": ["A"] * 4}).set_index(["patient_id", "study_id", "question_id"]) \
        .to_parquet(meta / "question_metadata.parquet")
    pd.DataFrame({"patient_id": [p for p, _ in keys], "study_id": [s for _, s in keys], "question_id": ["q1"] * 4,
                  "image_id": [f"img{i}" for i in range(4)]}).set_index(["patient_id", "study_id"]) \
        .to_parquet(meta / "question_image_metadata.parquet")
    pd.DataFrame({"patient_id": ["p10000001", "p10000002", "p10000003"], "split": ["train", "validate", "test"]}) \
        .set_index("patient_id").to_parquet(meta / "patient_metadata.parquet")
    with zipfile.ZipFile(tmp_path / "qa.zip", "w") as z:
        for p, s in keys:
            doc = {"questions": [{"question_id": "q1", "question": f"Any finding in {s}?",
                                  "answers": [{"text": f"No finding in {s}."}]}]}
            z.writestr(f"qa/{p[:3]}/{p}/{s}.qa.json", json.dumps(doc))
    return meta, tmp_path / "qa.zip"


@pytest.mark.parametrize("kwargs, column, values", [
    ({"study_ids": ["s50000002", "s50000004"]}, "study_id", {"50000002", "50000004"}),
    ({"study_ids": ["50000002", "50000004"]}, "study_id", {"50000002", "50000004"}),  # Q-12: both forms
    ({"patient_ids": ["p10000001"]}, "patient_id", {"10000001"}),
    ({"patient_ids": ["10000001"]}, "patient_id", {"10000001"}),
    ({"study_ids": ["s99999999"]}, "study_id", set()),  # no study matches: an empty table, no error
])
def test_mimic_ext_qba_id_filter_matches_filter_after(tmp_path, monkeypatch, kwargs, column, values):
    """D-02: study_ids= and patient_ids= give the rows of a filter after harmonize(), and qa.zip opens only those studies."""
    import zipfile

    from radharmony.harmonizer import MIMICExtCXRQBAHarmonizer

    meta, qa = _qba_folder(tmp_path)
    full = MIMICExtCXRQBAHarmonizer(metadata_dir=str(meta), qa_zip_path=str(qa)).harmonize()
    opened = []
    real_read = zipfile.ZipFile.read
    monkeypatch.setattr(zipfile.ZipFile, "read", lambda self, name, *a: opened.append(name) or real_read(self, name, *a))
    part = MIMICExtCXRQBAHarmonizer(metadata_dir=str(meta), qa_zip_path=str(qa)).harmonize(**kwargs)
    expected = full[full[column].isin(values)].reset_index(drop=True)
    if expected.empty:  # an empty table: same columns (the dtypes of an empty column can differ)
        assert part.empty and list(part.columns) == list(expected.columns)
    else:
        pd.testing.assert_frame_equal(part.reset_index(drop=True), expected)
    assert len(opened) == expected["study_id"].nunique()


def test_mimic_ext_qba_ids_are_digits_and_question_id_names_one_row(tmp_path):
    """Q-12 (D-01): MIMIC ids are digits. V-H1-30: the source question_id repeats in each study.

    So the row id joins the study id and the source question_id.
    """
    from radharmony.harmonizer import MIMICExtCXRQBAHarmonizer

    meta, qa = _qba_folder(tmp_path)
    df = MIMICExtCXRQBAHarmonizer(metadata_dir=str(meta), qa_zip_path=str(qa)).harmonize()
    assert df["patient_id"].tolist() == ["10000001", "10000001", "10000002", "10000003"]
    assert df["study_id"].tolist() == ["50000001", "50000002", "50000003", "50000004"]
    assert df["question_id"].tolist() == ["50000001_q1", "50000002_q1", "50000003_q1", "50000004_q1"]
    assert df["question"].tolist() == [f"Any finding in s5000000{i}?" for i in range(1, 5)]  # the text still joins
    assert df["image_path"].iloc[0].startswith("p10/p10000001/s50000001/img0")  # the path keeps the letters


def test_gemex_vqa_ids_are_digits():
    """Q-12 (D-01): GEMeX VQA takes the MIMIC ids from the image path, as digits with no p or s."""
    from radharmony.harmonizer.gemex_vqa.gemex_vqa import _parse_ids

    assert _parse_ids("p10/p10046166/s50051329/abc.jpg") == ("10046166", "50051329")


# ---------------------------------------------------------------------------
# D-09: the official MIMIC-Ext-CXR-QBA splits come from patient_metadata.parquet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("split, studies", [
    ("train", ["50000001", "50000002"]), ("validate", ["50000003"]), ("test", ["50000004"]),
])
def test_mimic_ext_qba_split_classes_keep_one_split(tmp_path, split, studies):
    """V-H1-29 (D-09): a split class gives only the patients of its split in patient_metadata.parquet."""
    import radharmony.dataset  # noqa: F401  (fills the registry)
    import radharmony.harmonizer as H
    from radharmony.registry import resolve_dataset

    meta, qa = _qba_folder(tmp_path)
    cls = getattr(H, f"MIMICExtCXRQBA{split.capitalize()}Harmonizer")
    df = cls(metadata_dir=str(meta), qa_zip_path=str(qa)).harmonize()
    assert df["study_id"].tolist() == studies
    assert set(df["split"]) == {split}
    # max_studies counts the studies of the split, so a small test run is not empty.
    assert cls(metadata_dir=str(meta), qa_zip_path=str(qa), max_studies=1).harmonize()["study_id"].tolist() == studies[:1]
    ds = resolve_dataset(f"mimic_ext_cxr_qba_{split}")(base_image_dir=str(tmp_path), metadata_dir=str(meta),
                                                        qa_zip_path=str(qa), cache_dir=None)
    assert ds.get_harmonized_df()["study_id"].tolist() == studies


def test_mimic_ext_qba_mixed_key_shows_the_split(tmp_path):
    """V-H1-29 (D-09): the old key keeps all rows, and its split column comes from patient_metadata.parquet."""
    from radharmony.harmonizer import MIMICExtCXRQBAHarmonizer

    meta, qa = _qba_folder(tmp_path)
    df = MIMICExtCXRQBAHarmonizer(metadata_dir=str(meta), qa_zip_path=str(qa)).harmonize()
    assert df["split"].tolist() == ["train", "train", "validate", "test"]


def test_mimic_ext_qba_split_class_needs_the_patient_file(tmp_path):
    """V-H1-29 (D-35): without patient_metadata.parquet, a split class names the missing file."""
    from radharmony.harmonizer import MIMICExtCXRQBATestHarmonizer

    meta, qa = _qba_folder(tmp_path)
    (meta / "patient_metadata.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="patient_metadata.parquet"):
        MIMICExtCXRQBATestHarmonizer(metadata_dir=str(meta), qa_zip_path=str(qa)).harmonize()
