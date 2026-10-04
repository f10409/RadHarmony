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
    """Own harmonize() that never calls the base one (like ChestX-ray14, BRAX)."""

    def harmonize(self, **kwargs):
        self._harmonized_df_override = None
        self.df = pd.read_csv(self.csv_path)
        self._build_patient_id()
        self._build_study_id()
        self._build_image_path()
        return self._select_harmonized_columns()


class _CustomChild(_CustomHarmonizer):
    """Calls super().harmonize() into a custom one (like ChestX-ray14 train)."""

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
