"""PadChest-GR harmonizer/dataset tests.

The synthetic tests do NOT skip: they pin the traps that fail silently -- the
``[x0, y0, x1, y1]`` box order, ``n_boxes == 0`` not meaning no finding,
multi-label sentences, and one-row-per-finding metadata. The ``needs_data`` tests skip unless ``PADCHEST_GR_DIR`` names
the extracted release (the directory holding ``grounded_reports_*.json``).
"""
import json
import os

import pandas as pd
import pytest

from radharmony.harmonizer import PadChestGRHarmonizer
from radharmony.harmonizer.padchest import PadChestHarmonizer
from radharmony.harmonizer.padchest_gr import (
    IMAGE_DIR, JSON_NAME, LABEL_COLUMNS, LABELS, MASTER_NAME, SPLITS)
from radharmony.harmonizer.padchest_gr.padchest_gr import _to_dim0_first
from radharmony.registry import list_datasets, resolve_dataset

ROOT = os.environ.get("PADCHEST_GR_DIR", "")
HAS_DATA = bool(ROOT) and os.path.isfile(os.path.join(ROOT, JSON_NAME))
needs_data = pytest.mark.skipif(
    not HAS_DATA, reason="set PADCHEST_GR_DIR to the extracted PadChest-GR release")


# --------------------------------------------------------------------------
# registration and vocabulary
# --------------------------------------------------------------------------
def test_registered():
    import radharmony.dataset  # noqa: F401  (importing it runs @register_dataset)

    assert "padchest_gr" in list_datasets()


def test_exported_from_package_namespaces():
    import radharmony.dataset as D
    import radharmony.harmonizer as H

    assert "PadChestGRDataset" in dir(D)
    assert "PadChestGRHarmonizer" in dir(H)


def test_label_vocabulary_is_a_padchest_subset():
    """The GR labels are PadChest's own, spelled identically -- that is what makes the
    two datasets joinable on ImageID."""
    assert len(LABELS) == 154 and len(set(LABELS)) == 154
    assert len(set(LABEL_COLUMNS)) == 154, "snake-casing must not merge two labels"
    assert set(LABELS) <= set(PadChestHarmonizer.LABEL_COLS)


def test_splits():
    assert SPLITS == ("train", "validation", "test")


def test_age_unit_is_declared():
    assert PadChestGRHarmonizer.AGE_SOURCE_COL == "patient_age"
    assert PadChestGRHarmonizer.AGE_SOURCE_UNIT == "years"


# --------------------------------------------------------------------------
# the box order
# --------------------------------------------------------------------------
def test_box_axes_are_swapped_to_dim0_first():
    """Released [x0, y0, x1, y1] -> RadHarmony [y0, y1, x0, x1]. A tall, narrow
    box makes a transposed conversion visible."""
    assert _to_dim0_first([0.1, 0.2, 0.3, 0.9]) == [0.2, 0.9, 0.1, 0.3]


# --------------------------------------------------------------------------
# a synthetic release
# --------------------------------------------------------------------------
def _finding(en, abnormal=True, labels=(), boxes=(), extra=(), locations=()):
    f = {"sentence_en": en, "sentence_es": "ES " + en, "abnormal": abnormal}
    if abnormal:
        f.update(labels=list(labels), boxes=[list(b) for b in boxes],
                 extra_boxes=[list(b) for b in extra], locations=list(locations),
                 progression=None)
    return f


def _write_release(tmp_path, reports, meta=None):
    """reports: list of (ImageID, [findings]); meta overrides per-image master fields."""
    rel = tmp_path / "release"
    (rel / IMAGE_DIR).mkdir(parents=True)
    js, rows = [], []
    for n, (iid, findings) in enumerate(reports):
        js.append({"StudyID": f"S{n}", "ImageID": iid, "PreviousStudyID": None,
                   "PreviousImageID": None, "findings": findings})
        base = {"ImageID": iid, "PatientID": f"P{n}", "split": ["train", "validation", "test"][n % 3],
                "PatientSex_DICOM": "F", "PatientAge": 60 + n, "StudyDate": 2014}
        base.update((meta or {}).get(iid, {}))
        for _ in findings or [None]:
            rows.append(dict(base))
    (rel / JSON_NAME).write_text(json.dumps(js))
    pd.DataFrame(rows).to_csv(rel / MASTER_NAME, index=False)
    return rel


def _standard(tmp_path):
    return _write_release(tmp_path, [
        # 0: two boxes on one sentence, a second sentence with no box, then a non-abnormal one
        ("a.png", [
            _finding("Apical thickening.", labels=["apical pleural thickening"],
                     boxes=[(0.1, 0.2, 0.3, 0.9), (0.5, 0.2, 0.7, 0.4)],
                     extra=[(0.05, 0.1, 0.35, 0.95)]),
            _finding("Blunting of the left angle.", labels=["costophrenic angle blunting"]),
            _finding("No other alterations.", abnormal=False),
        ]),
        # 1: no abnormal finding at all
        ("b.png", [_finding("Normal study.", abnormal=False)]),
        # 2: a sentence carrying two labels
        ("c.png", [
            _finding("Cardiomegaly with pulmonary vascular redistribution.",
                     labels=["cardiomegaly", "vascular redistribution"],
                     boxes=[(0.2, 0.3, 0.8, 0.8)]),
        ]),
        # 3: abnormal but no box at all
        ("d.png", [_finding("Pectus excavatum.", labels=["pectum excavatum"])]),
    ])


def _h(rel, **kw):
    return PadChestGRHarmonizer(base_image_dir=str(rel), **kw)


def test_one_row_per_image_not_per_finding(tmp_path):
    df = _h(_standard(tmp_path)).harmonize()
    assert list(df["image_path"]) == [f"{IMAGE_DIR}/{i}.png" for i in "abcd"]
    assert len(df) == 4


def test_boxes_are_dim0_first_and_json_encoded(tmp_path):
    df = _h(_standard(tmp_path)).harmonize()
    boxes = json.loads(df.loc[0, "bbox"])
    assert boxes == [[0.2, 0.9, 0.1, 0.3], [0.2, 0.4, 0.5, 0.7]]


def test_a_box_past_the_edge_is_clipped_and_a_box_with_no_area_is_dropped(tmp_path):
    """R-code-17 (D-07): clip to the image, and drop a box with no area with its label and finding."""
    rel = _write_release(tmp_path, [("a.png", [
        _finding("Past the edge.", labels=["cardiomegaly"], boxes=[(0.1, 0.2, 1.2, 0.9)],
                 extra=[(-0.1, 0.2, 0.4, 0.9)]),
        _finding("No width.", labels=["nodule"], boxes=[(0.3, 0.2, 0.3, 0.9)], extra=[(0.4, 0.2, 0.1, 0.9)]),
    ])])
    with pytest.warns(UserWarning, match="dropped 1 box"):
        df = _h(rel).harmonize()
    assert json.loads(df.loc[0, "bbox"]) == [[0.2, 0.9, 0.1, 1.0]]
    assert json.loads(df.loc[0, "bbox_labels"]) == ["cardiomegaly"]
    assert json.loads(df.loc[0, "bbox_findings"]) == ["Past the edge."]
    assert df.loc[0, "n_boxes"] == 1
    assert json.loads(df.loc[0, "extra_bbox"]) == [[0.2, 0.9, 0.0, 0.4]]  # the inverted extra box is dropped
    assert df.loc[0, "n_extra_boxes"] == 1


def test_sentence_box_counts_match_after_a_drop(tmp_path):
    """D-07: the box count of each sentence in ``findings`` adds up to ``n_boxes`` after a drop."""
    rel = _write_release(tmp_path, [("a.png", [
        _finding("No width.", labels=["nodule"], boxes=[(0.3, 0.2, 0.3, 0.9)]),
        _finding("Good.", labels=["cardiomegaly"], boxes=[(0.1, 0.2, 0.4, 0.9)]),
    ])])
    with pytest.warns(UserWarning, match="dropped 1 box"):
        df = _h(rel).harmonize()
    counts = [(f["sentence_en"], f["n_boxes"]) for f in json.loads(df.loc[0, "findings"])]
    assert counts == [("No width.", 0), ("Good.", 1)] and df.loc[0, "n_boxes"] == 1


def test_box_lists_are_parallel(tmp_path):
    df = _h(_standard(tmp_path)).harmonize()
    for _, r in df.iterrows():
        n = len(json.loads(r["bbox"]))
        assert len(json.loads(r["bbox_labels"])) == n
        assert len(json.loads(r["bbox_findings"])) == n
        assert r["n_boxes"] == n
    assert json.loads(df.loc[0, "bbox_findings"]) == ["Apical thickening."] * 2


def test_extra_boxes_are_not_merged_into_bbox(tmp_path):
    df = _h(_standard(tmp_path)).harmonize()
    assert df.loc[0, "n_boxes"] == 2 and df.loc[0, "n_extra_boxes"] == 1
    assert json.loads(df.loc[0, "extra_bbox"]) == [[0.1, 0.95, 0.05, 0.35]]


def test_an_image_with_no_abnormal_finding_is_kept_with_no_labels_or_boxes(tmp_path):
    df = _h(_standard(tmp_path)).harmonize()
    b = df.loc[1]
    assert b["bbox"] == "[]" and b["n_boxes"] == 0
    assert sum(b[c] for c in LABEL_COLUMNS) == 0


def test_no_box_does_not_mean_no_finding(tmp_path):
    """An image whose only abnormal finding cannot be localised has no box and a label."""
    df = _h(_standard(tmp_path)).harmonize()
    d = df.loc[3]
    assert d["n_boxes"] == 0
    assert d["pectum_excavatum"] == 1


def test_multi_label_sentence_joins_labels_and_sets_both_columns(tmp_path):
    df = _h(_standard(tmp_path)).harmonize()
    c = df.loc[2]
    assert json.loads(c["bbox_labels"]) == ["cardiomegaly + vascular_redistribution"]
    assert c["cardiomegaly"] == 1 and c["vascular_redistribution"] == 1


def test_label_columns_are_the_154_and_multi_hot(tmp_path):
    df = _h(_standard(tmp_path)).harmonize()
    assert all(c in df.columns for c in LABEL_COLUMNS)
    assert df.loc[0, LABEL_COLUMNS].sum() == 2


def test_a_non_abnormal_sentence_labels_nothing(tmp_path):
    """'No other alterations.' is in the report and in `findings`, but not in the labels."""
    df = _h(_standard(tmp_path)).harmonize()
    assert "No other alterations." in df.loc[0, "report"]
    fs = json.loads(df.loc[0, "findings"])
    assert [f["abnormal"] for f in fs] == [True, True, False]
    assert fs[2]["labels"] == []


def test_report_language(tmp_path):
    rel = _standard(tmp_path)
    assert _h(rel).harmonize().loc[1, "report"] == "Normal study."
    assert _h(rel, report_language="es").harmonize().loc[1, "report"] == "ES Normal study."
    with pytest.raises(ValueError, match="report_language"):
        _h(rel, report_language="fr")


def test_split_filter(tmp_path):
    rel = _standard(tmp_path)
    assert _h(rel, split="train").harmonize()["split"].unique().tolist() == ["train"]
    assert len(_h(rel, split="test").harmonize()) == 1
    with pytest.raises(ValueError, match="split"):
        _h(rel, split="val")


def test_age_is_normalised_to_years(tmp_path):
    df = _h(_standard(tmp_path)).harmonize()
    assert df["age_years"].tolist() == [60.0, 61.0, 62.0, 63.0]


def test_sex_comes_from_patient_sex_dicom(tmp_path):
    """D-26: one ``sex`` column, "M", "F" or NaN. The DICOM value O gives NaN."""
    rel = _write_release(tmp_path, [(f"{i}.png", [_finding("Normal.", abnormal=False)]) for i in "abc"],
                         meta={"b.png": {"PatientSex_DICOM": "M"}, "c.png": {"PatientSex_DICOM": "O"}})
    df = _h(rel).harmonize()
    assert df["sex"].iloc[:2].tolist() == ["F", "M"]
    assert pd.isna(df["sex"].iloc[2])
    assert list(df.columns).count("sex") == 1


# --------------------------------------------------------------------------
# refusals: things that would otherwise pass quietly
# --------------------------------------------------------------------------
def test_unknown_label_raises(tmp_path):
    rel = _write_release(tmp_path, [
        ("a.png", [_finding("x.", labels=["not a padchest label"])])])
    with pytest.raises(ValueError, match="not in PadChestGR LABELS"):
        _h(rel).harmonize()


def test_abnormal_finding_without_a_label_raises(tmp_path):
    rel = _write_release(tmp_path, [("a.png", [_finding("x.", labels=[])])])
    with pytest.raises(ValueError, match="without a label"):
        _h(rel).harmonize()


def test_duplicate_image_id_raises(tmp_path):
    f = _finding("x.", abnormal=False)
    rel = _write_release(tmp_path, [("a.png", [f]), ("a.png", [f])])
    with pytest.raises(ValueError, match="duplicate ImageID"):
        _h(rel).harmonize()


def test_metadata_disagreeing_across_an_images_rows_raises(tmp_path):
    """The master table is one row per finding; per-image fields must agree."""
    rel = _standard(tmp_path)
    m = pd.read_csv(rel / MASTER_NAME)
    first = m.index[m.ImageID == "a.png"][0]
    m.loc[first, "PatientAge"] = 99
    m.to_csv(rel / MASTER_NAME, index=False)
    with pytest.raises(ValueError, match="differs across"):
        _h(rel).harmonize()


def test_image_missing_from_master_table_raises(tmp_path):
    rel = _standard(tmp_path)
    m = pd.read_csv(rel / MASTER_NAME)
    m[m.ImageID != "d.png"].to_csv(rel / MASTER_NAME, index=False)
    with pytest.raises(ValueError, match="absent from the master table"):
        _h(rel).harmonize()


def test_needs_a_root_or_both_paths():
    with pytest.raises(ValueError, match="needs base_image_dir"):
        PadChestGRHarmonizer()


# --------------------------------------------------------------------------
# view_position from PadChest
# --------------------------------------------------------------------------
def test_view_position_from_padchest_csv(tmp_path):
    rel = _standard(tmp_path)
    pc = tmp_path / "padchest.csv"
    pd.DataFrame({"ImageID": ["a.png", "b.png", "c.png", "d.png"],
                  "Projection": ["PA", "AP", "PA", "AP_horizontal"]}).to_csv(pc, index=False)
    df = _h(rel, padchest_csv_path=str(pc)).harmonize()
    assert df["view_position"].tolist() == ["PA", "AP", "PA", "AP_HORIZONTAL"]  # upper case (Q-13)


def test_view_position_column_is_absent_without_padchest(tmp_path):
    """An optional column: not invented, not filled with NaN."""
    df = _h(_standard(tmp_path)).harmonize()
    assert "view_position" not in df.columns


def test_padchest_csv_missing_an_image_raises(tmp_path):
    rel = _standard(tmp_path)
    pc = tmp_path / "padchest.csv"
    pd.DataFrame({"ImageID": ["a.png"], "Projection": ["PA"]}).to_csv(pc, index=False)
    with pytest.raises(ValueError, match="not the matching PadChest label file"):
        _h(rel, padchest_csv_path=str(pc)).harmonize()


# --------------------------------------------------------------------------
# save / load and the dataset class
# --------------------------------------------------------------------------
def test_save_load_keeps_the_arguments(tmp_path):
    rel = _standard(tmp_path)
    h = _h(rel, split="train", report_language="es")
    df = h.harmonize()
    h.set_harmonized_df(df)
    h.save(str(tmp_path / "h.pkl"))
    g = PadChestGRHarmonizer.load_from_saved(str(tmp_path / "h.pkl"))
    assert (g.split, g.report_language) == ("train", "es")
    assert g.json_path == h.json_path


def test_dataset_declares_its_capabilities():
    import radharmony.dataset  # noqa: F401

    cls = resolve_dataset("padchest_gr")
    assert cls.SUPPORTED_OUTPUTS >= {"cls", "bbox", "report"}
    assert cls.SUPPORTS_BBOX_FINDINGS is True
    assert list(cls.LABEL_COLS) == LABEL_COLUMNS


def test_dataset_requires_a_root():
    import radharmony.dataset  # noqa: F401

    with pytest.raises(ValueError, match="base_image_dir"):
        resolve_dataset("padchest_gr")(output_cls=True)


# --------------------------------------------------------------------------
# the real release
# --------------------------------------------------------------------------
@needs_data
def test_counts_match_the_release():
    df = PadChestGRHarmonizer(base_image_dir=ROOT).harmonize()
    nb = df["bbox"].apply(lambda s: len(json.loads(s)))
    assert len(df) == 4555
    assert df["patient_id"].nunique() == 4459
    assert df["split"].value_counts().to_dict() == {
        "train": 3185, "test": 915, "validation": 455}
    assert nb.sum() == 7733 and (nb > 0).sum() == 3008
    assert (df[LABEL_COLUMNS].sum(axis=1) == 0).sum() == 1456
    assert df["n_extra_boxes"].sum() == 6265


@needs_data
def test_real_boxes_are_in_the_unit_square_and_dim0_first():
    df = PadChestGRHarmonizer(base_image_dir=ROOT).harmonize()
    for s in df["bbox"]:
        for y0, y1, x0, x1 in json.loads(s):
            assert 0.0 <= y0 < y1 <= 1.0 and 0.0 <= x0 < x1 <= 1.0


@needs_data
def test_real_patients_do_not_cross_splits():
    df = PadChestGRHarmonizer(base_image_dir=ROOT).harmonize()
    assert (df.groupby("patient_id")["split"].nunique() == 1).all()


@needs_data
def test_real_images_with_no_abnormal_finding_have_no_boxes():
    df = PadChestGRHarmonizer(base_image_dir=ROOT).harmonize()
    none = df[df[LABEL_COLUMNS].sum(axis=1) == 0]
    assert len(none) == 1456
    assert (none["n_boxes"] == 0).all()
    assert ((df["n_boxes"] == 0) & (df[LABEL_COLUMNS].sum(axis=1) > 0)).sum() == 91


@needs_data
def test_real_pngs_exist_for_a_sample():
    df = PadChestGRHarmonizer(base_image_dir=ROOT).harmonize()
    for p in df["image_path"].sample(25, random_state=0):
        assert os.path.isfile(os.path.join(ROOT, p)), p


@needs_data
def test_real_box_side_agrees_with_the_sentence():
    """The check that fixed the axis order: boxes whose sentence names one side
    sit on that side of the midline in the radiological convention."""
    df = PadChestGRHarmonizer(base_image_dir=ROOT).harmonize()
    agree = total = 0
    for _, r in df.iterrows():
        for (y0, y1, x0, x1), s in zip(json.loads(r["bbox"]), json.loads(r["bbox_findings"])):
            t = s.lower()
            left, right = "left" in t, "right" in t
            if left == right:
                continue
            total += 1
            centre_is_image_right = (x0 + x1) / 2 > 0.5
            agree += centre_is_image_right == left
    assert total > 2000 and agree / total > 0.90


@needs_data
def test_dataset_yields_image_boxes_and_report():
    import torch

    import radharmony.dataset  # noqa: F401

    B = resolve_dataset("padchest_gr")(
        base_image_dir=ROOT, cache_dir="/tmp/padchest_gr_cache_test",
        output_cls=True, output_bbox=True, output_report=True, split="test",
        dtype=torch.float32)
    df = B.get_harmonized_df()
    ds = B.get_datasets()
    pos = int(df.index[df["n_boxes"] > 0][0])
    s = ds[pos]
    assert set(s) >= {"img", "cls", "bbox", "bbox_labels", "bbox_findings", "report"}
    assert s["img"].shape[0] == 1 and s["cls"].shape == (154,)
    bare = int(df.index[df["n_boxes"] == 0][0])
    # D-28: the sample gives the boxes as an (N, 4) tensor and the labels as lists.
    n = int(df.loc[pos, "n_boxes"])
    assert tuple(s["bbox"].shape) == (n, 4) and len(s["bbox_labels"]) == n
    assert tuple(ds[bare]["bbox"].shape) == (0, 4)
    assert ds[bare]["bbox_labels"] == [] and ds[bare]["bbox_findings"] == []


def test_dataset_expands_a_home_folder(monkeypatch, tmp_path):
    """A ``~/...`` base_image_dir becomes the full path, as in the harmonizer. Before, image paths kept ``~``."""
    import radharmony.dataset  # noqa: F401  (fills the registry)

    monkeypatch.setenv("HOME", str(tmp_path))
    df = pd.DataFrame({"patient_id": ["1"], "study_id": ["1"], "image_path": ["a.png"]})
    ds = resolve_dataset("padchest_gr_train")(base_image_dir="~/gr", harmonized_df=df, cache_dir=None)
    assert ds.base_image_dir == str(tmp_path / "gr")
