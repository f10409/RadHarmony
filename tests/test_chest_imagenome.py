"""Chest ImaGenome harmonizer tests (gold and silver). No real data.

The tests build a tiny fake release in ``tmp_path``. The expected values come
from the fake source files and from D-06, D-07 and D-26, not from the code.

Run with::

    .venv/bin/python -m pytest tests/test_chest_imagenome.py -v
"""
from __future__ import annotations

import json
import math
import warnings
import zipfile

import pandas as pd
import pytest
from test_harmonizer_boxes import _write_dicom

from radharmony.harmonizer import (
    ChestImaGenomeGoldHarmonizer,
    ChestImaGenomeSilverHarmonizer,
    ChestImaGenomeSilverTrainHarmonizer,
    ChestImaGenomeSilverValidHarmonizer,
)

_ATTR_COLS = ["image_id", "bbox", "context", "categoryID", "label_name"]


def _record_list(ann, ids):
    """Write the record list for the images ``ids``, each a 100 x 200 (H x W) DICOM header."""
    (ann / "utils").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"dicom_id": ids, "path": [f"files/p10/p1/s{i}/{d}.dcm" for i, d in enumerate(ids)],
                       "subject_id": 1, "study_id": [10 + i for i in range(len(ids))], "ViewPosition": "PA"})
    df.to_csv(ann / "utils" / "cxr-record-list_view.csv", index=False)
    return df


def _gold(tmp_path, boxes, attrs, ids=("d1",)):
    """Write a fake gold release. ``boxes``: (dicom_id, region, x1, y1, x2, y2). Return the harmonizer."""
    ann, img = tmp_path / "ann", tmp_path / "files"
    rl = _record_list(ann, list(ids))
    for p in rl["path"]:
        _write_dicom(img / p.removeprefix("files/"), rows=100, cols=200)
    (ann / "gold_dataset").mkdir()
    pd.DataFrame([{"image_id": f"{d}.dcm", "bbox_name": r, "original_x1": x1, "original_y1": y1,
                   "original_x2": x2, "original_y2": y2} for d, r, x1, y1, x2, y2 in boxes]
                 ).to_csv(ann / "gold_dataset" / "gold_bbox_coordinate_annotations_1000images.csv", index=False)
    pd.DataFrame(attrs, columns=_ATTR_COLS).to_csv(
        ann / "gold_dataset" / "gold_object_attribute_with_coordinates.txt", sep="\t", index=False)
    return ChestImaGenomeGoldHarmonizer(annotation_dir=str(ann), base_image_dir=str(img))


# ---------------------------------------------------------------------------
# Gold
# ---------------------------------------------------------------------------

def test_gold_drops_an_inverted_box_with_the_d07_warning(tmp_path):
    """Q-9 (D-07): a source box with x1 > x2 is dropped with its label. Its corners are not swapped."""
    h = _gold(tmp_path, [("d1", "right lung", 20, 10, 100, 60),   # [0.1, 0.6, 0.1, 0.5]
                         ("d1", "right hilar structures", 150, 10, 30, 60)], [])
    with pytest.warns(UserWarning, match="dropped 1 boxes"):
        df = h.harmonize()
    assert df.loc[0, "bbox"] == [[0.1, 0.6, 0.1, 0.5]]
    assert df.loc[0, "bbox_labels"] == ["right_lung"]  # Q-17: snake case


def test_gold_warns_about_findings_on_a_region_with_no_box(tmp_path):
    """Q-8 (D-07): the gold box file has no "mediastinum" box. Its findings are not in the table, with a warning."""
    attrs = [("d1.dcm", "right lung", "yes", "disease", "pneumonia"),
             ("d1.dcm", "mediastinum", "yes", "anatomicalfinding", "widened mediastinum"),
             ("d1.dcm", "mediastinum", "yes", "disease", "mass")]
    h = _gold(tmp_path, [("d1", "right lung", 20, 10, 100, 60)], attrs)
    with pytest.warns(UserWarning, match=r"2 positive findings in 1 images .* region with no box \(mediastinum\)"):
        df = h.harmonize()
    assert df.loc[0, "bbox_labels"] == ["right_lung"]  # Q-17: snake case
    assert df.loc[0, "bbox_findings"] == ["pneumonia"]


def test_gold_no_box_warning_skips_an_image_with_no_box_row(tmp_path):
    """Q-8: d9 has a finding but no row in the box file. It does not count as a region with no box."""
    h = _gold(tmp_path, [("d1", "right lung", 20, 10, 100, 60)],
              [("d9.dcm", "right lung", "yes", "disease", "mass")])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        h.harmonize()
    assert not [x for x in w if "region with no box" in str(x.message)]


def test_gold_tells_an_image_with_no_finding_annotation_from_no_finding(tmp_path):
    """V-H1-14: the attribute file covers only some images. ``findings_annotated`` is True only for them.

    d1 is in the attribute file with no positive finding. d2 is not in the file at all.
    """
    attrs = [("d1.dcm", "right lung", "no", "disease", "pneumonia")]
    h = _gold(tmp_path, [("d1", "right lung", 20, 10, 100, 60), ("d2", "right lung", 20, 10, 100, 60)],
              attrs, ids=("d1", "d2"))
    df = h.harmonize().set_index("image_path")
    assert df["bbox_findings"].tolist() == [[""], [""]]
    assert df["findings_annotated"].to_dict() == {"p10/p1/s0/d1.dcm": True, "p10/p1/s1/d2.dcm": False}


# ---------------------------------------------------------------------------
# Silver
# ---------------------------------------------------------------------------

def _silver(tmp_path, graphs, split="valid"):
    """Write a fake silver release. ``graphs``: dicom_id -> scene graph (None: no file in the zip)."""
    ann, img = tmp_path / "ann", tmp_path / "files"
    rl = _record_list(ann, list(graphs))
    for p in rl["path"]:
        _write_dicom(img / p.removeprefix("files/"), rows=100, cols=200)
    splits = ann / "silver_dataset" / "splits"
    splits.mkdir(parents=True)
    rl.to_csv(splits / f"{split}.csv", index=False)
    pd.DataFrame({"dicom_id": []}).to_csv(splits / "images_to_avoid.csv", index=False)
    with zipfile.ZipFile(ann / "silver_dataset" / "scene_graph.zip", "w") as zf:
        for did, sg in graphs.items():
            if sg is not None:
                zf.writestr(f"scene_graph/{did}_SceneGraph.json", json.dumps(sg))
    return dict(annotation_dir=str(ann), base_image_dir=str(img))


_BOX = {"bbox_name": "right lung", "original_x1": 20, "original_y1": 10, "original_x2": 100, "original_y2": 60}


def _graph(gender, age):
    return {"gender": gender, "age_decile": age, "objects": [_BOX], "attributes": []}


def test_silver_gives_sex_and_the_low_edge_of_the_age_group(tmp_path):
    """V-H1-35 (D-06, D-26): ``sex`` from ``gender``, ``age_years`` the low edge of ``age_decile``.

    The release has "20-Oct" for the group 10-20 (a spreadsheet read it as a date), and ">=90".
    """
    graphs = {"a": _graph("M", "50-60"), "b": _graph("F", ">=90"), "c": _graph("F", "20-Oct"),
              "d": _graph(None, None)}
    df = ChestImaGenomeSilverHarmonizer(split="valid", **_silver(tmp_path, graphs)).harmonize()
    assert df["sex"].tolist()[:3] == ["M", "F", "F"] and pd.isna(df["sex"].iloc[3])
    assert df["age_years"].tolist()[:3] == [50.0, 90.0, 10.0] and math.isnan(df["age_years"].iloc[3])


def test_silver_warns_about_images_with_no_scene_graph(tmp_path):
    """V-H1-37: an image of the split list with no scene graph file is not in the table, with a warning."""
    graphs = {"a": _graph("M", "50-60"), "b": None}
    with pytest.warns(UserWarning, match="1 images of the split list have no scene graph"):
        df = ChestImaGenomeSilverHarmonizer(split="valid", **_silver(tmp_path, graphs)).harmonize()
    assert df["image_path"].tolist() == ["p10/p1/s0/a.dcm"]


def test_silver_split_class_loads_a_file_of_the_old_class(tmp_path):
    """V-H1-49 (D-09): a valid file that the mixed class saved loads in the valid class, not in train."""
    table = pd.DataFrame({"patient_id": ["1"], "study_id": ["2"], "image_path": ["p1/x.dcm"], "split": ["valid"]})
    old = ChestImaGenomeSilverHarmonizer(annotation_dir=str(tmp_path), split="valid")
    old.set_harmonized_df(table)
    old.save(str(tmp_path / "old.pkl"))
    h = ChestImaGenomeSilverValidHarmonizer.load_from_saved(str(tmp_path / "old.pkl"))
    assert type(h) is ChestImaGenomeSilverValidHarmonizer
    pd.testing.assert_frame_equal(h.harmonized_df, table)
    with pytest.raises(ValueError, match="has rows of the split"):
        ChestImaGenomeSilverTrainHarmonizer.load_from_saved(str(tmp_path / "old.pkl"))


def test_gold_warns_about_an_image_with_no_readable_size(tmp_path):
    """D-07: the box needs the image size. The harmonizer leaves out an image with no DICOM size, and warns.

    Before, it gave no warning.
    """
    h = _gold(tmp_path, [("d1", "right lung", 20, 10, 100, 60), ("d2", "right lung", 20, 10, 100, 60)], [],
              ids=("d1", "d2"))
    (tmp_path / "files" / "p10" / "p1" / "s1" / "d2.dcm").unlink()
    with pytest.warns(UserWarning, match=r"1 images have no DICOM header with Rows and Columns in base_image_dir"):
        df = h.harmonize()
    assert df["image_path"].tolist() == ["p10/p1/s0/d1.dcm"]


def test_silver_warns_about_an_image_with_no_readable_size(tmp_path):
    """D-07: the silver harmonizer leaves out an image with no DICOM file, and warns with the count."""
    graphs = {"a": _graph("M", "50-60"), "b": _graph("F", "50-60")}
    kw = _silver(tmp_path, graphs)
    (tmp_path / "files" / "p10" / "p1" / "s1" / "b.dcm").unlink()
    with pytest.warns(UserWarning, match=r"1 images have no DICOM header with Rows and Columns"):
        df = ChestImaGenomeSilverHarmonizer(split="valid", **kw).harmonize()
    assert df["image_path"].tolist() == ["p10/p1/s0/a.dcm"]
