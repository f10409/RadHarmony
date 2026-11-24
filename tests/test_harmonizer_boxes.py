"""Harmonizer box tests (D-07): fractions in [0, 1], finite, each min less than its max.

The tests build a tiny fake source folder in ``tmp_path``. The expected
values come from the fake source files, not from the harmonizer.

Run with::

    .venv/bin/python -m pytest tests/test_harmonizer_boxes.py -v
"""
from __future__ import annotations

import pandas as pd
import pytest


def _write_dicom(path, rows, cols, **tags):
    """Write a header-only DICOM file with the image size and extra tags."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = Dataset()
    ds.file_meta = meta
    ds.Rows, ds.Columns = rows, cols
    for k, v in tags.items():
        setattr(ds, k, v)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)


# --------------------------------------------------------------------------
# The shared clip step (R-code-17)
# --------------------------------------------------------------------------
def test_clip_boxes_follows_d07():
    """Clip to [0, 1], and drop a box with no area with its labels, findings and count (D-07).

    A cell keeps its form (list or JSON text). A box inside the image stays as it is, and a
    NaN box stays, so that the transform raises on it.
    """
    import json
    import math

    from radharmony.harmonizer._bbox import clip_boxes

    df = pd.DataFrame({
        "bbox": [[[0.1, 0.5, 0.2, 0.6], [0.3, 0.3, 0.1, 0.2], [-0.1, 0.4, 0.5, 1.2]],
                 json.dumps([[0, 1, 0, 1]]), None, [[float("nan"), 0.5, 0.1, 0.2]],
                 json.dumps([[0.2, 0.6, 0.1, 0.4, 0.0, 1.5], [0.5, 0.4, 0.1, 0.2, 0.3, 0.4]])],
        "bbox_labels": [["a", "b", "c"], json.dumps(["n"]), None, ["x"], json.dumps(["y", "z"])],
        "bbox_findings": [["fa", "fb", "fc"], json.dumps(["fn"]), None, ["fx"], json.dumps(["fy", "fz"])],
        "n_boxes": [3, 1, 0, 1, 2],
    })
    with pytest.warns(UserWarning, match="Fake: dropped 2 boxes"):
        assert clip_boxes(df, "Fake") == 2
    assert df.at[0, "bbox"] == [[0.1, 0.5, 0.2, 0.6], [0.0, 0.4, 0.5, 1.0]]
    assert df.at[0, "bbox_labels"] == ["a", "c"] and df.at[0, "bbox_findings"] == ["fa", "fc"]
    assert df.at[1, "bbox"] == json.dumps([[0, 1, 0, 1]])
    assert df.at[2, "bbox"] is None
    assert math.isnan(df.at[3, "bbox"][0][0])
    assert json.loads(df.at[4, "bbox"]) == [[0.2, 0.6, 0.1, 0.4, 0.0, 1.0]]  # 3-D, still JSON text
    assert json.loads(df.at[4, "bbox_labels"]) == ["y"] and json.loads(df.at[4, "bbox_findings"]) == ["fy"]
    assert df["n_boxes"].tolist() == [2, 1, 0, 1, 1]


def test_every_box_harmonizer_clips_its_boxes():
    """R-code-17, L-015 (D-07): each harmonizer that writes a ``bbox`` column uses the shared clip step.

    A static check: the code names ``clip_boxes`` or ``clip_box`` (a call, or an argument such as
    ``map(clip_box, boxes)``). A name in a comment or an import does not count. A call on a copy of
    the table passes too, so the tests of each harmonizer still check the boxes.
    """
    import ast
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "radharmony" / "harmonizer"
    writes = re.compile(r"""\[\s*["']bbox["']\s*\]\s*=|["']bbox["']\s*:|\bassign\(\s*bbox\s*=|"""
                        r"""BBOX_SOURCE_COL\s*=\s*["']""")
    missing = []
    for path in sorted(root.rglob("*.py")):
        if ".ipynb_checkpoints" in path.parts or path.name in ("base.py", "_bbox.py"):
            continue
        src = path.read_text()
        if not writes.search(src):
            continue
        names = {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        if not names & {"clip_boxes", "clip_box"}:
            missing.append(str(path.relative_to(root)))
    assert not missing, f"no clip_box or clip_boxes in the code: {missing}"


def test_ms_cxr_drops_the_phrase_of_a_dropped_box(tmp_path):
    """D-07: ``label_text`` has one phrase for each box, so a dropped box takes its phrase too."""
    from radharmony.harmonizer.ms_cxr.ms_cxr import MSCXRHarmonizer

    row = dict(dicom_id="d1", path="files/p10/p10000001/s50000001/d1.jpg", image_width=1000, image_height=1000,
               split="train")
    pd.DataFrame([dict(row, category_name="Pneumonia", label_text="no width", x=100, y=100, w=0, h=50),
                  dict(row, category_name="Cardiomegaly", label_text="enlarged heart", x=300, y=300, w=200, h=200)]
                 ).to_csv(tmp_path / "ms_cxr.csv", index=False)
    with pytest.warns(UserWarning, match="dropped 1 box"):
        df = MSCXRHarmonizer(csv_path=str(tmp_path / "ms_cxr.csv")).harmonize()
    assert _json_or_list(df.loc[0, "bbox_labels"]) == ["cardiomegaly"]  # snake case (Q-17)
    assert _json_or_list(df.loc[0, "label_text"]) == ["enlarged heart"]


def _json_or_list(cell):
    import json

    return json.loads(cell) if isinstance(cell, str) else list(cell)


def test_chestxray14_bbox_drops_an_image_whose_only_box_is_dropped(tmp_path):
    """D-07: the box subset keeps only images with a box, also after the clip step."""
    from PIL import Image

    from radharmony.harmonizer.chestxray14.chestxray14_bbox import ChestXray14BboxHarmonizer

    (tmp_path / "images_001" / "images").mkdir(parents=True)
    for name in ("a.png", "b.png"):
        Image.new("L", (100, 100)).save(tmp_path / "images_001" / "images" / name)
    pd.DataFrame([{"Image Index": n, "Finding Labels": "Mass", "Follow-up #": 0, "Patient ID": i, "Patient Age": 50,
                   "Patient Gender": "M", "View Position": "PA"} for i, n in enumerate(("a.png", "b.png"))]
                 ).to_csv(tmp_path / "de.csv", index=False)
    (tmp_path / "bb.csv").write_text("Image Index,Finding Label,Bbox [x,y,w,h],,,\n"
                                     "a.png,Mass,10,10,0,20,,,\nb.png,Mass,10,10,30,20,,,\n")  # a: no width
    with pytest.warns(UserWarning, match="dropped 1 box"):
        df = ChestXray14BboxHarmonizer(csv_path=str(tmp_path / "de.csv"), bbox_csv_path=str(tmp_path / "bb.csv"),
                                       base_image_dir=str(tmp_path)).harmonize()
    assert [p.rsplit("/", 1)[-1] for p in df["image_path"]] == ["b.png"]


@pytest.mark.parametrize("keep", [True, False])
def test_node21_keep_negatives_follows_the_label(tmp_path, keep):
    """D-07: keep_negatives=False drops the nodule-free controls only. A nodule image keeps its row
    when its only box is dropped."""
    import numpy as np
    import SimpleITK as sitk

    from radharmony.harmonizer.node21.node21 import Node21Harmonizer

    d = tmp_path / "cxr_images" / "proccessed_data"
    (d / "images").mkdir(parents=True)
    for name in ("n1.mha", "n2.mha", "c1.mha"):
        sitk.WriteImage(sitk.GetImageFromArray(np.zeros((100, 100), np.uint8)), str(d / "images" / name))
    pd.DataFrame([{"": 0, "img_name": "n1.mha", "label": 1, "x": 10, "y": 10, "width": 0, "height": 20},
                  {"": 1, "img_name": "n2.mha", "label": 1, "x": 10, "y": 10, "width": 20, "height": 20},
                  {"": 2, "img_name": "c1.mha", "label": 0, "x": 0, "y": 0, "width": 0, "height": 0}]
                 ).to_csv(d / "metadata.csv", index=False)
    with pytest.warns(UserWarning, match="dropped 1 box"):
        df = Node21Harmonizer(base_image_dir=str(tmp_path), keep_negatives=keep).harmonize()
    names = {p.rsplit("/", 1)[-1]: n for p, n in zip(df["image_path"], df["n_boxes"], strict=True)}
    assert names == ({"c1.mha": 0, "n1.mha": 0, "n2.mha": 1} if keep else {"n1.mha": 0, "n2.mha": 1})


def test_rsna_2022_cervical_box_with_nan_reaches_the_table(tmp_path):
    """D-07: a NaN box is not clipped to a number. It stays, so the transform raises on it."""
    import math

    from radharmony.harmonizer.rsna_2022_cervical_spine.rsna_2022_cervical_spine_bbox import (
        RSNA2022CervicalSpineBboxHarmonizer as H,
    )

    pd.DataFrame([{"StudyInstanceUID": "1.2.3", **{c: 1 for c in H.LABEL_COLS}}]).to_csv(tmp_path / "train.csv",
                                                                                       index=False)
    pd.DataFrame([{"StudyInstanceUID": "1.2.3", "x": 100, "y": 100, "width": float("nan"), "height": 50,
                   "slice_number": 3},
                  {"StudyInstanceUID": "1.2.3", "x": 100, "y": 100, "width": 50, "height": 50, "slice_number": 4}]
                 ).to_csv(tmp_path / "bb.csv", index=False)
    df = H(csv_path=str(tmp_path / "train.csv"), bbox_csv_path=str(tmp_path / "bb.csv"), num_workers=1).harmonize()
    boxes = _json_or_list(df.loc[0, "bbox"])
    assert len(boxes) == 2 and any(math.isnan(v) for v in boxes[0])


# --------------------------------------------------------------------------
# SIIM-COVID19 (V-H3-19)
# --------------------------------------------------------------------------

def test_siim_covid19_box_is_clipped_to_the_image(tmp_path):
    """V-H3-19: a source box that starts 4 px left of the image gets x_min = 0."""
    from radharmony.harmonizer import SIIMCOVID19TrainHarmonizer

    _write_dicom(tmp_path / "img" / "st1" / "se1" / "sop1.dcm", rows=100, cols=200, PatientID="p1")
    pd.DataFrame({"id": ["st1_study"], "Negative for Pneumonia": [0], "Typical Appearance": [1],
                  "Indeterminate Appearance": [0], "Atypical Appearance": [0]}
                 ).to_csv(tmp_path / "study.csv", index=False)
    # x = -4 and width 54: the box covers x = -4 .. 50 of a 200 px wide image.
    pd.DataFrame({"id": ["sop1_image"],
                  "boxes": ["[{'x': -4.0, 'y': 10.0, 'width': 54.0, 'height': 20.0}]"],
                  "label": ["opacity 1 -4 10 50 30"], "StudyInstanceUID": ["st1"]}
                 ).to_csv(tmp_path / "image.csv", index=False)
    df = SIIMCOVID19TrainHarmonizer(csv_path=str(tmp_path / "study.csv"),
                                    image_csv_path=str(tmp_path / "image.csv"),
                                    base_image_dir=str(tmp_path / "img"), num_workers=1).harmonize()
    assert df.loc[0, "bbox"] == [[0.1, 0.3, 0.0, 0.25]]


# --------------------------------------------------------------------------
# RAD-ChestCT (V-H2-6)
# --------------------------------------------------------------------------

def test_radchestct_extrema_with_inf_give_no_box(tmp_path):
    """V-H2-6: Extrema_35747.csv has inf for 173 volumes. Such a row has no box and no label."""
    from radharmony.harmonizer.radchestct.radchestct import RadChestCTHarmonizer

    pd.DataFrame({"NoteAcc_DEID": ["trn1", "trn2"], "atelectasis*lung": [1, 0]}
                 ).to_csv(tmp_path / "labels.csv", index=False)
    pd.DataFrame({"NoteAcc_DEID": ["trn1", "trn2"], "MRN_DEID": ["m1", "m2"]}
                 ).to_csv(tmp_path / "meta.csv", index=False)
    inf = float("inf")
    pd.DataFrame({"NoteAcc_DEID": ["trn1", "trn2"],
                  "sup_axis0min": [10, inf], "inf_axis0max": [90, -inf],
                  "ant_axis1min": [50, inf], "pos_axis1max": [450, -inf],
                  "rig_axis2min": [100, inf], "lef_axis2max": [400, -inf],
                  "shape0": [100, 100], "shape1": [500, 500], "shape2": [500, 500]}
                 ).to_csv(tmp_path / "extrema.csv", index=False)
    df = RadChestCTHarmonizer(csv_path=str(tmp_path / "meta.csv"),
                              label_csv_path=str(tmp_path / "labels.csv"),
                              bbox_csv_path=str(tmp_path / "extrema.csv")).harmonize()
    df = df.set_index("study_id")
    assert df.loc["trn1", "bbox"] == [0.1, 0.9, 0.1, 0.9, 0.2, 0.8]
    assert df.loc["trn1", "bbox_labels"] == ["lung"]
    assert df.loc["trn2", "bbox"] == [] and df.loc["trn2", "bbox_labels"] == []


# --------------------------------------------------------------------------
# Chest ImaGenome gold and silver (V-H1-13, V-H1-26)
# --------------------------------------------------------------------------

# Pixel boxes (x1, y1, x2, y2) on a 100 x 200 (H x W) image, like the source rows.
_CIG_BOXES = [
    ("right lung", (20, 10, 100, 60)),               # inside: [0.1, 0.6, 0.1, 0.5]
    ("trachea", (0, -20, 0, 50)),                    # no width (gold 41143630-...)
    ("left costophrenic angle", (210, 80, 250, 95)), # past the right edge (silver p10018712)
]


def _cig_folder(tmp_path):
    """Write the record list and one DICOM header. Return (annotation_dir, image_dir)."""
    ann, img = tmp_path / "ann", tmp_path / "files"
    (ann / "utils").mkdir(parents=True)
    pd.DataFrame({"dicom_id": ["d1"], "path": ["files/p10/p1/s1/d1.dcm"], "subject_id": [1],
                  "study_id": [11], "ViewPosition": ["PA"]}
                 ).to_csv(ann / "utils" / "cxr-record-list_view.csv", index=False)
    _write_dicom(img / "p10" / "p1" / "s1" / "d1.dcm", rows=100, cols=200)
    return ann, img


def _check_cig_row(df):
    assert df.loc[0, "bbox"] == [[0.1, 0.6, 0.1, 0.5]]
    assert df.loc[0, "bbox_labels"] == ["right_lung"]  # snake case (Q-17)
    assert df.loc[0, "bbox_findings"] == [""]


def test_chest_imagenome_gold_drops_boxes_with_no_area(tmp_path):
    """V-H1-13: a source box with no area inside the image is dropped with its label."""
    from radharmony.harmonizer import ChestImaGenomeGoldHarmonizer

    ann, img = _cig_folder(tmp_path)
    (ann / "gold_dataset").mkdir()
    pd.DataFrame([{"image_id": "d1.dcm", "bbox_name": name, "original_x1": b[0], "original_y1": b[1],
                   "original_x2": b[2], "original_y2": b[3]} for name, b in _CIG_BOXES]
                 ).to_csv(ann / "gold_dataset" / "gold_bbox_coordinate_annotations_1000images.csv",
                          index=False)
    pd.DataFrame(columns=["image_id", "bbox", "context", "categoryID", "label_name"]).to_csv(
        ann / "gold_dataset" / "gold_object_attribute_with_coordinates.txt", sep="\t", index=False)
    with pytest.warns(UserWarning, match="2 boxes"):
        df = ChestImaGenomeGoldHarmonizer(annotation_dir=str(ann), base_image_dir=str(img)).harmonize()
    _check_cig_row(df)


def test_chest_imagenome_silver_drops_boxes_with_no_area(tmp_path):
    """V-H1-26: the same rule for the silver scene graphs."""
    import json
    import zipfile

    from radharmony.harmonizer import ChestImaGenomeSilverHarmonizer

    ann, img = _cig_folder(tmp_path)
    splits = ann / "silver_dataset" / "splits"
    splits.mkdir(parents=True)
    pd.read_csv(ann / "utils" / "cxr-record-list_view.csv").to_csv(splits / "train.csv", index=False)
    pd.DataFrame({"dicom_id": []}).to_csv(splits / "images_to_avoid.csv", index=False)
    objects = [{"bbox_name": name, "original_x1": b[0], "original_y1": b[1], "original_x2": b[2],
                "original_y2": b[3]} for name, b in _CIG_BOXES]
    with zipfile.ZipFile(ann / "silver_dataset" / "scene_graph.zip", "w") as zf:
        zf.writestr("scene_graph/d1_SceneGraph.json", json.dumps({"objects": objects, "attributes": []}))
    with pytest.warns(UserWarning, match="2 boxes"):
        df = ChestImaGenomeSilverHarmonizer(annotation_dir=str(ann), base_image_dir=str(img),
                                            split="train").harmonize()
    _check_cig_row(df)


# --------------------------------------------------------------------------
# RSNA 2022 cervical spine (V-H3-8)
# --------------------------------------------------------------------------

def test_rsna_2022_cervical_box_uses_the_slice_size(tmp_path):
    """V-H3-8: x and y are fractions of each study's own Rows and Columns, not of 512."""
    from radharmony.harmonizer import RSNA2022CervicalSpineBboxHarmonizer

    for uid, size in (("st768", 768), ("st512", 512)):
        for sn in (1, 2):
            _write_dicom(tmp_path / "img" / uid / f"{sn}.dcm", rows=size, cols=size,
                         ImagePositionPatient=[0.0, 0.0, float(sn)])
    # The 768 x 768 box of the proof (study 1.2.826.0.1.3680043.22678): x 293, y 342, w 76, h 51.
    pd.DataFrame({"StudyInstanceUID": ["st768", "st512"], "x": [293.0, 256.0], "y": [342.0, 128.0],
                  "width": [76.0, 64.0], "height": [51.0, 32.0], "slice_number": [2, 1]}
                 ).to_csv(tmp_path / "boxes.csv", index=False)
    labels = {c: [1, 0] for c in ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]}
    pd.DataFrame({"StudyInstanceUID": ["st768", "st512"], **labels}).to_csv(tmp_path / "train.csv",
                                                                             index=False)
    df = RSNA2022CervicalSpineBboxHarmonizer(csv_path=str(tmp_path / "train.csv"),
                                             bbox_csv_path=str(tmp_path / "boxes.csv"),
                                             base_image_dir=str(tmp_path / "img"),
                                             num_workers=1).harmonize().set_index("study_id")
    assert _json_or_list(df.loc["st768", "bbox"])[0][2:] == [342 / 768, 393 / 768, 293 / 768, 369 / 768]
    assert _json_or_list(df.loc["st512", "bbox"])[0][2:] == [128 / 512, 160 / 512, 256 / 512, 320 / 512]
    # V-H3-23 (D-07): one label per box. Each source box marks a fracture, and the CSV has no class.
    assert all(_json_or_list(df.loc[s, "bbox_labels"]) == ["fracture"] * len(_json_or_list(df.loc[s, "bbox"]))
               for s in df.index)


# --------------------------------------------------------------------------
# VinDr-CXR train (V-H3-4, D-36)
# --------------------------------------------------------------------------

def test_vindr_cxr_train_groups_the_boxes_of_readers_by_overlap(tmp_path):
    """V-H3-4 (D-36): boxes of one class that overlap (IoU > 0.5) are one lesion.

    Image 100 x 200 (rows x columns). Class Nodule/Mass:
    - R1 draws 2 nodules: left [10, 10, 30, 30] and right [150, 50, 170, 70] (x_min, y_min, x_max, y_max).
    - R2 draws the left nodule only: [12, 10, 32, 30] (IoU 0.82 with R1 left).
    - R3 draws a third place: [100, 80, 110, 90].
    Only the left lesion has 2 readers. Its box is the mean of R1 and R2: [11, 10, 31, 30].
    Class Aortic enlargement: only R3 draws a box, so it has no box.
    The old code gave one box: the mean of all 4 nodule boxes, on no real nodule.
    """
    from radharmony.harmonizer.vindr_cxr.vindr_cxr_train import VinDrCXRTrainHarmonizer

    _write_dicom(tmp_path / "img0.dicom", 100, 200)
    pd.DataFrame({"image_id": ["img0"] * 3, "rad_id": ["R1", "R2", "R3"],
                  "Nodule/Mass": [1, 1, 1], "Aortic enlargement": [0, 0, 1]}).to_csv(tmp_path / "labels.csv",
                                                                                        index=False)
    boxes = [("R1", "Nodule/Mass", 10, 10, 30, 30), ("R1", "Nodule/Mass", 150, 50, 170, 70),
             ("R2", "Nodule/Mass", 12, 10, 32, 30), ("R3", "Nodule/Mass", 100, 80, 110, 90),
             ("R3", "Aortic enlargement", 50, 20, 90, 60)]
    pd.DataFrame(boxes, columns=["rad_id", "class_name", "x_min", "y_min", "x_max", "y_max"]).assign(
        image_id="img0").to_csv(tmp_path / "annotations.csv", index=False)
    df = VinDrCXRTrainHarmonizer(csv_path=str(tmp_path / "labels.csv"), bbox_csv_path=str(tmp_path / "annotations.csv"),
                                 base_image_dir=str(tmp_path)).harmonize()
    row = df.iloc[0]
    assert row["bbox_labels"] == ["nodule/mass"]
    assert len(row["bbox"]) == 1
    assert row["bbox"][0] == pytest.approx([10 / 100, 30 / 100, 11 / 200, 31 / 200])  # [y0, y1, x0, x1]


def test_vindr_cxr_train_box_is_clipped_to_the_image(tmp_path):
    """D-07: a lesion box that starts left of the image gets x_min = 0. A box with no area left is dropped."""
    from radharmony.harmonizer.vindr_cxr.vindr_cxr_train import VinDrCXRTrainHarmonizer

    _write_dicom(tmp_path / "img0.dicom", 100, 200)
    pd.DataFrame({"image_id": ["img0"] * 2, "rad_id": ["R1", "R2"], "Nodule/Mass": [1, 1],
                  "Aortic enlargement": [1, 1]}).to_csv(tmp_path / "labels.csv", index=False)
    boxes = [("R1", "Nodule/Mass", -10, 10, 30, 30), ("R2", "Nodule/Mass", -6, 10, 30, 30),
             ("R1", "Aortic enlargement", 210, 10, 230, 30), ("R2", "Aortic enlargement", 212, 10, 232, 30)]
    pd.DataFrame(boxes, columns=["rad_id", "class_name", "x_min", "y_min", "x_max", "y_max"]).assign(
        image_id="img0").to_csv(tmp_path / "annotations.csv", index=False)
    with pytest.warns(UserWarning, match="dropped 1 box"):
        df = VinDrCXRTrainHarmonizer(csv_path=str(tmp_path / "labels.csv"),
                                     bbox_csv_path=str(tmp_path / "annotations.csv"),
                                     base_image_dir=str(tmp_path)).harmonize()
    row = df.iloc[0]
    assert row["bbox_labels"] == ["nodule/mass"]  # the aortic box is right of the image: no area left
    assert row["bbox"][0] == pytest.approx([10 / 100, 30 / 100, 0.0, 30 / 200])


# --------------------------------------------------------------------------
# VinDr-PCXR (V-H3-12)
# --------------------------------------------------------------------------

def test_vindr_pcxr_row_with_empty_coordinates_gives_no_box(tmp_path):
    """V-H3-12: a "No finding" annotation row has no coordinates. It is not a full-image box."""
    import json

    from radharmony.harmonizer.vindr_pcxr.vindr_pcxr import _LABEL_MAP, VinDrPCXRHarmonizer

    for split in ("train", "test"):
        ids = [f"{split}_1", f"{split}_2"]
        for i in ids:
            _write_dicom(tmp_path / split / f"{i}.dicom", rows=2000, cols=1000)
        labels = pd.DataFrame({"image_id": ids, **{c: [0, 0] for c in _LABEL_MAP}})
        labels.loc[0, "No finding"] = 1
        labels.loc[1, "Pneumonia"] = 1
        labels.to_csv(tmp_path / f"image_labels_{split}.csv", index=False)
        nan = float("nan")
        pd.DataFrame({"image_id": ids, "class_name": ["No finding", "Consolidation"],
                      "x_min": [nan, 100], "y_min": [nan, 400], "x_max": [nan, 300], "y_max": [nan, 800]}
                     ).to_csv(tmp_path / f"annotations_{split}.csv", index=False)
    df = VinDrPCXRHarmonizer(base_dir=str(tmp_path)).harmonize().set_index("study_id")
    assert json.loads(df.loc["train_1", "bbox"]) == []
    assert json.loads(df.loc["train_1", "bbox_labels"]) == []
    assert json.loads(df.loc["train_2", "bbox"]) == [[0.2, 0.4, 0.1, 0.3]]
    assert json.loads(df.loc["train_2", "bbox_labels"]) == ["consolidation"]  # snake case (Q-17)


# --------------------------------------------------------------------------
# RSNA 2024 lumbar spine (V-H3-7)
# --------------------------------------------------------------------------

def _write_slice(path, instance, ipp, value, series_uid, study_uid):
    """Write one 8 x 8 sagittal MR slice with a constant pixel value."""
    import numpy as np
    from pydicom.uid import generate_uid

    _write_dicom(path, rows=8, cols=8, SOPClassUID="1.2.840.10008.5.1.4.1.1.4",
                 SOPInstanceUID=generate_uid(), StudyInstanceUID=study_uid,
                 SeriesInstanceUID=series_uid, Modality="MR", InstanceNumber=instance,
                 ImageOrientationPatient=[0, 1, 0, 0, 0, -1], ImagePositionPatient=list(ipp),
                 PixelSpacing=[1.0, 1.0], SliceThickness=4.0, SamplesPerPixel=1,
                 PhotometricInterpretation="MONOCHROME2", BitsAllocated=16, BitsStored=16,
                 HighBit=15, PixelRepresentation=0,
                 PixelData=np.full((8, 8), value, dtype=np.uint16).tobytes())


def test_rsna_2024_lumbar_box_lands_on_the_annotated_slice(tmp_path):
    """V-H3-7: after the 3-D load, the box sits on the slice with the annotated InstanceNumber.

    In this fake sagittal series, x and z both go down as InstanceNumber goes up (like
    series 227532076/1266383365). The loader stacks the slices in InstanceNumber order, so
    a rank by ascending ImagePositionPatient[2] puts the box on the mirrored slice.
    """
    import monai as mn
    from pydicom.uid import generate_uid

    from radharmony.dataset.transforms import _base_load_3d
    from radharmony.harmonizer import RSNA2024LumbarSpineTrainHarmonizer
    from radharmony.harmonizer.rsna_2024_lumbar_spine.rsna_2024_lumbar_spine import _BASE_TARGETS

    series, study = generate_uid(), generate_uid()
    for i in range(1, 6):
        _write_slice(tmp_path / "img" / "1" / "10" / f"{i}.dcm", i, (-4.0 * i, 0.0, 100 - 0.01 * i),
                     100 * i, series, study)
    pd.DataFrame({"study_id": [1], **{t: ["Normal/Mild"] for t in _BASE_TARGETS}}
                 ).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"study_id": [1], "series_id": [10], "series_description": ["Sagittal T1"]}
                 ).to_csv(tmp_path / "series.csv", index=False)
    pd.DataFrame({"study_id": [1], "series_id": [10], "instance_number": [2],
                  "condition": ["Spinal Canal Stenosis"], "level": ["L1/L2"], "x": [4.0], "y": [2.0]}
                 ).to_csv(tmp_path / "coords.csv", index=False)
    row = RSNA2024LumbarSpineTrainHarmonizer(
        csv_path=str(tmp_path / "train.csv"), series_description_csv_path=str(tmp_path / "series.csv"),
        coord_csv_path=str(tmp_path / "coords.csv"), base_image_dir=str(tmp_path / "img"),
        num_workers=1).harmonize().iloc[0]

    load = mn.transforms.Compose([t for t in _base_load_3d(["img"], hu_window=None)
                                  if not isinstance(t, mn.transforms.ScaleIntensityRangePercentilesD)])
    out = load({"img": str(tmp_path / "img" / row["image_path"]), "bbox": row["bbox"]})
    vol, box = out["img"][0], out["bbox"][0]
    centre = [min(int((box[2 * a] + box[2 * a + 1]) / 2 * n), n - 1) for a, n in enumerate(vol.shape)]
    assert float(vol[tuple(centre)]) == 200.0   # the pixel value of InstanceNumber 2
