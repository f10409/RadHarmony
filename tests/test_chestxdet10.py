"""ChestX-Det10 harmonizer tests — no real data.

The released annotations have four properties that fail silently rather than
loudly, so each gets a test: the ``train-old`` directory name, pixel->normalised
box conversion, negatives as real rows, and image-level labels derived from
box labels where one image may carry several boxes of the same class.

Run with::

    .venv/bin/python -m pytest tests/test_chestxdet10.py -v
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from radharmony.harmonizer import ChestXDet10Harmonizer
from radharmony.harmonizer.chestxdet10 import CLASSES, SPLIT_DIRS


@pytest.fixture()
def ds(tmp_path):
    return write_fake(tmp_path)


def write_fake(tmp_path):
    """Two positives (one with a repeated class) and one negative. Returns (JSON path, root)."""
    d = tmp_path / "train-old"
    d.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        Image.fromarray(np.zeros((200, 100), np.uint8)).save(d / name)  # W=100 H=200
    recs = [
        {"file_name": "a.png", "syms": ["Effusion", "Nodule"],
         "boxes": [[10, 20, 30, 60], [50, 100, 60, 120]]},
        {"file_name": "b.png", "syms": ["Nodule", "Nodule"],
         "boxes": [[0, 0, 10, 10], [90, 190, 100, 200]]},
        {"file_name": "c.png", "syms": [], "boxes": []},
    ]
    p = tmp_path / "train.json"
    p.write_text(json.dumps(recs))
    return str(p), str(tmp_path)


def test_split_dir_is_train_old(ds):
    """The archive extracts to train-old/, not train/. Assuming train/ yields
    an empty dataset with no error."""
    assert SPLIT_DIRS["train"] == "train-old"
    csv, root = ds
    df = ChestXDet10Harmonizer(csv_path=csv, base_image_dir=root).harmonize()
    assert df["image_path"].str.startswith("train-old/").all()


def test_boxes_converted_to_normalised_row_col(ds):
    """Released boxes are pixel [x1,y1,x2,y2]; RadHarmony wants normalised
    [dim0_min, dim0_max, dim1_min, dim1_max] with dim0 = row."""
    csv, root = ds
    df = ChestXDet10Harmonizer(csv_path=csv, base_image_dir=root).harmonize()
    box = json.loads(df.iloc[0]["bbox"])[0]
    # x1,y1,x2,y2 = 10,20,30,60 on a 100x200 (WxH) image
    assert box == pytest.approx([20 / 200, 60 / 200, 10 / 100, 30 / 100])
    every = [v for r in df["bbox"] for b in json.loads(r) for v in b]
    assert min(every) >= 0.0 and max(every) <= 1.0


def test_negatives_are_kept_by_default(ds):
    """764 images across the release carry no findings. They are the
    benchmark's negative controls, not missing data."""
    csv, root = ds
    df = ChestXDet10Harmonizer(csv_path=csv, base_image_dir=root).harmonize()
    assert len(df) == 3
    assert (df["n_boxes"] == 0).sum() == 1
    dropped = ChestXDet10Harmonizer(
        csv_path=csv, base_image_dir=root, keep_negatives=False).harmonize()
    assert len(dropped) == 2


def test_labels_are_image_level_not_instance_level(ds):
    """b.png has TWO nodule boxes but must contribute 1 to the nodule label."""
    csv, root = ds
    df = ChestXDet10Harmonizer(csv_path=csv, base_image_dir=root).harmonize()
    assert df["nodule"].sum() == 2          # images, not the 3 nodule instances
    assert df["effusion"].sum() == 1
    assert df.loc[df.index[1], "n_boxes"] == 2


def test_bbox_is_json_encoded(ds):
    """Ragged per-image box lists die in MONAI collation; JSON strings survive."""
    csv, root = ds
    df = ChestXDet10Harmonizer(csv_path=csv, base_image_dir=root).harmonize()
    assert df["bbox"].map(type).eq(str).all()
    assert json.loads(df.iloc[2]["bbox"]) == []


def test_ten_classes_all_present_as_columns(ds):
    csv, root = ds
    df = ChestXDet10Harmonizer(csv_path=csv, base_image_dir=root).harmonize()
    for c in CLASSES:
        assert c.lower() in df.columns
    assert len(CLASSES) == 10


def test_bad_split_rejected(ds):
    csv, root = ds
    with pytest.raises(ValueError):
        ChestXDet10Harmonizer(csv_path=csv, base_image_dir=root, split="valid")


@pytest.mark.parametrize("key", ["chestxdet10", "chestxdet10_train", "chestxdet10_test"])
def test_no_base_image_dir_raises(key):
    """L-006: the default is not a NAS path. With no folder and no table, the class raises a ValueError that
    names base_image_dir."""
    import radharmony.dataset  # noqa: F401  (fills the registry)
    from radharmony.registry import resolve_dataset

    with pytest.raises(ValueError, match="base_image_dir"):
        resolve_dataset(key)(cache_dir=None)
