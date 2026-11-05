"""NODE21 harmonizer/dataset tests.

Data-dependent tests skip when the release is not on disk, so the suite still
runs on a machine without it. The synthetic tests below do NOT skip: they pin
the traps that would otherwise fail silently — the zero-filled negative
encoding, one-row-per-box collapsing, and the top-left box conversion.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from radharmony.harmonizer import Node21Harmonizer
from radharmony.harmonizer.node21 import CLASSES, VARIANT_DIRS
from radharmony.registry import list_datasets, resolve_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dataset_paths import paths  # noqa: E402

# No entry (a clone with its own paths file): the empty path skips the data tests.
ROOT = paths("node21", missing_ok=True).get("base_image_dir", "")
HAS_DATA = os.path.isdir(os.path.join(ROOT, "cxr_images"))
needs_data = pytest.mark.skipif(not HAS_DATA, reason="NODE21 not on this machine")


def test_registered():
    # Importing the dataset package is what runs @register_dataset; without it
    # the registry is empty and this asserts nothing.
    import radharmony.dataset  # noqa: F401

    keys = list_datasets()
    assert "node21" in keys
    assert "node21_original" in keys


def test_exported_from_package_namespaces():
    import radharmony.dataset as D
    import radharmony.harmonizer as H

    assert "Node21Dataset" in dir(D)
    assert "Node21OriginalDataset" in dir(D)
    assert "Node21Harmonizer" in dir(H)


def test_single_class():
    assert CLASSES == ["nodule"]


def test_variant_dir_is_the_misspelled_one():
    # The release ships "proccessed_data" with two c's. If this ever changes,
    # the harmonizer breaks silently on a missing path, so pin it.
    assert VARIANT_DIRS["processed"] == "proccessed_data"
    assert VARIANT_DIRS["original"] == "original_data"


def test_rejects_unknown_variant():
    with pytest.raises(ValueError):
        Node21Harmonizer(base_image_dir=ROOT, variant="preprocessed")


def _synthetic(tmp_path):
    """A 3-row CSV covering both encodings: 2 boxes on one image, 1 control."""
    d = tmp_path / "cxr_images" / "proccessed_data"
    d.mkdir(parents=True)
    pd.DataFrame([
        {"": 0, "height": 10, "img_name": "n0001.mha", "label": 1,
         "width": 20, "x": 30, "y": 40},
        {"": 1, "height": 12, "img_name": "n0001.mha", "label": 1,
         "width": 22, "x": 32, "y": 42},
        {"": 2, "height": 0, "img_name": "c0001.mha", "label": 0,
         "width": 0, "x": 0, "y": 0},
    ]).to_csv(d / "metadata.csv", index=False)
    return d / "metadata.csv"


def test_rows_collapse_to_images_not_boxes(tmp_path, monkeypatch):
    """5,224 CSV rows describe 4,882 images; rows must not be counted as images."""
    csv = _synthetic(tmp_path)
    h = Node21Harmonizer(csv_path=str(csv), base_image_dir=str(tmp_path))
    df = h._read_source()
    assert len(df) == 2, "two distinct images, not three rows"
    n1 = df[df.img_name == "n0001.mha"].iloc[0]
    assert len(n1["boxes"]) == 2


def test_zero_filled_negative_yields_no_box(tmp_path):
    """label=0 rows carry x=y=w=h=0; kept literally that is a corner box."""
    csv = _synthetic(tmp_path)
    h = Node21Harmonizer(csv_path=str(csv), base_image_dir=str(tmp_path))
    df = h._read_source()
    c1 = df[df.img_name == "c0001.mha"].iloc[0]
    assert c1["boxes"] == []
    assert c1["label"] == 0


@needs_data
def test_counts_match_the_paper():
    h = Node21Harmonizer(base_image_dir=ROOT, variant="processed")
    df = h.harmonize()
    nb = df["bbox"].apply(lambda s: len(json.loads(s)))
    assert len(df) == 4882
    assert nb.sum() == 1476
    assert (nb > 0).sum() == 1134
    assert (nb == 0).sum() == 3748
    assert df["nodule"].sum() == 1134


@needs_data
def test_all_boxes_normalised_in_unit_range():
    h = Node21Harmonizer(base_image_dir=ROOT, variant="processed")
    df = h.harmonize()
    vals = [v for s in df["bbox"] for b in json.loads(s) for v in b]
    assert vals, "expected boxes"
    assert all(0.0 <= v <= 1.0 for v in vals)


@needs_data
def test_provenance_covers_positives_only():
    """The source-corpus map covers the 1,134 nodule images and no controls."""
    h = Node21Harmonizer(base_image_dir=ROOT, variant="processed")
    df = h.harmonize()
    known = df[df["source_dataset"] != "unknown"]
    assert len(known) == 1134
    assert set(known["source_dataset"]) == {
        "chestxray14", "padchest", "jsrt", "openi"}
    assert (df["source_dataset"] == "unknown").sum() == 3748


@needs_data
def test_keep_negatives_false_drops_controls():
    h = Node21Harmonizer(base_image_dir=ROOT, variant="processed",
                         keep_negatives=False)
    df = h.harmonize()
    assert len(df) == 1134
    assert (df["n_boxes"] == 0).sum() == 0


@needs_data
def test_variants_disagree_on_coordinates():
    """Boxes are only valid against their own variant's pixels."""
    a = Node21Harmonizer(base_image_dir=ROOT, variant="processed").harmonize()
    b = Node21Harmonizer(base_image_dir=ROOT, variant="original").harmonize()
    a = a.set_index("patient_id")["bbox"]
    b = b.set_index("patient_id")["bbox"]
    common = [i for i in a.index if a[i] != "[]"]
    identical = sum(1 for i in common if a[i] == b[i])
    assert identical / len(common) < 0.10, (
        "processed and original coordinates should differ for nearly every box")


@needs_data
def test_dataset_yields_image_and_boxes():
    import torch

    B = resolve_dataset("node21")(
        base_image_dir=ROOT, cache_dir="/tmp/n21cache_test", output_cls=True, output_bbox=True,
        dtype=torch.float32)
    df = B.get_harmonized_df()
    ds = B.get_datasets()
    nb = df["bbox"].apply(lambda s: len(json.loads(s)))
    i = int(nb[nb > 0].index[0])
    s = ds[i]
    assert set(s) >= {"img", "cls", "bbox", "bbox_labels"}
    assert s["img"].shape[0] == 1
    assert s["cls"].shape == (1,)
    boxes = json.loads(s["bbox"]) if isinstance(s["bbox"], str) else s["bbox"]
    assert len(boxes) >= 1


def test_no_base_image_dir_raises():
    """L-006: the default is not a NAS path. With no folder and no table, the class raises a ValueError that
    names base_image_dir."""
    from radharmony.dataset import Node21Dataset

    with pytest.raises(ValueError, match="base_image_dir"):
        Node21Dataset(cache_dir=None)
