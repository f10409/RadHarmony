"""ChestX-ray14 tests: the box subset and verify_images(). No real data.

The expected values come from the fake source files, not from the code.

Run with::

    .venv/bin/python -m pytest tests/test_chestxray14.py -v
"""
from __future__ import annotations

import pandas as pd
import pytest
from PIL import Image

from radharmony.harmonizer import ChestXray14BboxHarmonizer


def _fake(tmp_path):
    """One 1024 x 1024 PNG whose CSV gives an original size of 2500 x 2048, and one box. Return the arguments."""
    (tmp_path / "images_001" / "images").mkdir(parents=True)
    Image.new("L", (1024, 1024)).save(tmp_path / "images_001" / "images" / "a.png")
    pd.DataFrame([{"Image Index": "a.png", "Finding Labels": "Mass", "Follow-up #": 0, "Patient ID": 1,
                   "Patient Age": 50, "Patient Gender": "M", "View Position": "PA",
                   "OriginalImage[Width": 2500, "Height]": 2048}]).to_csv(tmp_path / "de.csv", index=False)
    (tmp_path / "bb.csv").write_text("Image Index,Finding Label,Bbox [x,y,w,h],,,\na.png,Mass,256,512,128,256,,,\n")
    return dict(csv_path=str(tmp_path / "de.csv"), bbox_csv_path=str(tmp_path / "bb.csv"),
                base_image_dir=str(tmp_path))


def test_box_is_a_fraction_of_the_png_size(tmp_path):
    """V-H1-46: the source box is in the pixels of the 1024 x 1024 PNG, not of the original image in the CSV."""
    df = ChestXray14BboxHarmonizer(**_fake(tmp_path)).harmonize()
    assert df.loc[0, "bbox"] == [[512 / 1024, 768 / 1024, 256 / 1024, 384 / 1024]]


def test_no_base_image_dir_raises(tmp_path):
    """V-H1-36: the box scale needs the PNG files. With no folder, harmonize() names base_image_dir."""
    args = {**_fake(tmp_path), "base_image_dir": None}
    with pytest.raises(ValueError, match="base_image_dir"):
        ChestXray14BboxHarmonizer(**args).harmonize()


def test_a_missing_png_drops_its_row(tmp_path):
    """A box image with no PNG on disk is not in the table. Before, it raised a TypeError with no cause."""
    args = _fake(tmp_path)
    de = pd.read_csv(args["csv_path"])
    pd.concat([de, de.assign(**{"Image Index": "b.png", "Patient ID": 2})]).to_csv(args["csv_path"], index=False)
    with open(args["bbox_csv_path"], "a") as f:
        f.write("b.png,Mass,256,512,128,256,,,\n")
    df = ChestXray14BboxHarmonizer(**args).harmonize()
    assert df["image_path"].tolist() == ["images_001/images/a.png"]


@pytest.mark.parametrize("cls_name", ["ChestXray14TrainDataset", "ChestXray14BboxDataset"])
def test_verify_images_keeps_a_gray_rgba_png(tmp_path, cls_name):
    """D-42: a gray RGBA PNG loads as 1 channel, so it is a good image. Before, verify_images() dropped it."""
    import radharmony.dataset as ds_mod

    Image.new("RGBA", (8, 8), (90, 90, 90, 255)).save(tmp_path / "rgba.png")
    Image.new("L", (8, 8)).save(tmp_path / "gray.png")
    (tmp_path / "broken.png").write_bytes(b"not a png")
    df = pd.DataFrame({"patient_id": ["1", "2", "3", "4"], "study_id": ["1", "2", "3", "4"],
                       "image_path": ["rgba.png", "gray.png", "broken.png", "missing.png"]})
    ds = getattr(ds_mod, cls_name)(base_image_dir=str(tmp_path), harmonized_df=df, cache_dir=None)
    bad = ds.verify_images()
    assert dict(zip(bad["image_path"], bad["verify_reason"], strict=True)) == {"broken.png": "unreadable",
                                                                                "missing.png": "missing"}
    assert ds._get_harmonized_df()["image_path"].tolist() == ["rgba.png", "gray.png"]
