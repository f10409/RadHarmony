"""Transform-pipeline tests — no real data, synthetic image/volume in tmp_path.

Exercises the deterministic preprocessing of RadiologyTransform2D / 3D
(load → normalise → resize → pad → tensor → select) end to end, plus the
fluent builder contract. Uses a throwaway PNG and NIfTI so it runs anywhere.

Run with::

    .venv/bin/python -m pytest tests/test_transforms.py -v
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch
from PIL import Image

import monai.transforms as mn
from radharmony.dataset.transforms import RadiologyTransform2D, RadiologyTransform3D


@pytest.fixture
def png_2d(tmp_path):
    """A 60x40 grayscale PNG (non-square, so resize/pad behaviour is observable)."""
    path = tmp_path / "img2d.png"
    Image.fromarray((np.random.rand(60, 40) * 255).astype(np.uint8)).save(path)
    return str(path)


@pytest.fixture
def nifti_3d(tmp_path):
    """A 20x30x40 synthetic volume written as NIfTI (needs SimpleITK)."""
    sitk = pytest.importorskip("SimpleITK")
    vol = (np.random.rand(20, 30, 40) * 100).astype(np.float32)
    path = tmp_path / "vol3d.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(vol), str(path))
    return str(path)


# --------------------------------------------------------------------------
# 2-D
# --------------------------------------------------------------------------

def test_2d_padded_shape_and_dtype(png_2d):
    t = RadiologyTransform2D(img_size=32, output_keys={"img", "cls"},
                             dtype=torch.float32).get_transform()
    out = t({"img": png_2d, "cls": [1.0]})
    assert tuple(out["img"].shape) == (1, 32, 32)   # padded to a square
    assert out["img"].dtype == torch.float32


def test_2d_intensity_normalised_to_unit_range(png_2d):
    t = RadiologyTransform2D(img_size=32, dtype=torch.float32).get_transform()
    img = t({"img": png_2d, "cls": [0.0]})["img"]
    assert img.min() >= -1.0 - 1e-4 and img.max() <= 1.0 + 1e-4


def test_2d_no_pad_preserves_aspect(png_2d):
    t = RadiologyTransform2D(img_size=32, pad=False, dtype=torch.float32).get_transform()
    img = t({"img": png_2d, "cls": [0.0]})["img"]
    spatial = tuple(img.shape[1:])
    assert max(spatial) == 32          # longest edge resized to img_size
    assert min(spatial) < 32           # shorter edge not padded out


def test_2d_select_items_keeps_only_requested_keys(png_2d):
    t = RadiologyTransform2D(img_size=32, output_keys={"img"},
                             dtype=torch.float32).get_transform()
    out = t({"img": png_2d, "cls": [1.0]})
    assert "img" in out and "cls" not in out   # cls dropped — not in output_keys


# --------------------------------------------------------------------------
# 3-D
# --------------------------------------------------------------------------

def test_3d_padded_cube_shape_and_dtype(nifti_3d):
    t = RadiologyTransform3D(img_size=16, output_keys={"img", "cls"},
                             dtype=torch.float32).get_transform()
    out = t({"img": nifti_3d, "cls": [1.0]})
    assert tuple(out["img"].shape) == (1, 16, 16, 16)   # padded to a cube
    assert out["img"].dtype == torch.float32


def test_3d_no_pad_longest_edge_matches(nifti_3d):
    t = RadiologyTransform3D(img_size=16, pad=False, dtype=torch.float32).get_transform()
    img = t({"img": nifti_3d, "cls": [0.0]})["img"]
    assert max(tuple(img.shape[1:])) == 16


# --------------------------------------------------------------------------
# Builder contract
# --------------------------------------------------------------------------

def test_default_output_keys():
    assert RadiologyTransform2D().output_keys == {"img", "cls"}


def test_builder_is_fluent_and_builds_compose():
    builder = RadiologyTransform2D(img_size=32)
    assert builder.with_flip(spatial_axis=1) is builder        # returns self
    assert builder.with_intensity_jitter(shift=0.05) is builder
    assert len(builder._aug) >= 2                              # each with_* queued ≥1 step
    assert isinstance(builder.get_transform(), mn.Compose)


# --------------------------------------------------------------------------
# Boxes (D-28). The expected values come from the spec, not from the code.
# tests/test_lessons.py (L-011) checks every spatial step on a white rectangle.
# --------------------------------------------------------------------------

@pytest.fixture
def png_tall(tmp_path):
    """A 200 x 100 (H x W) grayscale PNG, like the 2,000 x 1,000 example of D-28."""
    path = tmp_path / "tall.png"
    Image.fromarray((np.random.rand(200, 100) * 255).astype(np.uint8)).save(path)
    return str(path)


def _box_transform(output_keys=frozenset({"img", "bbox", "bbox_labels"}), img_size=64,
                   dtype=torch.float32, **kw):
    return RadiologyTransform2D(img_size=img_size, output_keys=set(output_keys),
                                dtype=dtype, **kw)


def test_box_moves_with_resize_and_pad(png_tall):
    """V-D-1, R-code-6: the D-28 example. 200 x 100 -> 64 x 32 -> pad 16 left and right."""
    out = _box_transform().get_transform()(
        {"img": png_tall, "bbox": [[0.0, 1.0, 0.0, 1.0]], "bbox_labels": ["all"]})
    assert out["bbox"].tolist() == [[0.0, 1.0, 0.25, 0.75]]


def test_box_moves_with_flip(png_2d):
    """V-D-18: a left-right flip mirrors x: [0.1, 0.3] -> [0.7, 0.9]."""
    t = _box_transform(pad=False).with_flip(spatial_axis=1, prob=1.0).get_transform()
    out = t({"img": png_2d, "bbox": [[0.2, 0.4, 0.1, 0.3]], "bbox_labels": ["a"]})
    assert torch.allclose(out["bbox"], torch.tensor([[0.2, 0.4, 0.7, 0.9]]))


def test_box_moves_with_transpose(png_2d):
    """V-D-18: a transpose of H and W swaps the y and x pairs."""
    t = _box_transform(pad=False).with_transpose(indices=[0, 2, 1]).get_transform()
    out = t({"img": png_2d, "bbox": [[0.2, 0.4, 0.1, 0.3]], "bbox_labels": ["a"]})
    assert torch.allclose(out["bbox"], torch.tensor([[0.1, 0.3, 0.2, 0.4]]))


@pytest.mark.parametrize("raw,n", [
    ([[0.1, 0.4, 0.2, 0.5]], 1),
    ([0.1, 0.4, 0.2, 0.5], 1),                                 # one flat box
    (list(np.array([0.1, 0.4, 0.2, 0.5])), 1),                 # numpy floats (RAD-ChestCT)
    (np.array([[0.1, 0.4, 0.2, 0.5], [0.5, 0.6, 0.5, 0.6]]), 2),
    ("[[0.1, 0.4, 0.2, 0.5], [0.5, 0.6, 0.5, 0.6]]", 2),       # JSON string (NODE21)
    ([], 0), ("[]", 0), ("nan", 0), (float("nan"), 0), (np.float32("nan"), 0), (None, 0),
])
def test_box_shape_is_n_by_4(png_2d, raw, n):
    """V-D-3, R-code-5: always (N, 4). No box gives (0, 4) and no fake [0, 0, 0, 0] box."""
    labels = ["a", "b"][:n]
    out = _box_transform().get_transform()({"img": png_2d, "bbox": raw, "bbox_labels": labels})
    assert tuple(out["bbox"].shape) == (n, 4)
    assert out["bbox_labels"] == labels


def test_json_labels_become_a_list(png_2d):
    """The per-box labels stay a list, in the order of the boxes."""
    out = _box_transform().get_transform()(
        {"img": png_2d, "bbox": "[[0.1, 0.4, 0.2, 0.5]]", "bbox_labels": '["nodule"]'})
    assert out["bbox_labels"] == ["nodule"]


def test_a_table_read_back_from_csv_gives_the_same_boxes(png_2d, tmp_path):
    """V-D-33: to_csv writes a list as Python text ("['a', 'b']"). The box step reads it as well as JSON."""
    import pandas as pd

    boxes, labels = [[0.1, 0.4, 0.2, 0.5], [0.5, 0.6, 0.5, 0.6]], ["nodule", "mass"]
    pd.DataFrame({"img": [png_2d], "bbox": [boxes], "bbox_labels": [labels]}).to_csv(tmp_path / "t.csv", index=False)
    row = pd.read_csv(tmp_path / "t.csv").iloc[0]
    assert row["bbox_labels"] == "['nodule', 'mass']"  # not JSON
    out = _box_transform(pad=False).get_transform()(
        {"img": row["img"], "bbox": row["bbox"], "bbox_labels": row["bbox_labels"]})
    assert torch.allclose(out["bbox"], torch.tensor(boxes))
    assert out["bbox_labels"] == labels


def test_a_box_text_that_is_not_a_list_raises(png_2d):
    """V-D-33: the Python-list fallback reads only literals. Other text raises a clear error."""
    with pytest.raises(RuntimeError) as err:   # MONAI wraps the ValueError
        _box_transform().get_transform()(
            {"img": png_2d, "bbox": "[[0.1, 0.4, 0.2, 0.5]]", "bbox_labels": "__import__('os')"})
    assert "cannot read" in str(err.value.__cause__)


def test_2d_dtype_is_for_the_image_only(png_2d):
    """V-B-16 (dataset side): with the default bfloat16, a label 37.3 and a reg value 1001 keep their values."""
    t = RadiologyTransform2D(img_size=32, output_keys={"img", "cls", "reg", "mask"}).get_transform()
    out = t({"img": png_2d, "mask": png_2d, "cls": [37.3, float("nan")], "reg": [1001.0]})
    assert out["img"].dtype == torch.bfloat16
    assert out["cls"].dtype == out["reg"].dtype == out["mask"].dtype == torch.float32
    assert out["cls"][0].item() == pytest.approx(37.3, abs=1e-5)
    assert torch.isnan(out["cls"][1])
    assert out["reg"].tolist() == [1001.0]


def test_3d_dtype_is_for_the_image_only(nifti_3d, tmp_path):
    """V-B-16 (dataset side): a 3-D mask id 257 and 1001 keep their values with the default bfloat16."""
    sitk = pytest.importorskip("SimpleITK")
    mask = np.zeros((20, 30, 40), dtype=np.float32)
    mask[:, :, :20], mask[:, :, 20:] = 257, 1001
    path = tmp_path / "mask3d.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(path))
    t = RadiologyTransform3D(img_size=16, output_keys={"img", "cls", "mask"}).get_transform()
    out = t({"img": nifti_3d, "mask": str(path), "cls": [37.3]})
    assert out["img"].dtype == torch.bfloat16
    assert out["mask"].dtype == out["cls"].dtype == torch.float32
    assert set(torch.unique(out["mask"]).tolist()) <= {0.0, 257.0, 1001.0}  # 0 is the pad
    assert {257.0, 1001.0} <= set(torch.unique(out["mask"]).tolist())
    assert out["cls"][0].item() == pytest.approx(37.3, abs=1e-5)


@pytest.mark.parametrize("text", ["'abc'", "{'a': 1}", "b'xy'"])
def test_a_box_text_that_is_a_literal_but_not_a_list_raises(png_2d, text):
    """V-D-33: the fallback takes only a list or a tuple. A string or a dict is not split into a fake list."""
    with pytest.raises(RuntimeError) as err:   # MONAI wraps the ValueError
        _box_transform().get_transform()(
            {"img": png_2d, "bbox": "[[0.1, 0.4, 0.2, 0.5]]", "bbox_labels": text})
    assert "cannot read" in str(err.value.__cause__)


def test_spacing_drops_a_box_on_the_cut_strip(tmp_path):
    """V-D-42 (D-07): with 8 mm slices to 1 mm, MONAI cuts about 3.5 mm off each z end.

    A box only on that strip has no area after the clip. It is dropped with its label.
    """
    sitk = pytest.importorskip("SimpleITK")
    img = sitk.GetImageFromArray(np.random.rand(6, 40, 40).astype(np.float32))
    img.SetSpacing((1.0, 1.0, 8.0))
    path = tmp_path / "thick.nii.gz"
    sitk.WriteImage(img, str(path))
    t = (RadiologyTransform3D(img_size=48, output_keys={"img", "bbox", "bbox_labels"}, dtype=torch.float32)
         .with_spacing((1.0, 1.0, 1.0)).get_transform())
    boxes = [[0.0, 0.05, 0.1, 0.5, 0.1, 0.5], [0.95, 1.0, 0.1, 0.5, 0.1, 0.5], [0.2, 0.6, 0.1, 0.5, 0.1, 0.5]]
    out = t({"img": str(path), "bbox": boxes, "bbox_labels": ["a", "b", "c"]})
    assert out["bbox_labels"] == ["c"]
    assert tuple(out["bbox"].shape) == (1, 6)


def test_box_is_float32_for_a_bfloat16_image(png_2d):
    """V-D-24: bfloat16 keeps 8 bits. The box stays float32."""
    box = [0.123456, 0.654321, 0.111111, 0.999999]
    out = _box_transform(pad=False, dtype=torch.bfloat16).get_transform()(
        {"img": png_2d, "bbox": [box], "bbox_labels": ["a"]})
    assert out["img"].dtype == torch.bfloat16
    assert out["bbox"].dtype == torch.float32
    assert torch.allclose(out["bbox"], torch.tensor([box]))


def test_with_bbox_as_mask_does_not_change_the_callers_set(png_2d):
    """V-D-26: a second builder that reuses the set still works."""
    keys = {"img", "bbox", "bbox_labels"}
    RadiologyTransform2D(output_keys=keys).with_bbox_as_mask(keep_mask=True).get_transform()
    assert keys == {"img", "bbox", "bbox_labels"}
    out = _box_transform(output_keys=keys).get_transform()(
        {"img": png_2d, "bbox": [[0.1, 0.4, 0.2, 0.5]], "bbox_labels": ["a"]})
    assert "bbox_mask" not in out


def test_box_that_is_not_finite_raises(png_2d):
    """D-07: box values are finite. A NaN box stops with a clear error, not a wrong box."""
    with pytest.raises(RuntimeError) as err:   # MONAI wraps the ValueError
        _box_transform().get_transform()(
            {"img": png_2d, "bbox": [[float("nan"), 0.5, 0.1, 0.2]], "bbox_labels": ["a"]})
    assert "not finite" in str(err.value.__cause__)


def test_old_transpose_keys_with_bbox_mask_still_work(png_2d):
    """D-09: old code passed "bbox_mask" in the transpose keys. The box still moves."""
    t = (_box_transform(pad=False).with_bbox_as_mask(keep_mask=True)
         .with_transpose(indices=[0, 2, 1], keys=["img", "bbox_mask"]).get_transform())
    out = t({"img": png_2d, "bbox": [[0.2, 0.4, 0.1, 0.3]], "bbox_labels": ["a"]})
    assert torch.allclose(out["bbox"], torch.tensor([[0.1, 0.3, 0.2, 0.4]]))
    assert tuple(out["bbox_mask"].shape[1:]) == tuple(out["img"].shape[1:])


def test_keep_mask_gives_one_channel_per_box(png_2d):
    """keep_mask=True gives one mask channel for each box, on the image grid."""
    t = _box_transform(img_size=32).with_bbox_as_mask(keep_mask=True).get_transform()
    out = t({"img": png_2d, "bbox": [[0.0, 0.5, 0.25, 0.75], [0.5, 1.0, 0.25, 0.75]],
             "bbox_labels": ["a", "b"]})
    assert tuple(out["bbox_mask"].shape) == (2, 32, 32)
    assert out["bbox_mask"][0, :16].sum() > 0 and out["bbox_mask"][0, 16:].sum() == 0


def test_mask_round_trip_keeps_the_box():
    """V-D-4: paint, then read back. The max edge is (index + 1) / n, so nothing shrinks."""
    from radharmony.dataset.transforms import _channel_to_box, _paint_box

    for box in ([0.25, 0.5, 0.25, 0.5], [0.0, 1.0, 0.0, 1.0], [0.125, 0.875, 0.5, 0.625]):
        chan = torch.zeros(8, 8)
        _paint_box(chan, box)
        assert _channel_to_box(chan) == box
    chan = torch.zeros(8, 8)
    _paint_box(chan, [0.0, 1.0003, 0.5, 0.75])   # a value above 1 is clipped, not dropped
    assert _channel_to_box(chan) == [0.0, 1.0, 0.5, 0.75]


def test_identity_warp_keeps_the_exact_box(png_2d):
    """V-D-4: a warp that does not move the box keeps its exact value."""
    t = _box_transform(pad=False).with_affine(
        translate_range=(0, 0), rotate_range=(0,), scale_range=(0, 0), shear_range=(0,),
        prob=1.0).get_transform()
    box = [0.123, 0.456, 0.211, 0.789]
    out = t({"img": png_2d, "bbox": [box], "bbox_labels": ["a"]})
    assert torch.allclose(out["bbox"], torch.tensor([box]))


def test_box_that_leaves_the_image_is_dropped_with_its_label():
    """R-code-5: an empty mask channel after a warp gives no box, not a [0, 0, 0, 0] box."""
    from radharmony.dataset.transforms import _BboxToTensorD, _MaskToBbox

    mask = torch.zeros(2, 8, 8)
    mask[1, 4:8, 0:4] = 1
    data = {"img": torch.zeros(1, 8, 8), "bbox": [[0.0, 0.25, 0.0, 0.25], [0.5, 1.0, 0.0, 0.5]],
            "bbox_labels": ["gone", "kept"], "bbox_mask": mask}
    out = _BboxToTensorD(2)(_MaskToBbox()(data))
    assert out["bbox"].tolist() == [[0.5, 1.0, 0.0, 0.5]]
    assert out["bbox_labels"] == ["kept"]
    data["bbox_mask"] = torch.zeros(2, 8, 8)
    out = _BboxToTensorD(2)(_MaskToBbox()(data))
    assert tuple(out["bbox"].shape) == (0, 4) and out["bbox_labels"] == []


def test_bbox_collate_batches_different_box_counts(png_2d):
    """V-D-2: images with 0, 1 and 2 boxes go into one batch."""
    from torch.utils.data import DataLoader

    from radharmony.dataset import bbox_collate

    t = _box_transform(img_size=32, output_keys={"img", "cls", "bbox", "bbox_labels"}).get_transform()
    boxes = [[], [[0.1, 0.4, 0.2, 0.5]], [[0.1, 0.4, 0.2, 0.5], [0.5, 0.6, 0.5, 0.6]]]
    samples = [t({"img": png_2d, "cls": [float(i)], "bbox": b, "bbox_labels": ["x"] * len(b)})
               for i, b in enumerate(boxes)]
    with pytest.raises(RuntimeError):
        next(iter(DataLoader(samples, batch_size=3)))       # the default collate fails
    batch = next(iter(DataLoader(samples, batch_size=3, collate_fn=bbox_collate)))
    assert tuple(batch["img"].shape) == (3, 1, 32, 32) and tuple(batch["cls"].shape) == (3, 1)
    assert [tuple(b.shape) for b in batch["bbox"]] == [(0, 4), (1, 4), (2, 4)]
    assert batch["bbox_labels"] == [[], ["x"], ["x", "x"]]


def test_box_steps_read_a_json_box_in_a_custom_pipeline():
    """A pipeline with its own preprocessing (no box parse step) can flip and warp a JSON box."""
    from radharmony.evaluator.transforms import RadiologyEncoderTransform

    def preprocess(d):
        return {**d, "img": torch.zeros(1, 16, 16)}

    with pytest.warns(UserWarning, match="D-28"):  # a custom preprocess has unknown steps
        t = (RadiologyEncoderTransform(preprocess=preprocess,
                                       output_keys={"img", "bbox", "bbox_labels"}, dtype=torch.float32)
             .with_flip(spatial_axis=1, prob=1.0).with_affine(prob=0.0).get_transform())
    out = t({"img": "unused.png", "bbox": "[[0.25, 0.5, 0.25, 0.5]]", "bbox_labels": '["a"]'})
    assert torch.allclose(torch.as_tensor(out["bbox"]), torch.tensor([[0.25, 0.5, 0.5, 0.75]]))


def test_2d_mask_file_with_255_gives_0_and_1(png_2d, tmp_path):
    """V-D-10 (D-14): a 2-D mask in a sample holds 0 and 1, also when the file holds 0 and 255."""
    mask = np.zeros((60, 40), dtype=np.uint8)
    mask[10:30, 5:25] = 255
    path = tmp_path / "mask2d.png"
    Image.fromarray(mask).save(path)
    t = RadiologyTransform2D(img_size=32, output_keys={"img", "mask"}, dtype=torch.float32).get_transform()
    out = t({"img": png_2d, "mask": str(path)})
    assert sorted(torch.unique(out["mask"]).tolist()) == [0.0, 1.0]


# A "spawn" or "forkserver" DataLoader worker pickles the transform. A lambda step cannot be pickled.
PICKLE_CASES = {
    "2d": lambda: RadiologyTransform2D(img_size=32),
    "2d_affine": lambda: RadiologyTransform2D(img_size=32).with_affine(prob=1.0),
    "2d_elastic": lambda: RadiologyTransform2D(img_size=32).with_elastic_deformation(prob=1.0),
    "3d_affine": lambda: RadiologyTransform3D(img_size=16).with_affine(prob=1.0),
    "3d_elastic": lambda: RadiologyTransform3D(img_size=16).with_elastic_deformation(prob=1.0),
}


@pytest.mark.parametrize("name", list(PICKLE_CASES))
def test_transform_pickles_for_spawn_workers(name, png_2d, nifti_3d):
    """The pickled transform gives the same sample as the original, with the same seed."""
    t = PICKLE_CASES[name]().get_transform()
    copy = pickle.loads(pickle.dumps(t))
    data = {"img": png_2d if name.startswith("2d") else nifti_3d, "cls": [1.0]}
    t.set_random_state(seed=0)
    copy.set_random_state(seed=0)
    assert torch.equal(t(dict(data))["img"], copy(dict(data))["img"])


# V-D-27 (D-42): one channel count for each dataset.
def _png(tmp_path, mode):
    """A 60x40 PNG in PIL ``mode``. Its 3 channels are independent, so a wrong gray weight shows."""
    rng = np.random.default_rng(0)
    rgb = Image.fromarray(rng.integers(0, 256, (60, 40, 3), dtype=np.uint8), "RGB")
    path = tmp_path / f"{mode}.png"
    (rgb if mode == "RGB" else rgb.convert(mode)).save(path)
    return str(path)


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "P"])
def test_a_gray_dataset_gives_1_channel_for_any_png(tmp_path, mode):
    """V-D-27 (D-42): RGB, RGBA, LA and palette PNGs give 1 channel, the gray image of PIL ``convert("L")``."""
    path = _png(tmp_path, mode)
    gray = str(tmp_path / "gray.png")
    Image.open(path).convert("RGB").convert("L").save(gray)
    t = RadiologyTransform2D(img_size=32, dtype=torch.float32).get_transform()
    img = t({"img": path, "cls": [0.0]})["img"]
    assert tuple(img.shape) == (1, 32, 32)
    assert torch.allclose(img, t({"img": gray, "cls": [0.0]})["img"], atol=0.02)


def test_a_gray_image_saved_as_rgba_keeps_its_exact_values(png_2d, tmp_path):
    """V-D-27 (D-42): as the RGBA PNGs of ChestX-ray14 and ChestX-Det10 (R = G = B, alpha 255)."""
    rgba = str(tmp_path / "rgba.png")
    Image.open(png_2d).convert("RGBA").save(rgba)
    t = RadiologyTransform2D(img_size=32, dtype=torch.float32).get_transform()
    assert torch.equal(t({"img": rgba, "cls": [0.0]})["img"], t({"img": png_2d, "cls": [0.0]})["img"])


@pytest.mark.parametrize("channels", [1, 3])
def test_an_rgb_dicom_skips_the_window_in_the_dataset_load(tmp_path, channels):
    """V-P-21 (D-13): the DICOM standard applies a window only to a MONOCHROME image.

    So an RGB DICOM with window tags gives the same sample as the same file with no window tags.
    """
    from synthetic_images import write_dicom

    gray = (np.add.outer(np.arange(30) * 3, np.arange(40) * 2) + 10).astype(np.uint8)
    rgb = np.stack([gray, gray // 2, 255 - gray], axis=-1)
    t = RadiologyTransform2D(img_size=32, channels=channels, dtype=torch.float32).get_transform()
    windowed, plain = (t({"img": write_dicom(tmp_path / f"{n}.dcm", rgb, window=w), "cls": [0.0]})["img"]
                       for n, w in (("windowed", (100, 50)), ("plain", None)))
    assert tuple(windowed.shape) == (channels, 32, 32)
    assert torch.equal(windowed, plain)


def test_the_channel_step_drops_alpha_and_checks_the_count():
    """V-D-27 (D-42): gray + alpha gives the gray channel. ITK drops the alpha of an LA PNG itself."""
    from radharmony.dataset.transforms import _SetChannelsD
    gray = torch.arange(12.0).view(1, 3, 4)
    la = torch.cat([gray, torch.full_like(gray, 255)])
    assert torch.equal(_SetChannelsD(1)({"img": la})["img"], gray)
    assert torch.equal(_SetChannelsD(3)({"img": la})["img"], gray.repeat(3, 1, 1))
    with pytest.raises(ValueError, match="1 or 3"):
        _SetChannelsD(2)


def test_a_color_dataset_repeats_a_gray_png_to_3_channels(png_2d):
    """V-D-27 (D-42): with ``channels=3``, a gray file gives 3 equal channels."""
    t = RadiologyTransform2D(img_size=32, channels=3, dtype=torch.float32).get_transform()
    img = t({"img": png_2d, "cls": [0.0]})["img"]
    assert tuple(img.shape) == (3, 32, 32)
    assert torch.equal(img[0], img[1]) and torch.equal(img[0], img[2])


def test_the_vqa_datasets_set_their_channel_count():
    """V-D-27 (D-42): the datasets of color figures give 3 channels, the CXR VQA datasets give 1."""
    import radharmony.dataset  # noqa: F401  (fills the registry)
    from radharmony.dataset.base_vqa import BaseVQADataset
    from radharmony.registry import list_datasets, resolve_dataset
    color = ("MedPMC", "ROCO", "VQARad")  # ROCOv2 starts with ROCO
    classes = {resolve_dataset(k) for k in list_datasets()}
    vqa = [c for c in classes if isinstance(c, type) and issubclass(c, BaseVQADataset)]
    assert len(vqa) >= 7
    for cls in vqa:
        assert cls.IMG_CHANNELS == (3 if cls.__name__.startswith(color) else 1), cls.__name__


def test_an_error_in_the_3d_box_step_is_not_hidden(monkeypatch):
    """R-code-18 (J-03, D-28): an error stops the run. The box does not keep the old axes while the image turns."""
    from monai.data import MetaTensor

    from radharmony.dataset.transforms import _ReorientBbox

    def broken(self, affine):
        raise RuntimeError("no orientation code")

    monkeypatch.setattr(_ReorientBbox, "_src_orientation_code", broken)
    data = {"img": MetaTensor(torch.zeros(1, 4, 4, 4), affine=torch.eye(4)), "bbox": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]}
    with pytest.raises(RuntimeError, match="no orientation code"):
        _ReorientBbox(axcodes="IPL")(data)


@pytest.mark.parametrize("raw", [[], "[]", "nan", float("nan"), None])
def test_a_3d_sample_with_no_box_gives_an_empty_box(nifti_3d, raw):
    """V-D-41 (D-28): a 3-D row with no box gives a (0, 6) box and no label. It does not raise IndexError."""
    t = RadiologyTransform3D(img_size=16, output_keys={"img", "bbox", "bbox_labels"},
                             dtype=torch.float32).get_transform()
    out = t({"img": nifti_3d, "bbox": raw, "bbox_labels": []})
    assert tuple(out["bbox"].shape) == (0, 6)
    assert out["bbox"].dtype == torch.float32
    assert out["bbox_labels"] == []


def test_collate_keeps_the_vqa_answer_struct_as_a_list():
    """V-D-21: a VQA batch with ``output_struct=True`` collates. Each ``answer_struct`` dict stays as it is."""
    import torch

    from radharmony.dataset import bbox_collate

    batch = [{"img": torch.zeros(1, 4, 4), "question": "q1", "answer_struct": {"yes": True, "no": False}},
             {"img": torch.zeros(1, 4, 4), "question": "q2", "answer_struct": {"regions": ["left lung", "heart"]}}]
    out = bbox_collate(batch)
    assert out["img"].shape == (2, 1, 4, 4) and out["question"] == ["q1", "q2"]
    assert out["answer_struct"] == [batch[0]["answer_struct"], batch[1]["answer_struct"]]
