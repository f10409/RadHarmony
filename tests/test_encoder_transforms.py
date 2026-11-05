"""Tests for the readers and the encoder transform in ``radharmony/evaluator/transforms.py``.

A stub processor stands in for a Hugging Face processor, so no model download
is needed. The expected reader output comes from D-27 in ``docs/wiki/design.md``.
A reader for an 8-bit processor scales a 16-bit image to 0 to 255 with the
minimum and maximum of that image. An 8-bit image stays as it is.

Run with::

    pytest tests/test_encoder_transforms.py -v
"""

import numpy as np
import pytest
import torch
from PIL import Image

from radharmony.evaluator import EncoderPreprocessTransform, RadiologyEncoderTransform
from synthetic_images import write_dicom


class _StubProcessor:
    """Act as an HF image processor: keep the input and give it back as a (C, H, W) tensor."""

    def __call__(self, images, return_tensors="pt"):
        arr = np.asarray(images, dtype=np.float32)
        if isinstance(images, Image.Image):
            arr = arr.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        return {"pixel_values": torch.from_numpy(arr)[None]}


def _read(path, reader) -> np.ndarray:
    """Return what the reader gives to the processor, as a (C, H, W) float array."""
    proc = _StubProcessor()
    pp = EncoderPreprocessTransform.from_huggingface(keys=["img"], processor=proc, reader=reader)
    return pp({"img": str(path)})["img"].numpy()


def _scaled_to_255(values: np.ndarray) -> np.ndarray:
    """D-27: scale with the minimum and maximum of the image."""
    values = values.astype(np.float64)
    return (values - values.min()) / (values.max() - values.min()) * 255


# 30 x 40 values from 1,000 to 3,675, different in every row and column.
HIGH_BIT = (np.add.outer(np.arange(30) * 25, np.arange(40) * 50) + 1000).astype(np.uint16)


def test_pil_reader_scales_a_16_bit_png(tmp_path):
    """V-P-14: a 16-bit PNG is scaled to 0 to 255, not clipped at 255 (D-27)."""
    path = tmp_path / "cxr16.png"
    Image.fromarray(HIGH_BIT).save(path)
    assert Image.open(path).mode.startswith("I;16")
    out = _read(path, "pil")
    assert out.shape == (3, 30, 40)
    for channel in out:
        np.testing.assert_allclose(channel, _scaled_to_255(HIGH_BIT), atol=0.5)


def test_pil_reader_keeps_an_8_bit_png(tmp_path):
    """D-27: an 8-bit image stays as it is."""
    gray = np.random.default_rng(0).integers(10, 200, size=(30, 40), dtype=np.uint8)
    path = tmp_path / "cxr8.png"
    Image.fromarray(gray).save(path)
    out = _read(path, "pil")
    for channel in out:
        np.testing.assert_array_equal(channel, gray)


def test_monai_reader_scales_a_12_bit_dicom(tmp_path):
    """V-P-9: reader='monai' gives 0 to 255 for a 12-bit DICOM, not the stored values (D-27)."""
    path = write_dicom(tmp_path / "cxr.dcm", HIGH_BIT, bits_stored=12)
    out = _read(path, "monai")
    assert out.shape == (3, 30, 40)
    for channel in out:
        np.testing.assert_allclose(channel, _scaled_to_255(HIGH_BIT), atol=0.5)


@pytest.mark.parametrize("photometric", ["MONOCHROME2", "MONOCHROME1"])
def test_monai_reader_applies_the_dicom_window(tmp_path, photometric):
    """D-13: reader='monai' applies the VOI LUT (the window) of a DICOM file, as the dataset load does."""
    from synthetic_images import dicom_display_values

    stored = np.tile(np.arange(0, 3000, 100, dtype=np.uint16), (4, 1))
    path = write_dicom(tmp_path / "cxr.dcm", stored, photometric=photometric, window=(1500, 1000))
    out = _read(path, "monai")
    for channel in out:
        np.testing.assert_allclose(channel, _scaled_to_255(dicom_display_values(path)), atol=0.5)


def test_monai_reader_keeps_an_8_bit_png(tmp_path):
    """D-27: an 8-bit image stays as it is, also with reader='monai'."""
    gray = np.random.default_rng(0).integers(10, 200, size=(30, 40), dtype=np.uint8)
    path = tmp_path / "cxr8.png"
    Image.fromarray(gray).save(path)
    out = _read(path, "monai")
    for channel in out:
        np.testing.assert_array_equal(channel, gray)


def _recipe_chain():
    """The image loader of the backbone recipes that give uint8 (V-B-12)."""
    import monai.transforms as mt

    from radharmony.evaluator.backbones._utils import (display_255, load_image, squeeze_frame_dim, to_3channel,
                                                       to_uint8)

    return mt.Compose([load_image(), mt.Lambda(squeeze_frame_dim), mt.Transpose(indices=[0, 2, 1]),
                       mt.Lambda(display_255), mt.Lambda(to_3channel), mt.Lambda(to_uint8)])


@pytest.mark.parametrize("photometric", ["MONOCHROME2", "MONOCHROME1"])
def test_recipe_loader_applies_the_dicom_window(tmp_path, photometric):
    """V-B-12 (D-13): the recipe loader applies the VOI window of a DICOM file, as reader='monai' does."""
    from synthetic_images import dicom_display_values

    stored = np.tile(np.arange(0, 3000, 100, dtype=np.uint16), (4, 1))
    path = write_dicom(tmp_path / "cxr.dcm", stored, photometric=photometric, window=(1500, 1000))
    out = _recipe_chain()(str(path)).numpy().astype(np.float64)
    for channel in out:
        np.testing.assert_allclose(channel, _scaled_to_255(dicom_display_values(path)), atol=1.0)


def test_recipe_loader_keeps_an_8_bit_png(tmp_path):
    """V-B-12 (D-27): an 8-bit image stays as it is in the recipe loader, as with reader='pil'."""
    gray = np.random.default_rng(0).integers(10, 200, size=(30, 40), dtype=np.uint8)
    path = tmp_path / "cxr8.png"
    Image.fromarray(gray).save(path)
    for channel in _recipe_chain()(str(path)).numpy():
        np.testing.assert_array_equal(channel, gray)


@pytest.mark.parametrize("loader", ["monai", "recipe"])
@pytest.mark.parametrize("window", [(100, 50), None])
def test_an_rgb_dicom_skips_the_window_and_stays_as_it_is(tmp_path, loader, window):
    """V-P-21, V-P-24 (D-13, D-27): the DICOM standard applies a window only to a MONOCHROME image.

    An 8-bit RGB DICOM with window tags gives its 3 channels as they are, as with no window tags.
    CXRPreprocessor also skips the window here.
    """
    gray = (np.add.outer(np.arange(30) * 3, np.arange(40) * 2) + 10).astype(np.uint8)
    rgb = np.stack([gray, gray // 2, 255 - gray], axis=-1)
    path = write_dicom(tmp_path / "rgb.dcm", rgb, window=window)
    out = _read(path, "monai") if loader == "monai" else _recipe_chain()(str(path)).numpy()
    np.testing.assert_array_equal(out, rgb.transpose(2, 0, 1))


def test_every_recipe_image_loader_has_the_display_step():
    """V-B-12: each recipe image loader runs display_255 right after the transpose, with no min-max step."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "radharmony/evaluator/backbones"
    found = 0
    for path in sorted(root.glob("[a-z]*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)):
                continue
            name = node.targets[0].id
            if not name.startswith("_loader") or "mask" in name or not isinstance(node.value, ast.Call):
                continue
            if not (node.value.args and isinstance(node.value.args[0], ast.List)):
                continue
            steps = [ast.unparse(e) for e in node.value.args[0].elts]
            found += 1
            assert steps[:4] == ["load_image()", "mt.Lambda(squeeze_frame_dim)", "mt.Transpose(indices=[0, 2, 1])",
                                 "mt.Lambda(display_255)"], (path.name, name, steps)
            assert "mt.Lambda(normalize_to_uint8)" not in steps, (path.name, name)
    assert found >= 13  # the image loaders on 2026-11-01


def _constant_pp(value: float, shape=(3, 16, 16)):
    """Return a preprocess step that gives a constant (C, H, W) image for any path."""
    return EncoderPreprocessTransform.from_custom(
        keys=["img"], preprocess=lambda _: torch.full(shape, value)
    )


def test_encoder_transform_keeps_reg():
    """V-P-10: 'reg' in output_keys stays in the sample, as in RadiologyTransform2D."""
    t = RadiologyEncoderTransform(
        preprocess=_constant_pp(0.0), output_keys={"img", "reg"}, dtype=torch.float32
    ).get_transform()
    out = t({"img": "x.png", "reg": [12.5]})
    assert set(out) == {"img", "reg"}
    assert out["reg"].tolist() == [12.5]


def test_encoder_transform_casts_only_the_image():
    """V-B-16: dtype is for the image. A mask id 257, a reg value 1001 and a label 37.3 keep their values."""
    mask = torch.zeros(1, 16, 16)
    mask[:, :, :8] = 257
    mask[:, :, 8:] = 1001
    t = RadiologyEncoderTransform(
        preprocess=_constant_pp(0.5), output_keys={"img", "cls", "reg", "mask"},
        mask_preprocess=lambda _: mask,
    ).get_transform()
    out = t({"img": "x.png", "mask": "m.png", "reg": [1001.0], "cls": [37.3, float("nan")]})
    assert out["img"].dtype == torch.bfloat16
    assert torch.unique(out["mask"]).tolist() == [257.0, 1001.0]
    assert out["reg"].tolist() == [1001.0]
    assert out["cls"][0].item() == pytest.approx(37.3, abs=1e-5)
    assert torch.isnan(out["cls"][1])


def test_encoder_transform_keeps_bbox_findings():
    """'bbox_findings' in output_keys stays in the sample, as in RadiologyTransform2D."""
    t = RadiologyEncoderTransform(
        preprocess=_constant_pp(0.0), output_keys={"img", "bbox_findings"}, dtype=torch.float32
    ).get_transform()
    out = t({"img": "x.png", "bbox_findings": ["nodule"]})
    assert out["bbox_findings"] == ["nodule"]


def test_encoder_transform_mask_needs_mask_preprocess():
    """V-P-10: 'mask' without mask_preprocess is an error (docstring: requires mask_preprocess)."""
    with pytest.raises(ValueError, match="mask_preprocess"):
        RadiologyEncoderTransform(preprocess=_constant_pp(0.0), output_keys={"img", "mask"})


def test_with_affine_fills_the_border_with_the_darkest_value_of_each_channel():
    """V-P-16 (D-37): the border of a shift gets the darkest value of each channel. The rest only moves.

    Each channel is a ramp from left to right, with its own darkest value (ImageNet black per channel).
    A shift of 4 pixels along the columns moves the image to the left, so the last 4 columns are border.
    """
    black = [(0 - m) / s for m, s in zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), strict=True)]
    ramp = torch.linspace(0, 1, 16).expand(16, 16)
    img = torch.stack([b + ramp for b in black])  # (3, 16, 16), channel c runs from black[c] to black[c] + 1
    pp = EncoderPreprocessTransform.from_custom(keys=["img"], preprocess=lambda _: img.clone())
    t = (
        RadiologyEncoderTransform(preprocess=pp, output_keys={"img"}, dtype=torch.float32)
        .with_affine(translate_range=((0.0, 0.0), (4.0, 4.0)), rotate_range=(0.0,), scale_range=(0.0, 0.0),
                     shear_range=(0.0,), prob=1.0)
        .get_transform()
    )
    out = t({"img": "x.png"})["img"]
    for c, b in enumerate(black):
        np.testing.assert_allclose(out[c, :, 12:].numpy(), b, atol=1e-5)  # the border
        np.testing.assert_allclose(out[c, :, :12].numpy(), img[c, :, 4:].numpy(), atol=1e-5)  # the image moved


def test_with_affine_fills_the_border_with_the_encoder_black():
    """V-P-16: the border that a shift adds has the encoder black value, not -1 (gray).

    The image is all black, so any fill rule that gives black passes (for example the
    image minimum, or (0 - mean) / std of the processor).
    """
    black = (0.0 - 0.5307) / 0.2583  # RAD-DINO: (0 - mean) / std
    t = (
        RadiologyEncoderTransform(
            preprocess=_constant_pp(black), output_keys={"img"}, dtype=torch.float32
        )
        .with_affine(translate_range=(5.0, 5.0), rotate_range=(0.0,), scale_range=(0.0, 0.0),
                     shear_range=(0.0,), prob=1.0)
        .get_transform()
    )
    t.set_random_state(seed=0)
    out = t({"img": "x.png"})["img"]
    np.testing.assert_allclose(out.numpy(), black, atol=1e-4)


@pytest.mark.parametrize("boxes, n", [("[]", 0), ("[[0.1, 0.3, 0.2, 0.4]]", 1)])
def test_encoder_transform_box_is_float32_with_4_columns(boxes, n):
    """D-28: the box is a float32 tensor of shape (N, 4), also for N = 0, as with RadiologyTransform2D."""
    with pytest.warns(UserWarning, match="D-28"):
        t = RadiologyEncoderTransform(
            preprocess=_constant_pp(0.0), output_keys={"img", "bbox", "bbox_labels"}
        ).get_transform()  # the default dtype is bfloat16 for the image
    out = t({"img": "x.png", "bbox": boxes, "bbox_labels": "[]" if n == 0 else '["nodule"]'})
    assert out["bbox"].dtype == torch.float32
    assert tuple(out["bbox"].shape) == (n, 4)


def test_monai_reader_window_does_not_wrap_an_integer_dicom(tmp_path):
    """D-13: the DICOM steps work in float. An 8-bit file with a 12-bit VOI LUT (0 to 4080) must not wrap at 255."""
    import pydicom
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence
    from synthetic_images import dicom_display_values

    ramp = np.tile(np.arange(0, 256, 8, dtype=np.uint8), (4, 1))
    path = write_dicom(tmp_path / "cxr8.dcm", ramp.astype(np.uint16), bits_stored=8)
    ds = pydicom.dcmread(path)
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelData = 8, 8, 7, ramp.tobytes()
    lut = Dataset()
    lut.LUTDescriptor = [256, 0, 12]
    lut.add_new(0x00283006, "OW", (np.arange(256) * 16).astype("<u2").tobytes())  # LUTData, 12-bit output
    ds.VOILUTSequence = Sequence([lut])
    ds.save_as(path)
    out = _read(path, "monai")
    for channel in out:
        np.testing.assert_allclose(channel, _scaled_to_255(dicom_display_values(path)), atol=0.5)


def test_encoder_transform_box_labels_are_lists():
    """D-28: bbox_labels and bbox_findings are lists, one item per box, as with RadiologyTransform2D."""
    with pytest.warns(UserWarning, match="D-28"):
        t = RadiologyEncoderTransform(
            preprocess=_constant_pp(0.0), output_keys={"img", "bbox", "bbox_labels", "bbox_findings"}
        ).get_transform()
    out = t({"img": "x.png", "bbox": "[[0.1, 0.3, 0.2, 0.4]]", "bbox_labels": '["nodule"]',
             "bbox_findings": '["calcified"]'})
    assert out["bbox_labels"] == ["nodule"]
    assert out["bbox_findings"] == ["calcified"]



class _CropProcessor(_StubProcessor):
    """Act as an HF processor that center crops, as RAD-DINO (short side 518, crop 518)."""

    do_center_crop = True


class _PadProcessor(_StubProcessor):
    """Act as an HF processor that pads the image to a square."""

    do_pad = True


class _CropPctProcessor(_StubProcessor):
    """Act as ConvNeXt, which crops with ``crop_pct`` and has no ``do_center_crop``."""

    crop_pct = 0.875


class _ResizeProcessor(_StubProcessor):
    """Act as an HF processor that only resizes to a fixed size, as SigLIP (no crop, no pad)."""

    do_center_crop = None
    do_pad = None


class _FullProcessor(_StubProcessor):
    """Act as an ``AutoProcessor``, which holds the image processor and a tokenizer."""

    def __init__(self, image_processor):
        self.image_processor = image_processor


def _box_transform(pp, keys=("img", "bbox")):
    """Build a RadiologyEncoderTransform with the given preprocess and output keys."""
    return RadiologyEncoderTransform(preprocess=pp, output_keys=set(keys))


def _hf(processor):
    return EncoderPreprocessTransform.from_huggingface(keys=["img"], processor=processor)


def _custom(processor):
    return EncoderPreprocessTransform.from_custom(keys=["img"], preprocess=lambda _: torch.zeros(3, 8, 8),
                                                  processor=processor)


@pytest.mark.parametrize(
    "pp",
    [
        _hf(_CropProcessor()),
        _hf(_PadProcessor()),
        _hf(_CropPctProcessor()),
        _hf(_FullProcessor(_CropProcessor())),
        _custom(_CropProcessor()),
        _constant_pp(0.0),
    ],
    ids=["crop", "pad", "crop_pct", "full-processor-crop", "custom-with-crop", "custom-unknown"],
)
def test_encoder_transform_warns_that_the_box_does_not_follow_the_processor(pp):
    """D-28 gap: a processor that crops or pads, or a preprocess with unknown steps, gives a warning for 'bbox'."""
    with pytest.warns(UserWarning, match="D-28"):
        _box_transform(pp)


@pytest.mark.parametrize(
    "pp, keys",
    [
        (_hf(_ResizeProcessor()), ("img", "bbox")),
        (_hf(_FullProcessor(_ResizeProcessor())), ("img", "bbox")),
        (_custom(_ResizeProcessor()), ("img", "bbox")),
        (_hf(_CropProcessor()), ("img", "cls")),
    ],
    ids=["resize-only", "full-processor-resize-only", "custom-with-resize-only", "no-bbox"],
)
def test_encoder_transform_gives_no_box_warning(pp, keys, recwarn):
    """D-28: a processor that only resizes keeps the box fractions, and no 'bbox' needs no warning."""
    _box_transform(pp, keys)
    assert not [w for w in recwarn if "D-28" in str(w.message)]


def test_processor_of_the_box_warning_does_not_change_the_cache_key():
    """D-15: the processor kept for the D-28 warning is private, so the cache description is the same."""
    from radharmony.dataset.base import _describe

    assert "processor" not in str(_describe(_custom(_CropProcessor())))
    assert "processor" not in str(_describe(_hf(_CropProcessor())))


def _inverted_palette_png(path, gray):
    """Write a palette PNG whose index i shows the gray value 255 - i, as half of the Shenzhen PNGs do."""
    img = Image.fromarray(gray, "P")
    img.putpalette([v for i in range(256) for v in (255 - i,) * 3])
    img.save(path)
    return path


GRAY = np.add.outer(np.arange(20) * 5, np.arange(30) * 3).astype(np.uint8)


def test_monai_reader_reads_a_palette_png_as_shown(tmp_path):
    """V-P-19 (D-13): reader='monai' gives the gray values that the palette shows, not the raw index."""
    path = _inverted_palette_png(tmp_path / "pal.png", GRAY)
    shown = np.asarray(Image.open(path).convert("L"), dtype=np.float32)  # what a viewer shows: 255 - GRAY
    out = _read(path, "monai")
    assert out.shape == (3, 20, 30)
    for channel in out:
        np.testing.assert_array_equal(channel, shown)


def test_monai_reader_drops_the_alpha_channel(tmp_path):
    """V-P-19: an RGBA PNG gives 3 channels, the RGB that it shows, not 4."""
    path = tmp_path / "rgba.png"
    Image.fromarray(GRAY).convert("RGBA").save(path)
    out = _read(path, "monai")
    assert out.shape == (3, 20, 30)
    for channel in out:
        np.testing.assert_array_equal(channel, GRAY)


def test_palette_loader_picks_the_reader_by_file_suffix(tmp_path):
    """D-13: only the PIL reader changes. Other files go to the same reader as in ``mt.LoadImage``.

    Example: a ``.dcm`` file that holds a NumPy array is not read with ``np.load``, which can
    run the code in a pickle. A TIFF with 5 pages keeps its 5 pages.
    """
    import monai.transforms as mt

    from radharmony.evaluator.transforms import _LoadImageAsShown

    np.save(tmp_path / "array.npy", GRAY)
    (tmp_path / "array.npy").rename(tmp_path / "array.dcm")
    with pytest.raises(RuntimeError):
        mt.LoadImage(image_only=True)(str(tmp_path / "array.dcm"))
    with pytest.raises(RuntimeError):
        _LoadImageAsShown(image_only=True)(str(tmp_path / "array.dcm"))

    pages = [Image.fromarray(GRAY + i) for i in range(5)]
    pages[0].save(tmp_path / "stack.tif", save_all=True, append_images=pages[1:])
    plain = mt.LoadImage(image_only=True, ensure_channel_first=True)(str(tmp_path / "stack.tif"))
    ours = _LoadImageAsShown(image_only=True, ensure_channel_first=True)(str(tmp_path / "stack.tif"))
    assert ours.shape == plain.shape
    np.testing.assert_array_equal(ours.numpy(), plain.numpy())

    # A reader option, for example reverse_indexing for PIL, is kept.
    Image.fromarray(GRAY).save(tmp_path / "gray.png")
    plain = mt.LoadImage(image_only=True, reverse_indexing=False)(str(tmp_path / "gray.png"))
    ours = _LoadImageAsShown(image_only=True, reverse_indexing=False)(str(tmp_path / "gray.png"))
    assert ours.shape == plain.shape == GRAY.shape


def test_palette_loader_gives_a_new_cache_key(monkeypatch):
    """D-15: a new reader gives a new key, so a cache with the inverted Shenzhen images is not used."""
    import monai.transforms as mt

    from radharmony.dataset.base import _transform_key
    from radharmony.evaluator import transforms
    from radharmony.evaluator.backbones._utils import load_image

    def recipe(first_step):
        loader = mt.Compose([first_step, mt.Transpose(indices=[0, 2, 1])])
        pp = EncoderPreprocessTransform.from_custom(keys=["img"], preprocess=lambda p: loader(str(p)))
        return RadiologyEncoderTransform(preprocess=pp, output_keys={"img"}).get_transform()

    def monai_reader():
        pp = EncoderPreprocessTransform.from_huggingface(keys=["img"], processor=_StubProcessor(), reader="monai")
        return RadiologyEncoderTransform(preprocess=pp, output_keys={"img"}).get_transform()

    new_recipe, new_monai = _transform_key(recipe(load_image())), _transform_key(monai_reader())
    monkeypatch.setattr(transforms, "_LoadImageAsShown", mt.LoadImage)
    assert new_recipe != _transform_key(recipe(mt.LoadImage(image_only=True, ensure_channel_first=True)))
    assert new_monai != _transform_key(monai_reader())


def test_recipe_image_loader_reads_palette_and_alpha_pngs_as_shown(tmp_path):
    """V-B-11 (D-13): the shared recipe image loader reads a palette PNG as shown and drops alpha."""
    from radharmony.evaluator.backbones._utils import load_image

    pal = _inverted_palette_png(tmp_path / "pal.png", GRAY)
    rgba = tmp_path / "rgba.png"
    Image.fromarray(GRAY).convert("RGBA").save(rgba)
    x = load_image()(str(pal))
    np.testing.assert_array_equal(x[0].numpy().T, 255 - GRAY)  # MONAI gives (C, W, H)
    assert load_image()(str(rgba)).shape[0] == 3
