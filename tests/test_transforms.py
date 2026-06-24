"""Transform-pipeline tests — no real data, synthetic image/volume in tmp_path.

Exercises the deterministic preprocessing of RadiologyTransform2D / 3D
(load → normalise → resize → pad → tensor → select) end to end, plus the
fluent builder contract. Uses a throwaway PNG and NIfTI so it runs anywhere.

Run with::

    .venv/bin/python -m pytest tests/test_transforms.py -v
"""
from __future__ import annotations

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
