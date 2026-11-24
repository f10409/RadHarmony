"""Tests for the CXR and CT preprocessors, with hand-made images (no NAS data).

The expected DICOM pixels come from the DICOM pipeline in pydicom
(``tests/synthetic_images.py``), not from the preprocessor.

Run with::

    pytest tests/test_preprocessor.py -v
"""

import os
import subprocess
import sys
from pathlib import Path

import monai.transforms as mt
import numpy as np
import pandas as pd
import pydicom
import pytest
import SimpleITK as sitk
from PIL import Image

from radharmony.preprocessor import CTPreprocessor, CXRPreprocessor, RadiologicPreprocessor
from synthetic_images import dicom_display_values, to_unit_range, write_dicom

ROOT = Path(__file__).resolve().parents[1]

# 30 x 40 stored values from 100 to 3,800 (12-bit), different in every row and column.
STORED = (np.add.outer(np.arange(30) * 20, np.arange(40) * 80) + 100).astype(np.uint16)


def _run_unresized(path) -> np.ndarray:
    """Run CXRPreprocessor with an output size equal to the input size (40 px, same spacing)."""
    arr = sitk.GetArrayFromImage(CXRPreprocessor(target_long_side=40)(str(path)))
    assert arr.shape == STORED.shape
    return arr.astype(np.float64)


@pytest.mark.parametrize("window", [None, (1500, 1000)], ids=["no_window", "window"])
def test_monochrome1_dicom_is_inverted_once(tmp_path, window):
    """V-P-1: a MONOCHROME1 DICOM gives the viewer image (bone bright), not a second inversion."""
    path = write_dicom(tmp_path / "mono1.dcm", STORED, photometric="MONOCHROME1", window=window)
    expected = to_unit_range(dicom_display_values(path))
    np.testing.assert_allclose(_run_unresized(path), expected, atol=1e-4)


def test_signed_monochrome1_dicom_with_window(tmp_path):
    """A signed MONOCHROME1 DICOM: GDCM inverts it as -1 - raw, not as 4095 - raw."""
    stored = (np.add.outer(np.arange(30) * 20, np.arange(40) * 40) - 1500).astype(np.int16)
    path = write_dicom(tmp_path / "signed.dcm", stored, photometric="MONOCHROME1", window=(0, 1000), signed=True)
    expected = to_unit_range(dicom_display_values(path))
    np.testing.assert_allclose(_run_unresized(path), expected, atol=1e-4)


@pytest.mark.parametrize("window", [None, (1500, 1000)], ids=["no_window", "window"])
@pytest.mark.parametrize("photometric", ["MONOCHROME1", "MONOCHROME2"])
def test_dicom_without_highbit_matches_viewer(tmp_path, photometric, window):
    """No HighBit tag (about 5% of CheXpert-Plus): GDCM reads a constant image, pydicom does not."""
    path = write_dicom(tmp_path / "no_highbit.dcm", STORED, photometric=photometric, window=window)
    ds = pydicom.dcmread(path)
    del ds.HighBit
    ds.save_as(path)
    expected = to_unit_range(dicom_display_values(path))
    np.testing.assert_allclose(_run_unresized(path), expected, atol=1e-4)


def test_rgb_dicom_with_window_tags(tmp_path):
    """The DICOM standard applies a window only to MONOCHROME images. An RGB file with window tags works."""
    gray = (STORED // 16).astype(np.uint8)
    path = write_dicom(tmp_path / "rgb.dcm", np.stack([gray] * 3, axis=-1), window=(100, 50))
    np.testing.assert_allclose(_run_unresized(path), to_unit_range(gray), atol=1e-4)


@pytest.mark.parametrize("name", ["image.dicom", "image_no_extension"])
@pytest.mark.parametrize("photometric", ["MONOCHROME1", "MONOCHROME2"])
def test_voi_lut_is_applied_for_any_dicom_file_name(tmp_path, name, photometric):
    """V-P-6: the window and the MONOCHROME1 step do not depend on a ``.dcm`` file name."""
    path = write_dicom(tmp_path / name, STORED, photometric=photometric, window=(1500, 1000))
    expected = to_unit_range(dicom_display_values(path))
    np.testing.assert_allclose(_run_unresized(path), expected, atol=1e-4)


def test_mha_copy_of_a_dicom_is_2d_and_not_read_with_pydicom(tmp_path, monkeypatch):
    """D-13: an .mha copy keeps the DICOM tag 0028|0004 and the 3-D shape with 1 slice.

    It is not a DICOM file. So pydicom does not read it, and the result is the DICOM result.
    """
    dcm = write_dicom(tmp_path / "a.dcm", STORED)
    mha = str(tmp_path / "a.mha")
    sitk.WriteImage(sitk.ReadImage(str(dcm)), mha)
    expected = sitk.GetArrayFromImage(CXRPreprocessor()(str(dcm)))

    def no_read(*args, **kwargs):
        raise AssertionError("pydicom read an .mha file")

    monkeypatch.setattr(pydicom, "dcmread", no_read)
    out = CXRPreprocessor()(mha)
    assert out.GetDimension() == 2
    np.testing.assert_allclose(sitk.GetArrayFromImage(out), expected)


def test_monochrome2_dicom_with_rescale_matches_viewer(tmp_path):
    """Control: a MONOCHROME2 DICOM with a Modality LUT and a window."""
    path = write_dicom(tmp_path / "mono2.dcm", STORED, window=(3000, 2000), rescale=(2.0, -50.0))
    expected = to_unit_range(dicom_display_values(path))
    np.testing.assert_allclose(_run_unresized(path), expected, atol=1e-4)


def test_dicom_cxr_gives_2d_image_with_target_long_side(tmp_path):
    """V-P-2: a 400 x 300 DICOM at 1 mm gives a 2-D 112 x 84 image (docstring: long side = target)."""
    stored = np.random.default_rng(0).integers(0, 4096, size=(300, 400)).astype(np.uint16)
    path = write_dicom(tmp_path / "cxr.dcm", stored, pixel_spacing=(1.0, 1.0))
    out = CXRPreprocessor(target_long_side=112)(path)
    assert out.GetDimension() == 2
    assert out.GetSize() == (112, round(300 / 400 * 112))


def test_uniform_image_gives_minus_one_not_nan(tmp_path):
    """V-P-17: a blank image stays in [-1, 1], with the value of the dataset transforms (D-14)."""
    blank = np.zeros((64, 50), np.uint8)
    path = tmp_path / "blank.png"
    Image.fromarray(blank).save(path)
    # D-14: the dataset transforms scale a 2-D image with MONAI. A uniform image gives b_min there.
    expected = mt.ScaleIntensityRangePercentiles(
        lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True
    )(blank[None].astype(np.float32))
    with pytest.warns(UserWarning, match="Uniform image"):
        out = sitk.GetArrayFromImage(CXRPreprocessor(target_long_side=32)(str(path)))
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out, float(np.unique(np.asarray(expected))[0]))


def test_ct_self_test_passes(tmp_path):
    """V-P-18: ``python -m radharmony.preprocessor.ct <volume>`` ends with exit code 0."""
    vol = np.full((20, 64, 64), -1000, np.int16)  # (z, y, x) air
    yy, xx = np.mgrid[:64, :64]
    vol[:, ((yy - 32) ** 2 / 25**2 + (xx - 32) ** 2 / 30**2) < 1] = 40  # soft tissue body
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((1.0, 1.0, 2.5))
    path = str(tmp_path / "phantom.nii.gz")
    sitk.WriteImage(img, path)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    proc = subprocess.run(
        [sys.executable, "-m", "radharmony.preprocessor.ct", path],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "run_batch : passed." in proc.stdout


# --------------------------------------------------------------------------
# Base-code fixes of 2026-10-29 in radharmony/preprocessor/base.py (approved by Frank Li).
# --------------------------------------------------------------------------


def test_resize_isotropic_long_side_is_the_physical_long_side():
    """V-P-11: 100 x 90 px at 0.1 x 0.2 mm is 10 x 18 mm, so y is the long side (docstring)."""
    img = sitk.GetImageFromArray(np.zeros((90, 100), np.float32))
    img.SetSpacing((0.1, 0.2))
    out = RadiologicPreprocessor(target_long_side=112)._resize_isotropic(img)
    assert out.GetSize() == (round(10 / 18 * 112), 112)


def test_crop_3d_keeps_the_object_when_an_axis_is_flipped():
    """V-P-5: the 3-D crop keeps the whole object for a flipped y axis, as for the identity."""
    arr = -np.ones((20, 160, 160), np.float32)
    arr[:, 70:120, 60:100] = 1.0  # the object
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.8, 0.8, 1.0))
    img.SetDirection((1, 0, 0, 0, -1, 0, 0, 0, 1))
    kept = sitk.GetArrayFromImage(CTPreprocessor()._crop_3d(img))
    assert (kept > 0).sum() == (arr > 0).sum()


def test_run_batch_raises_when_an_image_fails(tmp_path):
    """V-P-3: a missing image raises an error. It does not pass in silence."""
    df = pd.DataFrame({"image_path": [str(tmp_path / "missing.png")]})
    with pytest.raises(RuntimeError):
        CXRPreprocessor().run_batch(df, base_output_dir=str(tmp_path / "out"), num_workers=1)


def test_run_batch_writes_one_file_per_image(tmp_path):
    """V-P-4: 3 images with the same file name in 3 folders give 3 output files."""
    rng = np.random.default_rng(0)
    rel_paths = [f"{study}/view1_frontal.png" for study in ("s1", "s2", "s3")]
    for rel in rel_paths:
        (tmp_path / "img" / rel).parent.mkdir(parents=True)
        Image.fromarray(rng.integers(0, 256, (40, 30), dtype=np.uint8)).save(tmp_path / "img" / rel)
    out_dir = tmp_path / "out"
    CXRPreprocessor(target_long_side=16).run_batch(
        pd.DataFrame({"image_path": rel_paths}),
        base_output_dir=str(out_dir), base_image_dir=str(tmp_path / "img"), num_workers=1,
    )
    assert len([f for _, _, files in os.walk(out_dir) for f in files]) == 3


@pytest.mark.parametrize(("image_path", "expected"), [
    ("abc.dicom", "abc.dicom.nii"),  # VinDr-CXR and VinDr-PCXR
    ("train/a.DCM", "train/a.DCM.nii"),
    ("p1/1.2.840.10197", "p1/1.2.840.10197.nii"),  # a DICOM series folder keeps its name
    ("/abs/x.nii.gz", "abs/x.nii.gz.nii"),
])
def test_output_path_keeps_the_folders_and_the_file_ending(image_path, expected):
    """V-P-4, V-P-22: the copy keeps the folders and the file ending of image_path, and adds ``.nii``."""
    from radharmony.preprocessor.base import output_path

    assert output_path(image_path, "/out") == os.path.join("/out", expected)


def test_run_batch_gives_2_files_for_2_images_with_the_same_stem(tmp_path):
    """V-P-22: ``a/1.png`` and ``a/1.jpg`` are 2 images. Before, both wrote ``a/1.nii``."""
    rng = np.random.default_rng(0)
    (tmp_path / "img" / "a").mkdir(parents=True)
    for name in ("1.png", "1.jpg"):
        Image.fromarray(rng.integers(0, 256, (40, 30), dtype=np.uint8)).save(tmp_path / "img" / "a" / name)
    out_dir = tmp_path / "out"
    CXRPreprocessor(target_long_side=16).run_batch(
        pd.DataFrame({"image_path": ["a/1.png", "a/1.jpg"]}),
        base_output_dir=str(out_dir), base_image_dir=str(tmp_path / "img"), num_workers=1,
    )
    assert sorted(f for _, _, files in os.walk(out_dir) for f in files) == ["1.jpg.nii", "1.png.nii"]


@pytest.mark.parametrize("image_path", ["../other/2.png", "a/../../b.png", "a\\..\\..\\b.png"])
def test_a_path_with_dot_dot_raises_and_writes_nothing(tmp_path, image_path):
    """V-P-22: every output stays below the output folder. A ``..`` part raises before any file is written."""
    from radharmony.preprocessor.base import output_path

    with pytest.raises(ValueError, match=r"\.\."):
        output_path(image_path, str(tmp_path / "out"))
    (tmp_path / "other").mkdir()
    Image.fromarray(np.zeros((40, 30), dtype=np.uint8)).save(tmp_path / "other" / "2.png")
    with pytest.raises(ValueError, match=r"\.\."):
        CXRPreprocessor(target_long_side=16).run_batch(
            pd.DataFrame({"image_path": [image_path]}),
            base_output_dir=str(tmp_path / "out" / "run"), base_image_dir=str(tmp_path / "img"), num_workers=1,
        )
    assert not [f for _, _, files in os.walk(tmp_path / "out") for f in files]


@pytest.mark.parametrize("flip_x", [False, True])
def test_resample_keeps_the_image_centered(flip_x):
    """Q-20 (D-43): the output covers the same area as the input, so the content does not move.

    The D-43 example: 1,000 pixels at 1 mm cover -0.5 to 999.5 mm, also at the new spacing.
    """
    import SimpleITK as sitk

    from radharmony.preprocessor import CXRPreprocessor

    arr = np.zeros((1000, 1000), np.float32)
    yy, xx = np.mgrid[0:1000, 0:1000]
    arr += np.exp(-((xx - 300.0) ** 2 + (yy - 600.0) ** 2) / (2 * 40.0 ** 2))  # a blob at x 300, y 600
    img = sitk.GetImageFromArray(arr)
    img.SetDirection((-1.0, 0.0, 0.0, 1.0) if flip_x else (1.0, 0.0, 0.0, 1.0))

    def centre_mm(im):
        a = sitk.GetArrayFromImage(im).astype(np.float64)
        a = np.clip(a, 0, None)
        iy, ix = np.mgrid[0:a.shape[0], 0:a.shape[1]]
        idx = ((ix * a).sum() / a.sum(), (iy * a).sum() / a.sum())
        return np.array(im.TransformContinuousIndexToPhysicalPoint(idx))

    out = CXRPreprocessor(target_long_side=111)._resize_isotropic(img)
    sp = out.GetSpacing()[0]
    first = np.array(out.TransformContinuousIndexToPhysicalPoint((-0.5, -0.5)))
    last = np.array(out.TransformContinuousIndexToPhysicalPoint((out.GetSize()[0] - 0.5, out.GetSize()[1] - 0.5)))
    np.testing.assert_allclose(first, img.TransformContinuousIndexToPhysicalPoint((-0.5, -0.5)), atol=1e-6)
    np.testing.assert_allclose(last, img.TransformContinuousIndexToPhysicalPoint((999.5, 999.5)), atol=1e-6)
    np.testing.assert_allclose(centre_mm(out), centre_mm(img), atol=0.1 * sp)


# --------------------------------------------------------------------------
# CTPreprocessor fixes of 2026-11-03 (V-P-25, V-P-26) and CXRPreprocessor (V-P-27).
# --------------------------------------------------------------------------


def _body_volume() -> sitk.Image:
    """An LPS volume: 100 x 80 x 60 voxels (x, y, z) of 1 mm, an air box with an ellipsoid body."""
    zz, yy, xx = np.mgrid[:60, :80, :100]
    arr = np.full((60, 80, 100), -1000, np.int16)
    arr[((xx - 50) / 40) ** 2 + ((yy - 40) / 30) ** 2 + ((zz - 30) / 25) ** 2 < 1] = 40
    return sitk.GetImageFromArray(arr)


def test_ct_orientation_with_a_bad_code_raises():
    """V-P-25: a bad orientation code is an error, not a silent LPS output."""
    for bad in ("LIX", "LIPS", "LRP", "lip"):
        with pytest.raises(ValueError, match="orientation"):
            CTPreprocessor(orientation=bad)


def test_ct_long_side_is_axial_for_a_sagittal_volume(tmp_path):
    """V-P-26: the axial plane is L-R and A-P, whatever the index axes are."""
    axial = _body_volume()
    sag = sitk.PermuteAxes(axial, [2, 0, 1])  # same body, index axes (S, L, P)
    p_ax, p_sag = str(tmp_path / "ax.nii.gz"), str(tmp_path / "sag.nii.gz")
    sitk.WriteImage(axial, p_ax)
    sitk.WriteImage(sag, p_sag)
    out_ax = CTPreprocessor()(p_ax)
    out_sag = CTPreprocessor()(p_sag)
    # LIP: size = (L-R, S-I, A-P). The longest axial side is 112.
    assert max(out_sag.GetSize()[0], out_sag.GetSize()[2]) == 112
    assert out_sag.GetSize() == out_ax.GetSize()
    np.testing.assert_allclose(sitk.GetArrayFromImage(out_sag), sitk.GetArrayFromImage(out_ax), atol=1e-5)


def test_cxr_reads_a_gray_plus_alpha_png(tmp_path):
    """V-P-27: an LA PNG gives a 2-D gray image, as the MONAI reader does (D-13)."""
    gray = (np.random.default_rng(0).random((60, 80)) * 255).astype(np.uint8)
    path = tmp_path / "la.png"
    Image.fromarray(np.dstack([gray, np.full_like(gray, 255)]), "LA").save(path)
    out = CXRPreprocessor(target_long_side=80)(str(path))
    assert out.GetDimension() == 2 and out.GetSize() == (80, 60)
    got = sitk.GetArrayFromImage(out)
    assert np.corrcoef(got.ravel(), gray.ravel())[0, 1] > 0.999
