"""Tests for ``harmonize(preprocess_output_dir=..., mask_output_dir=...)``, with hand-made files.

A small harmonizer reads a CSV with one row per image. Its ``_decode_mask``
reads a box string, so the expected mask comes from the CSV.

Run with::

    pytest tests/test_harmonizer_preprocess.py -v
"""

import os

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
from fixtures.stub_preprocessor import StubPreprocessor
from PIL import Image

from radharmony.harmonizer.base import BaseHarmonizer
from radharmony.harmonizer.ms_cxr import MSCXRHarmonizer
from radharmony.preprocessor import CXRPreprocessor


class _TinyHarmonizer(BaseHarmonizer):
    """CSV columns: ``pid``, ``sid``, ``path`` and ``mask``.

    ``mask`` is ``"y0 y1 x0 x1"`` (a finding), ``"-1"`` (annotated, no finding, as in
    SIIM-ACR) or empty (not annotated, D-33).
    """

    MASK_SOURCE_COL = "mask"

    def _build_patient_id(self):
        self.df["patient_id"] = self.df["pid"]

    def _build_study_id(self):
        self.df["study_id"] = self.df["sid"]

    def _build_image_path(self):
        self.df["image_path"] = self.df["path"]

    def _decode_mask(self, value, width, height):
        if not isinstance(value, str) or value.strip() == "-1":
            return None
        y0, y1, x0, x1 = (int(v) for v in value.split())
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 1
        return mask


def _write_images(img_dir, rel_paths, size=(40, 30)):
    """Write one random 8-bit image (W x H = size) per path."""
    rng = np.random.default_rng(0)
    for rel in rel_paths:
        (img_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, size=(size[1], size[0]), dtype=np.uint8)).save(img_dir / rel)


def _tiny(tmp_path, rel_paths, masks=None, cls=None):
    """Write a CSV with one row per image and return a ``_TinyHarmonizer`` for it."""
    rows = {
        "pid": [f"p{i}" for i in range(len(rel_paths))],
        "sid": [f"s{i}" for i in range(len(rel_paths))],
        "path": rel_paths,
        "mask": masks if masks is not None else ["-1"] * len(rel_paths),
    }
    csv = tmp_path / "tiny.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return (cls or _TinyHarmonizer)(csv_path=str(csv))


def test_preprocess_masks_writes_zero_mask_for_annotated_image_with_no_finding(tmp_path):
    """D-33: an annotated image with no finding gets an all-zero mask (SIIM-ACR ``-1``)."""
    img_dir = tmp_path / "img"
    _write_images(img_dir, ["a.png"])
    h = _tiny(tmp_path, ["a.png"], masks=["-1"])
    df = h.harmonize(base_image_dir=str(img_dir), mask_output_dir=str(tmp_path / "masks"))
    mask = np.asarray(Image.open(df.loc[0, "mask_path"]))
    assert mask.shape == (30, 40) and mask.max() == 0


def test_mask_step_reads_from_the_folder_of_the_harmonizer(tmp_path):
    """D-04: with no ``base_image_dir`` in the call, the mask step reads the image size from ``h.base_image_dir``."""
    img_dir = tmp_path / "img"
    _write_images(img_dir, ["a.png"])
    h = _tiny(tmp_path, ["a.png"], masks=["2 6 3 9"])
    h.base_image_dir = str(img_dir)
    df = h.harmonize(mask_output_dir=str(tmp_path / "masks"))
    mask = np.asarray(Image.open(df.loc[0, "mask_path"]))
    assert mask.shape == (30, 40) and mask[2:6, 3:9].all()


def test_preprocess_masks_skips_image_that_nobody_annotated(tmp_path):
    """V-H2-4, D-33: an empty ``mask_path`` stays empty, and no PNG is written for it."""
    img_dir = tmp_path / "img"
    _write_images(img_dir, ["a.png", "b.png"])
    h = _tiny(tmp_path, ["a.png", "b.png"], masks=["2 6 3 9", None])
    df = h.harmonize(base_image_dir=str(img_dir), mask_output_dir=str(tmp_path / "masks"))
    mask_a = np.asarray(Image.open(df.loc[0, "mask_path"]))
    assert mask_a[2:6, 3:9].all() and mask_a.sum() == mask_a[2:6, 3:9].sum()
    assert pd.isna(df.loc[1, "mask_path"])
    assert not (tmp_path / "masks" / "b.png").exists()


def test_preprocess_gives_each_row_its_own_image(tmp_path):
    """V-H2-18: 3 images named ``view1_frontal.png`` in 3 folders keep 3 different outputs."""
    img_dir = tmp_path / "img"
    rel_paths = [f"{s}/view1_frontal.png" for s in ("s1", "s2", "s3")]
    _write_images(img_dir, rel_paths)
    pp = CXRPreprocessor(target_long_side=16)
    df = _tiny(tmp_path, rel_paths).harmonize(
        preprocess_output_dir=str(tmp_path / "out"), preprocessor=pp, base_image_dir=str(img_dir)
    )
    assert df["image_path"].nunique() == 3
    for rel, out_path in zip(rel_paths, df["image_path"], strict=True):
        expected = sitk.GetArrayFromImage(pp(str(img_dir / rel)))
        np.testing.assert_allclose(sitk.GetArrayFromImage(sitk.ReadImage(out_path)), expected)


def test_preprocess_rewrites_image_path_and_adds_no_column(tmp_path):
    """V-H2-37: preprocess() points ``image_path`` to the output file. It adds no column, and its docstring says so."""
    h = _tiny(tmp_path, ["s1/a.png"])
    h.harmonize()
    columns = list(h.df.columns)
    out = str(tmp_path / "out")
    h.preprocess(out, StubPreprocessor())
    assert list(h.df.columns) == columns
    assert h.df["image_path"].tolist() == [os.path.join(out, "s1", "a.png.nii")]
    assert "preprocessed_path" not in BaseHarmonizer.preprocess.__doc__
    assert "image_path" in BaseHarmonizer.preprocess.__doc__


def test_preprocess_raises_when_images_fail(tmp_path):
    """V-H2-19: a wrong ``base_image_dir`` raises. ``image_path`` does not point at missing files."""
    _write_images(tmp_path / "img", ["a.png"])
    with pytest.raises(RuntimeError):
        _tiny(tmp_path, ["a.png"]).harmonize(
            preprocess_output_dir=str(tmp_path / "out"),
            preprocessor=CXRPreprocessor(target_long_side=16),
            base_image_dir=str(tmp_path / "wrong_dir"),
        )


def test_masks_keep_the_source_size_after_preprocess(tmp_path):
    """V-H2-21: with both output folders, the mask has the size of the source image (40 x 30)."""
    img_dir = tmp_path / "img"
    _write_images(img_dir, ["a.png"])
    df = _tiny(tmp_path, ["a.png"], masks=["2 6 3 9"]).harmonize(
        preprocess_output_dir=str(tmp_path / "out"),
        preprocessor=CXRPreprocessor(target_long_side=16),
        base_image_dir=str(img_dir),
        mask_output_dir=str(tmp_path / "masks"),
    )
    mask = np.asarray(Image.open(df.loc[0, "mask_path"]))
    assert mask.shape == (30, 40)
    assert mask[2:6, 3:9].all()


def test_mask_of_absolute_image_path_goes_into_mask_output_dir(tmp_path):
    """V-H2-21: with an absolute ``image_path``, the mask is a new file inside ``mask_output_dir``."""
    _write_images(tmp_path / "img", ["a.png"])
    abs_path = str(tmp_path / "img" / "a.png")
    mask_dir = tmp_path / "masks"
    df = _tiny(tmp_path, [abs_path], masks=["2 6 3 9"]).harmonize(mask_output_dir=str(mask_dir))
    mask_path = df.loc[0, "mask_path"]
    assert mask_path != abs_path
    assert os.path.commonpath([mask_path, str(mask_dir)]) == str(mask_dir)


# --------------------------------------------------------------------------
# MS-CXR (the base harmonize(), with its own preprocess() and preprocess_masks())
# --------------------------------------------------------------------------


def _ms_cxr(tmp_path, n=2):
    """Write a 2-image MS-CXR CSV (the v1.1.0 columns) and the JPG files under ``files/``."""
    rows, rel_paths = [], []
    for i in range(n):
        rel = f"files/p10/p1000000{i}/s5000000{i}/img{i}.jpg"
        rel_paths.append(rel.removeprefix("files/"))
        rows.append({
            "dicom_id": f"img{i}", "category_name": "Pneumonia", "label_text": "opacity",
            "path": rel, "x": 4, "y": 6, "w": 10, "h": 8,
            "image_width": 40, "image_height": 30, "split": "train",
        })
    csv = tmp_path / "ms_cxr.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    _write_images(tmp_path / "files", rel_paths)
    return MSCXRHarmonizer(csv_path=str(csv), mimic_base_dir=str(tmp_path / "files"))


def test_ms_cxr_harmonize_preprocesses_images(tmp_path):
    """V-H2-26: ``preprocess_output_dir`` writes the images and updates ``image_path``."""
    out_dir = tmp_path / "out"
    df = _ms_cxr(tmp_path).harmonize(
        preprocess_output_dir=str(out_dir), preprocessor=CXRPreprocessor(target_long_side=16)
    )
    assert len(df) == 2
    for p in df["image_path"]:
        assert p.startswith(str(out_dir)) and os.path.isfile(p)


def test_ms_cxr_harmonize_sample_n_works_like_the_base(tmp_path):
    """V-H2-26: ``sample_n`` with ``preprocess_output_dir`` keeps that many rows, as in the base."""
    df = _ms_cxr(tmp_path).harmonize(
        preprocess_output_dir=str(tmp_path / "out"),
        preprocessor=CXRPreprocessor(target_long_side=16),
        sample_n=1,
    )
    assert len(df) == 1


def test_ms_cxr_harmonize_runs_the_preprocessor_once(tmp_path):
    """V-H2-27: the image copies run once."""
    pre = StubPreprocessor()
    _ms_cxr(tmp_path).harmonize(preprocess_output_dir=str(tmp_path / "out"), preprocessor=pre)
    assert pre.calls == 1


def test_ms_cxr_harmonize_rejects_mask_output_dir(tmp_path):
    """V-H2-26: MS-CXR has boxes, not masks. ``mask_output_dir`` is an error, not ignored."""
    with pytest.raises(ValueError, match="mask_output_dir"):
        _ms_cxr(tmp_path).harmonize(mask_output_dir=str(tmp_path / "masks"))


def test_shenzhen_tb_image_with_no_mask_file_is_not_annotated(tmp_path):
    """D-33: a TB-positive Shenzhen image with no finding mask has an empty mask_path, not an all-zero mask.

    CHNCXR_0001_0 is normal: an all-zero mask (annotated, no finding). CHNCXR_0002_1 has a finding mask:
    the mask. CHNCXR_0003_1 is TB-positive with no mask file: nobody drew it, so no mask.
    """
    from radharmony.harmonizer import ShenzhenCXRHarmonizer

    base = tmp_path / "CXR_png"
    (tmp_path / "Annotations-2" / "masks").mkdir(parents=True)
    base.mkdir()
    for stem in ("CHNCXR_0001_0", "CHNCXR_0002_1", "CHNCXR_0003_1"):
        Image.fromarray(np.full((8, 8), 100, dtype=np.uint8)).save(base / f"{stem}.png")
    lesion = np.zeros((8, 8), dtype=np.uint8)
    lesion[2:4, 2:4] = 255
    Image.fromarray(lesion).save(tmp_path / "Annotations-2" / "masks" / "CHNCXR_0002_1_Cavity_1.png")
    h = ShenzhenCXRHarmonizer(base_dir=str(base))
    h.harmonize()
    h.preprocess_masks(str(tmp_path / "out"))
    paths = dict(zip(h.df["image_path"], h.df["mask_path"], strict=True))
    assert not np.array(Image.open(paths["CHNCXR_0001_0.png"])).any()
    assert np.array(Image.open(paths["CHNCXR_0002_1.png"])).sum() == 4 * 255
    assert pd.isna(paths["CHNCXR_0003_1.png"])


def test_shenzhen_masks_with_no_mask_folder_raise(tmp_path):
    """D-35: with no Annotations-2/masks/ folder, the mask step raises and names the folder.

    Before, every TB image got an empty mask_path, so only the normal images stayed, and the only signal was a warning.
    """
    from radharmony.harmonizer import ShenzhenCXRHarmonizer

    base = tmp_path / "CXR_png"
    base.mkdir()
    for stem in ("CHNCXR_0001_0", "CHNCXR_0002_1"):
        Image.fromarray(np.full((8, 8), 100, dtype=np.uint8)).save(base / f"{stem}.png")
    h = ShenzhenCXRHarmonizer(base_dir=str(base))
    h.harmonize()
    with pytest.raises(FileNotFoundError, match="Annotations-2"):
        h.preprocess_masks(str(tmp_path / "out"))


def test_shenzhen_masks_of_normal_images_need_no_mask_folder(tmp_path):
    """A normal image gets an all-zero mask, so a table with no TB image needs no Annotations-2/masks/."""
    from radharmony.harmonizer import ShenzhenCXRHarmonizer

    base = tmp_path / "CXR_png"
    base.mkdir()
    Image.fromarray(np.full((8, 8), 100, dtype=np.uint8)).save(base / "CHNCXR_0001_0.png")
    h = ShenzhenCXRHarmonizer(base_dir=str(base))
    h.harmonize()
    h.preprocess_masks(str(tmp_path / "out"))
    assert not np.array(Image.open(h.df["mask_path"].iloc[0])).any()


# --------------------------------------------------------------------------
# Own harmonize() that never reaches the base one (V-H2-27, D-30)
# --------------------------------------------------------------------------


class _OwnHarmonize(_TinyHarmonizer):
    """Own harmonize() that never reaches the base one, as Montgomery had before D-30. It makes the masks itself."""

    def harmonize(self, *, mask_output_dir=None, base_image_dir=None, **kwargs):
        self._harmonized_df_override = None
        self.df = pd.read_csv(self.csv_path)
        self._build_patient_id()
        self._build_study_id()
        self._build_image_path()
        self._build_mask_path()
        if mask_output_dir is not None:
            self.preprocess_masks(output_dir=mask_output_dir, base_image_dir=base_image_dir)
        return self._select_harmonized_columns(self.df)


class _OwnOverride(_TinyHarmonizer):
    """Own harmonize() that gives its table with ``set_harmonized_df``. ``self.df`` keeps the raw CSV."""

    def harmonize(self, **kwargs):
        self.df = pd.read_csv(self.csv_path)
        table = pd.DataFrame({"patient_id": self.df["pid"], "study_id": self.df["sid"],
                              "image_path": self.df["path"]})
        self.set_harmonized_df(table)
        return self.harmonized_df


def test_own_harmonize_runs_the_preprocessor(tmp_path):
    """V-H2-27: harmonize() makes the image copies, as the base one does. ``harmonized_df`` is the result (D-24)."""
    img_dir = tmp_path / "img"
    rel_paths = ["s1/a.png", "s2/b.png"]
    _write_images(img_dir, rel_paths)
    out = str(tmp_path / "out")
    h = _tiny(tmp_path, rel_paths, cls=_OwnHarmonize)
    df = h.harmonize(preprocess_output_dir=out, preprocessor=CXRPreprocessor(target_long_side=16),
                     base_image_dir=str(img_dir))
    assert df["image_path"].tolist() == [os.path.join(out, "s1", "a.png.nii"),
                                         os.path.join(out, "s2", "b.png.nii")]
    assert all(os.path.isfile(p) for p in df["image_path"])
    pd.testing.assert_frame_equal(df, h.harmonized_df)


def test_own_harmonize_makes_the_masks_before_the_image_copies(tmp_path):
    """V-H2-21, V-H2-27: the mask has the size of the source image (40 x 30), not of the copy."""
    img_dir = tmp_path / "img"
    _write_images(img_dir, ["a.png"])
    df = _tiny(tmp_path, ["a.png"], masks=["2 6 3 9"], cls=_OwnHarmonize).harmonize(
        preprocess_output_dir=str(tmp_path / "out"), preprocessor=CXRPreprocessor(target_long_side=16),
        base_image_dir=str(img_dir), mask_output_dir=str(tmp_path / "masks"))
    mask = np.asarray(Image.open(df.loc[0, "mask_path"]))
    assert mask.shape == (30, 40)
    assert mask[2:6, 3:9].all()
    assert df.loc[0, "image_path"].startswith(str(tmp_path / "out"))


def test_own_harmonize_sample_n_keeps_that_many_rows(tmp_path):
    """V-H2-27: ``sample_n`` with ``preprocess_output_dir`` keeps 1 of 3 rows, as the base harmonize() does."""
    pre = StubPreprocessor()
    h = _tiny(tmp_path, ["a.png", "b.png", "c.png"], cls=_OwnHarmonize)
    df = h.harmonize(preprocess_output_dir=str(tmp_path / "out"), preprocessor=pre, sample_n=1)
    assert pre.calls == 1
    assert len(df) == 1 and len(h.harmonized_df) == 1


def test_own_harmonize_with_only_sample_n_warns(tmp_path):
    """V-H2-27: ``sample_n`` works only with ``preprocess_output_dir``. Alone, it warns and keeps all 3 rows."""
    h = _tiny(tmp_path, ["a.png", "b.png", "c.png"], cls=_OwnHarmonize)
    with pytest.warns(UserWarning, match="sample_n"):
        df = h.harmonize(sample_n=1)
    assert len(df) == 3


class _SuperHarmonize(_TinyHarmonizer):
    """Own harmonize() that calls the base one, as CT-RATE does. The wrapper runs around it."""

    def harmonize(self, *args, **kwargs):
        return super().harmonize(*args, **kwargs)


@pytest.mark.parametrize("cls", [None, _SuperHarmonize])
def test_base_harmonize_with_only_sample_n_warns_once(tmp_path, cls):
    """D-30: the base ``harmonize()`` warns for ``sample_n`` alone, one warning only, and keeps all 3 rows.

    With ``_SuperHarmonize``, the wrapper must not add a second warning.
    """
    h = _tiny(tmp_path, ["a.png", "b.png", "c.png"], cls=cls)
    with pytest.warns(UserWarning, match="sample_n") as record:
        df = h.harmonize(sample_n=1)
    assert len(df) == 3
    assert len([w for w in record if "sample_n" in str(w.message)]) == 1


def test_copy_step_reads_from_the_folder_of_the_harmonizer(tmp_path):
    """D-04: with no ``base_image_dir`` in the call, the copy step reads from ``h.base_image_dir``.

    After the copy, ``image_path`` is the absolute path of the copy. A folder in the call wins.
    """
    img_dir = tmp_path / "img"
    _write_images(img_dir, ["a.png"])
    pre = CXRPreprocessor(target_long_side=16)
    h = _tiny(tmp_path, ["a.png"])
    h.base_image_dir = str(img_dir)
    df = h.harmonize(preprocess_output_dir=str(tmp_path / "out"), preprocessor=pre)
    assert df.loc[0, "image_path"] == str(tmp_path / "out" / "a.png.nii")
    assert os.path.isfile(df.loc[0, "image_path"])

    # A read from the wrong folder raises, so the file proves that the folder of the call was used.
    h.base_image_dir = str(tmp_path / "wrong")
    df = h.harmonize(preprocess_output_dir=str(tmp_path / "out2"), preprocessor=pre, base_image_dir=str(img_dir))
    assert os.path.isfile(df.loc[0, "image_path"])


def test_own_harmonize_default_call_does_not_change(tmp_path, recwarn):
    """V-H2-27: a call with no image-copy argument gives the source paths and no warning."""
    df = _tiny(tmp_path, ["a.png", "b.png"], cls=_OwnHarmonize).harmonize()
    assert df["image_path"].tolist() == ["a.png", "b.png"]
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_own_harmonize_with_a_set_table_copies_the_images_of_that_table(tmp_path):
    """V-H2-27: a class that gives its table with ``set_harmonized_df`` gets the copies in that table."""
    pre = StubPreprocessor()
    out = str(tmp_path / "out")
    h = _tiny(tmp_path, ["a.png", "b.png"], cls=_OwnOverride)
    df = h.harmonize(preprocess_output_dir=out, preprocessor=pre)
    assert pre.calls == 1
    assert df["image_path"].tolist() == [os.path.join(out, "a.png.nii"), os.path.join(out, "b.png.nii")]
    pd.testing.assert_frame_equal(df, h.harmonized_df)


class _GivesIds(_TinyHarmonizer):
    """Own harmonize() that reaches the base one and gives it its own ``study_ids``."""

    def harmonize(self, **kwargs):
        return super().harmonize(study_ids=["s1"], **kwargs)


def test_ids_that_a_subclass_gives_the_base_stay_in_a_call_for_copies(tmp_path):
    """V-H2-27: a call for image copies has no ids, so the base keeps the ids of the subclass (study s1)."""
    pre = StubPreprocessor()
    df = _tiny(tmp_path, ["a.png", "b.png", "c.png"], cls=_GivesIds).harmonize(
        preprocess_output_dir=str(tmp_path / "out"), preprocessor=pre)
    assert df["study_id"].tolist() == ["s1"]
    assert pre.calls == 1
