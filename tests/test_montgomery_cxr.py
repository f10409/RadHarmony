"""Montgomery harmonizer on a small fake tree: masks when ManualMask/ is missing (D-35)."""

import numpy as np
import pytest
from fixtures.stub_preprocessor import StubPreprocessor
from PIL import Image


def _tree(root):
    """1 image in MontgomerySet/CXR_png/. No ManualMask/ folder."""
    img_dir = root / "MontgomerySet" / "CXR_png"
    img_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(img_dir / "MCUCXR_0001_0.png")
    return img_dir


def test_no_manual_mask_folder_gives_no_mask_columns(tmp_path):
    """V-H2-36: without ManualMask/, the table has no mask columns. The images still work."""
    from radharmony.harmonizer import MontgomeryCXRHarmonizer

    df = MontgomeryCXRHarmonizer(base_dir=str(_tree(tmp_path))).harmonize()
    assert df["image_path"].tolist() == ["MCUCXR_0001_0.png"]
    assert not {"mask_path", "mask_path_left", "mask_path_right"} & set(df.columns)


def test_masks_without_manual_mask_folder_name_it(tmp_path):
    """V-H2-36 (D-35): the masks need ManualMask/, so a call for masks raises FileNotFoundError that names it."""
    from radharmony.harmonizer import MontgomeryCXRHarmonizer

    h = MontgomeryCXRHarmonizer(base_dir=str(_tree(tmp_path)))
    with pytest.raises(FileNotFoundError, match="ManualMask"):
        h.harmonize(mask_output_dir=str(tmp_path / "out"))


def test_harmonize_runs_the_preprocessor(tmp_path):
    """V-H2-27 (D-30): harmonize(preprocess_output_dir=..., preprocessor=...) runs the preprocessor.

    The spec is the docstring of BaseHarmonizer.harmonize(): image_path then points to the output files.
    Before 2026-11-03, the Montgomery harmonize() dropped these arguments with no message.
    """
    from radharmony.harmonizer import MontgomeryCXRHarmonizer
    from radharmony.preprocessor.base import output_path

    pre = StubPreprocessor()
    out = str(tmp_path / "out")
    df = MontgomeryCXRHarmonizer(base_dir=str(_tree(tmp_path))).harmonize(
        preprocess_output_dir=out, preprocessor=pre)
    assert pre.calls == 1
    assert df["image_path"].tolist() == [output_path("MCUCXR_0001_0.png", out)]


def test_label_is_tuberculosis(tmp_path):
    """D-05 (2026-11-05): the label name follows the finding. Montgomery and Shenzhen both use ``tuberculosis``."""
    from radharmony.dataset import MontgomeryCXRDataset, ShenzhenCXRDataset
    from radharmony.harmonizer import MontgomeryCXRHarmonizer, ShenzhenCXRHarmonizer

    assert MontgomeryCXRHarmonizer.LABEL_COLS == ShenzhenCXRHarmonizer.LABEL_COLS == ["tuberculosis"]
    assert MontgomeryCXRDataset.LABEL_COLS == ShenzhenCXRDataset.LABEL_COLS == ["tuberculosis"]
    df = MontgomeryCXRHarmonizer(base_dir=str(_tree(tmp_path))).harmonize()
    assert df["tuberculosis"].tolist() == [0]  # the file name ends in _0
    assert "tb" not in df.columns
