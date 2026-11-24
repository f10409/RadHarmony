"""ROCO and ROCOv2 harmonizers on small fake trees (ROCO is not on the NAS, ROCOv2 is restricted)."""

import warnings

import pandas as pd
import pytest


def _rocov2_tree(root):
    """2 test images. The manual concept file lists IMG_A only."""
    (root / "hf" / "data").mkdir(parents=True)
    pd.DataFrame({"image_id": ["IMG_A", "IMG_B"], "caption": ["a", "b"],
                  "cui": [["C0040405"], ["C0024485"]]}).to_parquet(root / "hf" / "data" / "test-0.parquet")
    pd.DataFrame({"ID": ["IMG_A"], "CUIs": ["C0041618"]}).to_csv(root / "test_concepts_manual.csv", index=False)


def test_rocov2_modality_source_is_per_image(tmp_path):
    """V-H2-29: an image with no manual concept row takes the automatic CUIs, so its source is "auto"."""
    from radharmony.harmonizer.rocov2 import ROCOv2Harmonizer

    _rocov2_tree(tmp_path)
    df = ROCOv2Harmonizer(base_dir=str(tmp_path), splits=["test"]).harmonize().set_index("question_id")
    # IMG_A: manual C0041618 (Ultrasonography). IMG_B: automatic C0024485 (Magnetic Resonance Imaging).
    assert df.loc["IMG_A", ["cuis", "modality", "modality_source"]].tolist() == [["C0041618"], "ultrasound", "manual"]
    assert df.loc["IMG_B", ["cuis", "modality", "modality_source"]].tolist() == [["C0024485"], "mri", "auto"]


def _roco_split(root, split, captions=True):
    folder = root / "data" / split / "radiology"
    (folder / "images").mkdir(parents=True)
    if captions:
        (folder / "captions.txt").write_text(f"ROCO_{split}\t CT of the chest.\n")


def test_roco_split_with_no_captions_file_warns_and_names_it(tmp_path):
    """V-H2-45 (D-35): a split folder with no captions.txt gives a warning that names the file."""
    from radharmony.harmonizer import ROCOHarmonizer

    _roco_split(tmp_path, "train")
    _roco_split(tmp_path, "test", captions=False)
    with pytest.warns(UserWarning, match="data/test/radiology/captions.txt"):
        df = ROCOHarmonizer(base_dir=str(tmp_path), splits=["train", "test"]).harmonize()
    assert df["split"].tolist() == ["train"]


def test_roco_with_no_captions_file_at_all_raises(tmp_path):
    """V-H2-45 (D-35): when no folder has captions.txt, it raises FileNotFoundError that names the file."""
    from radharmony.harmonizer import ROCOHarmonizer

    _roco_split(tmp_path, "test", captions=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(FileNotFoundError, match="data/test/radiology/captions.txt"):
            ROCOHarmonizer(base_dir=str(tmp_path), splits=["test"]).harmonize()
