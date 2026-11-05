"""SIIM-ACR Pneumothorax: one row per image, with all mask regions (V-H3-2)."""

import numpy as np
import pandas as pd
import pytest

from radharmony.harmonizer import SIIMACRPTXHarmonizer


def test_regions_of_one_image_make_one_mask(tmp_path):
    """Two RLE lines of one image give one row, and its mask has both regions."""
    for name in ("a", "b"):
        folder = tmp_path / f"study_{name}" / "series"
        folder.mkdir(parents=True)
        (folder / f"{name}.dcm").touch()
    h = SIIMACRPTXHarmonizer(csv_path=None, dicom_dir=str(tmp_path))
    h.df = pd.DataFrame({"ImageId": ["a", "a", "b"], " EncodedPixels": [" 1 2", " 9 3", " -1"]})
    h._build_image_path()
    assert h.df["ImageId"].tolist() == ["a", "b"]
    mask = h._decode_mask(h.df.loc[0, "EncodedPixels"], width=4, height=4)
    # RLE spec: column-major index = col * 4 + row. "1 2" sets 1, 2 and "9 3" sets 9, 10, 11.
    expected = np.zeros((4, 4), dtype=np.uint8)
    for row, col in [(1, 0), (2, 0), (1, 2), (2, 2), (3, 2)]:
        expected[row, col] = 1
    np.testing.assert_array_equal(mask, expected)
    assert h._decode_mask(h.df.loc[1, "EncodedPixels"], width=4, height=4) is None


def _fake_release(tmp_path):
    """2 patients, 3 images (one with 2 RLE lines), in the real patient/study/image tree."""
    for patient, study, image in (("p1", "s1", "a"), ("p1", "s2", "b"), ("p2", "s3", "c")):
        folder = tmp_path / "dicom" / patient / study
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{image}.dcm").touch()
    pd.DataFrame({"ImageId": ["a", "a", "b", "c"], " EncodedPixels": [" 1 2", " 9 3", " -1", " -1"]}).to_csv(
        tmp_path / "train-rle.csv", index=False)
    return SIIMACRPTXHarmonizer(csv_path=str(tmp_path / "train-rle.csv"), dicom_dir=str(tmp_path / "dicom"))


@pytest.mark.parametrize("arg,ids,images", [("study_ids", ["s2"], ["b"]), ("patient_ids", ["p1"], ["a", "b"])])
def test_id_filter_gives_the_rows_of_a_filter_after(tmp_path, arg, ids, images):
    """V-H2-20, V-H3-16 (D-02): the filter gives the same rows as a filter after harmonize()."""
    full = _fake_release(tmp_path).harmonize()
    col = arg[:-1]  # study_id or patient_id
    expected = full[full[col].isin(ids)].reset_index(drop=True)
    got = _fake_release(tmp_path).harmonize(**{arg: ids})
    assert sorted(p.split("/")[-1][:-4] for p in got["image_path"]) == images
    pd.testing.assert_frame_equal(got, expected)
