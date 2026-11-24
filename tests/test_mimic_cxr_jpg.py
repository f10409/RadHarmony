"""MIMIC-CXR-JPG dataset classes: a missing label file raises an error (D-35)."""
from __future__ import annotations

import pytest

from radharmony.dataset import MIMICCXRJPGDataset, MIMICCXRJPGTestDataset, MIMICCXRJPGTrainDataset

METADATA = "mimic-cxr-2.0.0-metadata.csv"
CHEXPERT = "mimic-cxr-2.0.0-chexpert.csv"
TEST_LABELS = "mimic-cxr-2.1.0-test-set-labeled.csv"
SPLIT = "mimic-cxr-2.0.0-split.csv"


def _release(tmp_path, *files: str):
    """Make a release folder with the given CSV files and an empty ``files/`` folder."""
    for name in files:
        (tmp_path / name).write_text("dicom_id,subject_id,study_id\n")
    (tmp_path / "files").mkdir()
    return str(tmp_path / "files") + "/"


@pytest.mark.parametrize("cls", [MIMICCXRJPGDataset, MIMICCXRJPGTrainDataset])
def test_output_cls_with_no_label_file_raises(tmp_path, cls):
    """Before, it only warned, and the samples had no cls."""
    base = _release(tmp_path, METADATA, SPLIT)
    with pytest.raises(FileNotFoundError, match="label_csv_path"):
        cls(base_image_dir=base, output_cls=True, cache_dir=None)


def test_no_label_file_is_fine_without_output_cls(tmp_path):
    """D-35: the label file is needed only when a sample has cls."""
    base = _release(tmp_path, METADATA)
    ds = MIMICCXRJPGDataset(base_image_dir=base, cache_dir=None)
    assert ds._harmonizer.label_csv_path is None


def test_labeled_test_class_does_not_take_the_label_file_of_all_splits(tmp_path):
    """Before, with no test label file, the class read mimic-cxr-2.0.0-chexpert.csv (all splits)."""
    base = _release(tmp_path, METADATA, CHEXPERT)
    with pytest.raises(FileNotFoundError, match="label_csv_path"):
        MIMICCXRJPGTestDataset(base_image_dir=base, cache_dir=None)


def test_labeled_test_class_finds_its_own_label_file(tmp_path):
    base = _release(tmp_path, METADATA, CHEXPERT, TEST_LABELS)
    ds = MIMICCXRJPGTestDataset(base_image_dir=base, cache_dir=None)
    assert ds._harmonizer.label_csv_path.endswith(TEST_LABELS)
