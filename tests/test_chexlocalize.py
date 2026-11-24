"""CheXlocalize dataset classes: a missing mask file raises an error that names it (D-35). No real data."""
from __future__ import annotations

import pytest

from radharmony.dataset import CheXlocalizeDataset, CheXlocalizeValDataset


@pytest.mark.parametrize("cls,labels", [(CheXlocalizeDataset, "test_labels.csv"),
                                        (CheXlocalizeValDataset, "val_labels.csv")])
def test_output_mask_with_no_mask_file_raises(tmp_path, cls, labels):
    """Before, the class built, and harmonize() raised "No mask_path column found"."""
    (tmp_path / labels).write_text("Path\n")
    with pytest.raises(FileNotFoundError, match=r"gt_segmentations_(test|val)\.json"):
        cls(base_image_dir=str(tmp_path), output_mask=True, mask_output_dir=str(tmp_path / "m"), cache_dir=None)


def test_no_mask_file_is_fine_without_output_mask(tmp_path):
    """D-35: the mask file is needed only when a sample has a mask."""
    (tmp_path / "test_labels.csv").write_text("Path\n")
    ds = CheXlocalizeDataset(base_image_dir=str(tmp_path), cache_dir=None)
    assert ds._mask_json_path is None


def test_a_preset_table_needs_no_mask_file(tmp_path):
    """With harmonized_df, the class reads no mask file, so it does not look for one."""
    import pandas as pd

    df = pd.DataFrame({"patient_id": ["1"], "study_id": ["1"], "image_path": ["a.jpg"], "mask_path": ["m.png"]})
    ds = CheXlocalizeDataset(base_image_dir=str(tmp_path), harmonized_df=df, output_mask=True,
                             mask_output_dir=str(tmp_path / "m"), cache_dir=None)
    assert ds._mask_json_path is None
