"""RSNA 2022 cervical spine box tests on a tiny fake source folder in ``tmp_path``.

The expected values come from the fake source files, not from the harmonizer.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from test_harmonizer_boxes import _write_slice

_LABELS = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def _write_source(tmp_path):
    """One study ``st1`` of 4 slices of 8 x 8, with one fracture box on slice 2 (x 4, y 2, w 2, h 2)."""
    from pydicom.uid import generate_uid

    series, study = generate_uid(), generate_uid()
    for sn in range(1, 5):
        _write_slice(tmp_path / "img" / "st1" / f"{sn}.dcm", sn, (0.0, 0.0, float(sn)), 100 * sn,
                     series, study)
    pd.DataFrame({"StudyInstanceUID": ["st1"], "x": [4.0], "y": [2.0], "width": [2.0], "height": [2.0],
                  "slice_number": [2]}).to_csv(tmp_path / "boxes.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ["st1"], **{c: [1] for c in _LABELS}}
                 ).to_csv(tmp_path / "train.csv", index=False)


def _harmonizer(tmp_path, **kw):
    from radharmony.harmonizer import RSNA2022CervicalSpineBboxHarmonizer

    return RSNA2022CervicalSpineBboxHarmonizer(
        csv_path=str(tmp_path / "train.csv"), bbox_csv_path=str(tmp_path / "boxes.csv"),
        base_image_dir=str(tmp_path / "img"), num_workers=1, **kw)


def test_a_table_read_back_from_csv_gives_the_boxes(tmp_path):
    """V-H3-34: the wiki block "Load from saved harmonized CSV" with ``output_bbox=True`` works.

    Before, ``bbox_labels`` was a Python list. ``to_csv`` wrote it as ``['fracture']``, which is
    not JSON, so ``ds[0]`` raised. Now the harmonizer writes JSON text, as NODE21 does.
    """
    from radharmony.dataset import RSNA2022CervicalSpineBboxDataset

    _write_source(tmp_path)
    _harmonizer(tmp_path).harmonize().to_csv(tmp_path / "harmonized.csv", index=False)
    ds = RSNA2022CervicalSpineBboxDataset(
        base_image_dir=str(tmp_path / "img"), harmonized_df=pd.read_csv(tmp_path / "harmonized.csv"),
        output_cls=True, output_bbox=True, cache_dir=None).get_datasets()
    sample = ds[0]
    assert tuple(sample["bbox"].shape) == (1, 6)
    assert list(sample["bbox_labels"]) == ["fracture"]


def test_the_in_plane_box_comes_from_the_source_size(tmp_path):
    """V-H3-24: ``bbox_epsilon`` changes nothing, so a value gives a warning.

    The in-plane box is the source box: y 2 to 4 and x 4 to 6 of an 8 x 8 slice.
    """
    _write_source(tmp_path)
    with pytest.warns(UserWarning, match="bbox_epsilon has no effect"):
        h = _harmonizer(tmp_path, bbox_epsilon=0.3)
    box = json.loads(h.harmonize().loc[0, "bbox"])[0]
    assert box[2:] == [0.25, 0.5, 0.5, 0.75]


def test_mask_path_keeps_its_type_for_a_study_with_no_mask(tmp_path):
    """D-02: a study filter gives the same table as the filter after harmonize(), with the same dtypes.

    Before, a filter to studies with no mask gave mask_path the dtype object, not str.
    """
    from radharmony.harmonizer import RSNA2022CervicalSpineHarmonizer

    pd.DataFrame({"StudyInstanceUID": ["s1", "s2"], "patient_overall": [1, 0],
                  **{f"C{i}": [0, 0] for i in range(1, 8)}}).to_csv(tmp_path / "train.csv", index=False)
    seg = tmp_path / "segmentations"
    seg.mkdir()
    (seg / "s1.nii").write_bytes(b"")
    h = RSNA2022CervicalSpineHarmonizer(csv_path=str(tmp_path / "train.csv"), segmentation_dir=str(seg))
    full = h.harmonize()
    part = h.harmonize(study_ids=["s2"])
    assert part["mask_path"].dtype == full["mask_path"].dtype
    assert part["mask_path"].isna().all()


def test_a_box_on_a_slice_with_no_file_is_dropped_with_a_warning(tmp_path):
    """D-07: the harmonizer warns with the number of dropped boxes. Before, it dropped them with no warning.

    The study has slices 1 to 4. A box on slice 9 has no slice file, so it has no place in the volume.
    """
    _write_source(tmp_path)
    boxes = pd.read_csv(tmp_path / "boxes.csv")
    pd.concat([boxes, boxes.assign(slice_number=9)]).to_csv(tmp_path / "boxes.csv", index=False)
    with pytest.warns(UserWarning, match="1 box"):
        df = _harmonizer(tmp_path).harmonize()
    assert len(json.loads(df.loc[0, "bbox"])) == 1
