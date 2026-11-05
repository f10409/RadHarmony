"""MR-RATE harmonizer on a small fake tree (the real MR-RATE data is restricted)."""

import pandas as pd
import pytest
from fixtures.stub_preprocessor import StubPreprocessor


def _tree(root):
    """1 study with 1 T1w series in the train split, a report and 1 pathology label."""
    meta = root / "meta"
    (meta / "metadata").mkdir(parents=True)
    (meta / "reports").mkdir()
    (meta / "pathology_labels").mkdir()
    pd.DataFrame({"patient_uid": ["P1"], "study_uid": ["S1"], "batch_id": ["batch00"],
                  "split": ["train"]}).to_csv(meta / "splits.csv", index=False)
    pd.DataFrame({"patient_uid": ["P1"], "study_uid": ["S1"], "series_id": ["t1w-raw-sag"],
                  "classified_modality": ["T1w"], "is_center_modality": [True]}).to_csv(
        meta / "metadata" / "batch00_metadata.csv", index=False)
    pd.DataFrame({"study_uid": ["S1"], "report": ["No acute finding."], "findings": ["None."],
                  "impression": ["Normal."]}).to_csv(meta / "reports" / "batch00_reports.csv", index=False)
    pd.DataFrame({"study_uid": ["S1"], "Tumor": [0]}).to_csv(
        meta / "pathology_labels" / "mrrate_labels.csv", index=False)


@pytest.mark.parametrize("kwargs", [
    {"modalities": ["MRA"]},
    {"splits": ["test"]},
    {"modalities": ["MRA"], "require_images": True},
    {"modalities": ["MRA"], "require_images": True, "with_masks": False},
])
def test_a_filter_with_no_series_gives_an_empty_harmonized_table(tmp_path, kwargs):
    """V-H2-30 (D-01, D-24): a filter that keeps no series gives 0 rows with the harmonized columns."""
    from radharmony.harmonizer.mrrate import MRRATEHarmonizer

    _tree(tmp_path)
    # The same columns as a run with no filter. The fake tree has no image files, so no require_images here.
    full = MRRATEHarmonizer(base_dir=str(tmp_path), with_masks=kwargs.get("with_masks", True)).harmonize()
    assert len(full) == 1
    h = MRRATEHarmonizer(base_dir=str(tmp_path), **kwargs)
    df = h.harmonize()
    assert len(df) == 0
    assert df.columns.tolist() == full.columns.tolist()
    pd.testing.assert_frame_equal(df, h.harmonized_df)


def test_mask_output_dir_raises(tmp_path):
    """D-30: the masks are NIfTI files, so the base mask step would write empty PNGs. A call for it raises."""
    from radharmony.harmonizer.mrrate import MRRATEHarmonizer

    _tree(tmp_path)
    with pytest.raises(ValueError, match="mask_output_dir"):
        MRRATEHarmonizer(base_dir=str(tmp_path)).harmonize(mask_output_dir=str(tmp_path / "out"))


@pytest.mark.parametrize("image_dir", [None, "imgs"])
def test_copy_step_reads_from_the_image_root(tmp_path, image_dir):
    """D-04: the copy step reads ``image_path`` (``mri/...``) below ``base_image_dir``, else ``base_dir``."""
    from radharmony.harmonizer.mrrate import MRRATEHarmonizer

    _tree(tmp_path)
    root = str(tmp_path / image_dir) if image_dir else None
    pre = StubPreprocessor()
    MRRATEHarmonizer(base_dir=str(tmp_path), base_image_dir=root).harmonize(
        preprocess_output_dir=str(tmp_path / "out"), preprocessor=pre)
    assert pre.folders == [root or str(tmp_path)]
