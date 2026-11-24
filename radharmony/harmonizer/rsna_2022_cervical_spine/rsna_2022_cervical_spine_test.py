"""Test-split harmonizer for the RSNA 2022 Cervical Spine Fracture Detection Challenge."""

import pandas as pd

from .rsna_2022_cervical_spine import RSNA2022CervicalSpineHarmonizer


class RSNA2022CervicalSpineTestHarmonizer(RSNA2022CervicalSpineHarmonizer):
    """Test-split harmonizer for RSNA 2022 Cervical Spine.

    Reads ``test.csv`` (columns ``row_id``, ``StudyInstanceUID``, ``prediction_type``) and
    keeps one row per study. ``image_path`` is ``StudyUID``. There are no labels and no masks.

    Args:
        csv_path: Path to ``test.csv``.
        base_image_dir: Root of the test DICOM tree, usually ``<competition_root>/test_images/``.
    """

    LABEL_COLS = []

    def __init__(self, csv_path: str, base_image_dir: str = None, **kwargs):
        # The test split has no masks.
        super().__init__(csv_path=csv_path, base_image_dir=base_image_dir, segmentation_dir=None)

    def _read_source(self) -> pd.DataFrame:
        """Read ``test.csv`` and keep one row per study.

        ``test.csv`` has one line for each prediction type of a study.
        """
        df = pd.read_csv(self.csv_path)
        return df.drop_duplicates(subset=["StudyInstanceUID"], keep="first").reset_index(drop=True)
