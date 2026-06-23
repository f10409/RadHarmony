"""Test-split harmonizer for the RSNA 2022 Cervical Spine Fracture Detection Challenge."""

import pandas as pd

from .rsna_2022_cervical_spine import RSNA2022CervicalSpineHarmonizer


class RSNA2022CervicalSpineTestHarmonizer(RSNA2022CervicalSpineHarmonizer):
    """Test-split harmonizer for RSNA 2022 Cervical Spine.

    Reads ``test.csv`` (columns: ``row_id``, ``StudyInstanceUID``,
    ``prediction_type``), deduplicates to one row per study, and builds
    ``image_path = StudyUID``.  ``LABEL_COLS`` is empty; no segmentation masks.

    Args:
        csv_path: Path to ``test.csv``.
        base_image_dir: Root of the test DICOM tree (typically
            ``<competition_root>/test_images/``).
    """

    LABEL_COLS = []

    def __init__(self, csv_path: str, base_image_dir: str = None, **kwargs):
        # segmentation_dir is not applicable for test split
        super().__init__(csv_path=csv_path, base_image_dir=base_image_dir, segmentation_dir=None)

    def harmonize(self, **kwargs) -> pd.DataFrame:
        """Read ``test.csv``, deduplicate to one row per study, build paths."""
        self._harmonized_df_override = None
        self.df = pd.read_csv(self.csv_path)

        self.df = self.df.drop_duplicates(
            subset=["StudyInstanceUID"], keep="first"
        ).reset_index(drop=True)

        self._build_patient_id()
        self._build_study_id()
        self._build_image_path()

        self.df.dropna(subset=["image_path"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self._select_harmonized_columns(self.df)
