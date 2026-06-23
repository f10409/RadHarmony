"""Test-split harmonizer for the RSNA PE Detection Challenge (2020)."""

import pandas as pd

from .rsna_pe_detection import RSNAPEDetectionHarmonizer


class RSNAPEDetectionTestHarmonizer(RSNAPEDetectionHarmonizer):
    """Test-split harmonizer for RSNA PE Detection.

    Reads ``test.csv`` (columns: ``StudyInstanceUID``, ``SeriesInstanceUID``,
    ``SOPInstanceUID``), collapses per-slice rows to one row per study, and
    builds ``image_path = StudyUID/SeriesUID``.  ``LABEL_COLS`` is empty.

    Args:
        csv_path: Path to ``test.csv``.
        base_image_dir: Root of the test DICOM tree (typically
            ``<competition_root>/test/``).
    """

    LABEL_COLS = []

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
