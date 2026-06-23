"""Test-split harmonizer for the RSNA 2023 Abdominal Trauma Detection Challenge."""

import os

import pandas as pd

from .rsna_abdominal_trauma_2023 import RSNAAbdominalTrauma2023Harmonizer


class RSNAAbdominalTrauma2023TestHarmonizer(RSNAAbdominalTrauma2023Harmonizer):
    """Test-split harmonizer for RSNA 2023 Abdominal Trauma Detection.

    Reads ``test_series_meta.csv`` (columns: ``patient_id``, ``series_id``,
    ``aortic_hu``) — no patient-level label CSV, no join.  ``LABEL_COLS`` is
    empty; ``aortic_hu`` is preserved as an extra output column.

    Args:
        series_meta_csv_path: Path to ``test_series_meta.csv``.
        base_image_dir: Root of the test DICOM tree (typically
            ``<competition_root>/test_images/``).
    """

    LABEL_COLS = []
    EXTRA_OUTPUT_COLS = ("aortic_hu",)

    def __init__(self, series_meta_csv_path: str, base_image_dir: str = None, **kwargs):
        # csv_path (train_2024.csv) is not needed for test split
        super().__init__(
            csv_path=None,
            series_meta_csv_path=series_meta_csv_path,
            base_image_dir=base_image_dir,
        )

    def harmonize(self, **kwargs) -> pd.DataFrame:
        """Read ``test_series_meta.csv`` and build structural columns."""
        self._harmonized_df_override = None

        if self.series_meta_csv_path is None:
            raise ValueError(
                "series_meta_csv_path is required for "
                "RSNAAbdominalTrauma2023TestHarmonizer (test_series_meta.csv)."
            )

        self.df = pd.read_csv(self.series_meta_csv_path)
        self._build_patient_id()
        self._build_study_id()
        self._build_series_id()
        self._build_image_path()
        self._build_view_position()
        self._build_mask_path()
        self._build_bbox()
        self._build_report()

        self.df.dropna(subset=["image_path"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self._select_harmonized_columns(self.df)
