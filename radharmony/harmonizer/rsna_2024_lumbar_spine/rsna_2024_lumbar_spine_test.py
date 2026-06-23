"""Test-split harmonizer for the RSNA 2024 Lumbar Spine Degenerative Classification."""

import pandas as pd

from .rsna_2024_lumbar_spine import RSNA2024LumbarSpineHarmonizer


class RSNA2024LumbarSpineTestHarmonizer(RSNA2024LumbarSpineHarmonizer):
    """Test-split harmonizer for RSNA 2024 Lumbar Spine.

    Reads ``test_series_descriptions.csv`` only — no label CSV, no join, no
    severity expansion.  ``LABEL_COLS`` is empty; the harmonized DataFrame
    carries only the structural columns (``image_path``, ``patient_id``,
    ``study_id``, ``series_id``, ``view_position``).

    Args:
        series_description_csv_path: Path to ``test_series_descriptions.csv``.
        base_image_dir: Root of the test DICOM tree (typically
            ``<competition_root>/test_images/``).
    """

    LABEL_COLS = []

    def __init__(self, series_description_csv_path: str, base_image_dir: str = None, **kwargs):
        super().__init__(
            csv_path=None,
            series_description_csv_path=series_description_csv_path,
            base_image_dir=base_image_dir,
        )

    def harmonize(self, **kwargs) -> pd.DataFrame:
        """Read ``test_series_descriptions.csv`` and build structural columns."""
        self._harmonized_df_override = None

        if self.series_description_csv_path is None:
            raise ValueError(
                "series_description_csv_path is required for "
                "RSNA2024LumbarSpineTestHarmonizer (test_series_descriptions.csv)."
            )

        self.df = pd.read_csv(self.series_description_csv_path)
        self._build_patient_id()
        self._build_study_id()
        self._build_series_id()
        self._build_image_path()
        self._build_view_position()
        self._build_mask_path()
        self._build_report()

        self.df.dropna(subset=["image_path"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self._select_harmonized_columns(self.df)
