"""Test-split harmonizer for the RSNA 2023 Abdominal Trauma Detection Challenge."""

import pandas as pd

from .rsna_abdominal_trauma_2023 import RSNAAbdominalTrauma2023Harmonizer


class RSNAAbdominalTrauma2023TestHarmonizer(RSNAAbdominalTrauma2023Harmonizer):
    """Test-split harmonizer for RSNA 2023 Abdominal Trauma Detection.

    Reads ``test_series_meta.csv`` (columns ``patient_id``, ``series_id``, ``aortic_hu``).
    There is no label CSV and no join. The table has no labels, and ``aortic_hu`` is an extra
    output column.

    Args:
        series_meta_csv_path: Path to ``test_series_meta.csv``.
        base_image_dir: Root of the test DICOM tree, usually ``<competition_root>/test_images/``.
    """

    LABEL_COLS = []
    EXTRA_OUTPUT_COLS = ("aortic_hu",)

    def __init__(self, series_meta_csv_path: str, base_image_dir: str = None, **kwargs):
        # The test split does not use train_2024.csv.
        super().__init__(
            csv_path=None,
            series_meta_csv_path=series_meta_csv_path,
            base_image_dir=base_image_dir,
        )

    def _read_source(self) -> pd.DataFrame:
        """Read ``test_series_meta.csv`` (one row per series)."""
        if self.series_meta_csv_path is None:
            raise ValueError(
                "series_meta_csv_path is required for "
                "RSNAAbdominalTrauma2023TestHarmonizer (test_series_meta.csv)."
            )
        return pd.read_csv(self.series_meta_csv_path)
