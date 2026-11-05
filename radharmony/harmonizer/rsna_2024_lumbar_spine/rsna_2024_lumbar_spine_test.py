"""Test-split harmonizer for the RSNA 2024 Lumbar Spine Degenerative Classification."""

import pandas as pd

from .rsna_2024_lumbar_spine import RSNA2024LumbarSpineHarmonizer


class RSNA2024LumbarSpineTestHarmonizer(RSNA2024LumbarSpineHarmonizer):
    """Test-split harmonizer for RSNA 2024 Lumbar Spine.

    Reads only ``test_series_descriptions.csv``. There is no label CSV and no join. The
    table has no labels. It has only ``image_path``, ``patient_id``, ``study_id``,
    ``series_id`` and ``view_position``.

    Args:
        series_description_csv_path: Path to ``test_series_descriptions.csv``.
        base_image_dir: Root of the test DICOM tree, usually ``<competition_root>/test_images/``.
    """

    LABEL_COLS = []

    def __init__(self, series_description_csv_path: str, base_image_dir: str = None, **kwargs):
        super().__init__(
            csv_path=None,
            series_description_csv_path=series_description_csv_path,
            base_image_dir=base_image_dir,
        )

    def _read_source(self) -> pd.DataFrame:
        """Read ``test_series_descriptions.csv`` (one row per series)."""
        if self.series_description_csv_path is None:
            raise ValueError(
                "series_description_csv_path is required for "
                "RSNA2024LumbarSpineTestHarmonizer (test_series_descriptions.csv)."
            )
        return pd.read_csv(self.series_description_csv_path)
