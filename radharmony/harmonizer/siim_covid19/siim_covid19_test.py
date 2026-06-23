"""Test-split harmonizer for the SIIM COVID-19 Detection Challenge."""

import pandas as pd

from .siim_covid19 import SIIMCOVID19Harmonizer


class SIIMCOVID19TestHarmonizer(SIIMCOVID19Harmonizer):
    """Test-split harmonizer for SIIM COVID-19 (images only, no labels).

    Walks the ``test/`` directory tree (``study/series/sop.dcm``) instead of
    reading the train CSVs.  No class labels or bounding boxes are produced.

    Args:
        base_image_dir: Root of the test DICOM tree
            (typically ``<competition_root>/test/``).
    """

    LABEL_COLS = []

    def __init__(self, base_image_dir: str, **kwargs):
        super().__init__(
            csv_path=None,
            image_csv_path=None,
            base_image_dir=base_image_dir,
        )

    def harmonize(self, **kwargs) -> pd.DataFrame:
        """Walk the test directory and return one row per image (no labels)."""
        self._harmonized_df_override = None
        sop_to_path = self._build_sop_to_path()
        rows = []
        for rel_path in sop_to_path.values():
            study = rel_path.split("/")[0]
            rows.append({"patient_id": study, "study_id": study, "image_path": rel_path})
        self.df = pd.DataFrame(rows)
        return self._select_harmonized_columns(self.df)
