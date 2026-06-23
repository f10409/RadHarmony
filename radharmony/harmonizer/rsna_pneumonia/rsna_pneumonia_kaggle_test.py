"""Test-split harmonizer for the RSNA Pneumonia Detection Challenge (Kaggle Stage 2)."""

import os

import pandas as pd

from .rsna_pneumonia_kaggle import RSNAPneumoniaKaggleHarmonizer


class RSNAPneumoniaKaggleTestHarmonizer(RSNAPneumoniaKaggleHarmonizer):
    """Test-split harmonizer for RSNA Pneumonia Kaggle (images only, no labels).

    Lists flat ``<patientId>.dcm`` files in ``base_image_dir``.  No class
    labels or bounding boxes are produced.

    Args:
        base_image_dir: ``stage_2_test_images/`` directory — flat dir of
            ``<patientId>.dcm`` files (e.g.
            ``~/datasets/rsna-pneumonia-detection-challenge/stage_2_test_images/``).
    """

    LABEL_COLS = []

    def __init__(self, base_image_dir: str, **kwargs):
        super().__init__(csv_path=None, base_image_dir=base_image_dir)

    def harmonize(self, **kwargs) -> pd.DataFrame:
        """List the test images directory and return one row per image (no labels)."""
        self._harmonized_df_override = None
        if not self.base_image_dir or not os.path.isdir(self.base_image_dir):
            raise ValueError(
                f"base_image_dir {self.base_image_dir!r} is required and must exist. "
                "Point it at stage_2_test_images/ (e.g. "
                "rsna-pneumonia-detection-challenge/stage_2_test_images/)."
            )
        rows = []
        for fname in sorted(os.listdir(self.base_image_dir)):
            if fname.endswith(".dcm"):
                patient_id = fname[:-4]
                rows.append({
                    "patient_id": patient_id,
                    "study_id": patient_id,
                    "image_path": fname,
                })
        self.df = pd.DataFrame(rows)
        return self._select_harmonized_columns(self.df)
