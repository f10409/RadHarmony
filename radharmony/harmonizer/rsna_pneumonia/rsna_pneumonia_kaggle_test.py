"""Test-split harmonizer for the RSNA Pneumonia Detection Challenge (Kaggle Stage 2)."""

import os

import pandas as pd

from .rsna_pneumonia_kaggle import RSNAPneumoniaKaggleHarmonizer


class RSNAPneumoniaKaggleTestHarmonizer(RSNAPneumoniaKaggleHarmonizer):
    """Test-split harmonizer for RSNA Pneumonia Kaggle (images only, no labels).

    Lists the flat ``<patientId>.dcm`` files in ``base_image_dir``. There are no class labels
    and no boxes. ``sex`` and ``age_years`` come from the DICOM tags, as in the train class.

    Args:
        base_image_dir: The ``stage_2_test_images/`` folder (flat, ``<patientId>.dcm``), for
            example ``~/datasets/rsna-pneumonia-detection-challenge/stage_2_test_images/``.
    """

    LABEL_COLS = []

    def __init__(self, base_image_dir: str, **kwargs):
        super().__init__(csv_path=None, base_image_dir=base_image_dir)

    def _read_source(self) -> pd.DataFrame:
        """List the test images, one row per ``<patientId>.dcm`` (no labels)."""
        if not self.base_image_dir or not os.path.isdir(self.base_image_dir):
            raise ValueError(
                f"base_image_dir {self.base_image_dir!r} is required and must exist. "
                "Point it at stage_2_test_images/ (e.g. "
                "rsna-pneumonia-detection-challenge/stage_2_test_images/)."
            )
        ids = [f[:-4] for f in sorted(os.listdir(self.base_image_dir)) if f.endswith(".dcm")]
        if not ids:
            raise FileNotFoundError(
                f"No .dcm file in {self.base_image_dir}. Point base_image_dir at the "
                "stage_2_test_images/ folder itself (D-35)."
            )
        return pd.DataFrame({"patientId": ids})

    def _build_labels(self) -> None:
        # The test split has no class column.
        pass
