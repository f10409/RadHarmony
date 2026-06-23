"""Test-split harmonizer for the SIIM-ACR Pneumothorax dataset."""

import os
from glob import glob

import pandas as pd

from .siim_acr_ptx import SIIMACRPTXHarmonizer


class SIIMACRPTXTestHarmonizer(SIIMACRPTXHarmonizer):
    """Test-split harmonizer for SIIM-ACR PTX (images only, no labels).

    Globs the ``dicom-images-test/`` tree (``patient/study/sop.dcm``) to
    build image paths.  No RLE masks or pneumothorax labels are produced.

    Args:
        dicom_dir: Root of the test DICOM tree
            (typically ``<competition_root>/dicom-images-test/``).
    """

    LABEL_COLS = []
    MASK_SOURCE_COL = None

    def __init__(self, dicom_dir: str, **kwargs):
        super().__init__(csv_path=None, dicom_dir=dicom_dir)

    def harmonize(self, **kwargs) -> pd.DataFrame:
        """Glob the test directory and return one row per image (no labels)."""
        self._harmonized_df_override = None
        paths = glob(os.path.join(self.dicom_dir, "**", "*.dcm"), recursive=True)
        rel_paths = ["/".join(p.split("/")[-3:]) for p in paths]
        rows = [
            {
                "patient_id": rel.split("/")[0],
                "study_id": rel.split("/")[1],
                "image_path": rel,
            }
            for rel in rel_paths
        ]
        self.df = pd.DataFrame(rows)
        return self._select_harmonized_columns(self.df)
