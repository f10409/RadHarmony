"""Test-split harmonizer for the SIIM-ACR Pneumothorax dataset."""

import os
from glob import glob

import pandas as pd

from .siim_acr_ptx import SIIMACRPTXHarmonizer


class SIIMACRPTXTestHarmonizer(SIIMACRPTXHarmonizer):
    """Test-split harmonizer for SIIM-ACR PTX (images only, no labels).

    Globs the ``dicom-images-test/`` tree (``patient/study/sop.dcm``) to
    build image paths.  No RLE masks or pneumothorax labels are produced.
    ``sex`` and ``age_years`` come from the DICOM tags, as in the train class.

    Args:
        dicom_dir: Root of the test DICOM tree
            (typically ``<competition_root>/dicom-images-test/``).
    """

    LABEL_COLS = []
    MASK_SOURCE_COL = None

    def __init__(self, dicom_dir: str, **kwargs):
        super().__init__(csv_path=None, dicom_dir=dicom_dir)

    def _read_source(self) -> pd.DataFrame:
        """Glob the test directory and return one row per image (no labels)."""
        paths = glob(os.path.join(self.dicom_dir, "**", "*.dcm"), recursive=True)
        if not paths:
            raise FileNotFoundError(
                f"No .dcm file in {self.dicom_dir} or its sub folders. Check the path of the test image folder (D-35)."
            )
        rel_paths = ["/".join(p.split("/")[-3:]) for p in paths]
        rows = [
            {
                "patient_id": rel.split("/")[0],
                "study_id": rel.split("/")[1],
                "image_path": rel,
            }
            for rel in rel_paths
        ]
        return pd.DataFrame(rows)

    def _build_patient_id(self) -> None:
        pass  # _read_source() sets it

    def _build_image_path(self) -> None:
        """``_read_source()`` sets ``image_path``. Read the DICOM sex and age tags here.

        This step runs after the id filter, so only the kept rows are read.
        """
        self._build_patient_tags()

    def _build_labels(self) -> None:
        pass  # no labels

    def _build_view_position(self) -> None:
        pass  # no view
