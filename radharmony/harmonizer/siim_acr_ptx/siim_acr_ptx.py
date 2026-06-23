"""Harmonizer for the SIIM-ACR Pneumothorax dataset."""

import os
from glob import glob

import pandas as pd

from radharmony.utils.mask_utils import rle_to_mask

from ..base import BaseHarmonizer


class SIIMACRPTXHarmonizer(BaseHarmonizer):
    """Harmonizer for the SIIM-ACR Pneumothorax segmentation dataset.

    ``patient_id``, ``study_id``, and ``image_path`` are derived by globbing
    the DICOM directory tree and merging on ``ImageId``.  The RLE mask string
    (``EncodedPixels``) is kept as ``mask_path``; a binary ``pneumothorax``
    label (1 = mask present, 0 = no finding) is derived from it.

    Expected CSV columns: ``ImageId``, `` EncodedPixels`` (leading space is
    handled automatically).

    Args:
        csv_path: Path to ``train-rle.csv`` (or equivalent).
        dicom_dir: Root directory of the DICOM tree
            (e.g. ``dicom-images-train/``).  All ``*.dcm`` files are
            discovered recursively.
    """

    LABEL_COLS = ["pneumothorax"]
    MASK_SOURCE_COL = "EncodedPixels"  # space stripped during _build_image_path

    def __init__(self, csv_path: str, dicom_dir: str):
        super().__init__(csv_path=csv_path)
        self.dicom_dir = dicom_dir

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["dicom_dir"] = self.dicom_dir
        return snap

    def _build_patient_id(self) -> None:
        pass  # populated in _build_image_path

    def _build_study_id(self) -> None:
        pass  # populated in _build_image_path

    def _build_image_path(self) -> None:
        # Fix leading space in column name and strip whitespace from values
        self.df.rename(columns={" EncodedPixels": "EncodedPixels"}, inplace=True)
        self.df["EncodedPixels"] = self.df["EncodedPixels"].str.strip()

        # Glob all DICOM files and build the path lookup table
        paths = glob(os.path.join(self.dicom_dir, "**", "*.dcm"), recursive=True)
        rel_paths = ["/".join(p.split("/")[-3:]) for p in paths]
        df_paths = pd.DataFrame(
            {
                "ImageId": [os.path.basename(p)[:-4] for p in rel_paths],
                "patient_id": [p.split("/")[0] for p in rel_paths],
                "study_id": [p.split("/")[1] for p in rel_paths],
                "image_path": rel_paths,
            }
        )
        self.df = self.df.merge(df_paths, on="ImageId", how="left")

    def _build_labels(self) -> None:
        self.df["pneumothorax"] = (self.df["EncodedPixels"] != "-1").astype(int)

    def _build_view_position(self) -> None:
        self.df["view_position"] = "Frontal"

    def _decode_mask(self, rle: str, width: int, height: int):
        """Decode a SIIM-ACR RLE string; returns ``None`` for no-finding entries."""
        if rle.strip() == "-1":
            return None
        return rle_to_mask(rle, width=width, height=height)

    def preprocess_masks(self, output_dir: str, base_image_dir: str = None, num_cores: int = 1) -> None:
        """Decode RLE masks and save as PNGs, defaulting ``base_image_dir`` to ``dicom_dir``."""
        super().preprocess_masks(
            output_dir=output_dir,
            base_image_dir=base_image_dir or self.dicom_dir,
            num_cores=num_cores,
        )
