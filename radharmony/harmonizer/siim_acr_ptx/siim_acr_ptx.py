"""Harmonizer for the SIIM-ACR Pneumothorax dataset."""

import os
from glob import glob

import numpy as np
import pandas as pd

from radharmony.utils.mask_utils import rle_to_mask

from .._dicom_tags import add_patient_tags
from ..base import BaseHarmonizer


class SIIMACRPTXHarmonizer(BaseHarmonizer):
    """Harmonizer for the SIIM-ACR Pneumothorax segmentation dataset.

    ``patient_id``, ``study_id`` and ``image_path`` come from a glob of the DICOM tree,
    merged on ``ImageId``. The RLE string (``EncodedPixels``) is kept as ``mask_path``. The
    binary label ``pneumothorax`` comes from it (1 is a mask, 0 is no finding).

    ``train-rle.csv`` has one line for each pneumothorax region. The table has one row for
    each image, and ``mask_path`` joins the RLE strings of its regions with ``|``.

    The CSV columns are ``ImageId`` and `` EncodedPixels`` (with a leading space, which the
    code removes).

    ``sex`` and ``age_years`` come from the DICOM tags ``PatientSex`` and ``PatientAge``. The
    CSV of the release has neither (D-06, D-26). ``PatientAge`` is a plain number of years,
    for example ``"58"``. A few ages are above 130 (for example 413) and give NaN.

    Args:
        csv_path: Path to ``train-rle.csv`` (or a similar file).
        dicom_dir: Root of the DICOM tree, for example ``dicom-images-train/``. The code finds
            all ``*.dcm`` files below it.
    """

    LABEL_COLS = ["pneumothorax"]
    MASK_SOURCE_COL = "EncodedPixels"  # space stripped during _build_image_path
    RLE_SEP = "|"

    # DICOM tags, read by _build_patient_tags() (D-06, D-26).
    AGE_SOURCE_COL = "PatientAge"
    AGE_SOURCE_UNIT = "years"
    SEX_SOURCE_COL = "PatientSex"

    def __init__(self, csv_path: str, dicom_dir: str):
        super().__init__(csv_path=csv_path)
        self.dicom_dir = dicom_dir

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["dicom_dir"] = self.dicom_dir
        return snap

    def verify_images(self, base_image_dir: str = None, drop_missing: bool = True) -> pd.DataFrame:
        """Check the image files. The folder defaults to ``dicom_dir`` (D-04, V-H3-36)."""
        return super().verify_images(base_image_dir or self.dicom_dir, drop_missing)

    def _build_patient_id(self) -> None:
        # The IDs come from the DICOM tree with the paths. The base filter for study_ids= and
        # patient_ids= runs after this step, so build them here (D-02).
        self._build_image_path()

    def _build_study_id(self) -> None:
        pass  # built by _build_patient_id

    def _build_image_path(self) -> None:
        if "image_path" in self.df.columns:
            return  # _build_patient_id did it
        # The column name has a leading space.
        self.df.rename(columns={" EncodedPixels": "EncodedPixels"}, inplace=True)
        self.df["EncodedPixels"] = self.df["EncodedPixels"].str.strip()
        # One line for each region. Join the regions of an image into one row (D-03).
        self.df = self.df.groupby("ImageId", sort=False, as_index=False)["EncodedPixels"].agg(
            self.RLE_SEP.join
        )

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
        self._build_patient_tags()

    def _build_patient_tags(self) -> None:
        """Read ``PatientSex`` and ``PatientAge`` from the DICOM headers (D-06)."""
        self.df = add_patient_tags(self.df, self.dicom_dir)

    def _build_labels(self) -> None:
        self.df["pneumothorax"] = (self.df["EncodedPixels"] != "-1").astype(int)

    def _build_view_position(self) -> None:
        self.df["view_position"] = "Frontal"

    def _decode_mask(self, rle: str, width: int, height: int):
        """Decode a SIIM-ACR RLE string. Return ``None`` for no finding (``-1``).

        A string with ``RLE_SEP`` holds more than one region. The mask is the union.
        """
        if rle.strip() == "-1":
            return None
        parts = [rle_to_mask(p, width=width, height=height) for p in rle.split(self.RLE_SEP)]
        return np.bitwise_or.reduce(parts)

    def preprocess_masks(self, output_dir: str, base_image_dir: str = None, num_cores: int = 1) -> None:
        """Decode RLE masks and save as PNGs, defaulting ``base_image_dir`` to ``dicom_dir``."""
        super().preprocess_masks(
            output_dir=output_dir,
            base_image_dir=base_image_dir or self.dicom_dir,
            num_cores=num_cores,
        )
