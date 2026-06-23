"""Harmonizer for the MIMIC-CXR-JPG dataset (pre-converted JPEG images)."""

import numpy as np
import pandas as pd

from ..base import BaseHarmonizer


class MIMICCXRJPGHarmonizer(BaseHarmonizer):
    """Harmonizer for the MIMIC-CXR-JPG chest X-ray dataset.

    ``patient_id`` and ``study_id`` are taken directly from the CSV.
    ``image_path`` is built from ``patient_id``, ``study_id``, and ``dicom_id``.

    Expected CSV columns: ``patient_id``, ``study_id``, ``dicom_id``, ``ViewPosition``.

    Args:
        csv_path: Path to the MIMIC-CXR metadata CSV (e.g. ``mimic-cxr-2.0.0-metadata.csv``).
        label_csv_path: Path to the CheXpert label CSV (e.g. ``mimic-cxr-2.0.0-chexpert.csv``).
    """

    LABEL_COLS = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "No Finding",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices",
    ]

    LABEL_JOIN_COLS = ["patient_id", "study_id"]

    VIEW_POSITION_SOURCE_COL = "ViewPosition"

    def __init__(
        self, csv_path: str, label_csv_path: str = None, base_image_dir: str = None
    ):
        super().__init__(csv_path=csv_path, label_csv_path=label_csv_path)
        self.base_image_dir = base_image_dir

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        if self.base_image_dir is not None:
            snap["base_image_dir"] = self.base_image_dir
        return snap

    def harmonize(self, *args, drop_uncertain: bool = True, **kwargs):
        self.drop_uncertain = drop_uncertain
        return super().harmonize(*args, **kwargs)

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["subject_id"].astype(str)

    def _build_study_id(self) -> None:
        # study_id column is already present and correctly named
        pass

    def _build_image_path(self) -> None:
        self.df["patient_id"] = self.df["subject_id"].astype(str)
        self.df["study_id"] = self.df["study_id"].astype(str)
        self.df["image_path"] = self.df.apply(
            lambda x: f"p{x['patient_id'][:2]}/p{x['patient_id']}/s{x['study_id']}/{x['dicom_id']}.jpg",
            axis=1,
        )

    def _preprocess_label_df(self, label_df: pd.DataFrame) -> pd.DataFrame:
        label_df = label_df.fillna(0.0)
        label_df = label_df.replace({-1: np.nan})
        if self.drop_uncertain:
            label_df = label_df.dropna()
        label_df["patient_id"] = label_df["subject_id"].astype(str)
        label_df["study_id"] = label_df["study_id"].astype(str)
        return label_df
