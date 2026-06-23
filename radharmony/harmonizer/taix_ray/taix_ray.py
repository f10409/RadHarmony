"""Harmonizer for the TAIX-Ray bedside chest radiography dataset.

TAIX-Ray ships a single ``annotation.csv`` (produced by the download notebook
from the upstream HuggingFace parquet shards) plus a flat ``images/<UID>.png``
tree.  Labels are eight ordinal severity grades, all stored as raw integers
in this layer; the dataset class binarises them when ``label_mode="binary"``.

Source CSV columns:
    UID, Fold, Split, PatientID, PhysicianID, StudyDate, Age, Sex,
    HeartSize, PulmonaryCongestion,
    PleuralEffusion_Right, PleuralEffusion_Left,
    PulmonaryOpacities_Right, PulmonaryOpacities_Left,
    Atelectasis_Right, Atelectasis_Left

Severity encoding (per Truhn et al. 2026):
    HeartSize        : 0=Normal, 1=Borderline Enlarged, 2=Enlarged, 3=Massively Enlarged
    All other labels : 0=None, 1=Questionable, 2=Mild, 3=Moderate, 4=Severe
"""

import pandas as pd

from ..base import BaseHarmonizer


# Raw CSV → snake_case rename applied during _build_labels.
_RENAME_MAP = {
    "HeartSize": "heart_size",
    "PulmonaryCongestion": "pulmonary_congestion",
    "PleuralEffusion_Left": "pleural_effusion_left",
    "PleuralEffusion_Right": "pleural_effusion_right",
    "PulmonaryOpacities_Left": "pulmonary_opacities_left",
    "PulmonaryOpacities_Right": "pulmonary_opacities_right",
    "Atelectasis_Left": "atelectasis_left",
    "Atelectasis_Right": "atelectasis_right",
    "Sex": "sex",
    "Age": "age",
    "StudyDate": "study_date",
    "Fold": "fold",
    "Split": "split",
    "PhysicianID": "physician_id",
}


class TAIXRayHarmonizer(BaseHarmonizer):
    """Harmonize TAIX-Ray into the standard RadHarmony format.

    Args:
        csv_path: Path to ``annotation.csv`` (produced by
            ``notebooks/download_huggingface.ipynb``).
    """

    # snake_case names — already match what _build_labels will leave in self.df
    LABEL_COLS = [
        "atelectasis_left",
        "atelectasis_right",
        "heart_size",
        "pleural_effusion_left",
        "pleural_effusion_right",
        "pulmonary_congestion",
        "pulmonary_opacities_left",
        "pulmonary_opacities_right",
    ]

    # Carry per-radiograph metadata through to the harmonized output. Includes
    # `split` because the dataset's predefined train/val/test slicing reads it.
    EXTRA_OUTPUT_COLS = ["sex", "age", "study_date", "fold", "split", "physician_id"]

    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    def __init__(self, csv_path: str, base_image_dir: str = None):
        super().__init__(csv_path=csv_path)
        self.base_image_dir = base_image_dir

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["PatientID"].astype(str)

    def _build_study_id(self) -> None:
        # Each radiograph is its own "study" in this dataset (one image per UID,
        # no multi-view grouping). UID is the natural study identifier.
        self.df["study_id"] = self.df["UID"].astype(str)

    def _build_image_path(self) -> None:
        self.df["image_path"] = self.df["UID"].astype(str) + ".png"

    def _build_labels(self) -> None:
        # Rename raw camelCase columns to snake_case so they match LABEL_COLS
        # and survive _select_harmonized_columns via EXTRA_OUTPUT_COLS / split.
        self.df.rename(columns=_RENAME_MAP, inplace=True)
        # Store raw severity integers as float32 so they fit cleanly into the
        # MONAI cls tensor. The dataset class applies the binary/ordinal toggle.
        for col in self.LABEL_COLS:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("float32")
