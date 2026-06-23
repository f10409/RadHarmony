"""Harmonizer for the CheXpert dataset."""

import numpy as np
import pandas as pd

from ..base import BaseHarmonizer


class CheXpertHarmonizer(BaseHarmonizer):
    """Harmonizer for the CheXpert chest X-ray dataset.

    ``patient_id`` and ``study_id`` are extracted from the ``Path`` column.
    ``image_path`` is a relative path from the dataset root (prepend
    ``base_image_dir`` at dataset load time).

    Expected CSV columns: ``Path``, ``AP/PA``, and the 14 label columns.

    Args:
        csv_path: Path to the CheXpert metadata CSV (e.g. ``train.csv``).
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

    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = "AP/PA"
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None

    def __init__(self, csv_path: str, base_image_dir: str = None):
        super().__init__(csv_path=csv_path)
        self.base_image_dir = base_image_dir

    def harmonize(self, *args, drop_uncertain: bool = True, **kwargs):
        self.drop_uncertain = drop_uncertain
        return super().harmonize(*args, **kwargs)

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["Path"].apply(lambda x: x.split("/")[2])

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["Path"].apply(
            lambda x: f"{x.split('/')[2]}_{x.split('/')[3]}"
        )

    def _build_image_path(self) -> None:
        self.df["image_path"] = self.df["Path"].apply(
            lambda x: "/".join(x.split("/")[2:])
        )

    def _build_labels(self) -> None:
        for col in self.LABEL_COLS:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0.0).replace({-1: np.nan})
                self.df.rename(
                    columns={col: col.lower().replace(" ", "_")}, inplace=True
                )
        if self.drop_uncertain:
            self.df.dropna(inplace=True)

    def _build_view_position(self) -> None:
        if (
            self.VIEW_POSITION_SOURCE_COL
            and self.VIEW_POSITION_SOURCE_COL in self.df.columns
        ):
            self.df["view_position"] = self.df["AP/PA"].fillna("Lateral")
