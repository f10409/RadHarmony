"""Harmonizer for the CT-RATE dataset."""

import os

from ..base import BaseHarmonizer


class CTRATEHarmonizer(BaseHarmonizer):
    """Harmonizer for the CT-RATE chest CT dataset.

    Derives ``patient_id``, ``study_id``, and ``image_path`` from the
    ``VolumeName`` column.  Optionally merges a metadata CSV to add
    ``view_position`` (``ImageOrientationPatient``).

    Expected CSV columns: ``VolumeName`` + predicted label columns.

    Args:
        csv_path: Path to ``train_predicted_labels.csv`` (or equivalent).
        view_position_csv_path: Path to ``train_metadata.csv`` (or equivalent).
    """

    LABEL_COLS = [
        "Arterial wall calcification",
        "Atelectasis",
        "Bronchiectasis",
        "Cardiomegaly",
        "Consolidation",
        "Coronary artery wall calcification",
        "Emphysema",
        "Hiatal hernia",
        "Interlobular septal thickening",
        "Lung nodule",
        "Lung opacity",
        "Lymphadenopathy",
        "Medical material",
        "Mosaic attenuation pattern",
        "Peribronchial thickening",
        "Pericardial effusion",
        "Pleural effusion",
        "Pulmonary fibrotic sequela",
    ]

    VIEW_POSITION_JOIN_COLS = ["VolumeName"]
    VIEW_POSITION_SOURCE_COL = "ImageOrientationPatient"

    def __init__(self, csv_path: str, view_position_csv_path: str = None, base_image_dir: str = None):
        super().__init__(csv_path=csv_path, view_position_csv_path=view_position_csv_path)
        self.base_image_dir = base_image_dir

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["VolumeName"].apply(
            lambda x: "_".join(x.split("_")[:2])
        )

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["VolumeName"].apply(
            lambda x: "_".join(x.split("_")[:3])
        )

    def _build_image_path(self) -> None:
        def _rel(vol_name: str) -> str:
            return os.path.join(
                "_".join(vol_name.split("_")[:2]),
                "_".join(vol_name.split("_")[:3]),
                vol_name,
            )

        self.df["image_path"] = self.df["VolumeName"].apply(_rel)

