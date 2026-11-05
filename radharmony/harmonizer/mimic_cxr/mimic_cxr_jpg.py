"""Harmonizer for the MIMIC-CXR-JPG dataset (pre-converted JPEG images)."""

import pandas as pd

from .._labels import chexpert_codes, warn_images_without_labels
from ..base import BaseHarmonizer


class MIMICCXRJPGHarmonizer(BaseHarmonizer):
    """Harmonizer for the MIMIC-CXR-JPG chest X-ray dataset, all splits in one table.

    This class mixes the official splits and is not for benchmarks. Use
    :class:`MIMICCXRJPGTrainHarmonizer`, :class:`MIMICCXRJPGValidateHarmonizer`
    and :class:`MIMICCXRJPGTestHarmonizer`.

    ``patient_id`` and ``study_id`` are taken directly from the CSV.
    ``image_path`` is built from ``patient_id``, ``study_id``, and ``dicom_id``.

    Expected CSV columns: ``patient_id``, ``study_id``, ``dicom_id``, ``ViewPosition``.

    Args:
        csv_path: Path to the MIMIC-CXR metadata CSV (for example ``mimic-cxr-2.0.0-metadata.csv``).
        label_csv_path: Path to the CheXpert label CSV (for example ``mimic-cxr-2.0.0-chexpert.csv``).
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
        self.drop_uncertain, self.not_mentioned = True, "negative"  # set again by harmonize()

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        if self.base_image_dir is not None:
            snap["base_image_dir"] = self.base_image_dir
        return snap

    def harmonize(self, *args, drop_uncertain: bool = True, not_mentioned: str = "negative", **kwargs):
        """Run the base steps with the CheXpert-style label rule (D-05).

        Args:
            drop_uncertain: If ``True`` (the default), drop the rows with an uncertain label (-1).
            not_mentioned: ``"negative"`` (the default) gives 0 for a blank label, ``"nan"`` keeps
                NaN (D-05).
        """
        self.drop_uncertain, self.not_mentioned = drop_uncertain, not_mentioned
        return super().harmonize(*args, **kwargs)

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["subject_id"].astype(str)

    def _build_study_id(self) -> None:
        # study_id column is already present and correctly named
        pass

    def _build_image_path(self) -> None:
        self.df["patient_id"] = self.df["subject_id"].astype(str)
        self.df["study_id"] = self.df["study_id"].astype(str)
        # A list, not DataFrame.apply: apply on an empty table gives a table (V-H1-38).
        self.df["image_path"] = [
            f"p{p[:2]}/p{p}/s{s}/{d}.jpg"
            for p, s, d in zip(self.df["patient_id"], self.df["study_id"], self.df["dicom_id"], strict=True)
        ]

    def _preprocess_label_df(self, label_df: pd.DataFrame) -> pd.DataFrame:
        warn_images_without_labels(self.df, label_df, "MIMIC-CXR-JPG", self.label_csv_path)
        label_df = chexpert_codes(label_df, self.LABEL_COLS, self.not_mentioned, self.drop_uncertain)
        label_df["patient_id"] = label_df["subject_id"].astype(str)
        label_df["study_id"] = label_df["study_id"].astype(str)
        return label_df
