"""Harmonizer for the MIMIC-CXR v2 dataset (DICOM source, supports reports)."""

import os

import pandas as pd
import pydicom
from tqdm import tqdm

from .._labels import chexpert_codes, warn_images_without_labels
from ..base import BaseHarmonizer


class MIMICCXRHarmonizer(BaseHarmonizer):
    """Harmonize MIMIC-CXR v2 into the standard RadHarmony format, all splits in one table.

    This class mixes the official splits and is not for benchmarks. Use
    :class:`MIMICCXRTrainHarmonizer`, :class:`MIMICCXRValidateHarmonizer`
    and :class:`MIMICCXRTestHarmonizer`.

    Reads ``mimic-cxr-2.0.0-metadata.csv`` (or ``cxr-record-list.csv.gz``),
    which must contain a ``path`` column with relative image paths.

    Args:
        csv_path: Path to the metadata CSV (``mimic-cxr-2.0.0-metadata.csv``
            or ``cxr-record-list.csv.gz``).
        label_csv_path: Path to ``mimic-cxr-2.0.0-chexpert.csv`` (optional).
        dicom_base_dir: Root of the DICOM file tree (for example ``.../mimic-cxr/2.1.0/files/``).
            When provided, ``view_position`` is read directly from DICOM headers.
        report_csv_path: Path to ``cxr-study-list.csv.gz`` (optional).
        report_base_dir: Root directory prepended to relative report paths
            (for example ``.../mimic-cxr/2.1.0/files/``).
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

    VIEW_POSITION_SOURCE_COL = None  # populated via DICOM headers, not a CSV column
    VIEW_POSITION_JOIN_COLS = None

    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None

    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None

    REPORT_JOIN_COLS = ["patient_id", "study_id"]
    REPORT_PATH_COL = "path"

    def __init__(
        self,
        csv_path: str,
        label_csv_path: str = None,
        dicom_base_dir: str = None,
        report_csv_path: str = None,
        report_base_dir: str = None,
    ):
        super().__init__(
            csv_path=csv_path,
            label_csv_path=label_csv_path,
            report_csv_path=report_csv_path,
            report_base_dir=report_base_dir,
        )
        self.dicom_base_dir = dicom_base_dir
        self.drop_uncertain, self.not_mentioned = True, "negative"  # set again by harmonize()

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        if self.dicom_base_dir is not None:
            snap["dicom_base_dir"] = self.dicom_base_dir
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
        self.df["study_id"] = self.df["study_id"].astype(str)

    def _build_image_path(self) -> None:
        # Strip leading "files/" so paths are relative to the files/ subtree,
        # consistent with base_image_dir pointing at .../mimic-cxr/2.1.0/files/
        self.df["image_path"] = self.df["path"].str.removeprefix("files/")
        self.df.drop(columns=["path"], inplace=True)

    def _build_view_position(self) -> None:
        if self.dicom_base_dir is None:
            return

        def _read_vp(row):
            fp = os.path.join(
                self.dicom_base_dir, row["image_path"].replace(".jpg", ".dcm")
            )
            try:
                dcm = pydicom.dcmread(fp, stop_before_pixels=True)
                return dcm.ViewPosition
            except Exception:
                return None

        tqdm.pandas(desc="Reading ViewPosition")
        self.df["view_position"] = self.df.progress_apply(_read_vp, axis=1)

    def _preprocess_label_df(self, label_df: pd.DataFrame) -> pd.DataFrame:
        warn_images_without_labels(self.df, label_df, "MIMIC-CXR", self.label_csv_path)
        label_df = chexpert_codes(label_df, self.LABEL_COLS, self.not_mentioned, self.drop_uncertain)
        label_df["patient_id"] = label_df["subject_id"].astype(str)
        label_df["study_id"] = label_df["study_id"].astype(str)
        return label_df

    def _preprocess_report_df(self, report_df: pd.DataFrame) -> pd.DataFrame:
        report_df["patient_id"] = report_df["subject_id"].astype(str)
        report_df["study_id"] = report_df["study_id"].astype(str)
        # Strip leading "files/" to match base_image_dir pointing at .../files/
        report_df["path"] = report_df["path"].str.removeprefix("files/")
        return report_df
