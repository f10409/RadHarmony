"""Harmonizer for the CT-RATE dataset."""

import json
import os

import numpy as np
import pandas as pd

from ..base import BaseHarmonizer


class CTRATEHarmonizer(BaseHarmonizer):
    """Harmonizer for the CT-RATE chest CT dataset.

    Derives ``patient_id``, ``study_id``, and ``image_path`` from the
    ``VolumeName`` column.  Optionally merges a metadata CSV to add
    ``view_position`` (the plane: "axial", "coronal" or "sagittal"),
    ``image_orientation`` (the ``ImageOrientationPatient`` vector), ``sex`` and
    ``age_years``.

    Expected CSV columns: ``VolumeName`` + predicted label columns.

    Args:
        csv_path: Path to ``train_predicted_labels.csv`` (or equivalent).
        view_position_csv_path: Path to ``train_metadata.csv`` (or equivalent).
    """

    #: view_position is a 3-D plane or series name, not a 2-D view (Q-13, Q-18).
    VIEW_POSITION_NAMES = False

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

    # The metadata CSV (view_position_csv_path) has PatientSex (M or F) and
    # PatientAge as DICOM age strings, for example "049Y" or "018M" (D-06, D-26).
    AGE_SOURCE_COL = "PatientAge"
    AGE_SOURCE_UNIT = "dicom_as"
    SEX_SOURCE_COL = "PatientSex"

    VIEW_POSITION_JOIN_COLS = ["VolumeName"]
    VIEW_POSITION_SOURCE_COL = "ImageOrientationPatient"
    #: The ImageOrientationPatient vector. ``view_position`` holds its plane name (D-01).
    EXTRA_OUTPUT_COLS = ["image_orientation"]

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

    def harmonize(self, *args, **kwargs):
        """Run the base steps, then add the sex and age of the metadata CSV."""
        super().harmonize(*args, **kwargs)
        self._build_demographics()
        return self._select_harmonized_columns(self.df)

    def _build_demographics(self) -> None:
        """Join ``PatientSex`` and ``PatientAge`` of the metadata CSV on ``VolumeName``.

        The base view-position step takes only ``ImageOrientationPatient`` from that CSV.
        """
        if self.view_position_csv_path is None:
            return
        cols = {"VolumeName", self.SEX_SOURCE_COL, self.AGE_SOURCE_COL}
        meta = pd.read_csv(self.view_position_csv_path, usecols=lambda c: c in cols)
        if "VolumeName" in meta.columns and len(meta.columns) > 1:
            self.df = self.df.merge(meta.drop_duplicates("VolumeName"), on="VolumeName", how="left")

    def _build_view_position(self) -> None:
        """Give ``view_position`` as the nearest plane, and keep the vector in ``image_orientation`` (D-01).

        The base step merges ``ImageOrientationPatient`` as ``view_position``.
        Then ``image_orientation`` keeps that vector, and ``view_position``
        becomes "axial", "coronal" or "sagittal" (:func:`_nearest_plane`).
        """
        super()._build_view_position()
        if "view_position" in self.df.columns:
            self.df["image_orientation"] = self.df["view_position"]
            self.df["view_position"] = self.df["image_orientation"].map(_nearest_plane)

    def _preprocess_view_position_df(self, vp_df):
        """Write each ``ImageOrientationPatient`` in one form (V-H1-28).

        The metadata has 2 forms of the axial orientation:
        ``[1, 0, 0, 0, 1, 0]`` and ``[1.000000, 0.000000, ..., -0.000000]``.
        A whole number becomes an int and ``-0`` becomes ``0``. Other numbers
        keep all their digits.
        """
        col = self.VIEW_POSITION_SOURCE_COL
        if col in vp_df.columns:
            vp_df = vp_df.copy()
            vp_df[col] = vp_df[col].map(_one_form)
        return vp_df


def _nearest_plane(value):
    """Return "axial", "coronal" or "sagittal" for an ``ImageOrientationPatient`` string, else NaN.

    The slice normal is row x column. Its largest part along a patient axis
    gives the plane. The x axis (right to left) gives sagittal, the y axis
    (front to back) coronal, and the z axis (feet to head) axial. A tilted
    slice gets the nearest plane.
    """
    try:
        v = np.asarray(json.loads(value), dtype=float).reshape(6)
    except (TypeError, ValueError):  # not a list of 6 numbers
        return np.nan
    normal = np.abs(np.cross(v[:3], v[3:]))
    if not normal.any():
        return np.nan
    return ("sagittal", "coronal", "axial")[int(normal.argmax())]


def _one_form(value):
    """Return an orientation string such as ``"[1, 0, 0, 0, 1, 0]"``, or the value if it is not a list."""
    try:
        nums = [float(x) + 0.0 for x in json.loads(value)]  # + 0.0 turns -0.0 into 0.0
    except (TypeError, ValueError):
        return value
    return "[" + ", ".join(str(int(v)) if v.is_integer() else repr(v) for v in nums) + "]"
