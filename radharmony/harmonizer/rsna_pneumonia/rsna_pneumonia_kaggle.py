"""Harmonizer for the RSNA Pneumonia Detection Challenge dataset (Kaggle Stage 2).

The Kaggle release has two flat CSVs (``stage_2_detailed_class_info.csv`` and
``stage_2_train_labels.csv``) and a flat folder of 1024x1024 DICOM files
(``stage_2_train_images/<patientId>.dcm``). ``class`` has 3 values: "Normal",
"No Lung Opacity / Not Normal" and "Lung Opacity". An image with ``Target=1`` has one or
more pixel boxes. The harmonizer converts them to fractions
``[dim0_min, dim0_max, dim1_min, dim1_max]``.

``sex`` and ``age_years`` come from the DICOM tags ``PatientSex`` and ``PatientAge``
(D-06, D-26). The CSVs have no sex and no age. ``PatientAge`` is a plain number of years,
for example ``"58"``. A few ages are above 130, for example 155, and give NaN.
``view_position`` comes from the DICOM tag ``ViewPosition`` (PA or AP, D-01, Q-18). The
harmonizer reads the headers only if ``base_image_dir`` is set.
"""

import os

import pandas as pd

from .._bbox import clip_boxes
from .._dicom_tags import add_patient_tags
from ..base import BaseHarmonizer


class RSNAPneumoniaKaggleHarmonizer(BaseHarmonizer):
    """Harmonize RSNA Pneumonia Detection Challenge (Kaggle Stage 2) into RadHarmony format.

    Args:
        csv_path: Path to ``stage_2_detailed_class_info.csv``.
        base_image_dir: The ``stage_2_train_images/`` folder (flat, ``<patientId>.dcm``), for
            example ``~/datasets/rsna-pneumonia-detection-challenge/stage_2_train_images/``.
        bbox_csv_path: Path to ``stage_2_train_labels.csv``. Needed for boxes. Without it,
            the table has only the class labels.
    """

    LABEL_COLS = [
        "Lung Opacity",
        "No Lung Opacity / Not Normal",
        "Normal",
    ]

    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = "ViewPosition"  # the DICOM tag (D-01, Q-18)
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    # DICOM tags, read by _build_patient_tags() (D-06, D-26).
    AGE_SOURCE_COL = "PatientAge"
    AGE_SOURCE_UNIT = "years"
    SEX_SOURCE_COL = "PatientSex"

    # All images are 1024x1024.
    _IMG_SIZE = 1024

    def __init__(
        self,
        csv_path: str,
        base_image_dir: str = None,
        bbox_csv_path: str = None,
    ):
        super().__init__(csv_path=os.path.expanduser(csv_path) if csv_path else csv_path)
        self.base_image_dir = (
            os.path.expanduser(base_image_dir) if base_image_dir else base_image_dir
        )
        self.bbox_csv_path = (
            os.path.expanduser(bbox_csv_path) if bbox_csv_path else bbox_csv_path
        )

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["base_image_dir"] = self.base_image_dir
        snap["bbox_csv_path"] = self.bbox_csv_path
        return snap

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["patientId"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["patientId"].astype(str)

    def _build_image_path(self) -> None:
        self.df["image_path"] = self.df["patientId"].astype(str) + ".dcm"

    def _build_patient_tags(self) -> None:
        """Read ``PatientSex``, ``PatientAge`` and ``ViewPosition`` from the DICOM headers (D-06, Q-18)."""
        if self.base_image_dir:
            self.df = add_patient_tags(self.df, self.base_image_dir, extra_tags=(self.VIEW_POSITION_SOURCE_COL,))

    def _build_view_position(self) -> None:
        # The DICOM headers give ViewPosition, sex and age. Read them after the id filter.
        self._build_patient_tags()
        super()._build_view_position()

    def _build_labels(self) -> None:
        # One 0/1 column for each value of ``class``.
        for orig, snake in [
            ("Lung Opacity",                 "lung_opacity"),
            ("No Lung Opacity / Not Normal", "no_lung_opacity_/_not_normal"),
            ("Normal",                       "normal"),
        ]:
            self.df[snake] = (self.df["class"] == orig).astype(int)

    def _build_bbox(self) -> None:
        if self.bbox_csv_path is None:
            return

        bbox_df = pd.read_csv(self.bbox_csv_path)
        # Target=0 rows have NaN coordinates.
        bbox_df = bbox_df[bbox_df["Target"] == 1].dropna(
            subset=["x", "y", "width", "height"]
        )

        s = self._IMG_SIZE
        bbox_df = bbox_df.copy()
        bbox_df["norm_bbox"] = bbox_df.apply(
            lambda r: [
                r["y"] / s,                   # dim0_min (row top)
                (r["y"] + r["height"]) / s,   # dim0_max (row bottom)
                r["x"] / s,                   # dim1_min (col left)
                (r["x"] + r["width"]) / s,    # dim1_max (col right)
            ],
            axis=1,
        )

        agg = (
            bbox_df.groupby("patientId")["norm_bbox"]
            .apply(list)
            .reset_index()
            .rename(columns={"norm_bbox": "bbox"})
        )

        self.df = self.df.merge(agg, on="patientId", how="left")
        # An image with no box gets an empty list.
        self.df["bbox"] = self.df["bbox"].apply(
            lambda x: x if isinstance(x, list) else []
        )
        # All boxes have one class.
        self.df["bbox_labels"] = self.df["bbox"].apply(
            lambda boxes: ["lung_opacity"] * len(boxes)
        )
        clip_boxes(self.df, "RSNA Pneumonia (Kaggle)")  # D-07

    def _read_source(self) -> pd.DataFrame:
        """Read the class CSV and keep one row per image (D-03).

        The CSV has one line per box, and one line for an image with no box.
        """
        return pd.read_csv(self.csv_path).drop_duplicates("patientId", ignore_index=True)
