"""Harmonizer for the RSNA Pneumonia Detection Challenge dataset (Kaggle Stage 2).

The Kaggle distribution ships two flat CSVs and a flat DICOM directory:

  stage_2_detailed_class_info.csv  — patientId, class
  stage_2_train_labels.csv         — patientId, x, y, width, height, Target
  stage_2_train_images/            — <patientId>.dcm  (flat, 1024×1024)

``class`` takes three values: "Normal", "No Lung Opacity / Not Normal",
"Lung Opacity".  Images with ``Target=1`` in the labels CSV carry one or
more pixel-space bounding boxes; these are normalised to
``[dim0_min, dim0_max, dim1_min, dim1_max]`` fractional coordinates.
"""

import os

import pandas as pd

from ..base import BaseHarmonizer


class RSNAPneumoniaKaggleHarmonizer(BaseHarmonizer):
    """Harmonize RSNA Pneumonia Detection Challenge (Kaggle Stage 2) into RadHarmony format.

    Args:
        csv_path: Path to ``stage_2_detailed_class_info.csv``.
        base_image_dir: ``stage_2_train_images/`` directory — flat dir of
            ``<patientId>.dcm`` files (e.g.
            ``~/datasets/rsna-pneumonia-detection-challenge/stage_2_train_images/``).
        bbox_csv_path: Path to ``stage_2_train_labels.csv``.  Required when
            bounding boxes are needed; otherwise only classification labels
            are produced.
    """

    LABEL_COLS = [
        "Lung Opacity",
        "No Lung Opacity / Not Normal",
        "Normal",
    ]

    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    # All RSNA Pneumonia DICOMs are 1024×1024.
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
        # base_image_dir points at stage_2_train_images/, so image_path is
        # just <patientId>.dcm relative to that.
        self.df["image_path"] = self.df["patientId"].astype(str) + ".dcm"

    def _build_labels(self) -> None:
        # One-hot encode the 'class' column into three binary columns.
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
        # Only Target=1 rows carry actual boxes; Target=0 rows have NaN coords.
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
        # Images without bboxes (Normal / No Lung Opacity) get empty lists.
        self.df["bbox"] = self.df["bbox"].apply(
            lambda x: x if isinstance(x, list) else []
        )
        # All boxes belong to a single class; mirror the list length for bbox_labels.
        self.df["bbox_labels"] = self.df["bbox"].apply(
            lambda boxes: ["lung_opacity"] * len(boxes)
        )

    def harmonize(self, **kwargs) -> pd.DataFrame:
        """Read the class-info CSV, build all standard columns, and return harmonized DataFrame."""
        self._harmonized_df_override = None

        self.df = pd.read_csv(self.csv_path)

        self._build_patient_id()
        self._build_study_id()
        self._build_image_path()
        self._build_labels()
        self._build_bbox()
        self._build_view_position()
        self._build_mask_path()
        self._build_report()

        self.df.dropna(subset=["image_path"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self._select_harmonized_columns(self.df)
