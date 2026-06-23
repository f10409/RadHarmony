"""Harmonizer for the NIH ChestX-ray14 dataset (non-bbox images)."""

import os

import pandas as pd

from ..base import BaseHarmonizer


class ChestXray14Harmonizer(BaseHarmonizer):
    """Harmonize ChestX-ray14 into the standard RadHarmony format.

    Includes only images that do **not** have bounding-box annotations
    (~111,240 images).  For the ~880 images with bounding boxes, use
    :class:`ChestXray14BboxHarmonizer`.

    ChestX-ray14 has 112,120 frontal-view chest X-rays from 30,805 patients.
    Labels are NLP-derived from radiology reports and stored as pipe-separated
    strings in the ``Finding Labels`` column
    (e.g. ``"Atelectasis|Effusion"``).

    Images are PNG files distributed across 12 subdirectories
    (``images_001/images/`` … ``images_012/images/``) under ``base_image_dir``
    (e.g. ``.../NIH_CXR/CXR14/``).

    Note: ``Pleural_Thickening`` already uses an underscore in the raw CSV,
    so its snake_case form is ``pleural_thickening`` (no double-underscore).

    Args:
        csv_path: Path to ``Data_Entry_2017.csv``.
        bbox_csv_path: Path to ``BBox_List_2017.csv`` (used to exclude bbox images).
        base_image_dir: Root directory containing ``images_*/images/`` folders
            (e.g. ``.../NIH_CXR/CXR14/``).
    """

    LABEL_COLS = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Effusion",
        "Emphysema",
        "Fibrosis",
        "Hernia",
        "Infiltration",
        "Mass",
        "No Finding",
        "Nodule",
        "Pleural_Thickening",
        "Pneumonia",
        "Pneumothorax",
    ]

    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = "View Position"
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    def __init__(
        self, csv_path: str, bbox_csv_path: str = None, base_image_dir: str = None
    ):
        super().__init__(csv_path=csv_path, bbox_csv_path=bbox_csv_path)
        self.base_image_dir = base_image_dir

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["Patient ID"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = (
            self.df["Patient ID"].astype(str) + "_" + self.df["Follow-up #"].astype(str)
        )

    def _build_image_path(self) -> None:
        """Resolve image filenames to relative paths by scanning subdirectories.

        The CSV ``Image Index`` column contains just the filename
        (e.g. ``00000001_000.png``), but images live in
        ``images_001/images/`` … ``images_012/images/``.  We scan once to
        build a filename→relative-path lookup.
        """
        if self.base_image_dir is None:
            self.df["image_path"] = self.df["Image Index"]
            return

        lookup = {}
        for entry in sorted(os.listdir(self.base_image_dir)):
            subdir = os.path.join(self.base_image_dir, entry, "images")
            if os.path.isdir(subdir):
                for fname in os.listdir(subdir):
                    lookup[fname] = os.path.join(entry, "images", fname)

        self.df["image_path"] = self.df["Image Index"].map(lookup)

    def _build_labels(self) -> None:
        """One-hot encode the pipe-separated ``Finding Labels`` column."""
        for label in self.LABEL_COLS:
            self.df[label] = (
                self.df["Finding Labels"]
                .str.contains(
                    label.replace("_", "[_ ]"),
                    case=True,
                    regex=True,
                )
                .astype(int)
            )

        for col in self.LABEL_COLS:
            snake = col.lower().replace(" ", "_")
            if snake != col:
                self.df.rename(columns={col: snake}, inplace=True)

    def harmonize(self, **kwargs) -> pd.DataFrame:
        """Read CSV, one-hot encode labels, resolve image paths.

        If ``bbox_csv_path`` was provided, images with bounding boxes are
        excluded (those belong to :class:`ChestXray14BboxHarmonizer`).
        """
        self._harmonized_df_override = None
        self.df = pd.read_csv(self.csv_path)

        self._build_patient_id()
        self._build_study_id()
        self._build_image_path()
        self._build_labels()
        self._build_view_position()
        self._build_mask_path()
        self._build_report()

        self.df.dropna(subset=["image_path"], inplace=True)

        # Exclude images that have bounding-box annotations
        if self.bbox_csv_path is not None:
            bbox_df = pd.read_csv(
                self.bbox_csv_path,
                header=0,
                names=[
                    "image_id", "finding_label", "x", "y", "w", "h",
                    "_extra1", "_extra2", "_extra3",
                ],
                usecols=["image_id"],
            )
            bbox_image_ids = set(bbox_df["image_id"].unique())
            self.df = self.df[~self.df["Image Index"].isin(bbox_image_ids)]

        self.df.reset_index(drop=True, inplace=True)

        return self._select_harmonized_columns(self.df)
