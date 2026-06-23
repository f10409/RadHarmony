"""Harmonizer for the NIH ChestX-ray14 bounding-box subset."""

import os

import pandas as pd
from PIL import Image

from ..base import BaseHarmonizer


class ChestXray14BboxHarmonizer(BaseHarmonizer):
    """Harmonize the ChestX-ray14 bounding-box subset.

    Only the ~880 images that have radiologist-drawn bounding boxes in
    ``BBox_List_2017.csv`` are included.  Each image may have multiple
    boxes (984 total across 8 pathology classes).

    The bbox CSV has a malformed header (``Bbox [x,y,w,h]`` splits across
    multiple columns) so it is read with explicit column names.
    Coordinates are pixel-space ``(x, y, w, h)``; we normalize to
    ``[dim0_min, dim0_max, dim1_min, dim1_max]`` using the image
    dimensions from the primary CSV (``OriginalImage[Width`` and
    ``Height]`` columns).

    Args:
        csv_path: Path to ``Data_Entry_2017.csv``.
        bbox_csv_path: Path to ``BBox_List_2017.csv`` (required).
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

    def __init__(self, csv_path: str, bbox_csv_path: str, base_image_dir: str = None):
        super().__init__(csv_path=csv_path, bbox_csv_path=bbox_csv_path)
        self.base_image_dir = base_image_dir

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["Patient ID"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = (
            self.df["Patient ID"].astype(str) + "_" + self.df["Follow-up #"].astype(str)
        )

    def _build_image_path(self) -> None:
        """Resolve image filenames to relative paths by scanning subdirectories."""
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

    def _build_bbox(self) -> None:
        """Normalize bounding boxes from BBox_List_2017.csv."""
        bbox_df = pd.read_csv(
            self.bbox_csv_path,
            header=0,
            names=[
                "image_id",
                "finding_label",
                "x",
                "y",
                "w",
                "h",
                "_extra1",
                "_extra2",
                "_extra3",
            ],
            usecols=["image_id", "finding_label", "x", "y", "w", "h"],
        )

        # Build a lookup of actual PNG dimensions (width, height) per image.
        # Bbox coordinates are in the rescaled PNG pixel space.
        img_sizes = {}  # image_id -> (png_width, png_height)
        bbox_image_ids = set(bbox_df["image_id"].unique())
        for _, irow in self.df[self.df["Image Index"].isin(bbox_image_ids)].iterrows():
            iid = irow["Image Index"]
            if iid in img_sizes:
                continue
            rel_path = irow.get("image_path", iid)
            if self.base_image_dir and rel_path:
                full_path = os.path.join(self.base_image_dir, rel_path)
            else:
                full_path = rel_path
            if full_path and os.path.isfile(full_path):
                with Image.open(full_path) as im:
                    img_sizes[iid] = im.size  # (width, height)

        def _normalize_group(g):
            iid = g.name
            size = img_sizes.get(iid)
            if size is None:
                return pd.Series({"bbox": [], "bbox_labels": []})
            img_w, img_h = size
            boxes, labels = [], []
            for _, row in g.iterrows():
                x, y, w, h = row["x"], row["y"], row["w"], row["h"]
                boxes.append(
                    [
                        y / img_h,        # dim0_min (row)
                        (y + h) / img_h,  # dim0_max (row)
                        x / img_w,        # dim1_min (col)
                        (x + w) / img_w,  # dim1_max (col)
                    ]
                )
                labels.append(row["finding_label"].lower().replace(" ", "_"))
            return pd.Series({"bbox": boxes, "bbox_labels": labels})

        bbox_agg = (
            bbox_df.groupby("image_id")
            .apply(_normalize_group, include_groups=False)
            .reset_index()
            .rename(columns={"image_id": "Image Index"})
        )
        self.df = self.df.merge(bbox_agg, on="Image Index", how="left")
        self.df["bbox"] = self.df["bbox"].apply(
            lambda x: x if isinstance(x, list) else []
        )
        self.df["bbox_labels"] = self.df["bbox_labels"].apply(
            lambda x: x if isinstance(x, list) else []
        )

    def harmonize(self, **kwargs) -> pd.DataFrame:
        """Read CSV, build bboxes, filter to only images with bboxes."""
        self._harmonized_df_override = None
        self.df = pd.read_csv(self.csv_path)

        self._build_patient_id()
        self._build_study_id()
        self._build_image_path()
        self._build_labels()
        self._build_view_position()
        self._build_mask_path()
        self._build_bbox()
        self._build_report()

        self.df.dropna(subset=["image_path"], inplace=True)

        # Keep only images that have at least one bounding box
        self.df = self.df[self.df["bbox"].apply(lambda x: len(x) > 0)]
        self.df.reset_index(drop=True, inplace=True)

        return self._select_harmonized_columns(self.df)
