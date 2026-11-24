"""Harmonizer for the NIH ChestX-ray14 bounding-box subset."""

import os

import pandas as pd
from PIL import Image

from .._bbox import box_label, clip_boxes
from ..base import BaseHarmonizer


class ChestXray14BboxHarmonizer(BaseHarmonizer):
    """Harmonize the ChestX-ray14 bounding-box subset.

    Only the ~880 images that have radiologist-drawn bounding boxes in
    ``BBox_List_2017.csv`` are included.  Each image may have multiple
    boxes (984 total across 8 pathology classes).

    The bbox CSV has a malformed header (``Bbox [x,y,w,h]`` splits across
    multiple columns) so it is read with explicit column names.
    Coordinates are pixel-space ``(x, y, w, h)`` of the released PNG
    (1024 x 1024). They are divided by the size of the PNG file, to give
    ``[dim0_min, dim0_max, dim1_min, dim1_max]`` (D-07). The
    ``OriginalImage[Width`` and ``Height]`` columns of the primary CSV give
    the size before the resize to the PNG, so the code does not use them.

    Args:
        csv_path: Path to ``Data_Entry_2017.csv``.
        bbox_csv_path: Path to ``BBox_List_2017.csv`` (required).
        base_image_dir: Root directory containing ``images_*/images/`` folders
            (for example ``.../NIH_CXR/CXR14/``). ``harmonize()`` needs it, because it
            reads the size of each PNG. Without it, ``harmonize()`` raises
            ``ValueError``.
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

    #: ``BBox_List_2017.csv`` calls a finding "Infiltrate", but
    #: ``Data_Entry_2017.csv`` calls it "Infiltration". Box labels use the
    #: label column name, so code can match a box to its label column.
    _BBOX_LABEL_NAMES = {"Infiltrate": "Infiltration"}

    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = "View Position"
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    # Same source columns as ChestXray14Harmonizer (D-06, D-26).
    AGE_SOURCE_COL = "Patient Age"
    AGE_SOURCE_UNIT = "years"
    SEX_SOURCE_COL = "Patient Gender"

    def __init__(self, csv_path: str, bbox_csv_path: str, base_image_dir: str = None):
        super().__init__(csv_path=csv_path, bbox_csv_path=bbox_csv_path)
        self.base_image_dir = base_image_dir

    def _read_source(self) -> pd.DataFrame:
        """Read ``Data_Entry_2017.csv``. Raise ``ValueError`` if ``base_image_dir`` is not set."""
        if not self.base_image_dir:
            # V-H1-36: with no PNG, every box list was empty and the table had 0 rows, with no message.
            raise ValueError(
                "ChestXray14BboxHarmonizer needs base_image_dir, the folder with images_*/images/. "
                "It reads the size of each PNG to scale the boxes."
            )
        return super()._read_source()

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["Patient ID"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = (
            self.df["Patient ID"].astype(str) + "_" + self.df["Follow-up #"].astype(str)
        )

    def _build_image_path(self) -> None:
        """Resolve image filenames to relative paths by scanning subdirectories."""
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
        """Normalize the boxes of BBox_List_2017.csv, clip them (D-07), and keep only the images with a box."""
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
            if pd.isna(rel_path):
                continue  # no PNG found: harmonize() drops the row
            full_path = os.path.join(self.base_image_dir, rel_path)
            if os.path.isfile(full_path):
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
                name = self._BBOX_LABEL_NAMES.get(row["finding_label"], row["finding_label"])
                labels.append(box_label(name))
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
        clip_boxes(self.df, "ChestX-ray14")  # D-07, before the filter below
        # Keep only the images with at least one box.
        # astype(bool): on 0 rows, apply() gives an object Series, and df[...] then picks columns (V-H2-30).
        self.df = self.df[self.df["bbox"].apply(lambda x: len(x) > 0).astype(bool)]
