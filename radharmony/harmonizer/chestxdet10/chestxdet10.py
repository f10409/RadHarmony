"""Harmonizer for ChestX-Det10 (Liu, Lin and Yu, arXiv:2006.10550).

3,543 NIH ChestX-ray14 images with instance-level box annotations. Three
radiologists made them. There are 10 classes: Atelectasis, Calcification,
Consolidation, Effusion, Emphysema, Fibrosis, Fracture, Mass, Nodule and
Pneumothorax. Train has 3,001 images and test has 542 (checked against
paper Tables 1 and 2 on 2026-08-31).

Four things to know:

1. The training images unzip to ``train-old/``, not ``train/``. With
   ``train/`` the dataset is empty and gives no error.
2. You cannot find the images in an NIH ChestX-ray14 copy. File names
   differ (``36199.png`` and NIH ``00001075_000.png``). Use the Deepwise zips.
3. 40 of the 3,543 images are RGBA. The others are 8-bit grayscale.
4. 764 images have empty ``syms`` and ``boxes`` on purpose. They are the
   negative controls. Do not drop them.

``syms`` and ``boxes`` are parallel arrays of equal length. Several diseases
can share one box, so a box can have more than one label.

More: ``docs/wiki/reference/datasets/cxr-boxes-masks/chestxdet10.md``.
"""
from __future__ import annotations

import json
import os

import pandas as pd
from PIL import Image

from .._bbox import box_label, clip_boxes
from ..base import BaseHarmonizer

# Paper §1, in the order the paper lists them.
CLASSES = ["Atelectasis", "Calcification", "Consolidation", "Effusion",
           "Emphysema", "Fibrosis", "Fracture", "Mass", "Nodule", "Pneumothorax"]

# The split subdirectory inside the extracted archive. NOT "train"/"test".
SPLIT_DIRS = {"train": "train-old", "test": "test_data"}


class ChestXDet10Harmonizer(BaseHarmonizer):
    """Harmonize ChestX-Det10 box annotations.

    Args:
        csv_path: Path to ``train.json`` or ``test.json``. The name
            ``csv_path`` is kept for the base class.
        base_image_dir: Dataset root containing ``train-old/`` and ``test_data/``.
        split: ``train`` or ``test``. It selects the image folder.
        keep_negatives: Keep the 764 images with no findings (default True).
            They are the benchmark's negative controls, not missing data.
    """

    LABEL_COLS = CLASSES
    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None
    EXTRA_OUTPUT_COLS = ["split", "n_boxes"]

    def __init__(self, csv_path: str, base_image_dir: str = None,
                 split: str = "train", keep_negatives: bool = True):
        super().__init__(csv_path=csv_path)
        self.base_image_dir = base_image_dir
        if split not in SPLIT_DIRS:
            raise ValueError(f"split must be one of {sorted(SPLIT_DIRS)}, got {split!r}")
        self.split = split
        self.keep_negatives = keep_negatives

    def _harmonizer_init_snapshot(self) -> dict:
        """Add ``split`` and ``keep_negatives``, so a loaded object harmonizes the same way (D-08)."""
        snap = super()._harmonizer_init_snapshot()
        snap.update(split=self.split, keep_negatives=self.keep_negatives)
        return snap

    def _read_source(self) -> pd.DataFrame:
        """Read train.json or test.json (not a CSV). With keep_negatives=False, drop the images with no finding."""
        with open(self.csv_path) as fh:
            recs = json.load(fh)
        df = pd.DataFrame([
            {"file_name": r["file_name"],
             "syms": list(r.get("syms") or []),
             "boxes": list(r.get("boxes") or [])}
            for r in recs
        ])
        if not self.keep_negatives:
            # Use the findings, not the box count. An image keeps its row if
            # clip_boxes drops its only box (D-07).
            df = df[df["syms"].map(len) > 0].reset_index(drop=True)
        return df

    def _build_patient_id(self) -> None:
        # There is no patient ID, and the file names are not the NIH ones.
        # One patient per image shows that a patient-level split is not possible.
        self.df["patient_id"] = self.df["file_name"].str.replace(".png", "", regex=False)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["patient_id"]

    def _build_image_path(self) -> None:
        sub = SPLIT_DIRS[self.split]
        self.df["image_path"] = sub + "/" + self.df["file_name"]
        self.df["split"] = self.split

    def _build_labels(self) -> None:
        """Image-level multi-label presence, derived from the box labels."""
        for c in CLASSES:
            self.df[c] = self.df["syms"].apply(lambda s, c=c: int(c in s))
        self.df["n_boxes"] = self.df["boxes"].apply(len)
        for c in CLASSES:
            self.df.rename(columns={c: c.lower()}, inplace=True)

    def _build_bbox(self) -> None:
        """Convert pixel [x1, y1, x2, y2] to normalized [dim0_min, dim0_max, dim1_min, dim1_max].

        dim0 is the row. All images are 1024x1024, but the code reads the size
        of each image, so a new release cannot mis-scale the boxes.
        """
        boxes_out, labels_out = [], []
        for _, row in self.df.iterrows():
            if not row["boxes"]:
                boxes_out.append([]); labels_out.append([])
                continue
            path = row["image_path"]
            if self.base_image_dir:
                path = os.path.join(self.base_image_dir, path)
            with Image.open(path) as im:
                W, H = im.size
            bb, lb = [], []
            for (x1, y1, x2, y2), sym in zip(row["boxes"], row["syms"], strict=True):
                bb.append([y1 / H, y2 / H, x1 / W, x2 / W])
                lb.append(box_label(sym))
            boxes_out.append(bb); labels_out.append(lb)
        # Store JSON strings, as MS-CXR does. After a CSV save and load, a
        # label list is a Python repr such as "['mass']", which json.loads cannot read.
        self.df["bbox"] = [json.dumps(b) for b in boxes_out]
        self.df["bbox_labels"] = [json.dumps(l) for l in labels_out]
        clip_boxes(self.df, "ChestX-Det10")  # D-07
