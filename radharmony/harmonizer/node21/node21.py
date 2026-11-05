"""Harmonizer for NODE21 (Sogancioglu et al., arXiv:2401.02192).

A nodule detection benchmark on frontal chest radiographs. It has 4,882 images. 1,134 images
have 1,476 boxes drawn by radiologists. The other 3,748 images are controls. The images come
from ChestX-ray14, PadChest, JSRT and Open-I. The counts match the Zenodo release (DOI
10.5281/zenodo.5548363, md5 0bc7a2f16d6ad41b343bfb568964c6d8, checked on 2026-09-05).

Six facts about the data:

1. The release folder is misspelled ``proccessed_data`` (two c).
2. The images are in ``<variant>/images/``, not next to ``metadata.csv``.
3. A control is a real row with ``x=y=width=height=0``. The harmonizer drops these boxes, and
   ``n_boxes`` is 0 for them.
4. The CSV has one row for each box (5,224 rows for 4,882 images). A nodule image has up to 3
   rows.
5. ``original_data`` has many sizes (for example 1024x1024 and 2991x3000) and two dtypes.
   ``proccessed_data`` is always 1024x1024 uint16. It is the default.
6. The box coordinates belong to their variant. The same nodule is at y=538 in the original
   and at y=579 in the processed image (resize with padding). Do not mix the variants.

A box is ``(x, y, width, height)`` in pixels with a top-left origin. The release does not say
this. Boxes drawn on the largest positives all land in a lung field, which a bottom-left
origin would not give.

``filenames_orig_and_new.csv`` gives the source corpus only for the 1,134 nodule images. So
a control cannot be de-duplicated against a copy of ChestX-ray14, PadChest or Open-I.

Details: ``docs/wiki/reference/datasets/cxr-boxes-masks/node21.md``.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from .._bbox import clip_boxes
from ..base import BaseHarmonizer

# One class: an image has nodules or not.
CLASSES = ["nodule"]

# Both variants have the same file names. The release misspells "processed".
VARIANT_DIRS = {"processed": "proccessed_data", "original": "original_data"}


class Node21Harmonizer(BaseHarmonizer):
    """Harmonize NODE21 nodule boxes.

    Args:
        csv_path: Path to ``metadata.csv`` for the chosen variant. Defaults to
            the one inside ``base_image_dir/cxr_images/<variant dir>/``.
        base_image_dir: Dataset root containing ``cxr_images/``.
        variant: ``processed`` (default, uniform 1024x1024) or ``original``.
        keep_negatives: Keep the 3,748 nodule-free controls (default True).
            They are the benchmark's negative class, not missing data.
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
    EXTRA_OUTPUT_COLS = ["variant", "n_boxes", "source_dataset"]

    def __init__(self, csv_path: str = None, base_image_dir: str = None,
                 variant: str = "processed", keep_negatives: bool = True):
        if variant not in VARIANT_DIRS:
            raise ValueError(
                f"variant must be one of {sorted(VARIANT_DIRS)}, got {variant!r}")
        self.variant = variant
        self.base_image_dir = base_image_dir
        if csv_path is None:
            if not base_image_dir:
                raise ValueError("one of csv_path or base_image_dir is required")
            csv_path = os.path.join(
                base_image_dir, "cxr_images", VARIANT_DIRS[variant], "metadata.csv")
        super().__init__(csv_path=csv_path)
        self.keep_negatives = keep_negatives

    def _harmonizer_init_snapshot(self) -> dict:
        """Also keep ``variant`` and ``keep_negatives``, so a loaded object harmonizes the same (V-H2-16)."""
        snap = super()._harmonizer_init_snapshot()
        snap.update(variant=self.variant, keep_negatives=self.keep_negatives)
        return snap

    def _read_source(self) -> pd.DataFrame:
        """Make one row for each image from the one row for each box.

        It drops the first unnamed index column. A row with label 0 is a control with zero
        coordinates, so the harmonizer drops its box. With ``keep_negatives=False``, it drops
        the controls. The label decides, so a nodule image stays if ``clip_boxes`` drops its
        only box (D-07).
        """
        raw = pd.read_csv(self.csv_path)
        raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]
        recs = {}
        for r in raw.itertuples(index=False):
            rec = recs.setdefault(
                r.img_name, {"img_name": r.img_name, "label": 0, "boxes": []})
            if int(r.label) == 1:
                rec["label"] = 1
                rec["boxes"].append(
                    (float(r.x), float(r.y), float(r.width), float(r.height)))
        df = pd.DataFrame(list(recs.values()))
        if not self.keep_negatives:
            df = df[df["label"] > 0].reset_index(drop=True)
        return df

    def _build_patient_id(self) -> None:
        # The release has no patient ID. One image is one "patient", so nobody expects a
        # patient split.
        self.df["patient_id"] = self.df["img_name"].str.replace(
            ".mha", "", regex=False)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["patient_id"]

    def _build_image_path(self) -> None:
        sub = os.path.join("cxr_images", VARIANT_DIRS[self.variant], "images")
        self.df["image_path"] = sub + "/" + self.df["img_name"]
        self.df["variant"] = self.variant

    def _build_labels(self) -> None:
        self.df["nodule"] = self.df["label"].astype(int)
        self.df["n_boxes"] = self.df["boxes"].apply(len)
        self.df["source_dataset"] = self._source_map()

    def _source_map(self) -> pd.Series:
        """Return the source corpus of each image, if the release has it.

        ``filenames_orig_and_new.csv`` has only the 1,134 nodule images. A control gets
        "unknown".
        """
        mapping = {}
        if self.base_image_dir:
            p = os.path.join(self.base_image_dir, "cxr_images",
                             "original_data", "filenames_orig_and_new.csv")
            if os.path.isfile(p):
                m = pd.read_csv(p)
                mapping = dict(zip(m["node21_img_id"], m["orig_dataset"], strict=True))
        return self.df["patient_id"].map(mapping).fillna("unknown")

    def _build_bbox(self) -> None:
        """Convert the pixel boxes (x, y, w, h) to fractions [dim0_min, dim0_max, dim1_min, dim1_max].

        dim0 is the row. The image size comes from each file, because the original variant has
        many sizes. A fixed 1024 would scale the boxes wrong without an error.
        """
        import SimpleITK as sitk

        boxes_out, labels_out = [], []
        for row in self.df.itertuples(index=False):
            if not row.boxes:
                boxes_out.append([])
                labels_out.append([])
                continue
            path = row.image_path
            if self.base_image_dir:
                path = os.path.join(self.base_image_dir, path)
            reader = sitk.ImageFileReader()
            reader.SetFileName(path)
            reader.ReadImageInformation()
            W, H = reader.GetSize()  # ITK gives (x, y)
            bb, lb = [], []
            for (x, y, w, h) in row.boxes:
                bb.append([y / H, (y + h) / H, x / W, (x + w) / W])
                lb.append("nodule")
            boxes_out.append(bb)
            labels_out.append(lb)
        # JSON text, as in MS-CXR and ChestX-Det10. After a CSV save and load, a label list
        # is a Python repr (for example "['nodule']") that json.loads cannot read.
        self.df["bbox"] = [json.dumps(b) for b in boxes_out]
        self.df["bbox_labels"] = [json.dumps(l) for l in labels_out]
        clip_boxes(self.df, "NODE21")  # D-07
