"""Harmonizer for the VinDr-PCXR pediatric chest X-ray dataset.

A pediatric CXR set with 15 image labels, and boxes of 36 findings, annotated by
radiologists. Source (PhysioNet, credentialed access): https://physionet.org/content/vindr-pcxr/
Cite: Pham HH, et al. VinDr-PCXR: An open, large-scale pediatric chest X-ray dataset for
interpretation of common thoracic diseases. *PhysioNet* 2022.

Layout::

    <base_dir>/
      train/<image_id>.dicom
      test/<image_id>.dicom
      image_labels_{train,test}.csv
      annotations_{train,test}.csv    # boxes

The table has one row for each image (about 9,125: 7,728 train and 1,397 test). The 15 labels
are binary and belong to the image.

A box is in ``annotations_{split}.csv`` as pixels ``x_min, y_min, x_max, y_max``. The harmonizer
converts it to ``[y_min, y_max, x_min, x_max]`` with values from 0 to 1, with the image size
from the DICOM header. The boxes of an image go to JSON lists. ``bbox_labels`` has the class
names in snake case, for example ``bronchial_thickening`` (D-07). A boxed image with an
unreadable DICOM file gets no box, and a warning gives the count.

The DICOM headers have no PatientID and no StudyInstanceUID, so ``patient_id`` and ``study_id``
are the ``image_id``. ``sex`` and ``age_years`` come from the DICOM tags ``PatientSex`` and
``PatientAge``. The release has neither in a CSV (D-06).

Details: ``docs/wiki/reference/datasets/cxr-boxes-masks/vindr_pcxr.md``.
"""

import json
import math
import os
import warnings

import pandas as pd
import pydicom

from .._bbox import box_label, clip_boxes
from .._dicom_tags import add_patient_tags
from ..base import BaseHarmonizer


# Label columns of the image: CSV name and snake_case name.
_LABEL_MAP = {
    "No finding":            "no_finding",
    "Bronchitis":            "bronchitis",
    "Brocho-pneumonia":      "brocho_pneumonia",
    "Other disease":         "other_disease",
    "Bronchiolitis":         "bronchiolitis",
    "Situs inversus":        "situs_inversus",
    "Pneumonia":             "pneumonia",
    "Pleuro-pneumonia":      "pleuro_pneumonia",
    "Diagphramatic hernia":  "diagphramatic_hernia",
    "Tuberculosis":          "tuberculosis",
    "Congenital emphysema":  "congenital_emphysema",
    "CPAM":                  "cpam",
    "Hyaline membrane disease": "hyaline_membrane_disease",
    "Mediastinal tumor":     "mediastinal_tumor",
    "Lung tumor":            "lung_tumor",
}

_LABEL_COLS = list(_LABEL_MAP.values())


def _read_dicom_dims(path: str) -> tuple[int, int]:
    """Return (height, width) from a DICOM header."""
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    return int(ds.Rows), int(ds.Columns)


class VinDrPCXRHarmonizer(BaseHarmonizer):
    """Harmonize the VinDr-PCXR Pediatric Chest X-ray dataset.

    Args:
        base_dir: Dataset root, with ``train/``, ``test/`` and the four CSV files.
    """

    LABEL_COLS = _LABEL_COLS

    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    EXTRA_OUTPUT_COLS = ["split", "image_width", "image_height"]

    # DICOM tags, read by _build_patient_tags() (D-06, D-26, V-H3-31). A young child has an age in M or D ("006M").
    AGE_SOURCE_COL = "PatientAge"
    AGE_SOURCE_UNIT = "dicom_as"
    SEX_SOURCE_COL = "PatientSex"

    # The official splits to read. A split class keeps one of them.
    SPLITS = ("train", "test")

    def __init__(self, base_dir: str):
        self.base_dir = os.path.expanduser(base_dir)
        super().__init__(csv_path=None)

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["base_dir"] = self.base_dir
        snap.pop("csv_path", None)
        return snap

    def verify_images(self, base_image_dir: str = None, drop_missing: bool = True) -> pd.DataFrame:
        """Check the image files. The folder defaults to ``base_dir`` (D-04, V-H3-36)."""
        return super().verify_images(base_image_dir or self.base_dir, drop_missing)

    def _read_source(self) -> pd.DataFrame:
        """Read the label and box CSVs of each split in ``SPLITS``. Return one row for each image.

        The rows have the IDs, the image path, the labels, the image size and the boxes. The
        image size comes from the DICOM header.
        """
        rows = []
        for split in self.SPLITS:
            labels_csv = os.path.join(self.base_dir, f"image_labels_{split}.csv")
            annot_csv  = os.path.join(self.base_dir, f"annotations_{split}.csv")
            img_dir    = os.path.join(self.base_dir, split)

            if not os.path.isfile(labels_csv):
                raise FileNotFoundError(f"{labels_csv} not found.")
            if not os.path.isdir(img_dir):
                raise FileNotFoundError(f"{img_dir}/ not found.")

            labels_df = pd.read_csv(labels_csv)
            annot_df  = pd.read_csv(annot_csv) if os.path.isfile(annot_csv) else pd.DataFrame()

            # Box lookup: image_id to a list of [y_min, y_max, x_min, x_max] (fractions).
            bbox_map: dict[str, list] = {}
            blabel_map: dict[str, list] = {}
            # The size cache serves the box pass and the label pass, so no file is read twice.
            dims: dict[str, tuple[int, int]] = {}

            if not annot_df.empty:
                # A "No finding" row has no coordinates, so only these images lose a box.
                coords = annot_df[["x_min", "y_min", "x_max", "y_max"]].astype(float)
                boxed_ids = set(annot_df.loc[coords.notna().all(axis=1), "image_id"])
                unread = []  # missing or unreadable DICOM files of boxed images (R-code-19)
                for img_id in annot_df["image_id"].unique():
                    dcm_path = os.path.join(img_dir, f"{img_id}.dicom")
                    try:
                        dims[img_id] = _read_dicom_dims(dcm_path)
                    except Exception:
                        if img_id in boxed_ids:
                            unread.append(dcm_path)
                if unread:
                    warnings.warn(
                        f"VinDr-PCXR {split}: {len(unread)} images with boxes have no readable DICOM "
                        f"file, so they get no box (first: {unread[0]}).",
                        UserWarning,
                        stacklevel=2,
                    )

                for img_id, group in annot_df.groupby("image_id"):
                    if img_id not in dims:
                        continue
                    h, w = dims[img_id]
                    norm_boxes = []
                    labels_list = []
                    for _, row in group.iterrows():
                        # A "No finding" row has empty coordinates: it has no box.
                        y0, y1, x0, x1 = (
                            float(row[c]) for c in ("y_min", "y_max", "x_min", "x_max")
                        )
                        if not all(math.isfinite(v) for v in (y0, y1, x0, x1)):
                            continue
                        y_min, y_max = y0 / h, y1 / h
                        x_min, x_max = x0 / w, x1 / w
                        # _build_bbox() clips the box to [0, 1] (D-07).
                        norm_boxes.append([y_min, y_max, x_min, x_max])
                        labels_list.append(box_label(row["class_name"]))
                    bbox_map[img_id]   = norm_boxes
                    blabel_map[img_id] = labels_list

            for _, lrow in labels_df.iterrows():
                img_id = str(lrow["image_id"])
                dcm_rel = f"{split}/{img_id}.dicom"
                dcm_abs = os.path.join(self.base_dir, dcm_rel)

                row: dict = {
                    "patient_id":   img_id,
                    "study_id":     img_id,
                    "image_path":   dcm_rel,
                    "split":        split,
                }

                for orig_col, snake_col in _LABEL_MAP.items():
                    row[snake_col] = int(lrow.get(orig_col, 0))

                # Image size: use the cache of the box pass, or read the file.
                if img_id in dims:
                    row["image_height"], row["image_width"] = dims[img_id]
                elif os.path.isfile(dcm_abs):
                    try:
                        h, w = _read_dicom_dims(dcm_abs)
                        dims[img_id] = (h, w)
                        row["image_height"] = h
                        row["image_width"]  = w
                    except Exception:
                        row["image_height"] = None
                        row["image_width"]  = None
                else:
                    row["image_height"] = None
                    row["image_width"]  = None

                if img_id in bbox_map:
                    row["bbox"]        = json.dumps(bbox_map[img_id])
                    row["bbox_labels"] = json.dumps(blabel_map[img_id])
                else:
                    row["bbox"]        = None
                    row["bbox_labels"] = None

                rows.append(row)

        return pd.DataFrame(rows)

    # _read_source() builds these columns.
    def _build_patient_id(self) -> None:
        pass

    def _build_study_id(self) -> None:
        pass

    def _build_image_path(self) -> None:
        """``_read_source()`` sets ``image_path``. Read the DICOM sex and age tags here.

        This step runs after the id filter, so only the kept rows are read.
        """
        self._build_patient_tags()

    def _build_bbox(self) -> None:
        """``_read_source()`` makes the boxes. Clip them here (D-07)."""
        clip_boxes(self.df, "VinDr-PCXR")  # D-07

    def _build_patient_tags(self) -> None:
        """Read ``PatientSex`` and ``PatientAge`` from the DICOM headers (D-06).

        A table with no row (an id filter) gets the 2 columns too. A source with no row has no
        columns, so it is skipped.
        """
        if "image_path" in self.df.columns:
            self.df = add_patient_tags(self.df, self.base_dir)
