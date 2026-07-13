"""Harmonizer for the VinDr-PCXR Pediatric Chest X-ray dataset.

A large-scale pediatric CXR dataset annotated by experienced radiologists
with image-level labels and bounding boxes for 15 thoracic conditions.

**Source (PhysioNet — requires credentialed access):**
  https://physionet.org/content/vindr-pcxr/

**Cite:**
  Pham HH, et al. VinDr-PCXR: An open, large-scale pediatric chest X-ray
  dataset for interpretation of common thoracic diseases. *PhysioNet*. 2022.

**Layout after download:**

    <base_dir>/
      train/
        <image_id>.dicom
        ...
      test/
        <image_id>.dicom
        ...
      image_labels_train.csv
      image_labels_test.csv
      annotations_train.csv    ← bounding boxes (train split)
      annotations_test.csv     ← bounding boxes (test split)

**One row per image** (~9,125 total: 7,728 train + 1,397 test).

**Labels:** 15 binary conditions (image-level), one annotation per image.

**Bounding boxes:** stored in ``annotations_{split}.csv`` with pixel-space
``x_min, y_min, x_max, y_max`` coordinates. Normalized to the RadHarmony
convention ``[y_min, y_max, x_min, x_max]`` (values in [0, 1]) using DICOM
image dimensions read at harmonize time. Multiple boxes per image are
aggregated into JSON lists.

**DICOM note:** DICOM headers in this dataset do not contain PatientID or
StudyInstanceUID. ``patient_id`` and ``study_id`` are both set to
``image_id``.
"""

import json
import os

import pandas as pd
import pydicom

from ..base import BaseHarmonizer


# Image-level label columns (original CSV names → snake_case)
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
        base_dir: Dataset root — the directory containing ``train/``,
            ``test/``, and the four CSV files.
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

    def __init__(self, base_dir: str):
        self.base_dir = os.path.expanduser(base_dir)
        super().__init__(csv_path=None)

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["base_dir"] = self.base_dir
        snap.pop("csv_path", None)
        return snap

    def harmonize(self, *args, **kwargs) -> pd.DataFrame:
        rows = []
        for split in ("train", "test"):
            labels_csv = os.path.join(self.base_dir, f"image_labels_{split}.csv")
            annot_csv  = os.path.join(self.base_dir, f"annotations_{split}.csv")
            img_dir    = os.path.join(self.base_dir, split)

            if not os.path.isfile(labels_csv):
                raise FileNotFoundError(f"{labels_csv} not found.")
            if not os.path.isdir(img_dir):
                raise FileNotFoundError(f"{img_dir}/ not found.")

            labels_df = pd.read_csv(labels_csv)
            annot_df  = pd.read_csv(annot_csv) if os.path.isfile(annot_csv) else pd.DataFrame()

            # Build a bbox lookup: image_id → list of [y_min, y_max, x_min, x_max] (normalized)
            bbox_map: dict[str, list] = {}
            blabel_map: dict[str, list] = {}
            # dims cache shared between bbox pass and label pass (avoid double reads)
            dims: dict[str, tuple[int, int]] = {}

            if not annot_df.empty:
                annotated_ids = annot_df["image_id"].unique()
                for img_id in annotated_ids:
                    dcm_path = os.path.join(img_dir, f"{img_id}.dicom")
                    if os.path.isfile(dcm_path):
                        try:
                            dims[img_id] = _read_dicom_dims(dcm_path)
                        except Exception:
                            pass

                for img_id, group in annot_df.groupby("image_id"):
                    if img_id not in dims:
                        continue
                    h, w = dims[img_id]
                    norm_boxes = []
                    labels_list = []
                    for _, row in group.iterrows():
                        y_min = float(row["y_min"]) / h
                        y_max = float(row["y_max"]) / h
                        x_min = float(row["x_min"]) / w
                        x_max = float(row["x_max"]) / w
                        norm_boxes.append([
                            max(0.0, y_min), min(1.0, y_max),
                            max(0.0, x_min), min(1.0, x_max),
                        ])
                        labels_list.append(str(row["class_name"]))
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

                # Binary labels
                for orig_col, snake_col in _LABEL_MAP.items():
                    row[snake_col] = int(lrow.get(orig_col, 0))

                # Image dimensions — reuse cached dims from bbox pass; read fresh otherwise
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

                # Bounding boxes
                if img_id in bbox_map:
                    row["bbox"]        = json.dumps(bbox_map[img_id])
                    row["bbox_labels"] = json.dumps(blabel_map[img_id])
                else:
                    row["bbox"]        = None
                    row["bbox_labels"] = None

                rows.append(row)

        self.df = pd.DataFrame(rows)
        self.df.dropna(subset=["image_path"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self._select_harmonized_columns(self.df)
