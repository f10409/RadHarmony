"""Harmonizer for the MS-CXR Local Alignment dataset.

Phrase grounding benchmark for chest X-rays with explicit phrase-to-bounding-box
annotations. 1,047 images from MIMIC-CXR-JPG with 1,448 annotated
(phrase, bounding box) pairs across 8 pathology categories.

**Source (PhysioNet — requires MIMIC credentialed access):**
  https://physionet.org/content/ms-cxr/

**Cite:**
  Boecking B, et al. Making the Most of Text Semantics to Improve Biomedical
  Vision-Language Processing. ECCV 2022.

**Files needed:**
  MS_CXR_Local_Alignment_v1.1.0.csv  (primary — from ms-cxr PhysioNet)

**Images:** MIMIC-CXR-JPG — ``base_image_dir`` must point at the MIMIC-CXR-JPG
  2.0.0 root (the directory containing ``files/``), e.g.
  ``.../mimic-cxr-jpg/2.0.0/``.

**Layout:**

    <csv_path>   ← MS_CXR_Local_Alignment_v1.1.0.csv

    <base_image_dir>/
      files/
        p10/
          p10233088/
            s54276838/
              675d792f-a3521e48-5eec8573-1e81d644-e60c34f8.jpg
          ...

**One row per image** (1,047 rows). Multiple annotations per image are
aggregated into JSON lists.

**Harmonized columns:**
  ``patient_id``, ``study_id``, ``image_path``, ``split``,
  ``bbox`` (JSON list of [x,y,w,h] per annotation),
  ``bbox_labels`` (JSON list of category names),
  ``label_text`` (JSON list of phrases),
  ``image_width``, ``image_height``,
  ``Atelectasis``, ``Cardiomegaly``, ``Consolidation``, ``Edema``,
  ``Lung Opacity``, ``Pleural Effusion``, ``Pneumonia``, ``Pneumothorax``
"""

import json
import os

import pandas as pd

from radharmony.harmonizer.base import BaseHarmonizer


# Original category names from the CSV
_CATEGORY_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Lung Opacity", "Pleural Effusion", "Pneumonia", "Pneumothorax",
]

# snake_cased column names used in the harmonized DataFrame
FINDING_COLS = [c.lower().replace(" ", "_") for c in _CATEGORY_NAMES]

# Map CSV category name → snake_case column name
_CAT_TO_COL = {cat: cat.lower().replace(" ", "_") for cat in _CATEGORY_NAMES}


class MSCXRHarmonizer(BaseHarmonizer):
    """Harmonize the MS-CXR Local Alignment dataset.

    Args:
        csv_path: Path to ``MS_CXR_Local_Alignment_v1.1.0.csv``.
        mimic_base_dir: Root of MIMIC-CXR-JPG 2.0.0 (contains ``files/``).
            Used only to resolve absolute image paths; not required if you
            only need the harmonized DataFrame without image loading.
    """

    LABEL_COLS = FINDING_COLS

    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    EXTRA_OUTPUT_COLS = ["split", "label_text", "image_width", "image_height"]

    def __init__(self, csv_path: str, mimic_base_dir: str = None):
        self.mimic_base_dir = os.path.expanduser(mimic_base_dir) if mimic_base_dir else None
        super().__init__(csv_path=csv_path)

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["mimic_base_dir"] = self.mimic_base_dir
        return snap

    def harmonize(self, *args, **kwargs) -> pd.DataFrame:
        raw = pd.read_csv(self.csv_path)

        # --- aggregate per image ---
        rows = []
        for dicom_id, group in raw.groupby("dicom_id", sort=False):
            first = group.iloc[0]

            # binary finding flags
            findings = {col: 0 for col in FINDING_COLS}
            for cat in group["category_name"]:
                col = _CAT_TO_COL.get(cat)
                if col:
                    findings[col] = 1

            # aggregated bbox and phrase lists
            # Convert pixel [x, y, w, h] → normalized [y_min, y_max, x_min, x_max]
            # to match the RadHarmony bbox convention expected by the transform pipeline.
            img_w = int(first["image_width"])
            img_h = int(first["image_height"])
            norm_bboxes = [
                [
                    row_b["y"] / img_h,
                    (row_b["y"] + row_b["h"]) / img_h,
                    row_b["x"] / img_w,
                    (row_b["x"] + row_b["w"]) / img_w,
                ]
                for _, row_b in group[["x", "y", "w", "h"]].iterrows()
            ]
            phrases = group["label_text"].tolist()
            categories = group["category_name"].tolist()

            # image_path relative to mimic_base_dir
            image_path = first["path"]

            row = {
                "patient_id": str(first["subject_id"]),
                "study_id":   str(first["study_id"]),
                "image_path": image_path,
                "split":      first["split"],
                "bbox":       json.dumps(norm_bboxes),
                "bbox_labels": json.dumps(categories),
                "label_text": json.dumps(phrases),
                "image_width":  int(first["image_width"]),
                "image_height": int(first["image_height"]),
            }
            row.update(findings)
            rows.append(row)

        self.df = pd.DataFrame(rows)
        self.df.reset_index(drop=True, inplace=True)

        return self._select_harmonized_columns(self.df)
