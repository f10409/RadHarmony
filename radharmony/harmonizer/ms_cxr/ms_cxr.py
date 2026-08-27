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

**Images:** ``base_image_dir`` must point at the MIMIC-CXR ``files/`` directory
  (the leading ``files/`` in the CSV ``path`` column is stripped). This may be
  either a MIMIC-CXR-JPG tree (``.jpg``, as named in the CSV) or the MIMIC-CXR
  DICOM tree (``.dcm``) — the extension is auto-detected by probing one file,
  so no separate flag is needed.

**Layout:**

    <csv_path>   ← MS_CXR_Local_Alignment_v1.1.0.csv

    <base_image_dir>/           ← the files/ directory
      p10/
        p10233088/
          s54276838/
            675d792f-a3521e48-5eec8573-1e81d644-e60c34f8.dcm   (or .jpg)
        ...

**No subject_id / study_id columns:** the v1.1.0 CSV only carries a MIMIC-style
``path``; ``patient_id`` / ``study_id`` are derived from its components.

**One row per image** (1,047 rows). Multiple annotations per image are
aggregated into parallel Python lists (``bbox`` / ``bbox_labels`` /
``label_text``).

**Harmonized columns:**
  ``patient_id``, ``study_id``, ``image_path``, ``split``,
  ``bbox`` (JSON list of [x,y,w,h] per annotation),
  ``bbox_labels`` (JSON list of category names),
  ``label_text`` (JSON list of phrases),
  ``image_width``, ``image_height``,
  ``Atelectasis``, ``Cardiomegaly``, ``Consolidation``, ``Edema``,
  ``Lung Opacity``, ``Pleural Effusion``, ``Pneumonia``, ``Pneumothorax``
"""

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
        mimic_base_dir: MIMIC-CXR ``files/`` directory (JPG or DICOM tree).
            Used to resolve image paths and auto-detect the on-disk extension;
            not required if you only need the harmonized DataFrame without
            image loading (defaults the extension to ``.jpg``).
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

    def _resolve_image_ext(self, sample_path: str) -> str:
        """Return the on-disk image extension under ``mimic_base_dir``.

        The CSV names ``.jpg`` files (MIMIC-CXR-JPG). When ``base_image_dir``
        instead points at the MIMIC-CXR DICOM tree, only ``.dcm`` files exist;
        detect that by probing one file and fall back to ``.dcm``. Keeps ``.jpg``
        when the base dir can't be checked so the DataFrame is still buildable
        without images.
        """
        default_ext = os.path.splitext(sample_path)[1] or ".jpg"
        if not self.mimic_base_dir:
            return default_ext
        rel = sample_path.removeprefix("files/")
        stem = os.path.splitext(rel)[0]
        for ext in (default_ext, ".dcm", ".jpg"):
            if os.path.exists(os.path.join(self.mimic_base_dir, stem + ext)):
                return ext
        return default_ext

    def harmonize(self, *args, **kwargs) -> pd.DataFrame:
        raw = pd.read_csv(self.csv_path)

        # The v1.1.0 CSV has no subject_id / study_id columns — only a MIMIC-style
        # ``path`` (files/p10/p10233088/s54276838/<dicom_id>.jpg). Derive the IDs
        # from the path components (strip the leading p/s to match MIMIC-CXR's
        # numeric subject_id / study_id convention).
        path_parts = raw["path"].str.split("/", expand=True)
        raw["_patient_id"] = path_parts[2].str.lstrip("p")
        raw["_study_id"] = path_parts[3].str.lstrip("s")

        # Decide the on-disk image extension. base_image_dir may point at either
        # a MIMIC-CXR-JPG tree (.jpg, as named in the CSV) or the MIMIC-CXR DICOM
        # tree (.dcm). Probe one file and swap the extension across the board when
        # only the DICOM variant is present.
        img_ext = self._resolve_image_ext(raw["path"].iloc[0])

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

            # image_path relative to base_image_dir (strip leading "files/",
            # matching base_image_dir = .../files) with the resolved extension.
            image_path = first["path"].removeprefix("files/")
            image_path = os.path.splitext(image_path)[0] + img_ext

            row = {
                "patient_id": str(first["_patient_id"]),
                "study_id":   str(first["_study_id"]),
                "image_path": image_path,
                "split":      first["split"],
                # Stored as Python lists (VinDr convention) so the transform
                # pipeline and app receive parsed boxes/labels, not JSON strings.
                "bbox":       norm_bboxes,
                "bbox_labels": categories,
                "label_text": phrases,
                "image_width":  int(first["image_width"]),
                "image_height": int(first["image_height"]),
            }
            row.update(findings)
            rows.append(row)

        self.df = pd.DataFrame(rows)
        self.df.reset_index(drop=True, inplace=True)

        return self._select_harmonized_columns(self.df)
