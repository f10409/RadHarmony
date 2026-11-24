"""Harmonizer for the MS-CXR Local Alignment dataset.

Phrase grounding benchmark for chest X-rays with explicit phrase-to-bounding-box
annotations. 1,047 images from MIMIC-CXR-JPG with 1,448 annotated
(phrase, bounding box) pairs across 8 pathology categories.

Source: https://physionet.org/content/ms-cxr/ (MIMIC credentialed access).
Cite: Boecking et al., Making the Most of Text Semantics to Improve Biomedical
Vision-Language Processing, ECCV 2022. File needed:
``MS_CXR_Local_Alignment_v1.1.0.csv``.

Images: ``base_image_dir`` is the MIMIC-CXR ``files/`` folder (the code removes
the leading ``files/`` of the CSV ``path`` column). It can be a MIMIC-CXR-JPG
tree (``.jpg``, as in the CSV) or the MIMIC-CXR DICOM tree (``.dcm``). The
code tests one file to find the extension, so no flag is needed.

Layout:

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

- ``patient_id``, ``study_id``, ``image_path``, ``split``.
- ``bbox``: one ``[y_min, y_max, x_min, x_max]`` box per annotation, as fractions (D-07).
- ``bbox_labels``: one category name per box. ``label_text``: one phrase per box.
- ``image_width``, ``image_height``.
- 8 labels: ``atelectasis``, ``cardiomegaly``, ``consolidation``, ``edema``,
  ``lung_opacity``, ``pleural_effusion``, ``pneumonia``, ``pneumothorax``.
"""

import os

import pandas as pd

from radharmony.harmonizer._bbox import box_label, clip_boxes
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
        mimic_base_dir: MIMIC-CXR ``files/`` folder (JPG or DICOM tree). It
            resolves image paths and finds the file extension. Without it the
            extension is ``.jpg``, which is fine if you do not load images.
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

        The CSV names ``.jpg`` files (MIMIC-CXR-JPG). If ``base_image_dir`` is
        the MIMIC-CXR DICOM tree, only ``.dcm`` files exist. The code tests one
        file and then uses ``.dcm``. It keeps ``.jpg`` if the folder cannot be
        checked, so the table still builds without images.
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

    def preprocess(self, output_dir: str, preprocessor, base_image_dir: str = None, **kwargs) -> None:
        """Run :meth:`BaseHarmonizer.preprocess`. ``base_image_dir`` defaults to ``mimic_base_dir``."""
        super().preprocess(output_dir, preprocessor, base_image_dir=base_image_dir or self.mimic_base_dir,
                           **kwargs)

    def preprocess_masks(self, *args, **kwargs) -> None:
        """Raise ``ValueError``: MS-CXR has boxes, not masks, so ``mask_output_dir`` is an error."""
        raise ValueError("MS-CXR has boxes, not masks. Leave mask_output_dir as None.")

    def _read_source(self) -> pd.DataFrame:
        """Read the CSV and return one row per image, with its boxes, phrases and labels."""
        raw = super()._read_source()

        # The v1.1.0 CSV has no subject_id or study_id. It has only a MIMIC
        # ``path`` (files/p10/p10233088/s54276838/<dicom_id>.jpg). Take the IDs
        # from the path. Remove the leading p and s to match the MIMIC numbers.
        path_parts = raw["path"].str.split("/", expand=True)
        raw["_patient_id"] = path_parts[2].str.lstrip("p")
        raw["_study_id"] = path_parts[3].str.lstrip("s")

        # base_image_dir can be a JPG tree or a DICOM tree. Test one file, and
        # use .dcm for all paths if only DICOM files exist.
        img_ext = self._resolve_image_ext(raw["path"].iloc[0])

        rows = []
        for _, group in raw.groupby("dicom_id", sort=False):
            first = group.iloc[0]

            # Binary finding flags.
            findings = {col: 0 for col in FINDING_COLS}
            for cat in group["category_name"]:
                col = _CAT_TO_COL.get(cat)
                if col:
                    findings[col] = 1

            # Lists of boxes and phrases.
            # Convert pixel [x, y, w, h] to normalized [y_min, y_max, x_min, x_max],
            # the bbox format of the transform pipeline.
            # Python floats, not numpy floats: a CSV export then keeps JSON text (V-H2-35).
            img_w = int(first["image_width"])
            img_h = int(first["image_height"])
            norm_bboxes = [
                [
                    float(row_b["y"] / img_h),
                    float((row_b["y"] + row_b["h"]) / img_h),
                    float(row_b["x"] / img_w),
                    float((row_b["x"] + row_b["w"]) / img_w),
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
                "bbox_labels": [box_label(c) for c in categories],
                "label_text": phrases,
                "image_width":  int(first["image_width"]),
                "image_height": int(first["image_height"]),
            }
            row.update(findings)
            rows.append(row)

        return pd.DataFrame(rows)

    # _read_source() gives patient_id, study_id and image_path.
    def _build_patient_id(self) -> None:
        pass

    def _build_study_id(self) -> None:
        pass

    def _build_image_path(self) -> None:
        pass

    def _build_bbox(self) -> None:
        clip_boxes(self.df, "MS-CXR", per_box=("bbox_labels", "label_text"))  # D-07
