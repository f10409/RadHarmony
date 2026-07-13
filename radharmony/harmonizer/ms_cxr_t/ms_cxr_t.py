"""Harmonizer for MS-CXR-T Temporal Sentences and Image Pairs.

Temporal benchmark built on top of MS-CXR. 1,045 longitudinal image pairs
from MIMIC-CXR-JPG, annotated with progression labels (stable / improving /
worsening) for up to 5 pathology findings. Each row links a *current* image
to a *previous* image from the same patient.

**Source (PhysioNet — requires MIMIC credentialed access):**
  https://physionet.org/content/ms-cxr-t/

**Cite:**
  Dalla Serra F, et al. Making the Most of Text Semantics to Improve
  Biomedical Vision-Language Processing. ECCV 2022.

**Files needed:**
  MS_CXR_T_temporal_image_classification_v1.0.0.csv

**Images:** MIMIC-CXR-JPG 2.0.0 root (same as MS-CXR). ``base_image_dir``
  must point at the directory containing ``files/``.

**Layout:**

    <csv_path>   ← MS_CXR_T_temporal_image_classification_v1.0.0.csv

    <base_image_dir>/
      files/
        p10/
          p10002428/
            s55758034/
              3bea0373-....jpg
          ...

**One row per image pair** (1,045 rows).

**Harmonized columns:**
  ``patient_id``, ``study_id``, ``image_path`` (current image),
  ``previous_image_path``, ``previous_study_id``,
  ``consolidation_progression``, ``edema_progression``,
  ``pleural_effusion_progression``, ``pneumonia_progression``,
  ``pneumothorax_progression``  (int: -1=improving, 0=stable, 1=worsening, NaN=not annotated),
  ``consolidation_label_quality``, ``edema_label_quality``,
  ``pleural_effusion_label_quality``, ``pneumonia_label_quality``,
  ``pneumothorax_label_quality``  (str: one_expert / multiple_experts / disagreement / NaN)
"""

import os

import pandas as pd

from radharmony.harmonizer.base import BaseHarmonizer


_PROGRESSION_FINDINGS = [
    "consolidation",
    "edema",
    "pleural_effusion",
    "pneumonia",
    "pneumothorax",
]

_PROGRESSION_MAP = {"improving": -1, "stable": 0, "worsening": 1}

_PROGRESSION_COLS = [f"{f}_progression" for f in _PROGRESSION_FINDINGS]
_QUALITY_COLS = [f"{f}_label_quality" for f in _PROGRESSION_FINDINGS]


class MSCXRTHarmonizer(BaseHarmonizer):
    """Harmonize the MS-CXR-T temporal image pairs dataset.

    Args:
        csv_path: Path to ``MS_CXR_T_temporal_image_classification_v1.0.0.csv``.
        mimic_base_dir: Root of MIMIC-CXR-JPG 2.0.0 (contains ``files/``).
            Used only to resolve absolute image paths.
    """

    LABEL_COLS = []

    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    EXTRA_OUTPUT_COLS = (
        ["previous_image_path", "previous_study_id"]
        + _PROGRESSION_COLS
        + _QUALITY_COLS
    )

    def __init__(self, csv_path: str, mimic_base_dir: str = None):
        self.mimic_base_dir = os.path.expanduser(mimic_base_dir) if mimic_base_dir else None
        super().__init__(csv_path=csv_path)

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["mimic_base_dir"] = self.mimic_base_dir
        return snap

    @staticmethod
    def _dicom_id_to_path(dicom_id: str) -> str:
        """Convert ``p10/pXXX/sYYY/<hash>`` to ``files/p10/pXXX/sYYY/<hash>.jpg``."""
        return "files/" + dicom_id + ".jpg"

    def harmonize(self, *args, **kwargs) -> pd.DataFrame:
        raw = pd.read_csv(self.csv_path)

        image_path = raw["dicom_id"].apply(self._dicom_id_to_path)
        previous_image_path = raw["previous_dicom_id"].apply(self._dicom_id_to_path)

        df = pd.DataFrame({
            "patient_id":           raw["subject_id"].astype(str),
            "study_id":             raw["study_id"].astype(str),
            "image_path":           image_path,
            "previous_image_path":  previous_image_path,
            "previous_study_id":    raw["previous_study_id"].astype(str),
        })

        for finding in _PROGRESSION_FINDINGS:
            prog_col = f"{finding}_progression"
            qual_col = f"{finding}_label_quality"
            df[prog_col] = raw[prog_col].map(_PROGRESSION_MAP)
            df[qual_col] = raw[qual_col]

        self.df = df.reset_index(drop=True)
        return self._select_harmonized_columns(self.df)
