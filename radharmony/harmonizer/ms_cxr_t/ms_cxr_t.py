"""Harmonizer for MS-CXR-T Temporal Sentences and Image Pairs.

Temporal benchmark. Its image task extends the progression labels of
Chest ImaGenome. It has 1,045 image pairs from
MIMIC-CXR-JPG, with progression labels (stable, improving, worsening) for up
to 5 findings. Each row links a current image to a previous image of the same
patient.

Source: https://physionet.org/content/ms-cxr-t/ (MIMIC credentialed access).
Cite: Bannur et al., Learning to Exploit Temporal Structure for Biomedical
Vision-Language Processing, CVPR 2023. File needed:
``MS_CXR_T_temporal_image_classification_v1.0.0.csv``.

Images: MIMIC-CXR-JPG 2.0.0. ``base_image_dir`` is its ``files/`` folder, the
same as for MS-CXR and MIMIC-CXR-JPG (D-04).

Layout:

    <csv_path>   ← MS_CXR_T_temporal_image_classification_v1.0.0.csv

    <base_image_dir>/          ← MIMIC-CXR-JPG 2.0.0 files/
      p10/
        p10002428/
          s55758034/
            3bea0373-....jpg
        ...

One row for each image pair (1,045 rows).

Harmonized columns:
  ``patient_id``, ``study_id``, ``image_path`` (current image),
  ``previous_image_path``, ``previous_study_id``,
  ``consolidation_progression``, ``edema_progression``,
  ``pleural_effusion_progression``, ``pneumonia_progression``,
  ``pneumothorax_progression``.
  The progression columns are float, because of NaN: -1 improving, 0 stable, 1 worsening,
  NaN not annotated.
  ``consolidation_label_quality``, ``edema_label_quality``,
  ``pleural_effusion_label_quality``, ``pneumonia_label_quality``,
  ``pneumothorax_label_quality``.
  The quality columns are str: one_expert, multiple_experts, disagreement or NaN.
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
        mimic_base_dir: The ``files/`` folder of MIMIC-CXR-JPG 2.0.0. The
            harmonizer stores it for save and load. It does not read it.
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
        """Convert ``p10/pXXX/sYYY/<hash>`` to ``p10/pXXX/sYYY/<hash>.jpg``.

        The path is relative to the ``files/`` folder, like the other
        MIMIC-based tables (D-04). Before 2026-10-28 it started with
        ``files/`` (V-H2-8).
        """
        return dicom_id + ".jpg"

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["subject_id"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["study_id"].astype(str)
        self.df["previous_study_id"] = self.df["previous_study_id"].astype(str)

    def _build_image_path(self) -> None:
        self.df["image_path"] = self.df["dicom_id"].apply(self._dicom_id_to_path)
        self.df["previous_image_path"] = self.df["previous_dicom_id"].apply(self._dicom_id_to_path)

    def _read_source(self) -> pd.DataFrame:
        """Read the CSV. Map each progression word to its code (-1, 0, 1).

        The map runs before the id filter, so a column keeps the dtype of the full table. The
        quality columns keep the source text.
        """
        df = super()._read_source()
        for col in _PROGRESSION_COLS:
            df[col] = df[col].map(_PROGRESSION_MAP)
        return df
