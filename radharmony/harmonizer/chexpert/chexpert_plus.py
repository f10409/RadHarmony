"""Harmonizer for the CheXpert-Plus dataset (DICOM source, inline reports)."""

import json

import numpy as np
import pandas as pd

from ..base import BaseHarmonizer


class CheXpertPlusHarmonizer(BaseHarmonizer):
    """Harmonize CheXpert-Plus into the standard RadHarmony format.

    Reads ``df_chexpert_plus_240401.csv`` as the primary CSV and an optional
    ``report_fixed.json`` (JSONL) for CheXpert-style labels keyed by
    ``path_to_image``.

    ``image_path`` is set to ``<path_to_dcm>`` (e.g.
    ``train/patient42142/study5/view1_frontal.dcm``), making it directly
    usable against a ``base_image_dir`` that points at
    ``chexpertplus/DICOM/Uncompressed/``.

    The ``report`` column from the main CSV is preserved as-is.

    Args:
        csv_path: Path to ``df_chexpert_plus_240401.csv``.
        label_json_path: Path to ``report_fixed.json`` (JSONL).  When
            ``None``, no classification labels are loaded.
    """

    LABEL_COLS = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "No Finding",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices",
    ]

    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = "ap_pa"
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None

    def __init__(
        self, csv_path: str, label_json_path: str = None, base_image_dir: str = None
    ):
        super().__init__(csv_path=csv_path)
        self.label_json_path = label_json_path
        self.base_image_dir = base_image_dir
        self.drop_uncertain = True

    def harmonize(self, *args, drop_uncertain: bool = True, **kwargs):
        self.drop_uncertain = drop_uncertain
        return super().harmonize(*args, **kwargs)

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        if self.label_json_path is not None:
            snap["label_json_path"] = self.label_json_path
        return snap

    def _build_patient_id(self) -> None:
        # path_to_image: "train/patient42142/study5/view1_frontal.jpg"
        self.df["patient_id"] = self.df["path_to_image"].apply(
            lambda x: x.split("/")[1]
        )

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["path_to_image"].apply(
            lambda x: f"{x.split('/')[1]}_{x.split('/')[2]}"
        )

    def _build_image_path(self) -> None:
        # path_to_dcm: "train/patient42142/study5/view1_frontal.dcm"
        # Relative to DICOM/Uncompressed/
        self.df["image_path"] = self.df["path_to_dcm"]

    def _build_labels(self) -> None:
        if self.label_json_path is None:
            return

        records = []
        with open(self.label_json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        label_df = pd.DataFrame(records)

        present_label_cols = [c for c in self.LABEL_COLS if c in label_df.columns]
        self.df = self.df.merge(
            label_df[["path_to_image"] + present_label_cols],
            on="path_to_image",
            how="left",
        )

        # NaN (not annotated) → 0.0 first, then handle -1 (uncertain)
        for col in present_label_cols:
            self.df[col] = self.df[col].fillna(0.0)
            self.df[col] = self.df[col].replace(-1, np.nan)

        if self.drop_uncertain:
            self.df.dropna(subset=present_label_cols, how="any", inplace=True)

        for col in present_label_cols:
            self.df.rename(columns={col: col.lower().replace(" ", "_")}, inplace=True)

    def _select_harmonized_columns(self, df=None) -> pd.DataFrame:
        result = super()._select_harmonized_columns(df)
        src = self.df if df is None else df
        if src is not None and "split" in src.columns:
            result = result.copy()
            result["split"] = src["split"].values
        return result

    def _build_view_position(self) -> None:
        # Frontal views: use ap_pa (AP / PA); Lateral views: "Lateral"
        def _vp(row):
            lat = row.get("frontal_lateral", "")
            if isinstance(lat, str) and lat.lower() == "lateral":
                return "Lateral"
            ap = row.get("ap_pa", "")
            return ap if isinstance(ap, str) and ap else "Frontal"

        self.df["view_position"] = self.df.apply(_vp, axis=1)
