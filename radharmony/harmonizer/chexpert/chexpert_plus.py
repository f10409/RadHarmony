"""Harmonizer for the CheXpert-Plus dataset (DICOM source, inline reports)."""

import json
import os
import warnings

import pandas as pd

from .._labels import chexpert_codes
from ..base import BaseHarmonizer


class CheXpertPlusHarmonizer(BaseHarmonizer):
    """Harmonize CheXpert-Plus into the standard RadHarmony format.

    Reads ``df_chexpert_plus_240401.csv`` as the primary CSV and an optional
    ``impression_fixed.json`` (JSONL) for CheXpert-style labels keyed by
    ``path_to_image``. The labels come from the impression, as in CheXpert v1.
    The labels of ``report_fixed.json`` do not follow the image (D-48).

    ``image_path`` is set to ``<path_to_dcm>`` (for example
    ``train/patient42142/study5/view1_frontal.dcm``), making it directly
    usable against a ``base_image_dir`` that points at
    ``chexpertplus/DICOM/Uncompressed/``.

    The ``report`` column from the main CSV is preserved as-is.

    Args:
        csv_path: Path to ``df_chexpert_plus_240401.csv``.
        label_json_path: Path to ``impression_fixed.json`` (JSONL).  When
            ``None``, no classification labels are loaded.
    """

    #: ``age``, ``sex``, ``race`` and ``ethnicity`` are plain columns of
    #: ``df_chexpert_plus_240401.csv``. No join is needed.
    EXTRA_OUTPUT_COLS = ["age", "race", "ethnicity"]

    #: ``age`` is in years (0 to 89). The base step makes ``age_years`` (D-06).
    AGE_SOURCE_COL = "age"
    AGE_SOURCE_UNIT = "years"
    #: ``sex`` is Male, Female or Unknown. The base step maps it in place (D-26).
    SEX_SOURCE_COL = "sex"

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
        self.drop_uncertain, self.not_mentioned = True, "negative"  # set again by harmonize()

    def harmonize(self, *args, drop_uncertain: bool = True, not_mentioned: str = "negative", **kwargs):
        """Run the base steps with the CheXpert-style label rule (D-05).

        Args:
            drop_uncertain: If ``True`` (the default), drop the rows with an uncertain label (-1).
            not_mentioned: ``"negative"`` (the default) gives 0 for a blank label, ``"nan"`` keeps
                NaN (D-05).
        """
        self.drop_uncertain, self.not_mentioned = drop_uncertain, not_mentioned
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
        self._warn_missing_patient_dirs()

    def _warn_missing_patient_dirs(self) -> None:
        """Warn when rows have no ``<split>/<patient>/`` folder under ``base_image_dir``.

        A copy of the DICOM release can miss some patients. For example, the
        NAS copy misses 1,777 of 64,725 patients. One ``listdir`` per split
        folder keeps the check fast. The rows stay in the table (D-02).
        """
        # An empty table has no parts to split (V-H1-38).
        if self.df.empty or not self.base_image_dir or not os.path.isdir(self.base_image_dir):
            return
        # A split class has only its own rows here, because _KeepSplitMixin cuts them first (V-H1-45, D-30).
        parts = self.df["image_path"].str.split("/", n=2, expand=True)
        on_disk = {}
        for split in parts[0].dropna().unique():
            d = os.path.join(self.base_image_dir, split)
            on_disk[split] = set(os.listdir(d)) if os.path.isdir(d) else set()
        missing = [p not in on_disk.get(s, ()) for s, p in zip(parts[0], parts[1], strict=True)]
        n = sum(missing)
        if n:
            warnings.warn(
                f"CheXpert Plus: {n} of {len(missing)} source rows have no patient folder "
                f"under {self.base_image_dir}. Call verify_images() on the dataset "
                "to drop them.",
                UserWarning,
                stacklevel=2,
            )

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

        self.df = chexpert_codes(self.df, present_label_cols, self.not_mentioned, self.drop_uncertain)

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
        # Frontal views: AP or PA from ap_pa. Lateral views: LATERAL (Q-13). A frontal view
        # with LL in ap_pa has no known AP or PA, so it gives NaN, as for Emory (Q-14).
        def _vp(row):
            lat = row.get("frontal_lateral", "")
            if isinstance(lat, str) and lat.lower() == "lateral":
                return "LATERAL"
            ap = row.get("ap_pa", "")
            if not (isinstance(ap, str) and ap):
                return "Frontal"
            return ap if ap.strip().upper() in ("AP", "PA") else None

        self.df["view_position"] = self.df.apply(_vp, axis=1)
