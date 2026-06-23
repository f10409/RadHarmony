"""Harmonizer for the RAD-ChestCT dataset."""

import os

import pandas as pd

from ..base import BaseHarmonizer


class RadChestCTHarmonizer(BaseHarmonizer):
    """Harmonize a single RAD-ChestCT split into the standard RadHarmony format.

    Args:
        csv_path: Path to the metadata CSV (``CT_Scan_Metadata_Complete_35747.csv``).
            Must contain ``MRN_DEID`` and ``NoteAcc_DEID`` columns.
        label_csv_path: Path to one raw findings CSV
            (e.g. ``imgtrain_Abnormality_and_Location_Labels.csv``).
        image_base_dir: Root directory containing ``<NoteAcc_DEID>.npz`` files.
            When provided, subjects without an existing ``.npz`` file are dropped.
        bbox_csv_path: Path to the extrema CSV (``Extrema_35747.csv``).
            When provided, the 6 lung-boundary columns are normalised by the
            original volume shape and packed into a ``bbox`` list:
            ``[axis0_min, axis0_max, axis1_min, axis1_max, axis2_min, axis2_max]``
            with all values in ``[0, 1]``.  Multiply by ``(D, H, W)`` of the
            loaded volume to recover voxel coordinates.
    """

    LABEL_COLS = [
        "air_trapping",
        "airspace_disease",
        "aneurysm",
        "arthritis",
        "aspiration",
        "atelectasis",
        "atherosclerosis",
        "bandlike_or_linear",
        "breast_implant",
        "breast_surgery",
        "bronchial_wall_thickening",
        "bronchiectasis",
        "bronchiolectasis",
        "bronchiolitis",
        "bronchitis",
        "cabg",
        "calcification",
        "cancer",
        "cardiomegaly",
        "catheter_or_port",
        "cavitation",
        "chest_tube",
        "clip",
        "congestion",
        "consolidation",
        "coronary_artery_disease",
        "cyst",
        "debris",
        "deformity",
        "density",
        "dilation_or_ectasia",
        "distention",
        "emphysema",
        "fibrosis",
        "fracture",
        "gi_tube",
        "granuloma",
        "groundglass",
        "hardware",
        "heart_failure",
        "heart_valve_replacement",
        "hemothorax",
        "hernia",
        "honeycombing",
        "infection",
        "infiltrate",
        "inflammation",
        "interstitial_lung_disease",
        "lesion",
        "lucency",
        "lung_resection",
        "lymphadenopathy",
        "mass",
        "mucous_plugging",
        "nodule",
        "nodulegr1cm",
        "opacity",
        "other_path",
        "pacemaker_or_defib",
        "pericardial_effusion",
        "pericardial_thickening",
        "plaque",
        "pleural_effusion",
        "pleural_thickening",
        "pneumonia",
        "pneumonitis",
        "pneumothorax",
        "postsurgical",
        "pulmonary_edema",
        "reticulation",
        "scarring",
        "scattered_calc",
        "scattered_nod",
        "secretion",
        "septal_thickening",
        "soft_tissue",
        "staple",
        "stent",
        "sternotomy",
        "suture",
        "tracheal_tube",
        "transplant",
        "tree_in_bud",
        "tuberculosis",
    ]
    LABEL_JOIN_COLS = []

    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_JOIN_COLS = ["NoteAcc_DEID"]
    BBOX_SOURCE_COL = "bbox"
    REPORT_JOIN_COLS = []
    REPORT_PATH_COL = None

    _EXTREMA_COLS = [
        "sup_axis0min",
        "inf_axis0max",
        "ant_axis1min",
        "pos_axis1max",
        "rig_axis2min",
        "lef_axis2max",
    ]

    def __init__(
        self, csv_path, label_csv_path, image_base_dir=None, bbox_csv_path=None
    ):
        super().__init__(
            csv_path=csv_path,
            label_csv_path=label_csv_path,
            bbox_csv_path=bbox_csv_path,
        )
        self.image_base_dir = image_base_dir

    def _label_columns_for_output(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self.LABEL_COLS if c in df.columns]

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        if self.image_base_dir is not None:
            snap["image_base_dir"] = self.image_base_dir
        return snap

    def _prepare_df(self) -> pd.DataFrame:
        """Process label CSV, merge with metadata, return flat DataFrame."""
        # --- label CSV: melt → aggregate → pivot ---
        lbl = pd.read_csv(self.label_csv_path, dtype=str)
        df_long = lbl.melt(
            id_vars=["NoteAcc_DEID"], var_name="Finding", value_name="Presence"
        )
        df_long["Finding"] = df_long["Finding"].apply(lambda x: x.split("*")[0])
        df_long = (
            df_long.groupby(["NoteAcc_DEID", "Finding"])["Presence"].max().reset_index()
        )
        df_flat = df_long.pivot(
            index="NoteAcc_DEID", columns="Finding", values="Presence"
        ).reset_index()
        df_flat.columns.name = None

        finding_cols = [c for c in df_flat.columns if c != "NoteAcc_DEID"]
        for col in finding_cols:
            df_flat[col] = df_flat[col].astype(float)
        self.LABEL_COLS = sorted(finding_cols)

        # --- merge with metadata to get MRN_DEID ---
        meta = pd.read_csv(self.csv_path, dtype=str)[["NoteAcc_DEID", "MRN_DEID"]]
        df_flat = df_flat.merge(meta, on="NoteAcc_DEID", how="left")

        # --- filter to existing .npz files ---
        if self.image_base_dir is not None:
            exists = df_flat["NoteAcc_DEID"].apply(
                lambda x: os.path.exists(os.path.join(self.image_base_dir, x + ".npz"))
            )
            df_flat = df_flat[exists].reset_index(drop=True)

        return df_flat

    def _preprocess_bbox_df(self, bbox_df: pd.DataFrame) -> pd.DataFrame:
        """Normalise the 6 extrema columns by original volume shape → [0, 1]."""
        cols = ["NoteAcc_DEID"] + self._EXTREMA_COLS + ["shape0", "shape1", "shape2"]
        bbox_df = bbox_df[cols].copy()
        for col in self._EXTREMA_COLS + ["shape0", "shape1", "shape2"]:
            bbox_df[col] = pd.to_numeric(bbox_df[col], errors="coerce")

        bbox_df["bbox"] = bbox_df.apply(
            lambda r: [
                r["sup_axis0min"] / r["shape0"],
                r["inf_axis0max"] / r["shape0"],
                r["ant_axis1min"] / r["shape1"],
                r["pos_axis1max"] / r["shape1"],
                r["rig_axis2min"] / r["shape2"],
                r["lef_axis2max"] / r["shape2"],
            ],
            axis=1,
        )
        return bbox_df[["NoteAcc_DEID", "bbox"]]

    def harmonize(self, **kwargs) -> pd.DataFrame:
        self._harmonized_df_override = None
        self.df = self._prepare_df()

        self._build_patient_id()
        self._build_study_id()
        self._build_image_path()
        self._build_view_position()
        self._build_mask_path()
        self._build_bbox()
        # Single bbox per image represents the lung region
        if "bbox" in self.df.columns:
            self.df["bbox_labels"] = self.df["bbox"].apply(
                lambda x: ["lung"] if isinstance(x, list) and len(x) > 0 else []
            )
        self._build_report()

        self.df.dropna(subset=["image_path"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self._select_harmonized_columns(self.df)

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["MRN_DEID"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["NoteAcc_DEID"].astype(str)

    def _build_image_path(self) -> None:
        self.df["image_path"] = self.df["NoteAcc_DEID"] + ".npz"
