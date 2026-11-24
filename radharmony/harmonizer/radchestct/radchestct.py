"""Harmonizer for the RAD-ChestCT dataset."""

import math
import os

import pandas as pd

from .._bbox import clip_boxes
from ..base import BaseHarmonizer


class RadChestCTHarmonizer(BaseHarmonizer):
    """Harmonize a single RAD-ChestCT split into the standard RadHarmony format.

    Args:
        csv_path: Path to the metadata CSV (``CT_Scan_Metadata_Complete_35747.csv``).
            Must contain ``MRN_DEID`` and ``NoteAcc_DEID`` columns.
        label_csv_path: Path to one raw findings CSV
            (for example ``imgtrain_Abnormality_and_Location_Labels.csv``).
        image_base_dir: Old name of ``base_image_dir``. It still works.
        bbox_csv_path: Path to the extrema CSV (``Extrema_35747.csv``). If given, the 6 lung
            boundary columns are divided by the original volume shape. They become a ``bbox``
            list ``[axis0_min, axis0_max, axis1_min, axis1_max, axis2_min, axis2_max]`` with
            values from 0 to 1. Multiply by ``(D, H, W)`` of the loaded volume to get voxels.
        base_image_dir: Root folder with the ``<NoteAcc_DEID>.npz`` files. If given, a subject
            with no ``.npz`` file is dropped. ``verify_images()`` uses it by default (D-04).
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
        self, csv_path, label_csv_path, image_base_dir=None, bbox_csv_path=None, base_image_dir=None
    ):
        super().__init__(
            csv_path=csv_path,
            label_csv_path=label_csv_path,
            bbox_csv_path=bbox_csv_path,
        )
        if image_base_dir and base_image_dir and image_base_dir != base_image_dir:
            raise ValueError(
                f"base_image_dir={base_image_dir!r} and its old name image_base_dir={image_base_dir!r} "
                "name 2 folders. Pass base_image_dir only."
            )
        # All other harmonizers and verify_images() use the name base_image_dir (V-H2-33).
        self.base_image_dir = base_image_dir or image_base_dir
        self.image_base_dir = self.base_image_dir

    @classmethod
    def load_from_saved(cls, path: str, **init_overrides):
        """Load a saved harmonizer. A new folder in either name replaces the saved folder.

        A file saved before 2026-11-03 holds ``image_base_dir``. Without this step, a new
        ``base_image_dir`` and the saved ``image_base_dir`` name 2 folders and raise.
        """
        for name, other in (("base_image_dir", "image_base_dir"), ("image_base_dir", "base_image_dir")):
            if name in init_overrides and other not in init_overrides:
                init_overrides[other] = init_overrides[name]
        return super().load_from_saved(path, **init_overrides)

    def _label_columns_for_output(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self.LABEL_COLS if c in df.columns]

    def _read_source(self) -> pd.DataFrame:
        """Read the label CSV, merge it with the metadata and return one flat table.

        It also makes the label columns, so ``_build_labels()`` has nothing to do.
        """
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

        # The metadata gives MRN_DEID.
        meta = pd.read_csv(self.csv_path, dtype=str)[["NoteAcc_DEID", "MRN_DEID"]]
        df_flat = df_flat.merge(meta, on="NoteAcc_DEID", how="left")

        # Keep only the subjects with a .npz file.
        if self.base_image_dir is not None:
            exists = df_flat["NoteAcc_DEID"].apply(
                lambda x: os.path.exists(os.path.join(self.base_image_dir, x + ".npz"))
            )
            df_flat = df_flat[exists].reset_index(drop=True)

        return df_flat

    def _preprocess_bbox_df(self, bbox_df: pd.DataFrame) -> pd.DataFrame:
        """Divide the 6 extrema columns by the original volume shape, to get values from 0 to 1.

        A row with a value that is not finite gets no box (``[]``). The CSV has
        ``inf`` and ``-inf`` for 173 of its 35,747 volumes.
        """
        cols = ["NoteAcc_DEID"] + self._EXTREMA_COLS + ["shape0", "shape1", "shape2"]
        bbox_df = bbox_df[cols].copy()
        for col in self._EXTREMA_COLS + ["shape0", "shape1", "shape2"]:
            bbox_df[col] = pd.to_numeric(bbox_df[col], errors="coerce")

        def _box(r) -> list:
            # Python floats, so a table saved as CSV keeps a JSON-readable box.
            box = [
                float(r["sup_axis0min"] / r["shape0"]),
                float(r["inf_axis0max"] / r["shape0"]),
                float(r["ant_axis1min"] / r["shape1"]),
                float(r["pos_axis1max"] / r["shape1"]),
                float(r["rig_axis2min"] / r["shape2"]),
                float(r["lef_axis2max"] / r["shape2"]),
            ]
            return box if all(math.isfinite(v) for v in box) else []

        bbox_df["bbox"] = bbox_df.apply(_box, axis=1)
        return bbox_df[["NoteAcc_DEID", "bbox"]]

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["MRN_DEID"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["NoteAcc_DEID"].astype(str)

    def _build_image_path(self) -> None:
        self.df["image_path"] = self.df["NoteAcc_DEID"] + ".npz"

    def _build_labels(self) -> None:
        """``_read_source()`` makes the label columns."""

    def _build_bbox(self) -> None:
        """Merge the boxes, give each box the label ``lung``, and clip the boxes (D-07)."""
        super()._build_bbox()
        # One box for each image: the lung region.
        if "bbox" in self.df.columns:
            self.df["bbox_labels"] = self.df["bbox"].apply(
                lambda x: ["lung"] if isinstance(x, list) and len(x) > 0 else []
            )
        clip_boxes(self.df, "RAD-ChestCT")  # D-07
