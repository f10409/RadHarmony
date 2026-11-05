"""Harmonizer for the TAIX-Ray bedside chest radiography dataset.

TAIX-Ray has one ``annotation.csv`` (made by the download notebook from the HuggingFace parquet
shards) and a flat ``images/<UID>.png`` folder. The 8 labels are ordinal severity grades. This
layer keeps them as raw integers. The dataset class makes them binary when ``label_mode="binary"``.

CSV columns: UID, Fold, Split, PatientID, PhysicianID, StudyDate, Age, Sex, HeartSize,
PulmonaryCongestion, PleuralEffusion_{Right,Left}, PulmonaryOpacities_{Right,Left},
Atelectasis_{Right,Left}.

Grades (Truhn et al. 2026): HeartSize is 0 Normal, 1 Borderline Enlarged, 2 Enlarged,
3 Massively Enlarged. All other labels are 0 None, 1 Questionable, 2 Mild, 3 Moderate, 4 Severe.
"""

from ..base import BaseHarmonizer


# CSV names and their snake_case names. _build_labels renames the columns.
_RENAME_MAP = {
    "HeartSize": "heart_size",
    "PulmonaryCongestion": "pulmonary_congestion",
    "PleuralEffusion_Left": "pleural_effusion_left",
    "PleuralEffusion_Right": "pleural_effusion_right",
    "PulmonaryOpacities_Left": "pulmonary_opacities_left",
    "PulmonaryOpacities_Right": "pulmonary_opacities_right",
    "Atelectasis_Left": "atelectasis_left",
    "Atelectasis_Right": "atelectasis_right",
    "Sex": "sex",
    "Age": "age",
    "StudyDate": "study_date",
    "Fold": "fold",
    "Split": "split",
    "PhysicianID": "physician_id",
}


class TAIXRayHarmonizer(BaseHarmonizer):
    """Harmonize TAIX-Ray into the standard RadHarmony format.

    Args:
        csv_path: Path to ``annotation.csv`` (produced by
            ``notebooks/download_huggingface.ipynb``).
    """

    # The snake_case names that _build_labels leaves in self.df.
    LABEL_COLS = [
        "atelectasis_left",
        "atelectasis_right",
        "heart_size",
        "pleural_effusion_left",
        "pleural_effusion_right",
        "pulmonary_congestion",
        "pulmonary_opacities_left",
        "pulmonary_opacities_right",
    ]

    # Metadata for each image. ``split`` is here because the split classes read it.
    EXTRA_OUTPUT_COLS = ["age", "study_date", "fold", "split", "physician_id"]

    #: ``Age`` in annotation.csv is in DAYS (median 24,923, which is 68.2 years). Read as years,
    #: 214,893 of 215,381 rows are above 110.
    AGE_SOURCE_COL = "age"
    AGE_SOURCE_UNIT = "days"
    #: annotation.csv ``Sex`` (M or F), renamed to ``sex``. The base step maps it (D-26).
    SEX_SOURCE_COL = "sex"
    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    def __init__(self, csv_path: str, base_image_dir: str = None):
        super().__init__(csv_path=csv_path)
        self.base_image_dir = base_image_dir

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["PatientID"].astype(str)

    def _build_study_id(self) -> None:
        # One image for each UID, so the UID is the study ID.
        self.df["study_id"] = self.df["UID"].astype(str)

    def _build_image_path(self) -> None:
        self.df["image_path"] = self.df["UID"].astype(str) + ".png"

    def _build_labels(self) -> None:
        # The names must match LABEL_COLS and EXTRA_OUTPUT_COLS, so that
        # _select_harmonized_columns keeps the columns.
        self.df.rename(columns=_RENAME_MAP, inplace=True)
        # float32 fits the MONAI cls tensor. The dataset class chooses binary or ordinal.
        for col in self.LABEL_COLS:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("float32")
