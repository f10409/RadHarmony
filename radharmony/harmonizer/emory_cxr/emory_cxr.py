"""Harmonizer for the EmoryCXR v2 de-identified chest radiography dataset.

EmoryCXR v2 is an internal Emory dataset (~2.4M images from ~360K patients)
with de-identified PNGs, study-level CheXpert-aligned finding labels, and
de-identified free-text reports.

Layout::

    <base_image_dir>/
        <empi_anon>/<AccessionNumber_anon>/<SOP>.png

    TABLES/
        metadata.csv          # image-level metadata (primary CSV)
        finding_labels.csv    # study-level labels
        reports.csv           # study-level de-identified reports

The metadata CSV contains one row per SOP (image); the label and report CSVs
are at the accession (study) level and are left-merged on ``AccessionNumber_anon``
so every image row inherits its study's labels and report.

Labels use binary encoding (``1`` = positive, ``0`` = negative, ``NaN`` =
not mentioned).  There are no uncertain (``-1``) cells in this release.

Paths in the harmonized DataFrame are stored **relative to** ``base_image_dir``;
join them with ``base_image_dir`` to obtain absolute paths.
"""

import os

import pandas as pd

from radharmony.harmonizer.base import BaseHarmonizer


_LABEL_TITLE_CASE = [
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

_LABEL_RENAME = {c: c.lower().replace(" ", "_") for c in _LABEL_TITLE_CASE}


class EmoryCXRHarmonizer(BaseHarmonizer):
    """Harmonize EmoryCXR v2 into the standard RadHarmony format.

    Args:
        csv_path: Path to the image-level metadata CSV (e.g.
            ``metadata.csv``).
        base_image_dir: Root of the PNG image tree (the directory containing
            patient sub-folders).  Stored for reference only; ``image_path``
            in the harmonized DataFrame is relative to this directory.
        label_csv_path: Path to the study-level finding label CSV (e.g.
            ``finding_labels.csv``).  When ``None``, no
            label columns are added.
        report_csv_path: Path to the de-identified report CSV (e.g.
            ``reports.csv``).  When ``None``, no
            ``report`` column is added.
    """

    LABEL_COLS = sorted(list(_LABEL_RENAME.values()))

    LABEL_JOIN_COLS = ["AccessionNumber_anon"]
    VIEW_POSITION_SOURCE_COL = "ViewPosition"
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None  # handled in _build_report override
    REPORT_PATH_COL = None

    EXTRA_OUTPUT_COLS = [
        "sex",
        "age",
        "bmi",
        "race",
        "ethnicity",
        "frontal",
        "institution",
        "study_date",
    ]

    def __init__(
        self,
        csv_path: str,
        base_image_dir: str = None,
        label_csv_path: str = None,
        report_csv_path: str = None,
    ):
        super().__init__(
            csv_path=os.path.expanduser(csv_path) if csv_path else csv_path,
            label_csv_path=(
                os.path.expanduser(label_csv_path) if label_csv_path else label_csv_path
            ),
            report_csv_path=(
                os.path.expanduser(report_csv_path) if report_csv_path else report_csv_path
            ),
        )
        self.base_image_dir = (
            os.path.expanduser(base_image_dir) if base_image_dir else base_image_dir
        )

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["base_image_dir"] = self.base_image_dir
        return snap

    # ------------------------------------------------------------------
    # Build hooks
    # ------------------------------------------------------------------

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["empi_anon"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["AccessionNumber_anon"].astype(str)

    def _build_image_path(self) -> None:
        # ImagePath in the metadata CSV already encodes the relative path
        # <empi_anon>/<AccessionNumber_anon>/<SOP>.png.
        self.df["image_path"] = self.df["ImagePath"].astype(str)

    def _build_labels(self) -> None:
        if self.label_csv_path is None:
            return
        label_df = pd.read_csv(self.label_csv_path)
        label_df["AccessionNumber_anon"] = label_df["AccessionNumber_anon"].astype(str)
        label_df.rename(columns=_LABEL_RENAME, inplace=True)
        snake_cols = [c for c in _LABEL_RENAME.values() if c in label_df.columns]
        keep = ["AccessionNumber_anon"] + snake_cols
        # Left merge: images without a label row keep NaN labels rather than
        # being dropped. Enables using the full image pool for non-label tasks.
        self.df = self.df.merge(label_df[keep], on="AccessionNumber_anon", how="left")

    def _build_report(self) -> None:
        if self.report_csv_path is None:
            return
        report_df = pd.read_csv(self.report_csv_path, dtype=str)
        if "deid_report" not in report_df.columns:
            return
        report_df = report_df[["AccessionNumber_anon", "deid_report"]].rename(
            columns={"deid_report": "report"}
        )
        self.df = self.df.merge(report_df, on="AccessionNumber_anon", how="left")

    def _build_extras(self) -> None:
        rename = {
            "Sex": "sex",
            "Age": "age",
            "BMI": "bmi",
            "Race": "race",
            "Ethnicity": "ethnicity",
            "Frontal": "frontal",
            "InstitutionName_anon": "institution",
            "StudyDate_anon": "study_date",
        }
        for src, dst in rename.items():
            if src in self.df.columns:
                self.df.rename(columns={src: dst}, inplace=True)

    def harmonize(self, *args, **kwargs) -> pd.DataFrame:
        self._harmonized_df_override = None
        self.df = pd.read_csv(self.csv_path)
        # Normalise join key to str so merges with other CSVs don't fail on
        # int64/object type mismatch (accession numbers are numeric in metadata
        # but read as object when other CSVs use dtype=str).
        self.df["AccessionNumber_anon"] = self.df["AccessionNumber_anon"].astype(str)

        self._build_patient_id()
        self._build_study_id()
        self._build_image_path()
        self.LABEL_COLS = sorted(self.LABEL_COLS)
        self._build_labels()
        self._build_view_position()
        self._build_report()
        self._build_extras()
        self._build_mask_path()
        self._build_bbox()

        self.df.dropna(subset=["image_path"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self._select_harmonized_columns(self.df)
