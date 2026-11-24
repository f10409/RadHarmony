"""Harmonizer for the CheXpert dataset."""

import os

import pandas as pd

from .._labels import chexpert_codes
from ..base import BaseHarmonizer


class CheXpertHarmonizer(BaseHarmonizer):
    """Harmonizer for the CheXpert chest X-ray dataset.

    ``patient_id`` and ``study_id`` are extracted from the ``Path`` column.
    ``image_path`` is a relative path from the dataset root (prepend
    ``base_image_dir`` at dataset load time).

    Expected CSV columns: ``Path``, ``AP/PA``, and the 14 label columns.

    Args:
        csv_path: Path to the CheXpert metadata CSV (for example ``train.csv``).
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

    #: Race and ethnicity are in a separate workbook, ``Chexpert race labels.xlsx``,
    #: for 65,401 patients. Its columns are PATIENT, GENDER, AGE_AT_CXR,
    #: PRIMARY_RACE and ETHNICITY. The join key is the patient folder from ``Path``.
    EXTRA_OUTPUT_COLS = ["race", "ethnicity"]

    #: ``Age`` is in the metadata CSV, in whole years, on every row (223,414 of
    #: 223,414 in train.csv). The base step makes ``age_years`` from it. It is
    #: a wider source than the workbook ``AGE_AT_CXR`` (65,401 patients, needs
    #: ``demographics_path``). Both can exist and they do not conflict.
    AGE_SOURCE_COL = "Age"
    AGE_SOURCE_UNIT = "years"

    #: ``Sex`` in the CSV (Male, Female or Unknown) is on every row. The base
    #: step maps it to ``sex`` (D-26). The workbook ``GENDER`` covers fewer
    #: patients, so the harmonizer does not read it.
    SEX_SOURCE_COL = "Sex"

    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = "AP/PA"
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None

    def __init__(
        self,
        csv_path: str,
        base_image_dir: str = None,
        demographics_path: str = None,
    ):
        """
        Args:
            csv_path: CheXpert metadata CSV (for example ``train.csv``).
            base_image_dir: dataset root to prepend to ``image_path``.
            demographics_path: ``Chexpert race labels.xlsx``. Optional. Without it,
                the demographic columns are absent.
        """
        super().__init__(csv_path=csv_path)
        self.base_image_dir = base_image_dir
        self.drop_uncertain, self.not_mentioned = True, "negative"  # set again by harmonize()
        self.demographics_path = (
            os.path.expanduser(demographics_path) if demographics_path else None
        )

    def _harmonizer_init_snapshot(self) -> dict:
        """Add ``demographics_path``, so a loaded object harmonizes the same way (D-08)."""
        snap = super()._harmonizer_init_snapshot()
        # Only if set. Subclasses such as CheXphoto do not take this argument.
        if getattr(self, "demographics_path", None):
            snap["demographics_path"] = self.demographics_path
        return snap

    def harmonize(self, *args, drop_uncertain: bool = True, not_mentioned: str = "negative", **kwargs):
        """Run the base steps. Then, if ``demographics_path`` is set, add the demographics columns.

        The base ``harmonize()`` has no extras step, so this method calls
        ``_build_extras``. Without this call, the columns are missing and no
        warning shows. The join needs ``patient_id``, so it runs after the base steps.

        Args:
            drop_uncertain: If ``True`` (the default), drop the rows with an uncertain label (-1).
            not_mentioned: ``"negative"`` (the default) gives 0 for a blank label, ``"nan"`` keeps
                NaN (D-05).

        Returns:
            The harmonized table.
        """
        self.drop_uncertain, self.not_mentioned = drop_uncertain, not_mentioned
        out = super().harmonize(*args, **kwargs)
        self._build_extras()
        return self._select_harmonized_columns(self.df) if self.demographics_path else out

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["Path"].apply(lambda x: x.split("/")[2])

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["Path"].apply(
            lambda x: f"{x.split('/')[2]}_{x.split('/')[3]}"
        )

    def _build_image_path(self) -> None:
        self.df["image_path"] = self.df["Path"].apply(
            lambda x: "/".join(x.split("/")[2:])
        )

    def _build_labels(self) -> None:
        # D-05. Only the label columns count: AP/PA is blank on every lateral image.
        self.df = chexpert_codes(self.df, self.LABEL_COLS, self.not_mentioned, self.drop_uncertain)
        for col in self.LABEL_COLS:
            if col in self.df.columns:
                self.df.rename(columns={col: col.lower().replace(" ", "_")}, inplace=True)

    def _build_extras(self) -> None:
        """Join the demographics workbook on patient_id and print the coverage.

        It does two things on purpose:

        * It keeps ``PRIMARY_RACE`` as text, such as "White, non-Hispanic".
          The caller groups it as needed.
        * It has no default path. Without ``demographics_path`` the columns are absent.
        """
        if self.demographics_path is None:
            return
        if not os.path.isfile(self.demographics_path):
            raise FileNotFoundError(
                f"CheXpert demographics not found: {self.demographics_path}"
            )
        demo = pd.read_excel(self.demographics_path)
        # AGE_AT_CXR is one age for each patient. It is wrong for other studies (V-H1-40).
        # age_years uses the Age of each image in the CSV (D-06).
        rename = {
            "PATIENT": "patient_id",
            "PRIMARY_RACE": "race",
            "ETHNICITY": "ethnicity",
        }
        missing = [c for c in rename if c not in demo.columns]
        if missing:
            raise ValueError(
                f"CheXpert demographics workbook missing columns {missing}; "
                f"found {list(demo.columns)}"
            )
        demo = demo[list(rename)].rename(columns=rename)
        demo["patient_id"] = demo["patient_id"].astype(str)
        demo = demo.drop_duplicates("patient_id")
        before = len(self.df)
        self.df["patient_id"] = self.df["patient_id"].astype(str)
        self.df = self.df.merge(demo, on="patient_id", how="left")
        assert len(self.df) == before, (
            "demographics join changed the row count -- duplicate patient_id in the workbook"
        )
        matched = float(self.df["race"].notna().mean())
        print(
            f"[chexpert] demographics joined on {matched:.1%} of "
            f"{before:,} rows ({demo.shape[0]:,} patients in the workbook)"
        )

    def _build_view_position(self) -> None:
        if (
            self.VIEW_POSITION_SOURCE_COL
            and self.VIEW_POSITION_SOURCE_COL in self.df.columns
        ):
            # AP or PA for a frontal image, LATERAL for a lateral one (Q-13). 17 train images are frontal
            # with LL or RL in AP/PA. Their AP or PA is unknown, so they give NaN, as for Emory (Q-14).
            ap = self.df["AP/PA"]
            lateral = self.df["Frontal/Lateral"].eq("Lateral") if "Frontal/Lateral" in self.df else ap.isna()
            self.df["view_position"] = ap.where(ap.isin(["AP", "PA"]), None).mask(lateral, "LATERAL")
