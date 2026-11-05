"""Harmonizer for the RSNA 2023 Abdominal Trauma Detection Challenge.

Per-series CT volumes. Each series is a folder of per-slice DICOM files,
``train_images/<patient_id>/<series_id>/<instance_number>.dcm``.

The harmonizer uses two CSVs:

- ``train_2024.csv``: one row for each patient (3,147 rows), with 14 binary label columns for
  5 organs. ``bowel`` and ``extravasation`` have healthy and injury columns. ``kidney``,
  ``liver`` and ``spleen`` have healthy, low and high columns. ``any_injury`` is a summary flag.
- ``train_series_meta.csv``: one row for each series (4,711 rows), with ``aortic_hu`` (contrast
  phase) and ``incomplete_organ``.

A patient has about 1.5 series (often one non-contrast and one portal-venous scan). The table
has one row for each series, and the patient labels repeat on each series of the patient.

The release also has ``image_level_labels_2024.csv`` (15,632 slice labels: 8,400
``Active_Extravasation`` and 7,232 ``Bowel``) and ``segmentations/`` (organ masks for some
series). The harmonizer does not use them.

Source: https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection
Details: ``docs/wiki/reference/datasets/ct-mri-other/rsna_abdominal_trauma_2023.md``.
"""

import os
import warnings

import pandas as pd

from ..base import BaseHarmonizer


class RSNAAbdominalTrauma2023Harmonizer(BaseHarmonizer):
    """Harmonize RSNA 2023 Abdominal Trauma Detection into the standard format.

    It gives one row for each series. The patient labels of ``train_2024.csv`` join to each
    series of the patient by ``patient_id``. ``aortic_hu`` and ``incomplete_organ`` come
    from ``train_series_meta.csv``.

    Args:
        csv_path: Path to ``train_2024.csv`` (the patient labels).
        series_meta_csv_path: Path to ``train_series_meta.csv``. Required, because it lists
            the series.
        base_image_dir: Root of the DICOM tree, usually ``train_images/``. ``save()``,
            ``load_from_saved()`` and ``verify_images`` use it.
    """

    LABEL_COLS = [
        "any_injury",
        "bowel_healthy",
        "bowel_injury",
        "extravasation_healthy",
        "extravasation_injury",
        "kidney_healthy",
        "kidney_high",
        "kidney_low",
        "liver_healthy",
        "liver_high",
        "liver_low",
        "spleen_healthy",
        "spleen_high",
        "spleen_low",
    ]

    SERIES_ID_SOURCE_COL = "series_id"

    # These columns are in the output but are not labels.
    EXTRA_OUTPUT_COLS = ("aortic_hu", "incomplete_organ")

    # Age and sex come from train_demographics_2024.csv, next to csv_path (D-06, V-H3-32).
    AGE_SOURCE_COL = "Age"
    AGE_SOURCE_UNIT = "years"
    SEX_SOURCE_COL = "Sex"
    DEMOGRAPHICS_FILE = "train_demographics_2024.csv"

    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    def __init__(
        self,
        csv_path: str,
        series_meta_csv_path: str,
        base_image_dir: str = None,
    ):
        super().__init__(
            csv_path=os.path.expanduser(csv_path) if csv_path else csv_path
        )
        self.series_meta_csv_path = (
            os.path.expanduser(series_meta_csv_path)
            if series_meta_csv_path
            else series_meta_csv_path
        )
        self.base_image_dir = (
            os.path.expanduser(base_image_dir) if base_image_dir else base_image_dir
        )

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        if self.series_meta_csv_path is not None:
            snap["series_meta_csv_path"] = self.series_meta_csv_path
        if self.base_image_dir is not None:
            snap["base_image_dir"] = self.base_image_dir
        return snap

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["patient_id"].astype(str)

    def _build_study_id(self) -> None:
        # The release has no study ID. The patient is the study.
        self.df["study_id"] = self.df["patient_id"].astype(str)

    def _build_image_path(self) -> None:
        # A series folder, not one file.
        self.df["image_path"] = (
            self.df["patient_id"].astype(str)
            + "/"
            + self.df["series_id"].astype(str)
        )

    def _add_demographics(self) -> None:
        """Join ``Age`` and ``Sex`` from the demographics file next to ``csv_path``, if it is there.

        "90+" (an age that the release caps at 90) gives 90. "Unknown" gives NaN (D-06).
        A missing file gives a warning and no age or sex columns.
        """
        age, sex = self.AGE_SOURCE_COL, self.SEX_SOURCE_COL
        path = os.path.join(os.path.dirname(self.csv_path), self.DEMOGRAPHICS_FILE)
        if not os.path.isfile(path):
            warnings.warn(f"RSNA 2023 abdominal trauma: {path} not found, so the table has no age_years "
                          "and no sex.", UserWarning, stacklevel=4)
            return
        demo = pd.read_csv(path, dtype={age: str, sex: str})[["patient_id", age, sex]]
        demo = demo.drop_duplicates("patient_id")  # one row per patient, so the merge keeps the row count
        demo[age] = demo[age].str.strip().replace({"90+": "90"})
        self.df = self.df.merge(demo, on="patient_id", how="left")

    def _read_source(self) -> pd.DataFrame:
        """Join the patient labels to each series, then add age and sex."""
        labels = pd.read_csv(self.csv_path)        # one row for each patient
        series = pd.read_csv(self.series_meta_csv_path)  # one row for each series

        # The inner join drops a series whose patient has no label row.
        # _add_demographics() reads and changes self.df.
        self.df = series.merge(labels, on="patient_id", how="inner")
        self._add_demographics()
        return self.df

    def verify_images(
        self, base_image_dir: str = None, drop_missing: bool = True
    ) -> pd.DataFrame:
        """Check that each ``image_path`` folder exists.

        The base method checks files. Here ``image_path`` is a series folder.
        """
        df = self.harmonized_df
        base = base_image_dir or getattr(self, "base_image_dir", None)
        # The folder rule of the base verify_images (D-04, V-H3-22).
        if not base and not df["image_path"].astype(str).map(os.path.isabs).all():
            raise ValueError(
                "image_path is relative to base_image_dir (D-04). Pass "
                "verify_images(base_image_dir=...), or every row looks missing."
            )

        def _exists(p):
            fp = os.path.join(base, p) if base else p
            return os.path.isdir(fp)

        from tqdm import tqdm

        tqdm.pandas(desc="Verifying series dirs")
        mask = df["image_path"].progress_apply(_exists)
        missing_df = df[~mask].copy()

        if not missing_df.empty and drop_missing:
            kept = df[mask].reset_index(drop=True)
            self.set_harmonized_df(kept)
            print(
                f"Dropped {len(missing_df)} rows with missing series dirs "
                f"({len(kept)} remaining)."
            )
        elif missing_df.empty:
            print("All series directories exist.")
        else:
            print(
                f"Found {len(missing_df)} rows with missing series dirs "
                "(not dropped)."
            )

        return missing_df
