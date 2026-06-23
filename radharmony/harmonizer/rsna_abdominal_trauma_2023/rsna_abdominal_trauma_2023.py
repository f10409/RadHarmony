"""Harmonizer for the RSNA 2023 Abdominal Trauma Detection Challenge.

Per-series CT volumes from a multi-institution release.  On disk each patient
has one or more series, stored as a directory of per-slice DICOM files::

    <base_image_dir>/                                  # train_images/
      <patient_id>/
        <series_id>/
          <instance_number>.dcm
          ...

The release ships two CSVs we care about for the basic harmonized table:

- ``train_2024.csv`` — one row per *patient* (3,147 rows), with 14 binary
  label columns covering 5 organs.  ``bowel`` and ``extravasation`` are
  binary (healthy vs. injury); ``kidney``, ``liver``, ``spleen`` each have
  three one-hot columns (healthy / low-grade / high-grade); plus an
  ``any_injury`` summary flag.
- ``train_series_meta.csv`` — one row per *series* (4,711 rows), with
  ``aortic_hu`` (contrast-phase indicator) and ``incomplete_organ``.

Patients can have multiple series (mean ~1.5/patient) — typically one
non-contrast and one portal-venous-phase scan.  We emit one harmonized
row per series and replicate the patient-level labels across that
patient's series, matching the way the original challenge framed the
task ("predict patient-level injury from any of the patient's series").

The dataset also ships ``image_level_labels_2024.csv`` (15,632 rows of
per-slice ``Active_Extravasation`` point annotations) and
``segmentations/`` (organ segmentation masks for a subset of series)
— neither is wired into this initial harmonizer.  See
``docs/proposals/`` for follow-up work if these become useful.

See: https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection
"""

import os

import pandas as pd

from ..base import BaseHarmonizer


class RSNAAbdominalTrauma2023Harmonizer(BaseHarmonizer):
    """Harmonize RSNA 2023 Abdominal Trauma Detection into the standard format.

    Emits one row per series.  Patient-level labels from ``train_2024.csv``
    are joined onto each of that patient's series by ``patient_id``.
    Series-level metadata (``aortic_hu``, ``incomplete_organ``) is joined
    from ``train_series_meta.csv``.

    Args:
        csv_path: Path to ``train_2024.csv`` (the patient-level label CSV).
        series_meta_csv_path: Path to ``train_series_meta.csv``.  Required —
            no fallback, since the per-series row enumeration comes from here.
        base_image_dir: Root of the DICOM tree (typically ``train_images/``).
            Retained on the instance for ``save()`` / ``load_from_saved()``
            round-trips and for ``verify_images``.
    """

    LABEL_COLS = [
        # Sorted alphabetically — order is normalized to a deterministic
        # output column order in the harmonized df.
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

    # Frank's series_id refactor (commit ef8f089) — automatic via base class.
    SERIES_ID_SOURCE_COL = "series_id"

    # Surface aortic_hu + incomplete_organ as extra output columns so they
    # ride along in the harmonized df without being label/reg columns.
    EXTRA_OUTPUT_COLS = ("aortic_hu", "incomplete_organ")

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
        # No separate study UID in this release — the patient is the study
        # for harmonized-df purposes.  Patient-level splits stay correct
        # because get_datasets() splits on patient_id.
        self.df["study_id"] = self.df["patient_id"].astype(str)

    def _build_image_path(self) -> None:
        # DICOM series DIRECTORY: <patient_id>/<series_id>/.  MONAI's
        # ITKReader assembles the per-slice .dcm files into a 3-D volume.
        self.df["image_path"] = (
            self.df["patient_id"].astype(str)
            + "/"
            + self.df["series_id"].astype(str)
        )

    # ------------------------------------------------------------------
    # Custom harmonize — join patient labels onto per-series rows
    # ------------------------------------------------------------------

    def harmonize(self, **kwargs) -> pd.DataFrame:
        """Read both CSVs, join patient labels onto each series, build paths."""
        self._harmonized_df_override = None

        labels = pd.read_csv(self.csv_path)        # one row per patient
        series = pd.read_csv(self.series_meta_csv_path)  # one row per series

        # Series-level table is the row spine — merge patient labels onto it.
        # Inner join: drop series whose patient has no label row (shouldn't
        # happen in this release but defensive).
        self.df = series.merge(labels, on="patient_id", how="inner")

        self._build_patient_id()
        self._build_study_id()
        self._build_series_id()
        self._build_image_path()
        self.LABEL_COLS = sorted(self.LABEL_COLS)
        self._build_labels()
        self._build_view_position()
        self._build_mask_path()
        self._build_bbox()
        self._build_report()

        self.df.dropna(subset=["image_path"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self._select_harmonized_columns(self.df)

    def verify_images(
        self, base_image_dir: str = None, drop_missing: bool = True
    ) -> pd.DataFrame:
        """Check that every ``image_path`` **directory** exists.

        Overrides the base implementation because ``image_path`` is a DICOM
        series *directory*, not a single file (so ``os.path.isfile`` would
        always return ``False``).
        """
        df = self.harmonized_df
        base = base_image_dir or self.base_image_dir

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
