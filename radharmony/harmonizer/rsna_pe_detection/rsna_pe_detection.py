"""Harmonizer for the RSNA/STR Pulmonary Embolism Detection Challenge (2020).

CT Pulmonary Angiography (CTPA) dataset from five international institutions.
On disk each study has exactly one series, stored as a directory of
per-slice DICOM files::

    <base_image_dir>/                                 # train/
      <StudyInstanceUID>/
        <SeriesInstanceUID>/
          <SOPInstanceUID>.dcm
          ...

``train.csv`` in the release is one row per **slice**, but 13 of the 14
labels are actually study-level (``pe_present_on_image`` is the sole
per-slice flag) and identical across every slice of a study.  This
harmonizer collapses those per-slice rows into one row per study and
treats each sample as a 3-D volume loaded from the series directory
(MONAI's ``ITKReader`` handles DICOM series assembly).  The slice-level
``pe_present_on_image`` flag is dropped because it does not apply at
study/volume level — the study-level PE presence is already covered by
``leftsided_pe`` / ``rightsided_pe`` / ``central_pe``.

See: Colak et al., "The RSNA Pulmonary Embolism CT Dataset",
Radiology: AI 2021; 3(2):e200254.
"""

import os

import pandas as pd

from ..base import BaseHarmonizer


class RSNAPEDetectionHarmonizer(BaseHarmonizer):
    """Harmonize RSNA PE Detection (2020) into the standard RadHarmony format.

    Emits one row per ``StudyInstanceUID`` (= one CT volume), with
    ``image_path`` pointing at the DICOM series directory relative to
    ``base_image_dir``.

    Args:
        csv_path: Path to ``train.csv``.
        base_image_dir: Root of the DICOM tree (typically the ``train/``
            directory).  Retained on the instance for ``save()`` /
            ``load_from_saved()`` round-trips.
    """

    LABEL_COLS = [
        # 13 study-level labels — already snake_case in the CSV, sorted
        # alphabetically.  pe_present_on_image (slice-level) is dropped.
        "acute_and_chronic_pe",
        "central_pe",
        "chronic_pe",
        "flow_artifact",
        "indeterminate",
        "leftsided_pe",
        "negative_exam_for_pe",
        "qa_contrast",
        "qa_motion",
        "rightsided_pe",
        "rv_lv_ratio_gte_1",
        "rv_lv_ratio_lt_1",
        "true_filling_defect_not_pe",
    ]

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
        super().__init__(
            csv_path=os.path.expanduser(csv_path) if csv_path else csv_path
        )
        self.base_image_dir = (
            os.path.expanduser(base_image_dir) if base_image_dir else base_image_dir
        )

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        if self.base_image_dir is not None:
            snap["base_image_dir"] = self.base_image_dir
        return snap

    def _build_patient_id(self) -> None:
        # No separate patient column in this release.  StudyInstanceUID
        # serves as the patient key so patient-level splits keep a study
        # together (moot here since we also emit one row per study, but
        # kept for consistency with the rest of the pipeline).
        self.df["patient_id"] = self.df["StudyInstanceUID"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["StudyInstanceUID"].astype(str)

    def _build_image_path(self) -> None:
        # Directory of the DICOM series (not an individual .dcm file).
        # MONAI's ITKReader handles the directory → 3-D volume assembly.
        self.df["image_path"] = (
            self.df["StudyInstanceUID"].astype(str)
            + "/"
            + self.df["SeriesInstanceUID"].astype(str)
        )

    # ------------------------------------------------------------------
    # Custom harmonize — collapse per-slice CSV to one row per study
    # ------------------------------------------------------------------

    def harmonize(self, **kwargs) -> pd.DataFrame:
        """Read train.csv, deduplicate to one row per study, build volume paths.

        The default :meth:`BaseHarmonizer.harmonize` would emit ~1.79M slice
        rows; for 3-D per-volume processing we need one row per study.  The
        13 study-level labels are identical across slices of a study so
        ``drop_duplicates`` keeps the first row's values.  The slice-level
        ``pe_present_on_image`` column is dropped because it does not apply
        at volume granularity.
        """
        self._harmonized_df_override = None
        self.df = pd.read_csv(self.csv_path)

        # Collapse per-slice rows → per-study rows.  Labels are replicated
        # across every slice of a study, so the first row per study carries
        # the correct values.
        self.df = self.df.drop_duplicates(
            subset=["StudyInstanceUID"], keep="first"
        ).reset_index(drop=True)

        # Slice-level flag — not meaningful once we collapse to volumes.
        if "pe_present_on_image" in self.df.columns:
            self.df = self.df.drop(columns=["pe_present_on_image"])

        self._build_patient_id()
        self._build_study_id()
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

    def verify_images(self, base_image_dir: str = None, drop_missing: bool = True) -> pd.DataFrame:
        """Check that every ``image_path`` **directory** exists on disk.

        Overrides the base implementation because ``image_path`` is a DICOM
        series *directory*, not a single file, so ``os.path.isfile`` would
        always return ``False``.
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
                f"Found {len(missing_df)} rows with missing series dirs (not dropped)."
            )

        return missing_df
