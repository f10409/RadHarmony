"""Harmonizer for the RSNA/STR Pulmonary Embolism Detection Challenge (2020).

CT Pulmonary Angiography (CTPA) from five institutions (Colak et al., Radiology: AI 2021).
Each study has one series: a folder of per-slice DICOM files.

``train.csv`` has one row per slice. But 13 of the 14 labels belong to the study and repeat
on every slice. The harmonizer keeps one row per study, a 3-D volume that MONAI's
``ITKReader`` loads from the series folder. The slice flag ``pe_present_on_image`` is dropped.
The study labels ``leftsided_pe``, ``rightsided_pe`` and ``central_pe`` already say if PE is present.

Layout and columns: ``docs/wiki/reference/datasets/ct-mri-other/rsna_pe_detection.md``.
"""

import os

import pandas as pd

from ..base import BaseHarmonizer


class RSNAPEDetectionHarmonizer(BaseHarmonizer):
    """Harmonize RSNA PE Detection (2020) into the standard RadHarmony format.

    It gives one row per ``StudyInstanceUID`` (one CT volume). ``image_path`` is the DICOM
    series folder, relative to ``base_image_dir``.

    Args:
        csv_path: Path to ``train.csv``.
        base_image_dir: Root of the DICOM tree, usually ``train/``. ``save()`` and
            ``load_from_saved()`` keep it.
    """

    LABEL_COLS = [
        # 13 study labels, already snake_case in the CSV.
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
        # The release has no patient column.
        self.df["patient_id"] = self.df["StudyInstanceUID"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["StudyInstanceUID"].astype(str)

    def _build_image_path(self) -> None:
        # A series folder, not one .dcm file.
        self.df["image_path"] = (
            self.df["StudyInstanceUID"].astype(str)
            + "/"
            + self.df["SeriesInstanceUID"].astype(str)
        )

    def _read_source(self) -> pd.DataFrame:
        """Read the slice CSV and keep one row per study.

        The CSV has one row per slice, about 1.79M rows for train. The study labels repeat on
        each slice, so the first row of a study has the right values.
        """
        df = pd.read_csv(self.csv_path)
        df = df.drop_duplicates(subset=["StudyInstanceUID"], keep="first").reset_index(drop=True)

        # A slice flag has no meaning for a volume.
        if "pe_present_on_image" in df.columns:
            df = df.drop(columns=["pe_present_on_image"])
        return df

    def verify_images(self, base_image_dir: str = None, drop_missing: bool = True) -> pd.DataFrame:
        """Check that each ``image_path`` folder exists on disk.

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
                f"Found {len(missing_df)} rows with missing series dirs (not dropped)."
            )

        return missing_df
