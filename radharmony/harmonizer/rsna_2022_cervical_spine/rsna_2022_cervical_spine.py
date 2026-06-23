"""Harmonizer for the RSNA 2022 Cervical Spine Fracture Detection Challenge.

2,019 cervical-spine CT scans with 8 binary labels per study (one
``patient_overall`` flag plus per-vertebra fracture flags ``C1``–``C7``).
Each study is a directory of per-slice DICOMs:

    <base_image_dir>/                                 # train_images/
      <StudyInstanceUID>/
        <slice_number>.dcm                            # e.g. 100.dcm
        ...

Note the directory layout is **flat** (one level of nesting), unlike the
RSNA PE Detection release which adds an intermediate ``<SeriesInstanceUID>/``.
MONAI's ``ITKReader`` reads the directory and assembles a 3-D volume.

A subset of 87 studies (out of 2,019) ship with whole-volume segmentation
masks as NIfTI files under ``segmentations/<StudyInstanceUID>.nii``.
``mask_path`` is populated for those studies and left ``NaN`` for the
rest; downstream code that requests ``output_mask=True`` naturally
restricts the dataset to the mask-having subset via the framework's
``dropna`` on ``*_path`` columns.

See: <https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection>.
"""

import os

import pandas as pd

from ..base import BaseHarmonizer


class RSNA2022CervicalSpineHarmonizer(BaseHarmonizer):
    """Harmonize RSNA 2022 Cervical Spine into the standard RadHarmony format.

    Emits one row per ``StudyInstanceUID`` (= one CT volume), with
    ``image_path`` pointing at the per-study DICOM directory relative to
    ``base_image_dir``.

    Args:
        csv_path: Path to ``train.csv``.
        base_image_dir: Root of the DICOM tree (typically ``train_images/``).
            Retained on the instance for ``save()`` / ``load_from_saved()``
            round-trips.
        segmentation_dir: Optional directory containing
            ``<StudyInstanceUID>.nii`` segmentation files (typically
            ``segmentations/`` next to ``train_images/``).  When ``None``,
            ``mask_path`` is left empty for every study.
    """

    LABEL_COLS = [
        # 8 study-level binary labels (uppercase 'C1'..'C7' in the CSV; the
        # base harmonizer rename is .lower() → 'c1'..'c7').  Sorted by their
        # post-rename name so harmonize()'s sort is a no-op.
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "patient_overall",
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

    def __init__(
        self,
        csv_path: str,
        base_image_dir: str = None,
        segmentation_dir: str = None,
    ):
        super().__init__(
            csv_path=os.path.expanduser(csv_path) if csv_path else csv_path
        )
        self.base_image_dir = (
            os.path.expanduser(base_image_dir) if base_image_dir else base_image_dir
        )
        self.segmentation_dir = (
            os.path.expanduser(segmentation_dir) if segmentation_dir else segmentation_dir
        )

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        if self.base_image_dir is not None:
            snap["base_image_dir"] = self.base_image_dir
        if self.segmentation_dir is not None:
            snap["segmentation_dir"] = self.segmentation_dir
        return snap

    def _build_patient_id(self) -> None:
        # No distinct patient column — one study per patient in this release.
        self.df["patient_id"] = self.df["StudyInstanceUID"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["StudyInstanceUID"].astype(str)

    def _build_image_path(self) -> None:
        # Per-study DICOM directory.  Single level of nesting (no SeriesUID).
        # MONAI's ITKReader assembles a 3-D volume from all .dcm files inside.
        self.df["image_path"] = self.df["StudyInstanceUID"].astype(str)

    def _build_mask_path(self) -> None:
        # Absolute path to segmentations/<StudyUID>.nii when present;
        # NaN otherwise.  Only ~87 of 2,019 studies have segmentations.
        if not self.segmentation_dir:
            return

        def _resolve(study_uid: str) -> str | None:
            p = os.path.join(self.segmentation_dir, f"{study_uid}.nii")
            return p if os.path.isfile(p) else None

        self.df["mask_path"] = self.df["study_id"].map(_resolve)

    def verify_images(self, base_image_dir: str = None, drop_missing: bool = True) -> pd.DataFrame:
        """Check that every ``image_path`` **directory** exists on disk.

        Overrides the base implementation because ``image_path`` is a DICOM
        study *directory*, not a single file, so ``os.path.isfile`` would
        always return ``False``.
        """
        df = self.harmonized_df
        root = base_image_dir or self.base_image_dir

        from tqdm import tqdm

        def _exists(p):
            full = os.path.join(root, p) if root else p
            return os.path.isdir(full)

        tqdm.pandas(desc="Verifying study directories")
        mask = df["image_path"].progress_apply(_exists)
        missing_df = df[~mask].copy()

        if not missing_df.empty and drop_missing:
            kept = df[mask].reset_index(drop=True)
            self.set_harmonized_df(kept)
            print(
                f"Dropped {len(missing_df)} rows with missing study directories "
                f"({len(kept)} remaining)."
            )
        elif missing_df.empty:
            print("All study directories exist.")
        else:
            print(f"Found {len(missing_df)} missing study directories (not dropped).")

        return missing_df
