"""Harmonizer for the RSNA 2022 Cervical Spine Fracture Detection Challenge.

2,019 cervical spine CT scans. Each study has 8 binary labels: ``patient_overall`` and a
fracture flag for each vertebra ``C1`` to ``C7``. A study is a folder of per-slice DICOM
files, ``<StudyInstanceUID>/<slice_number>.dcm`` (no series folder). MONAI's ``ITKReader``
builds the 3-D volume from the folder.

87 studies have a whole-volume NIfTI mask, ``segmentations/<StudyInstanceUID>.nii``.
``mask_path`` is NaN for the other studies.

Source: <https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection>.
Details: ``docs/wiki/reference/datasets/ct-mri-other/rsna_2022_cervical_spine.md``.
"""

import os

import pandas as pd

from ..base import BaseHarmonizer


class RSNA2022CervicalSpineHarmonizer(BaseHarmonizer):
    """Harmonize RSNA 2022 Cervical Spine into the standard RadHarmony format.

    It gives one row per ``StudyInstanceUID`` (one CT volume). ``image_path`` is the study
    folder, relative to ``base_image_dir``.

    Args:
        csv_path: Path to ``train.csv``.
        base_image_dir: Root of the DICOM tree, usually ``train_images/``. ``save()`` and
            ``load_from_saved()`` keep it.
        segmentation_dir: Folder with the ``<StudyInstanceUID>.nii`` masks, usually
            ``segmentations/`` next to ``train_images/``. If ``None``, ``mask_path`` is
            empty for all studies.
    """

    LABEL_COLS = [
        # The base class lowers the names ("C1" becomes "c1"). This order is the sorted order.
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
        # The release has no patient column.
        self.df["patient_id"] = self.df["StudyInstanceUID"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["StudyInstanceUID"].astype(str)

    def _build_image_path(self) -> None:
        # A study folder, not one file.
        self.df["image_path"] = self.df["StudyInstanceUID"].astype(str)

    def _build_mask_path(self) -> None:
        # Only 87 of 2,019 studies have a mask.
        if not self.segmentation_dir:
            return

        def _resolve(study_uid: str) -> str | None:
            p = os.path.join(self.segmentation_dir, f"{study_uid}.nii")
            return p if os.path.isfile(p) else None

        # str, also when no study of a filtered table has a mask (D-02: same dtype as the full table).
        self.df["mask_path"] = self.df["study_id"].map(_resolve).astype("str")

    def verify_images(self, base_image_dir: str = None, drop_missing: bool = True) -> pd.DataFrame:
        """Check that each ``image_path`` folder exists on disk.

        The base method checks files. Here ``image_path`` is a study folder.
        """
        df = self.harmonized_df
        root = base_image_dir or getattr(self, "base_image_dir", None)
        # The folder rule of the base verify_images (D-04, V-H3-22).
        if not root and not df["image_path"].astype(str).map(os.path.isabs).all():
            raise ValueError(
                "image_path is relative to base_image_dir (D-04). Pass "
                "verify_images(base_image_dir=...), or every row looks missing."
            )

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
