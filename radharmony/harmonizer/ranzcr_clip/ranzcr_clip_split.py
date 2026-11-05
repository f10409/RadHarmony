"""Harmonizers for the RANZCR CLiP labeled train rows and unlabeled test images."""

import os

import pandas as pd

from .ranzcr_clip import RANZCRClipHarmonizer


class RANZCRClipTrainHarmonizer(RANZCRClipHarmonizer):
    """RANZCR CLiP labeled rows of ``train.csv`` only (30,083 images).

    Args:
        csv_path: Path to ``train.csv``.
        base_image_dir: Root of the Kaggle download (contains ``train/``).
        annotations_csv_path: Path to ``train_annotations.csv`` (optional, for masks).
        mask_line_thickness: Pixel width of the rasterized polylines.
    """

    # One split per class, so the table needs no ``split`` column.
    EXTRA_OUTPUT_COLS = []

    def __init__(
        self,
        csv_path: str,
        base_image_dir: str = None,
        annotations_csv_path: str = None,
        mask_line_thickness: int = 15,
    ):
        super().__init__(
            csv_path=csv_path,
            base_image_dir=base_image_dir,
            annotations_csv_path=annotations_csv_path,
            mask_line_thickness=mask_line_thickness,
            include_test_split=False,
        )

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap.pop("include_test_split")
        return snap


class RANZCRClipTestHarmonizer(RANZCRClipHarmonizer):
    """RANZCR CLiP Kaggle test images (3,582), with no labels (D-25).

    Lists ``<base_image_dir>/test/*.jpg``. Kaggle gives no labels and no
    patient ID for these images, so ``patient_id`` is the
    ``StudyInstanceUID``.

    Args:
        base_image_dir: Root of the Kaggle download (contains ``test/``).
    """

    LABEL_COLS = []
    EXTRA_OUTPUT_COLS = []

    def __init__(self, base_image_dir: str):
        super().__init__(csv_path=None, base_image_dir=base_image_dir)

    def _harmonizer_init_snapshot(self) -> dict:
        return {"base_image_dir": self.base_image_dir}

    def _read_source(self) -> pd.DataFrame:
        test_dir = os.path.join(self.base_image_dir or "", "test")
        df = self._build_test_rows()
        if df.empty:
            raise FileNotFoundError(f"No JPEG found in {test_dir!r}. Set base_image_dir to the Kaggle root.")
        return df
