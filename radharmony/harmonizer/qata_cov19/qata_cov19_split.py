"""Harmonizers for the official QaTa-COV19 v2 train and test splits."""

import os

from .qata_cov19 import QaTaCOV19Harmonizer


class _QaTaCOV19SplitHarmonizer(QaTaCOV19Harmonizer):
    """QaTa-COV19 harmonizer for one split. ``base_dir`` must be ``<SPLIT_DIR>/Images/``.

    The folder decides the rows. With the folder of the other split, the class raises ``ValueError``.
    """

    SPLIT_DIR: str = None

    def __init__(self, base_dir: str, masks_dir: str = None):
        parts = os.path.normpath(os.path.expanduser(base_dir)).split(os.sep)
        if self.SPLIT_DIR not in parts:
            raise ValueError(f"{type(self).__name__} needs a base_dir in {self.SPLIT_DIR!r}/, got {base_dir!r}.")
        super().__init__(base_dir=base_dir, masks_dir=masks_dir)

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap.pop("csv_path", None)
        return snap


class QaTaCOV19TrainHarmonizer(_QaTaCOV19SplitHarmonizer):
    """QaTa-COV19 v2 official train split (``Train Set/Images/``)."""

    SPLIT_DIR = "Train Set"


class QaTaCOV19TestHarmonizer(_QaTaCOV19SplitHarmonizer):
    """QaTa-COV19 v2 official test split (``Test Set/Images/``)."""

    SPLIT_DIR = "Test Set"
