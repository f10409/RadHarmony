"""Harmonizers for the official RAD-ChestCT train, valid and test splits."""

import os

from .radchestct import RadChestCTHarmonizer


class _RadChestCTSplitHarmonizer(RadChestCTHarmonizer):
    """RAD-ChestCT harmonizer for one split. The label CSV must be ``img<SPLIT>_*.csv``.

    The label CSV decides the rows. With a label CSV of another split, the class raises ``ValueError``.
    """

    SPLIT: str = None

    def _read_source(self):
        name = os.path.basename(str(self.label_csv_path))
        if not name.startswith(f"img{self.SPLIT}_"):
            raise ValueError(
                f"{type(self).__name__} needs img{self.SPLIT}_Abnormality_and_Location_Labels.csv, got {name}.")
        return super()._read_source()


class RadChestCTTrainHarmonizer(_RadChestCTSplitHarmonizer):
    """RAD-ChestCT official train split (``imgtrain_Abnormality_and_Location_Labels.csv``)."""

    SPLIT = "train"


class RadChestCTValidHarmonizer(_RadChestCTSplitHarmonizer):
    """RAD-ChestCT official valid split (``imgvalid_Abnormality_and_Location_Labels.csv``)."""

    SPLIT = "valid"


class RadChestCTTestHarmonizer(_RadChestCTSplitHarmonizer):
    """RAD-ChestCT official test split (``imgtest_Abnormality_and_Location_Labels.csv``)."""

    SPLIT = "test"
