"""MONAI dataset for the NIH ChestX-ray14 test split."""

import pandas as pd

from radharmony.harmonizer import ChestXray14TestHarmonizer
from radharmony.registry import register_dataset
from .chestxray14 import ChestXray14Dataset


@register_dataset("chestxray14_test")
class ChestXray14TestDataset(ChestXray14Dataset):
    """ChestX-ray14 test split (~25,596 non-bbox images).

    Filtered to images listed in ``test_list.txt`` (NIH official split).
    All 15 labels from ``Data_Entry_2017.csv`` are available.

    All constructor arguments are identical to :class:`ChestXray14Dataset`.
    """

    _HARMONIZER_CLS = ChestXray14TestHarmonizer

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        harmonizer = ChestXray14TestHarmonizer(
            csv_path=self._csv_path,
            bbox_csv_path=self._bbox_csv_path,
            base_image_dir=self.base_image_dir,
        )
        return harmonizer.harmonize()
