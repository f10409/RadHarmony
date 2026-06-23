"""Harmonizer for the NIH ChestX-ray14 test split."""

import os
import warnings

import pandas as pd

from .chestxray14 import ChestXray14Harmonizer

_TEST_LIST = "test_list.txt"


class ChestXray14TestHarmonizer(ChestXray14Harmonizer):
    """ChestX-ray14 test split (25,596 images).

    Extends :class:`ChestXray14Harmonizer` by filtering the harmonized
    DataFrame to images listed in ``test_list.txt``, which NIH ships
    alongside the dataset.  If the file is not found, all images are returned
    with a warning.

    Both train and test splits carry full labels from ``Data_Entry_2017.csv``.

    Args:
        csv_path: Path to ``Data_Entry_2017.csv``.
        bbox_csv_path: Path to ``BBox_List_2017.csv`` (to exclude bbox images).
        base_image_dir: Root directory containing ``images_*/images/`` folders.
    """

    def harmonize(self, **kwargs) -> pd.DataFrame:
        df = super().harmonize(**kwargs)

        if self.base_image_dir is None:
            warnings.warn(
                "base_image_dir is None; cannot locate test_list.txt. "
                "Returning all images.",
                stacklevel=2,
            )
            return df

        list_path = os.path.join(self.base_image_dir, _TEST_LIST)
        if not os.path.isfile(list_path):
            warnings.warn(
                f"{_TEST_LIST!r} not found under {self.base_image_dir!r}. "
                "Returning all images without train/test filtering.",
                stacklevel=2,
            )
            return df

        allowed = set(open(list_path).read().splitlines())
        mask = df["image_path"].apply(os.path.basename).isin(allowed)
        return df[mask].reset_index(drop=True)
