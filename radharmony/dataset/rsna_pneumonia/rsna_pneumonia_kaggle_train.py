"""MONAI dataset for the RSNA Pneumonia Detection Challenge — train split."""

import pandas as pd

from radharmony.harmonizer import RSNAPneumoniaKaggleTrainHarmonizer
from radharmony.registry import register_dataset
from .rsna_pneumonia_kaggle import RSNAPneumoniaKaggleDataset


@register_dataset("rsna_pneumonia_kaggle_train")
class RSNAPneumoniaKaggleTrainDataset(RSNAPneumoniaKaggleDataset):
    """Train split of :class:`RSNAPneumoniaKaggleDataset`.

    Identical to the base class; uses :class:`RSNAPneumoniaKaggleTrainHarmonizer`
    for correct save/load round-trips via ``load_from_saved()``.
    All constructor arguments are the same as :class:`RSNAPneumoniaKaggleDataset`.
    """

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        harmonizer = RSNAPneumoniaKaggleTrainHarmonizer(
            csv_path=self._csv_path,
            base_image_dir=self.base_image_dir,
            bbox_csv_path=self._bbox_csv_path,
        )
        return harmonizer.harmonize()
