"""MONAI dataset for the RSNA 2023 Abdominal Trauma train split."""

import pandas as pd

from radharmony.harmonizer import RSNAAbdominalTrauma2023TrainHarmonizer
from radharmony.registry import register_dataset
from .rsna_abdominal_trauma_2023 import RSNAAbdominalTrauma2023Dataset


@register_dataset("rsna_abdominal_trauma_2023_train")
class RSNAAbdominalTrauma2023TrainDataset(RSNAAbdominalTrauma2023Dataset):
    """Train split of :class:`RSNAAbdominalTrauma2023Dataset`.

    Identical to the base class; uses
    :class:`RSNAAbdominalTrauma2023TrainHarmonizer` for correct save/load
    round-trips via ``load_from_saved()``.
    All constructor arguments are the same as
    :class:`RSNAAbdominalTrauma2023Dataset`.
    """

    _HARMONIZER_CLS = RSNAAbdominalTrauma2023TrainHarmonizer

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        harmonizer = RSNAAbdominalTrauma2023TrainHarmonizer(
            csv_path=self._csv_path,
            series_meta_csv_path=self._series_meta_csv_path,
            base_image_dir=self.base_image_dir,
        )
        return harmonizer.harmonize()
