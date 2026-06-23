"""MONAI dataset for the SIIM COVID-19 Detection Challenge — train split."""

import pandas as pd

from radharmony.harmonizer import SIIMCOVID19TrainHarmonizer
from radharmony.registry import register_dataset
from .siim_covid19 import SIIMCOVID19Dataset


@register_dataset("siim_covid19_train")
class SIIMCOVID19TrainDataset(SIIMCOVID19Dataset):
    """Train split of :class:`SIIMCOVID19Dataset`.

    Identical to the base class; uses :class:`SIIMCOVID19TrainHarmonizer`
    for correct save/load round-trips via ``load_from_saved()``.
    All constructor arguments are the same as :class:`SIIMCOVID19Dataset`.
    """

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        harmonizer = SIIMCOVID19TrainHarmonizer(
            csv_path=self._harmonizer.csv_path,
            image_csv_path=self._harmonizer.image_csv_path,
            base_image_dir=self._harmonizer.base_image_dir,
        )
        return harmonizer.harmonize()
