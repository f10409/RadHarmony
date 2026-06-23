"""MONAI dataset for the RSNA 2022 Cervical Spine train split."""

import pandas as pd

from radharmony.harmonizer import RSNA2022CervicalSpineTrainHarmonizer
from radharmony.registry import register_dataset
from .rsna_2022_cervical_spine import RSNA2022CervicalSpineDataset


@register_dataset("rsna_2022_cervical_spine_train")
class RSNA2022CervicalSpineTrainDataset(RSNA2022CervicalSpineDataset):
    """Train split of :class:`RSNA2022CervicalSpineDataset`.

    Identical to the base class; uses :class:`RSNA2022CervicalSpineTrainHarmonizer`
    for correct save/load round-trips via ``load_from_saved()``.
    All constructor arguments are the same as :class:`RSNA2022CervicalSpineDataset`.
    """

    _HARMONIZER_CLS = RSNA2022CervicalSpineTrainHarmonizer

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        harmonizer = RSNA2022CervicalSpineTrainHarmonizer(
            csv_path=self._csv_path,
            base_image_dir=self.base_image_dir,
            segmentation_dir=self._segmentation_dir,
        )
        return harmonizer.harmonize()
