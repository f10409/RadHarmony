"""MONAI dataset for the RSNA PE Detection train split."""

import pandas as pd

from radharmony.harmonizer import RSNAPEDetectionTrainHarmonizer
from radharmony.registry import register_dataset
from .rsna_pe_detection import RSNAPEDetectionDataset


@register_dataset("rsna_pe_detection_train")
class RSNAPEDetectionTrainDataset(RSNAPEDetectionDataset):
    """Train split of :class:`RSNAPEDetectionDataset`.

    Identical to the base class; uses :class:`RSNAPEDetectionTrainHarmonizer`
    for correct save/load round-trips via ``load_from_saved()``.
    All constructor arguments are the same as :class:`RSNAPEDetectionDataset`.
    """

    _HARMONIZER_CLS = RSNAPEDetectionTrainHarmonizer

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        harmonizer = RSNAPEDetectionTrainHarmonizer(
            csv_path=self._csv_path,
            base_image_dir=self.base_image_dir,
        )
        return harmonizer.harmonize()
