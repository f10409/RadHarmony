"""MONAI dataset for the SIIM-ACR Pneumothorax dataset — train split."""

import pandas as pd

from radharmony.harmonizer import SIIMACRPTXTrainHarmonizer
from radharmony.registry import register_dataset
from .siim_acr_ptx import SIIMACRPTXDataset


@register_dataset("siim_acr_ptx_train")
class SIIMACRPTXTrainDataset(SIIMACRPTXDataset):
    """Train split of :class:`SIIMACRPTXDataset`.

    Identical to the base class; uses :class:`SIIMACRPTXTrainHarmonizer`
    for correct save/load round-trips via ``load_from_saved()``.
    All constructor arguments are the same as :class:`SIIMACRPTXDataset`.
    """

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        harmonizer = SIIMACRPTXTrainHarmonizer(
            csv_path=self._harmonizer.csv_path,
            dicom_dir=self._harmonizer.dicom_dir,
        )
        return harmonizer.harmonize(
            mask_output_dir=self._mask_output_dir,
            mask_num_cores=self._mask_num_cores,
        )
