"""MONAI datasets for the official VinDr-PCXR train and test splits."""

from radharmony.harmonizer.vindr_pcxr import VinDrPCXRTestHarmonizer, VinDrPCXRTrainHarmonizer
from radharmony.registry import register_dataset
from .vindr_pcxr import VinDrPCXRDataset


@register_dataset("vindr_pcxr_train")
class VinDrPCXRTrainDataset(VinDrPCXRDataset):
    """VinDr-PCXR official train split. The arguments are the same as :class:`VinDrPCXRDataset`."""

    _HARMONIZER_CLS = VinDrPCXRTrainHarmonizer


@register_dataset("vindr_pcxr_test")
class VinDrPCXRTestDataset(VinDrPCXRDataset):
    """VinDr-PCXR official test split. The arguments are the same as :class:`VinDrPCXRDataset`."""

    _HARMONIZER_CLS = VinDrPCXRTestHarmonizer
