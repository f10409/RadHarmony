"""MONAI datasets for the official CheXpert Plus train and valid splits."""

from radharmony.harmonizer import CheXpertPlusTrainHarmonizer, CheXpertPlusValidHarmonizer
from radharmony.registry import register_dataset
from .chexpert_plus import CheXpertPlusDataset


@register_dataset("chexpert_plus_train")
class CheXpertPlusTrainDataset(CheXpertPlusDataset):
    """CheXpert Plus official train split. The arguments are the same as :class:`CheXpertPlusDataset`."""

    _HARMONIZER_CLS = CheXpertPlusTrainHarmonizer


@register_dataset("chexpert_plus_valid")
class CheXpertPlusValidDataset(CheXpertPlusDataset):
    """CheXpert Plus official valid split. The arguments are the same as :class:`CheXpertPlusDataset`."""

    _HARMONIZER_CLS = CheXpertPlusValidHarmonizer
