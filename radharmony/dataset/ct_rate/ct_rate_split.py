"""MONAI datasets for the official CT-RATE train and valid splits."""

from radharmony.harmonizer import CTRATETrainHarmonizer, CTRATEValidHarmonizer
from radharmony.registry import register_dataset
from .ct_rate import CTRATEDataset


@register_dataset("ct_rate_train")
class CTRATETrainDataset(CTRATEDataset):
    """CT-RATE official train split. It looks for the train tables only.

    The arguments are the same as :class:`CTRATEDataset`.
    """

    _SPLIT = "train"
    _HARMONIZER_CLS = CTRATETrainHarmonizer


@register_dataset("ct_rate_valid")
class CTRATEValidDataset(CTRATEDataset):
    """CT-RATE official valid split. It looks for the valid tables only.

    The arguments are the same as :class:`CTRATEDataset`.
    """

    _SPLIT = "valid"
    _HARMONIZER_CLS = CTRATEValidHarmonizer
