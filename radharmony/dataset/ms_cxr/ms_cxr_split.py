"""MONAI datasets for the official MS-CXR train, val and test splits."""

from radharmony.harmonizer import MSCXRTestHarmonizer, MSCXRTrainHarmonizer, MSCXRValHarmonizer
from radharmony.registry import register_dataset
from .ms_cxr import MSCXRDataset


@register_dataset("ms_cxr_train")
class MSCXRTrainDataset(MSCXRDataset):
    """MS-CXR official train split (737 images). The arguments are the same as :class:`MSCXRDataset`."""

    _HARMONIZER_CLS = MSCXRTrainHarmonizer


@register_dataset("ms_cxr_val")
class MSCXRValDataset(MSCXRDataset):
    """MS-CXR official val split (155 images). The arguments are the same as :class:`MSCXRDataset`."""

    _HARMONIZER_CLS = MSCXRValHarmonizer


@register_dataset("ms_cxr_test")
class MSCXRTestDataset(MSCXRDataset):
    """MS-CXR official test split (155 images). The arguments are the same as :class:`MSCXRDataset`."""

    _HARMONIZER_CLS = MSCXRTestHarmonizer
