"""MONAI datasets for the official RAD-ChestCT train, valid and test splits."""

from radharmony.harmonizer import RadChestCTTestHarmonizer, RadChestCTTrainHarmonizer, RadChestCTValidHarmonizer
from radharmony.registry import register_dataset
from .radchestct import RadChestCTDataset


@register_dataset("radchestct_train")
class RadChestCTTrainDataset(RadChestCTDataset):
    """RAD-ChestCT official train split. The arguments are the same as :class:`RadChestCTDataset`."""

    _LABEL_NAMES = ("imgtrain_Abnormality_and_Location_Labels.csv",)
    _HARMONIZER_CLS = RadChestCTTrainHarmonizer


@register_dataset("radchestct_valid")
class RadChestCTValidDataset(RadChestCTDataset):
    """RAD-ChestCT official valid split. The arguments are the same as :class:`RadChestCTDataset`."""

    _LABEL_NAMES = ("imgvalid_Abnormality_and_Location_Labels.csv",)
    _HARMONIZER_CLS = RadChestCTValidHarmonizer


@register_dataset("radchestct_test")
class RadChestCTTestDataset(RadChestCTDataset):
    """RAD-ChestCT official test split. The arguments are the same as :class:`RadChestCTDataset`."""

    _LABEL_NAMES = ("imgtest_Abnormality_and_Location_Labels.csv",)
    _HARMONIZER_CLS = RadChestCTTestHarmonizer
