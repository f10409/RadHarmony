"""VQA datasets for the official VQA-RAD train and test questions."""

from radharmony.harmonizer import VQARadTestHarmonizer, VQARadTrainHarmonizer
from radharmony.registry import register_dataset
from .vqa_rad import VQARadDataset


@register_dataset("vqa_rad_train")
class VQARadTrainDataset(VQARadDataset):
    """VQA-RAD train questions (1,797). The arguments are the same as :class:`VQARadDataset`."""

    _HARMONIZER_CLS = VQARadTrainHarmonizer


@register_dataset("vqa_rad_test")
class VQARadTestDataset(VQARadDataset):
    """VQA-RAD test questions (451). 202 of the 203 test images also have train questions."""

    _HARMONIZER_CLS = VQARadTestHarmonizer
