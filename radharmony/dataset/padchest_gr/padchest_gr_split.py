"""Datasets for the official PadChest-GR splits."""

from radharmony.harmonizer import PadChestGRTestHarmonizer, PadChestGRTrainHarmonizer, PadChestGRValidationHarmonizer
from radharmony.harmonizer._split import _FixedSplitArgMixin
from radharmony.registry import register_dataset
from .padchest_gr import PadChestGRDataset


@register_dataset("padchest_gr_train")
class PadChestGRTrainDataset(_FixedSplitArgMixin, PadChestGRDataset):
    """PadChest-GR official ``train`` split.

    The arguments are the same as :class:`PadChestGRDataset`, without ``split``.
    """

    SPLIT = "train"
    _HARMONIZER_CLS = PadChestGRTrainHarmonizer


@register_dataset("padchest_gr_validation")
class PadChestGRValidationDataset(_FixedSplitArgMixin, PadChestGRDataset):
    """PadChest-GR official ``validation`` split.

    The arguments are the same as :class:`PadChestGRDataset`, without ``split``.
    """

    SPLIT = "validation"
    _HARMONIZER_CLS = PadChestGRValidationHarmonizer


@register_dataset("padchest_gr_test")
class PadChestGRTestDataset(_FixedSplitArgMixin, PadChestGRDataset):
    """PadChest-GR official ``test`` split.

    The arguments are the same as :class:`PadChestGRDataset`, without ``split``.
    """

    SPLIT = "test"
    _HARMONIZER_CLS = PadChestGRTestHarmonizer
