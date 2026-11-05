"""Datasets for the official ROCO splits."""

from radharmony.harmonizer import ROCOTestHarmonizer, ROCOTrainHarmonizer, ROCOValidationHarmonizer
from radharmony.harmonizer._split import _FixedSplitArgMixin
from radharmony.registry import register_dataset
from .roco import ROCODataset


@register_dataset("roco_train")
class ROCOTrainDataset(_FixedSplitArgMixin, ROCODataset):
    """ROCO official ``train`` split. The arguments are the same as :class:`ROCODataset`, without ``splits``."""

    SPLIT = "train"
    SPLIT_ARG = "splits"
    _HARMONIZER_CLS = ROCOTrainHarmonizer


@register_dataset("roco_validation")
class ROCOValidationDataset(_FixedSplitArgMixin, ROCODataset):
    """ROCO official ``validation`` split. The arguments are the same as :class:`ROCODataset`, without ``splits``."""

    SPLIT = "validation"
    SPLIT_ARG = "splits"
    _HARMONIZER_CLS = ROCOValidationHarmonizer


@register_dataset("roco_test")
class ROCOTestDataset(_FixedSplitArgMixin, ROCODataset):
    """ROCO official ``test`` split. The arguments are the same as :class:`ROCODataset`, without ``splits``."""

    SPLIT = "test"
    SPLIT_ARG = "splits"
    _HARMONIZER_CLS = ROCOTestHarmonizer
