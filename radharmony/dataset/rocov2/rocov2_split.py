"""Datasets for the official ROCOv2 splits."""

from radharmony.harmonizer import ROCOv2TestHarmonizer, ROCOv2TrainHarmonizer, ROCOv2ValidationHarmonizer
from radharmony.harmonizer._split import _FixedSplitArgMixin
from radharmony.registry import register_dataset
from .rocov2 import ROCOv2Dataset


@register_dataset("rocov2_train")
class ROCOv2TrainDataset(_FixedSplitArgMixin, ROCOv2Dataset):
    """ROCOv2 official ``train`` split. The arguments are the same as :class:`ROCOv2Dataset`, without ``splits``."""

    SPLIT = "train"
    SPLIT_ARG = "splits"
    _HARMONIZER_CLS = ROCOv2TrainHarmonizer


@register_dataset("rocov2_validation")
class ROCOv2ValidationDataset(_FixedSplitArgMixin, ROCOv2Dataset):
    """ROCOv2 official ``validation`` split.

    The arguments are the same as :class:`ROCOv2Dataset`, without ``splits``.
    """

    SPLIT = "validation"
    SPLIT_ARG = "splits"
    _HARMONIZER_CLS = ROCOv2ValidationHarmonizer


@register_dataset("rocov2_test")
class ROCOv2TestDataset(_FixedSplitArgMixin, ROCOv2Dataset):
    """ROCOv2 official ``test`` split. The arguments are the same as :class:`ROCOv2Dataset`, without ``splits``."""

    SPLIT = "test"
    SPLIT_ARG = "splits"
    _HARMONIZER_CLS = ROCOv2TestHarmonizer
