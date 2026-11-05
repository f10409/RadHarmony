"""Harmonizers for the official ROCOv2 splits."""

from .._split import _FixedSplitArgMixin
from .rocov2 import ROCOv2Harmonizer


class ROCOv2TrainHarmonizer(_FixedSplitArgMixin, ROCOv2Harmonizer):
    """ROCOv2 official ``train`` split. The arguments are the same as :class:`ROCOv2Harmonizer`, without ``splits``."""

    SPLIT = "train"
    SPLIT_ARG = "splits"


class ROCOv2ValidationHarmonizer(_FixedSplitArgMixin, ROCOv2Harmonizer):
    """ROCOv2 official ``validation`` split.

    The arguments are the same as :class:`ROCOv2Harmonizer`, without ``splits``.
    """

    SPLIT = "validation"
    SPLIT_ARG = "splits"


class ROCOv2TestHarmonizer(_FixedSplitArgMixin, ROCOv2Harmonizer):
    """ROCOv2 official ``test`` split. The arguments are the same as :class:`ROCOv2Harmonizer`, without ``splits``."""

    SPLIT = "test"
    SPLIT_ARG = "splits"
