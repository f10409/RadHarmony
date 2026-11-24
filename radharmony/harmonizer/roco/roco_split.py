"""Harmonizers for the official ROCO splits."""

from .._split import _FixedSplitArgMixin
from .roco import ROCOHarmonizer


class ROCOTrainHarmonizer(_FixedSplitArgMixin, ROCOHarmonizer):
    """ROCO official ``train`` split. The arguments are the same as :class:`ROCOHarmonizer`, without ``splits``."""

    SPLIT = "train"
    SPLIT_ARG = "splits"


class ROCOValidationHarmonizer(_FixedSplitArgMixin, ROCOHarmonizer):
    """ROCO official ``validation`` split. The arguments are the same as :class:`ROCOHarmonizer`, without ``splits``."""

    SPLIT = "validation"
    SPLIT_ARG = "splits"


class ROCOTestHarmonizer(_FixedSplitArgMixin, ROCOHarmonizer):
    """ROCO official ``test`` split. The arguments are the same as :class:`ROCOHarmonizer`, without ``splits``."""

    SPLIT = "test"
    SPLIT_ARG = "splits"
