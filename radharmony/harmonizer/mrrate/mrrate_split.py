"""Harmonizers for the official MR-RATE splits."""

from .._split import _FixedSplitArgMixin
from .mrrate import MRRATEHarmonizer


class MRRATETrainHarmonizer(_FixedSplitArgMixin, MRRATEHarmonizer):
    """MR-RATE official ``train`` split. The arguments are the same as :class:`MRRATEHarmonizer`, without ``splits``."""

    SPLIT = "train"
    SPLIT_ARG = "splits"


class MRRATEValHarmonizer(_FixedSplitArgMixin, MRRATEHarmonizer):
    """MR-RATE official ``val`` split. The arguments are the same as :class:`MRRATEHarmonizer`, without ``splits``."""

    SPLIT = "val"
    SPLIT_ARG = "splits"


class MRRATETestHarmonizer(_FixedSplitArgMixin, MRRATEHarmonizer):
    """MR-RATE official ``test`` split. The arguments are the same as :class:`MRRATEHarmonizer`, without ``splits``."""

    SPLIT = "test"
    SPLIT_ARG = "splits"
