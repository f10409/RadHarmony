"""Harmonizers for the official PadChest-GR splits."""

from .._split import _FixedSplitArgMixin
from .padchest_gr import PadChestGRHarmonizer


class PadChestGRTrainHarmonizer(_FixedSplitArgMixin, PadChestGRHarmonizer):
    """PadChest-GR official ``train`` split.

    The arguments are the same as :class:`PadChestGRHarmonizer`, without ``split``.
    """

    SPLIT = "train"


class PadChestGRValidationHarmonizer(_FixedSplitArgMixin, PadChestGRHarmonizer):
    """PadChest-GR official ``validation`` split.

    The arguments are the same as :class:`PadChestGRHarmonizer`, without ``split``.
    """

    SPLIT = "validation"


class PadChestGRTestHarmonizer(_FixedSplitArgMixin, PadChestGRHarmonizer):
    """PadChest-GR official ``test`` split.

    The arguments are the same as :class:`PadChestGRHarmonizer`, without ``split``.
    """

    SPLIT = "test"
