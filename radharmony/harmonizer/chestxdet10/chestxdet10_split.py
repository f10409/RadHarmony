"""Harmonizers for the official ChestX-Det10 splits."""

from .._split import _FixedSplitArgMixin, _LoadOldFileMixin
from .chestxdet10 import ChestXDet10Harmonizer


class ChestXDet10TrainHarmonizer(_LoadOldFileMixin, _FixedSplitArgMixin, ChestXDet10Harmonizer):
    """ChestX-Det10 official ``train`` split.

    The arguments are the same as :class:`ChestXDet10Harmonizer`, without ``split``.
    ``load_from_saved()`` also loads a train file that :class:`ChestXDet10Harmonizer` saved.
    """

    SPLIT = "train"
    OLD_FILE_CLASS = ChestXDet10Harmonizer


class ChestXDet10TestHarmonizer(_LoadOldFileMixin, _FixedSplitArgMixin, ChestXDet10Harmonizer):
    """ChestX-Det10 official ``test`` split.

    The arguments are the same as :class:`ChestXDet10Harmonizer`, without ``split``.
    ``load_from_saved()`` also loads a test file that :class:`ChestXDet10Harmonizer` saved.
    """

    SPLIT = "test"
    OLD_FILE_CLASS = ChestXDet10Harmonizer
