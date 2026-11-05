"""Harmonizers for the official Chest ImaGenome silver splits."""

from .._split import _FixedSplitArgMixin, _LoadOldFileMixin
from .chest_imagenome import ChestImaGenomeSilverHarmonizer


class ChestImaGenomeSilverTrainHarmonizer(_LoadOldFileMixin, _FixedSplitArgMixin, ChestImaGenomeSilverHarmonizer):
    """Chest ImaGenome silver official ``train`` split.

    The arguments are the same as :class:`ChestImaGenomeSilverHarmonizer`, without ``split``.
    ``load_from_saved()`` also loads a train file that :class:`ChestImaGenomeSilverHarmonizer` saved.
    """

    SPLIT = "train"
    OLD_FILE_CLASS = ChestImaGenomeSilverHarmonizer


class ChestImaGenomeSilverValidHarmonizer(_LoadOldFileMixin, _FixedSplitArgMixin, ChestImaGenomeSilverHarmonizer):
    """Chest ImaGenome silver official ``valid`` split.

    The arguments are the same as :class:`ChestImaGenomeSilverHarmonizer`, without ``split``.
    ``load_from_saved()`` also loads a valid file that :class:`ChestImaGenomeSilverHarmonizer` saved.
    """

    SPLIT = "valid"
    OLD_FILE_CLASS = ChestImaGenomeSilverHarmonizer


class ChestImaGenomeSilverTestHarmonizer(_LoadOldFileMixin, _FixedSplitArgMixin, ChestImaGenomeSilverHarmonizer):
    """Chest ImaGenome silver official ``test`` split.

    The arguments are the same as :class:`ChestImaGenomeSilverHarmonizer`, without ``split``.
    ``load_from_saved()`` also loads a test file that :class:`ChestImaGenomeSilverHarmonizer` saved.
    """

    SPLIT = "test"
    OLD_FILE_CLASS = ChestImaGenomeSilverHarmonizer
