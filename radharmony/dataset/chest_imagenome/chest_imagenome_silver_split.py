"""Datasets for the official Chest ImaGenome silver splits."""

from radharmony.harmonizer import (
    ChestImaGenomeSilverTestHarmonizer,
    ChestImaGenomeSilverTrainHarmonizer,
    ChestImaGenomeSilverValidHarmonizer,
)
from radharmony.harmonizer._split import _FixedSplitArgMixin
from radharmony.registry import register_dataset
from .chest_imagenome_silver import ChestImaGenomeSilverDataset


@register_dataset("chest_imagenome_silver_train")
class ChestImaGenomeSilverTrainDataset(_FixedSplitArgMixin, ChestImaGenomeSilverDataset):
    """Chest ImaGenome silver official ``train`` split.

    The arguments are the same as :class:`ChestImaGenomeSilverDataset`, without ``split``.
    """

    SPLIT = "train"
    _HARMONIZER_CLS = ChestImaGenomeSilverTrainHarmonizer


@register_dataset("chest_imagenome_silver_valid")
class ChestImaGenomeSilverValidDataset(_FixedSplitArgMixin, ChestImaGenomeSilverDataset):
    """Chest ImaGenome silver official ``valid`` split.

    The arguments are the same as :class:`ChestImaGenomeSilverDataset`, without ``split``.
    """

    SPLIT = "valid"
    _HARMONIZER_CLS = ChestImaGenomeSilverValidHarmonizer


@register_dataset("chest_imagenome_silver_test")
class ChestImaGenomeSilverTestDataset(_FixedSplitArgMixin, ChestImaGenomeSilverDataset):
    """Chest ImaGenome silver official ``test`` split.

    The arguments are the same as :class:`ChestImaGenomeSilverDataset`, without ``split``.
    """

    SPLIT = "test"
    _HARMONIZER_CLS = ChestImaGenomeSilverTestHarmonizer
