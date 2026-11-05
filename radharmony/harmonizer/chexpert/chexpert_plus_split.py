"""Harmonizers for the official CheXpert Plus train and valid splits."""

from .._split import _KeepSplitMixin
from .chexpert_plus import CheXpertPlusHarmonizer


class CheXpertPlusTrainHarmonizer(_KeepSplitMixin, CheXpertPlusHarmonizer):
    """CheXpert Plus official train split (``split == "train"`` in the source CSV)."""

    SPLIT = "train"


class CheXpertPlusValidHarmonizer(_KeepSplitMixin, CheXpertPlusHarmonizer):
    """CheXpert Plus official valid split (``split == "valid"`` in the source CSV)."""

    SPLIT = "valid"
