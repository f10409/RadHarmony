"""Harmonizers for the official MS-CXR train, val and test splits."""

from .._split import _KeepSplitMixin
from .ms_cxr import MSCXRHarmonizer


class MSCXRTrainHarmonizer(_KeepSplitMixin, MSCXRHarmonizer):
    """MS-CXR official train split (``split == "train"`` in the source CSV)."""

    SPLIT = "train"


class MSCXRValHarmonizer(_KeepSplitMixin, MSCXRHarmonizer):
    """MS-CXR official val split (``split == "val"`` in the source CSV)."""

    SPLIT = "val"


class MSCXRTestHarmonizer(_KeepSplitMixin, MSCXRHarmonizer):
    """MS-CXR official test split (``split == "test"`` in the source CSV)."""

    SPLIT = "test"
