"""Harmonizers for the official TAIX-Ray train, val and test splits."""

from .._split import _KeepSplitMixin
from .taix_ray import TAIXRayHarmonizer


class TAIXRayTrainHarmonizer(_KeepSplitMixin, TAIXRayHarmonizer):
    """TAIX-Ray official train split (``Split == "train"`` in ``annotation.csv``)."""

    SPLIT = "train"


class TAIXRayValHarmonizer(_KeepSplitMixin, TAIXRayHarmonizer):
    """TAIX-Ray official val split (``Split == "val"`` in ``annotation.csv``)."""

    SPLIT = "val"


class TAIXRayTestHarmonizer(_KeepSplitMixin, TAIXRayHarmonizer):
    """TAIX-Ray official test split (``Split == "test"`` in ``annotation.csv``)."""

    SPLIT = "test"
