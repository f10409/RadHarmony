"""MONAI datasets for the official TAIX-Ray train, val and test splits.

Each class gives the rows of one split of the paper's patient-level
partition. No patient is in two splits.
"""

from radharmony.harmonizer import TAIXRayTestHarmonizer, TAIXRayTrainHarmonizer, TAIXRayValHarmonizer
from radharmony.harmonizer._split import _OneSplitDatasetMixin
from radharmony.registry import register_dataset
from .taix_ray import TAIXRay512Dataset, TAIXRayDataset


@register_dataset("taix_ray_512_train")
class TAIXRay512TrainDataset(_OneSplitDatasetMixin, TAIXRay512Dataset):
    """TAIX-Ray 512-pixel PNGs, official train split."""

    _HARMONIZER_CLS = TAIXRayTrainHarmonizer


@register_dataset("taix_ray_512_val")
class TAIXRay512ValDataset(_OneSplitDatasetMixin, TAIXRay512Dataset):
    """TAIX-Ray 512-pixel PNGs, official val split."""

    _HARMONIZER_CLS = TAIXRayValHarmonizer


@register_dataset("taix_ray_512_test")
class TAIXRay512TestDataset(_OneSplitDatasetMixin, TAIXRay512Dataset):
    """TAIX-Ray 512-pixel PNGs, official test split."""

    _HARMONIZER_CLS = TAIXRayTestHarmonizer


@register_dataset("taix_ray_train")
class TAIXRayTrainDataset(_OneSplitDatasetMixin, TAIXRayDataset):
    """TAIX-Ray original-resolution PNGs, official train split."""

    _HARMONIZER_CLS = TAIXRayTrainHarmonizer


@register_dataset("taix_ray_val")
class TAIXRayValDataset(_OneSplitDatasetMixin, TAIXRayDataset):
    """TAIX-Ray original-resolution PNGs, official val split."""

    _HARMONIZER_CLS = TAIXRayValHarmonizer


@register_dataset("taix_ray_test")
class TAIXRayTestDataset(_OneSplitDatasetMixin, TAIXRayDataset):
    """TAIX-Ray original-resolution PNGs, official test split."""

    _HARMONIZER_CLS = TAIXRayTestHarmonizer
