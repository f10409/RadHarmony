"""MONAI dataset for the CheXphoto validation split."""

from radharmony.harmonizer import CheXphotoValidHarmonizer
from radharmony.registry import register_dataset

from .chexphoto import CheXphotoDataset


@register_dataset("chexphoto_valid")
class CheXphotoValidDataset(CheXphotoDataset):
    """CheXphoto validation split: 702 images / 200 patients from ``valid.csv``.

    Composition: ``natural/oneplus`` (234) plus ``synthetic/digital`` (234) and
    ``synthetic/photographic`` (234) renderings of the same 234 source x-rays.
    Contains no uncertain labels, so ``drop_uncertain`` is a no-op here.

    All constructor arguments are identical to :class:`CheXphotoDataset`.
    """

    _HARMONIZER_CLS = CheXphotoValidHarmonizer
    _SPLIT_CSV = "valid.csv"
