"""MONAI dataset for the CheXphoto training split."""

from radharmony.harmonizer import CheXphotoTrainHarmonizer
from radharmony.registry import register_dataset

from .chexphoto import CheXphotoDataset


@register_dataset("chexphoto_train")
class CheXphotoTrainDataset(CheXphotoDataset):
    """CheXphoto training split: 32,521 images / 3,000 patients from ``train.csv``.

    Composition: ``natural/iphone`` (1,000), ``natural/nokia`` (10,507),
    ``synthetic/digital`` (10,507) and ``synthetic/photographic`` (10,507).

    All constructor arguments are identical to :class:`CheXphotoDataset`.
    """

    _HARMONIZER_CLS = CheXphotoTrainHarmonizer
    _SPLIT_CSV = "train.csv"
