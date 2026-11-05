"""Harmonizer for the CheXphoto training split (train.csv)."""

from .._split import _LoadOldFileMixin
from .chexphoto import CheXphotoHarmonizer


class CheXphotoTrainHarmonizer(_LoadOldFileMixin, CheXphotoHarmonizer):
    """CheXphoto training split harmonizer.

    Identical logic to :class:`CheXphotoHarmonizer`; exists as a distinct class
    so that saved harmonizers can be round-tripped back to the correct
    train-specific dataset class.

    ``train.csv`` has 32,521 images / 3,000 patients: ``natural/{iphone,nokia}``
    plus ``synthetic/{digital,photographic}``. The default
    ``harmonize(drop_uncertain=True)`` drops the images with an uncertain
    (-1) label and gives 20,034 images / 2,279 patients. Use
    ``harmonize(drop_uncertain=False)`` to keep all 32,521 images.

    ``load_from_saved()`` also loads a file that :class:`CheXphotoHarmonizer`
    saved from ``train.csv`` (D-09).

    Args:
        csv_path: Path to ``train.csv``.
        base_image_dir: Split root containing ``natural/`` and ``synthetic/``.
        subset: ``all`` (default), ``natural`` or ``synthetic``.
    """

    SPLIT = "train"
    OLD_FILE_CLASS = CheXphotoHarmonizer
