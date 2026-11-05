"""Harmonizer for the CheXphoto validation split (valid.csv)."""

from .._split import _LoadOldFileMixin
from .chexphoto import CheXphotoHarmonizer


class CheXphotoValidHarmonizer(_LoadOldFileMixin, CheXphotoHarmonizer):
    """CheXphoto validation split harmonizer.

    Reads ``valid.csv``: 702 images of 200 patients. They are 234 natural
    OnePlus photographs and ``synthetic/{digital,photographic}`` images of the
    same 234 x-rays. The logic is the same as :class:`CheXphotoHarmonizer`.
    A separate class keeps ``load_from_saved()`` correct.

    ``load_from_saved()`` also loads a file that :class:`CheXphotoHarmonizer`
    saved from ``valid.csv`` (D-09).

    Args:
        csv_path: Path to ``valid.csv``.
        base_image_dir: Split root containing ``natural/`` and ``synthetic/``.
        subset: ``all`` (default), ``natural`` or ``synthetic``.
    """

    SPLIT = "valid"
    OLD_FILE_CLASS = CheXphotoHarmonizer
