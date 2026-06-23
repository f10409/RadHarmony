"""Harmonizer for the CheXpert validation split (valid.csv)."""

from .chexpert import CheXpertHarmonizer


class CheXpertValidHarmonizer(CheXpertHarmonizer):
    """CheXpert validation split harmonizer.

    Reads ``valid.csv`` (200 carefully curated studies, the official
    validation set).  Identical harmonization logic to
    :class:`CheXpertHarmonizer`; the distinct class enables correct
    round-trips via ``load_from_saved()``.

    Args:
        csv_path: Path to ``valid.csv``.
        base_image_dir: The ``valid/`` directory (directly contains ``patient*/`` folders).
    """
