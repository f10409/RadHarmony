"""Harmonizer for the CheXpert training split (train.csv)."""

from .chexpert import CheXpertHarmonizer


class CheXpertTrainHarmonizer(CheXpertHarmonizer):
    """CheXpert training split harmonizer.

    Identical logic to :class:`CheXpertHarmonizer`; exists as a distinct class
    so that saved harmonizers can be round-tripped back to the correct
    train-specific dataset class.

    Args:
        csv_path: Path to ``train.csv``.
        base_image_dir: Root of the image tree (parent of ``train/``).
    """
