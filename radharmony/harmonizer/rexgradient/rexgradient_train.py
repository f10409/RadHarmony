"""Harmonizer for the ReXGradient-160K training split."""

from .rexgradient import ReXGradientHarmonizer


class ReXGradientTrainHarmonizer(ReXGradientHarmonizer):
    """ReXGradient-160K training split harmonizer.

    Identical logic to :class:`ReXGradientHarmonizer`; exists as a
    distinct class so that saved harmonizers round-trip back to the
    train-specific dataset class.

    Args:
        csv_path: Path to ``train_metadata_view_position.json``.
        base_image_dir: Root of the extracted PNG tree (e.g.
            ``.../ReXGradient-160K/deid_png/``).
    """
