"""Harmonizer for the ReXGradient-160K training split."""

from .rexgradient import ReXGradientHarmonizer


class ReXGradientTrainHarmonizer(ReXGradientHarmonizer):
    """ReXGradient-160K training split harmonizer.

    It has the same logic as :class:`ReXGradientHarmonizer`. It is a separate class so
    that a saved harmonizer loads back into the train dataset class.

    Args:
        csv_path: Path to ``train_metadata_view_position.json``.
        base_image_dir: Root of the extracted PNG tree (for example
            ``.../ReXGradient-160K/deid_png/``).
    """

    SPLIT = "train"
