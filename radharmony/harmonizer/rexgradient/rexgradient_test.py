"""Harmonizer for the ReXGradient-160K public test split."""

from .rexgradient import ReXGradientHarmonizer


class ReXGradientTestHarmonizer(ReXGradientHarmonizer):
    """ReXGradient-160K public-test split harmonizer.

    It has the same logic as :class:`ReXGradientHarmonizer`. It is a separate class so
    that a saved harmonizer loads back into the test dataset class.

    Args:
        csv_path: Path to ``test_metadata_view_position.json``.
        base_image_dir: Root of the extracted PNG tree (for example
            ``.../ReXGradient-160K/deid_png/``).
    """

    SPLIT = "test"
