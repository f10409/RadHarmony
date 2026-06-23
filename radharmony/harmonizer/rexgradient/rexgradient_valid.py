"""Harmonizer for the ReXGradient-160K validation split."""

from .rexgradient import ReXGradientHarmonizer


class ReXGradientValidHarmonizer(ReXGradientHarmonizer):
    """ReXGradient-160K validation split harmonizer.

    Identical logic to :class:`ReXGradientHarmonizer`; exists as a
    distinct class so that saved harmonizers round-trip back to the
    valid-specific dataset class.

    Args:
        csv_path: Path to ``valid_metadata_view_position.json``.
        base_image_dir: Root of the extracted PNG tree (e.g.
            ``.../ReXGradient-160K/deid_png/``).
    """
