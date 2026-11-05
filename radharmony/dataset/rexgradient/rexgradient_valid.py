"""MONAI dataset for the ReXGradient-160K validation split."""

from radharmony.harmonizer import ReXGradientValidHarmonizer
from radharmony.registry import register_dataset
from .rexgradient import ReXGradientDataset


@register_dataset("rexgradient_valid")
class ReXGradientValidDataset(ReXGradientDataset):
    """ReXGradient-160K validation split (17,007 images, 10K studies).

    Pinned to ``valid_metadata_view_position.json``. All constructor
    arguments are identical to :class:`ReXGradientDataset`.
    """

    _HARMONIZER_CLS = ReXGradientValidHarmonizer

    @classmethod
    def _json_filename_variants(cls) -> tuple[str, ...]:
        return ("valid_metadata_view_position.json",)
