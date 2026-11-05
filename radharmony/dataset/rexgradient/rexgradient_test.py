"""MONAI dataset for the ReXGradient-160K public test split."""

from radharmony.harmonizer import ReXGradientTestHarmonizer
from radharmony.registry import register_dataset
from .rexgradient import ReXGradientDataset


@register_dataset("rexgradient_test")
class ReXGradientTestDataset(ReXGradientDataset):
    """ReXGradient-160K public-test split (17,029 images, 10K studies).

    Pinned to ``test_metadata_view_position.json``. All constructor
    arguments are identical to :class:`ReXGradientDataset`.
    """

    _HARMONIZER_CLS = ReXGradientTestHarmonizer

    @classmethod
    def _json_filename_variants(cls) -> tuple[str, ...]:
        return ("test_metadata_view_position.json",)
