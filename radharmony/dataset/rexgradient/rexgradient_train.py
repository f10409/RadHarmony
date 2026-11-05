"""MONAI dataset for the ReXGradient-160K training split."""

from radharmony.harmonizer import ReXGradientTrainHarmonizer
from radharmony.registry import register_dataset
from .rexgradient import ReXGradientDataset


@register_dataset("rexgradient_train")
class ReXGradientTrainDataset(ReXGradientDataset):
    """ReXGradient-160K training split (238,968 images, 140K studies).

    Pinned to ``train_metadata_view_position.json``. All constructor
    arguments are identical to :class:`ReXGradientDataset`.
    """

    _HARMONIZER_CLS = ReXGradientTrainHarmonizer

    @classmethod
    def _json_filename_variants(cls) -> tuple[str, ...]:
        return ("train_metadata_view_position.json",)
