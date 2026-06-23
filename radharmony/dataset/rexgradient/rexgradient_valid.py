"""MONAI dataset for the ReXGradient-160K validation split."""

import pandas as pd

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

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        harmonizer = ReXGradientValidHarmonizer(
            csv_path=self._csv_path,
            base_image_dir=self.base_image_dir,
        )
        return harmonizer.harmonize()
