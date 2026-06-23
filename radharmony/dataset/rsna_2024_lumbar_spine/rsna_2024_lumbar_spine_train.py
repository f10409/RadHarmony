"""MONAI dataset for the RSNA 2024 Lumbar Spine train split."""

from radharmony.harmonizer import RSNA2024LumbarSpineTrainHarmonizer
from radharmony.registry import register_dataset
from .rsna_2024_lumbar_spine import RSNA2024LumbarSpineDataset


@register_dataset("rsna_2024_lumbar_spine_train")
class RSNA2024LumbarSpineTrainDataset(RSNA2024LumbarSpineDataset):
    """Train split of :class:`RSNA2024LumbarSpineDataset`.

    Identical to the base class; uses :class:`RSNA2024LumbarSpineTrainHarmonizer`
    for correct save/load round-trips via ``load_from_saved()``.
    All constructor arguments are the same as :class:`RSNA2024LumbarSpineDataset`.
    """

    _HARMONIZER_CLS = RSNA2024LumbarSpineTrainHarmonizer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, "_harmonizer") and not isinstance(
            self._harmonizer, RSNA2024LumbarSpineTrainHarmonizer
        ):
            self._harmonizer = RSNA2024LumbarSpineTrainHarmonizer(
                csv_path=self._harmonizer.csv_path,
                series_description_csv_path=self._harmonizer.series_description_csv_path,
                coord_csv_path=self._harmonizer.coord_csv_path,
                base_image_dir=self._harmonizer.base_image_dir,
            )
