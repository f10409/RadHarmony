"""MONAI dataset for the CheXpert training split."""

from radharmony.harmonizer import CheXpertTrainHarmonizer
from radharmony.registry import register_dataset
from .chexpert import CheXpertDataset


@register_dataset("chexpert_train")
class CheXpertTrainDataset(CheXpertDataset):
    """CheXpert training split (~220 k images from ``train.csv``).

    Drop-in replacement for :class:`CheXpertDataset` that is pinned to the
    training CSV and uses :class:`CheXpertTrainHarmonizer` for correct
    ``save()`` / ``load_from_saved()`` round-trips.

    All constructor arguments are identical to :class:`CheXpertDataset`.
    """

    _HARMONIZER_CLS = CheXpertTrainHarmonizer

    def __init__(self, base_image_dir=None, csv_path=None, **kwargs):
        super().__init__(base_image_dir=base_image_dir, csv_path=csv_path, **kwargs)
        # Upgrade the harmonizer to the train-specific subclass so that
        # save() records the correct class for round-trip loading.
        if hasattr(self, "_harmonizer") and not isinstance(
            self._harmonizer, CheXpertTrainHarmonizer
        ):
            self._harmonizer = CheXpertTrainHarmonizer(
                csv_path=self._harmonizer.csv_path,
                base_image_dir=self.base_image_dir,
            )
