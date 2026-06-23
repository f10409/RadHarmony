"""MONAI dataset for the CheXpert validation split."""

from radharmony.harmonizer import CheXpertValidHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from .chexpert import CheXpertDataset


@register_dataset("chexpert_valid")
class CheXpertValidDataset(CheXpertDataset):
    """CheXpert test/validation split (200 images from ``valid.csv``).

    The official hold-out set released alongside the CheXpert dataset.
    Carries the same 14 labels as the training split.

    ``base_image_dir`` should point at the ``valid/`` directory (the directory
    that directly contains the ``patient*/`` image folders, e.g.
    ``/data/CheXpert-v1.0/valid/``).  ``valid.csv`` is discovered
    automatically via :func:`~radharmony.utils.infer.infer_path`.

    All other constructor arguments are identical to :class:`CheXpertDataset`.
    """

    _HARMONIZER_CLS = CheXpertValidHarmonizer

    def __init__(self, base_image_dir=None, csv_path=None, **kwargs):
        # Pre-infer valid.csv so the base class's train.csv search is bypassed.
        harmonizer_path = kwargs.get("harmonizer_path")
        harmonized_df = kwargs.get("harmonized_df")
        harmonizer = kwargs.get("harmonizer")
        if (
            csv_path is None
            and base_image_dir is not None
            and harmonizer_path is None
            and harmonized_df is None
            and harmonizer is None
        ):
            csv_path = (
                infer_path(
                    base_image_dir,
                    "valid.csv",
                    "CheXpert-v1.0-small/valid.csv",
                    user_path="",
                )
                or None
            )

        super().__init__(base_image_dir=base_image_dir, csv_path=csv_path, **kwargs)

        # Upgrade the harmonizer to the valid-specific subclass.
        if hasattr(self, "_harmonizer") and not isinstance(
            self._harmonizer, CheXpertValidHarmonizer
        ):
            self._harmonizer = CheXpertValidHarmonizer(
                csv_path=self._harmonizer.csv_path,
                base_image_dir=self.base_image_dir,
            )
