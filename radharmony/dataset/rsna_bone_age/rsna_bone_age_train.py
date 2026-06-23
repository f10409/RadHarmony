"""MONAI dataset for the RSNA Bone Age training split."""

from radharmony.harmonizer import RSNABoneAgeTrainHarmonizer
from radharmony.registry import register_dataset
from .rsna_bone_age import RSNABoneAgeDataset


@register_dataset("rsna_bone_age_train")
class RSNABoneAgeTrainDataset(RSNABoneAgeDataset):
    """RSNA Bone Age training split (12,611 images, ``train.csv``).

    Hardcodes ``split="train"`` and uses
    :class:`RSNABoneAgeTrainHarmonizer` for correct
    ``save()`` / ``load_from_saved()`` round-trips.

    All constructor arguments are identical to :class:`RSNABoneAgeDataset`
    except that ``split`` is not accepted (it is always ``"train"``).
    """

    _HARMONIZER_CLS = RSNABoneAgeTrainHarmonizer

    def __init__(self, base_image_dir=None, csv_path=None, **kwargs):
        kwargs.pop("split", None)
        super().__init__(
            base_image_dir=base_image_dir,
            csv_path=csv_path,
            split="train",
            **kwargs,
        )
        # Upgrade the harmonizer to the train-specific subclass.
        if hasattr(self, "_harmonizer") and not isinstance(
            self._harmonizer, RSNABoneAgeTrainHarmonizer
        ):
            self._harmonizer = RSNABoneAgeTrainHarmonizer(
                base_image_dir=self._harmonizer.base_image_dir,
                csv_path=self._harmonizer.csv_path,
            )
