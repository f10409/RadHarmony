"""MONAI dataset for the RSNA Bone Age validation split."""

from radharmony.harmonizer._split import _FixedSplitArgMixin
from radharmony.harmonizer import RSNABoneAgeValHarmonizer
from radharmony.registry import register_dataset
from .rsna_bone_age import RSNABoneAgeDataset


@register_dataset("rsna_bone_age_val")
class RSNABoneAgeValDataset(_FixedSplitArgMixin, RSNABoneAgeDataset):
    """RSNA Bone Age validation split (1,425 images, ``Validation Dataset.csv``).

    Hardcodes ``split="val"`` and uses
    :class:`RSNABoneAgeValHarmonizer` for correct
    ``save()`` / ``load_from_saved()`` round-trips.

    All constructor arguments are identical to :class:`RSNABoneAgeDataset`
    except ``split``: it is always ``"val"``, and another value raises ``ValueError``.
    """

    SPLIT = "val"
    _HARMONIZER_CLS = RSNABoneAgeValHarmonizer

    def __init__(self, base_image_dir=None, csv_path=None, **kwargs):
        # _FixedSplitArgMixin checks split= and passes this class's split.
        super().__init__(base_image_dir=base_image_dir, csv_path=csv_path, **kwargs)
        # Upgrade the harmonizer to the val-specific subclass.
        if hasattr(self, "_harmonizer") and not isinstance(
            self._harmonizer, RSNABoneAgeValHarmonizer
        ):
            self._harmonizer = RSNABoneAgeValHarmonizer(
                base_image_dir=self._harmonizer.base_image_dir,
                csv_path=self._harmonizer.csv_path,
            )
