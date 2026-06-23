"""Harmonizer for the RSNA Bone Age training split."""

from .rsna_bone_age import RSNABoneAgeHarmonizer


class RSNABoneAgeTrainHarmonizer(RSNABoneAgeHarmonizer):
    """RSNA Bone Age training split harmonizer (``train.csv``, 12,611 images).

    Thin subclass of :class:`RSNABoneAgeHarmonizer` that hardcodes
    ``split="train"``.  Exists as a distinct class so that saved harmonizers
    can be round-tripped back to :class:`RSNABoneAgeTrainDataset`.

    Args:
        base_image_dir: Parent directory of the extracted dataset.
        csv_path: Optional explicit override for ``train.csv``.
    """

    def __init__(self, base_image_dir: str, csv_path: str = None):
        super().__init__(base_image_dir=base_image_dir, csv_path=csv_path, split="train")
