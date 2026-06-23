"""Harmonizer for the RSNA Bone Age validation split."""

from .rsna_bone_age import RSNABoneAgeHarmonizer


class RSNABoneAgeValHarmonizer(RSNABoneAgeHarmonizer):
    """RSNA Bone Age validation split harmonizer (``Validation Dataset.csv``, 1,425 images).

    Thin subclass of :class:`RSNABoneAgeHarmonizer` that hardcodes
    ``split="val"``.  Exists as a distinct class so that saved harmonizers
    can be round-tripped back to :class:`RSNABoneAgeValDataset`.

    Args:
        base_image_dir: Parent directory of the extracted dataset.
        csv_path: Optional explicit override for ``Validation Dataset.csv``.
    """

    def __init__(self, base_image_dir: str, csv_path: str = None):
        super().__init__(base_image_dir=base_image_dir, csv_path=csv_path, split="val")
