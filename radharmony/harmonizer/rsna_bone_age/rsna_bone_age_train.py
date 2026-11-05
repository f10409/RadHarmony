"""Harmonizer for the RSNA Bone Age training split."""

from .rsna_bone_age import RSNABoneAgeHarmonizer


class RSNABoneAgeTrainHarmonizer(RSNABoneAgeHarmonizer):
    """RSNA Bone Age train split (``train.csv``, 12,611 images).

    It is :class:`RSNABoneAgeHarmonizer` with ``split="train"``. It is a separate class so
    that a saved harmonizer loads back into :class:`RSNABoneAgeTrainDataset`.

    Args:
        base_image_dir: The folder with the train PNGs.
        csv_path: Path of ``train.csv``. Optional.
    """

    def __init__(self, base_image_dir: str, csv_path: str = None):
        super().__init__(base_image_dir=base_image_dir, csv_path=csv_path, split="train")

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap.pop("split", None)  # fixed by this class, not an argument (V-H3-9)
        return snap
