"""MONAI dataset for the RSNA Bone Age test images (no labels, D-25)."""

import os
import warnings

import torch

from radharmony.harmonizer import RSNABoneAgeTestHarmonizer
from radharmony.registry import register_dataset
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D
from .rsna_bone_age import RSNABoneAgeDataset


@register_dataset("rsna_bone_age_test")
class RSNABoneAgeTestDataset(RSNABoneAgeDataset):
    """RSNA Bone Age test split: the 200 challenge test images, images only.

    The release gives no bone age for them, so the class gives no ``reg``.
    Every ``output_*`` flag gives a warning and no key (D-11).

    Args:
        base_image_dir: The folder that holds the test PNGs.
        transform: MONAI transform. Defaults to the standard 2-D pipeline.
        cache_dir: PersistentDataset cache directory. ``None`` disables caching.
        output_reg: Not supported (no labels).
        harmonized_df, harmonizer, harmonizer_path: Standard presets.
        dtype: Dtype of ``img``. The labels stay float32. Default ``torch.bfloat16``.
        output_cls / output_mask / output_report / output_bbox: Not supported.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset()
    REG_COLS = []
    _HARMONIZER_CLS = RSNABoneAgeTestHarmonizer

    def __init__(self, base_image_dir: str = None, transform=None, cache_dir: str = "./cache",
                 output_reg: bool = False, harmonized_df=None, harmonizer=None,
                 harmonizer_path: str = None, dtype=torch.bfloat16, output_cls: bool = False,
                 output_mask: bool = False, output_report: bool = False, output_bbox: bool = False):
        flags = {"output_reg": output_reg, "output_cls": output_cls, "output_mask": output_mask,
                 "output_report": output_report, "output_bbox": output_bbox}
        for name in [n for n, on in flags.items() if on]:
            warnings.warn(f"The RSNA Bone Age test images have no bone age and no other label. {name} is ignored.",
                          UserWarning, stacklevel=2)
        if harmonizer_path is not None and base_image_dir is None:
            base_image_dir = self._HARMONIZER_CLS.load_from_saved(harmonizer_path).base_image_dir
        if base_image_dir:
            base_image_dir = os.path.expanduser(base_image_dir)
        if transform is None:
            transform = RadiologyTransform2D(img_size=224, output_keys={"img"}, dtype=dtype).get_transform()
        # The parent constructor looks for a label CSV, so call the base constructor.
        BaseRadiologicalDataset.__init__(
            self, base_image_dir=base_image_dir, transform=transform, cache_dir=cache_dir,
            harmonized_df=harmonized_df, harmonizer=harmonizer, harmonizer_path=harmonizer_path)

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        return self._HARMONIZER_CLS(base_image_dir=self.base_image_dir).harmonize()
