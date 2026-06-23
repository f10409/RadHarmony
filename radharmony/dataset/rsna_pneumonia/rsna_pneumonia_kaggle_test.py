"""MONAI dataset for the RSNA Pneumonia Detection Challenge — test split."""

import torch

from radharmony.harmonizer import RSNAPneumoniaKaggleTestHarmonizer
from radharmony.registry import register_dataset
from .rsna_pneumonia_kaggle import RSNAPneumoniaKaggleDataset


@register_dataset("rsna_pneumonia_kaggle_test")
class RSNAPneumoniaKaggleTestDataset(RSNAPneumoniaKaggleDataset):
    """Test split of the RSNA Pneumonia Detection Challenge — images only, no labels.

    Lists ``stage_2_test_images/`` under ``base_image_dir``.
    ``output_cls`` and ``output_bbox`` are not supported by this split.

    Args:
        base_image_dir: Root directory containing ``stage_2_test_images/``
            (e.g. ``~/datasets/rsna-pneumonia-detection-challenge/``).
        All other args identical to :class:`RSNAPneumoniaKaggleDataset`.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls"})

    def __init__(
        self,
        base_image_dir=None,
        csv_path=None,
        bbox_csv_path=None,
        transform=None,
        cache_dir="./cache",
        output_cls=False,
        output_mask=False,
        output_report=False,
        output_bbox=False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path=None,
        dtype=torch.bfloat16,
    ):
        super().__init__(
            base_image_dir=base_image_dir,
            csv_path=csv_path,
            bbox_csv_path=bbox_csv_path,
            transform=transform,
            cache_dir=cache_dir,
            output_cls=False,
            output_mask=False,
            output_report=False,
            output_bbox=False,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
            dtype=dtype,
        )

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        return RSNAPneumoniaKaggleTestHarmonizer(
            base_image_dir=self.base_image_dir
        ).harmonize()
