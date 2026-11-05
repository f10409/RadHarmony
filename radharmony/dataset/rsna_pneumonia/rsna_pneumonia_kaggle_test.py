"""MONAI dataset for the RSNA Pneumonia Detection Challenge: test split."""

import warnings

import torch

from radharmony.harmonizer import RSNAPneumoniaKaggleTestHarmonizer
from radharmony.registry import register_dataset
from .rsna_pneumonia_kaggle import RSNAPneumoniaKaggleDataset


@register_dataset("rsna_pneumonia_kaggle_test")
class RSNAPneumoniaKaggleTestDataset(RSNAPneumoniaKaggleDataset):
    """Test split of the RSNA Pneumonia Detection Challenge: images only, no labels.

    Lists the ``.dcm`` files in ``base_image_dir``.
    The split has no labels and no boxes, so every ``output_*`` flag gives a
    warning and no key (D-11, D-25).

    Args:
        base_image_dir: The ``stage_2_test_images/`` folder itself (for example
            ``~/datasets/rsna-pneumonia-detection-challenge/stage_2_test_images/``).
        All other args identical to :class:`RSNAPneumoniaKaggleDataset`.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset()
    _HARMONIZER_CLS = RSNAPneumoniaKaggleTestHarmonizer
    _TRAIN_FILES_REQUIRED = False  # this class reads only the test images

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
        output_reg=False,
        output_bbox=False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path=None,
        dtype=torch.bfloat16,
    ):
        flags = {"output_cls": output_cls, "output_mask": output_mask,
                 "output_report": output_report, "output_bbox": output_bbox,
                 "output_reg": output_reg}
        for name in [n for n, on in flags.items() if on]:
            warnings.warn(
                f"{name}=True is not supported by RSNAPneumoniaKaggleTestDataset "
                "(the test split has no labels or boxes); ignoring.",
                UserWarning,
                stacklevel=2,
            )
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
