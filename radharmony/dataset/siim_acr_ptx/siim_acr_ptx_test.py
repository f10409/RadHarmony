"""MONAI dataset for the SIIM-ACR Pneumothorax dataset: test split."""

import warnings

import torch

from radharmony.harmonizer import SIIMACRPTXTestHarmonizer
from radharmony.registry import register_dataset
from .siim_acr_ptx import SIIMACRPTXDataset


@register_dataset("siim_acr_ptx_test")
class SIIMACRPTXTestDataset(SIIMACRPTXDataset):
    """Test split of the SIIM-ACR Pneumothorax dataset: images only, no labels.

    Globs the ``dicom-images-test/`` DICOM tree.  Every ``output_*`` flag
    gives a warning and no key.

    Args:
        base_image_dir: Root of the test DICOM tree, typically
            ``<competition_root>/dicom-images-test/``.
        All other args identical to :class:`SIIMACRPTXDataset`.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset()
    _HARMONIZER_CLS = SIIMACRPTXTestHarmonizer

    _TRAIN_FILES_REQUIRED = False  # this class reads only the test files

    def __init__(
        self,
        base_image_dir=None,
        csv_path=None,
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
        for flag, name in [
            (output_cls, "output_cls"),
            (output_mask, "output_mask"),
            (output_report, "output_report"),
            (output_reg, "output_reg"),
            (output_bbox, "output_bbox"),
        ]:
            if flag:
                warnings.warn(
                    f"{name}=True is not supported by SIIMACRPTXTestDataset "
                    "(test split has no labels or masks); ignoring.",
                    stacklevel=2,
                )

        super().__init__(
            base_image_dir=base_image_dir,
            csv_path=csv_path,
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
        return SIIMACRPTXTestHarmonizer(dicom_dir=self.base_image_dir).harmonize()
