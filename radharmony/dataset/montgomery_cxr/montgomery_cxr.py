"""MONAI dataset for the NLM Montgomery County Chest X-Ray Set.

138 PA chest radiographs (80 normal, 58 TB-positive) from Montgomery County
Department of Health and Human Services, Maryland.  Binary TB label is
encoded in each filename; no separate label CSV is required.

``base_image_dir`` must point at the ``CXR_png/`` images directory itself
— the last common folder containing every image (matching the convention
used by other RadHarmony datasets). ``ClinicalReadings/`` and
``ManualMask/`` are read as sibling directories. ``image_path`` entries
in the harmonized DataFrame are bare filenames (e.g. ``MCUCXR_0001_0.png``);
mask paths use ``../ManualMask/...`` to reach the sibling tree.

This mirrors :class:`ShenzhenCXRDataset` (the companion NLM TB dataset).
"""

import os

import pandas as pd
import torch

from radharmony.harmonizer.montgomery_cxr import MontgomeryCXRHarmonizer
from radharmony.registry import register_dataset
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


_LABEL_COLS = ["tuberculosis"]


@register_dataset("montgomery_cxr")
class MontgomeryCXRDataset(BaseRadiologicalDataset):
    """PyTorch/MONAI dataset for the Montgomery County CXR Set.

    Args:
        base_image_dir: The ``CXR_png/`` images directory (the last common
            folder containing every image), e.g.
            ``.../Montgomery-CXR/MontgomerySet/CXR_png/``. Sibling
            ``ClinicalReadings/`` and ``ManualMask/`` directories are read
            relative to it. Required unless ``harmonizer_path`` or
            ``harmonized_df`` is provided.
        transform: MONAI Compose transform.  Defaults to the standard 2-D
            224 px pipeline.
        cache_dir: MONAI PersistentDataset cache directory.  ``None`` disables
            caching.
        mask_output_dir: Directory to write fused left+right lung mask PNGs
            into. Required when ``output_mask=True`` (unless a pre-harmonized
            df is supplied). One PNG per image, idempotent on re-runs.
        mask_num_cores: Accepted for signature parity with SIIM-ACR PTX;
            currently unused (the fusion loop is small).
        output_cls: Yield TB label vector under the ``"cls"`` key.
        output_mask: Yield fused left+right lung mask under the ``"mask"`` key.
            Requires ``mask_output_dir``.
        output_report: Yield clinical reading text under the ``"report"`` key.
        output_bbox: Not supported; ignored with a warning.
        dtype: Output tensor dtype.  Default ``torch.bfloat16``.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "mask", "report"})
    LABEL_COLS = _LABEL_COLS
    _HARMONIZER_CLS = MontgomeryCXRHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        transform=None,
        cache_dir: str = "./cache",
        mask_output_dir: str = None,
        mask_num_cores: int = 1,
        output_cls: bool = False,
        output_mask: bool = False,
        output_report: bool = False,
        output_bbox: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        if base_image_dir:
            base_image_dir = os.path.expanduser(base_image_dir)

        if harmonizer_path is not None and base_image_dir is None:
            _h = MontgomeryCXRHarmonizer.load_from_saved(harmonizer_path)
            base_image_dir = getattr(_h, "base_dir", None)

        if (
            base_image_dir is None
            and harmonizer_path is None
            and harmonized_df is None
            and harmonizer is None
        ):
            raise ValueError(
                "base_image_dir is required when harmonizer_path is not provided. "
                "Pass the path to the Montgomery County CXR_png/ images directory "
                "(e.g. .../Montgomery-CXR/MontgomerySet/CXR_png/)."
            )

        if (
            output_mask
            and mask_output_dir is None
            and harmonizer_path is None
            and harmonized_df is None
            and harmonizer is None
        ):
            raise ValueError(
                "output_mask=True requires mask_output_dir to be set. "
                "Montgomery's left+right lung masks are fused into a single "
                "binary PNG per image and saved to mask_output_dir before "
                "MONAI can load them. Pass a directory:\n"
                "    MontgomeryCXRDataset(..., output_mask=True, "
                "mask_output_dir='/path/to/fused_masks')"
            )

        if transform is None:
            output_keys = {"img"}
            if output_cls:
                output_keys.add("cls")
            if output_mask:
                output_keys.add("mask")
            transform = RadiologyTransform2D(
                img_size=224,
                output_keys=output_keys,
                dtype=dtype,
            ).get_transform()

        super().__init__(
            base_image_dir=base_image_dir,
            transform=transform,
            cache_dir=cache_dir,
            output_cls=output_cls,
            output_mask=output_mask,
            output_report=output_report,
            output_bbox=output_bbox,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
        )
        self._mask_output_dir = mask_output_dir
        self._mask_num_cores = mask_num_cores

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        h = MontgomeryCXRHarmonizer(base_dir=self.base_image_dir)
        return h.harmonize(
            mask_output_dir=self._mask_output_dir,
            mask_num_cores=self._mask_num_cores,
        )
