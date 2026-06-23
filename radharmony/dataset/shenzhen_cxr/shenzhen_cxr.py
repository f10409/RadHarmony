"""MONAI dataset for the NLM Shenzhen Hospital Chest X-Ray Set.

662 frontal CXRs (336 TB-positive, 326 normal) from Shenzhen No.3 People's
Hospital.  Binary TB label is encoded in each filename; no separate label CSV
is required.

``base_image_dir`` must point at the ``CXR_png/`` images directory itself
— the last common folder containing every image (matching the convention
used by other RadHarmony datasets). ``ClinicalReadings/`` and
``Annotations-2/`` are read as sibling directories. ``image_path`` entries
in the harmonized DataFrame are bare filenames (e.g. ``CHNCXR_00001_1.png``);
finding-mask paths use ``../Annotations-2/...`` to reach the sibling tree.

This mirrors :class:`MontgomeryCXRDataset` (the companion NLM TB dataset).
"""

import os

import pandas as pd
import torch

from radharmony.harmonizer.shenzhen_cxr import ShenzhenCXRHarmonizer
from radharmony.registry import register_dataset
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


_LABEL_COLS = ["tuberculosis"]


@register_dataset("shenzhen_cxr")
class ShenzhenCXRDataset(BaseRadiologicalDataset):
    """PyTorch/MONAI dataset for the Shenzhen Hospital CXR Set.

    Args:
        base_image_dir: The ``CXR_png/`` images directory (the last common
            folder containing every image), e.g.
            ``.../Shenzhen-Hospital-CXR-Set/CXR_png/``. Sibling
            ``ClinicalReadings/`` and ``Annotations-2/`` directories are
            read relative to it. Required unless ``harmonizer_path`` or
            ``harmonized_df`` is provided.
        transform: MONAI Compose transform.  Defaults to the standard 2-D
            224 px pipeline.
        cache_dir: MONAI PersistentDataset cache directory.  ``None`` disables
            caching.
        mask_output_dir: Directory to write unioned per-finding "TB region"
            PNGs into. Required when ``output_mask=True`` (unless a
            pre-harmonized df is supplied). One PNG per image: TB-positive
            cases get the union of their finding masks; TB-negative cases
            get a same-size all-zero PNG (matches SIIM-ACR PTX convention).
            Idempotent on re-runs.
        mask_num_cores: Accepted for signature parity with SIIM-ACR PTX;
            currently unused (the union loop is small).
        output_cls: Yield TB label vector under the ``"cls"`` key.
        output_report: Yield clinical reading text under the ``"report"`` key.
            Populated automatically when ``ClinicalReadings/`` is present as
            a sibling of ``base_image_dir``.
        output_mask: Yield fused per-finding "TB region" mask under the
            ``"mask"`` key. Requires ``mask_output_dir``. TB-negative rows
            get an all-zero mask (kept in the sample stream).
        output_bbox: Not supported; ignored with a warning.
        dtype: Output tensor dtype.  Default ``torch.bfloat16``.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "report", "mask"})
    LABEL_COLS = _LABEL_COLS
    _HARMONIZER_CLS = ShenzhenCXRHarmonizer

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
            _h = ShenzhenCXRHarmonizer.load_from_saved(harmonizer_path)
            base_image_dir = getattr(_h, "base_dir", None)

        if (
            base_image_dir is None
            and harmonizer_path is None
            and harmonized_df is None
            and harmonizer is None
        ):
            raise ValueError(
                "base_image_dir is required when harmonizer_path is not provided. "
                "Pass the path to the Shenzhen CXR_png/ images directory "
                "(e.g. .../Shenzhen-Hospital-CXR-Set/CXR_png/)."
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
                "Shenzhen's per-finding mask PNGs are unioned into a single "
                "binary 'TB region' mask per TB-positive image and saved to "
                "mask_output_dir before MONAI can load them. TB-negative "
                "rows get a same-size all-zero PNG (matches SIIM-ACR PTX "
                "convention). Pass a directory:\n"
                "    ShenzhenCXRDataset(..., output_mask=True, "
                "mask_output_dir='/path/to/tb_region_masks')"
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
        h = ShenzhenCXRHarmonizer(base_dir=self.base_image_dir)
        return h.harmonize(
            mask_output_dir=self._mask_output_dir,
            mask_num_cores=self._mask_num_cores,
        )
