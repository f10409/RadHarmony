"""Eval intensity scaling that matches the RH-DINO pretraining (a module of its own: the image cache key holds
the module name of this function, so the key stays the same for every caller)."""

from __future__ import annotations

import numpy as np
import torch

from radharmony.dataset.base import _apply_voi_lut, _fix_missing_highbit
from radharmony.evaluator.transforms import _float_for_dicom_steps


def percentile_255(x: torch.Tensor) -> torch.Tensor:
    """Run the DICOM steps of ``display_255`` (HighBit fix, VOI window), then scale as in pretraining.

    Pretraining (``_base_load_2d`` and ``to_uint8_rgb``): gray (RGB with the PIL ``convert("L")`` weights, alpha
    dropped), then the 0.5 and 99.5 percentiles go to 0 and 255, with clipping. Unlike ``display_255``, an 8-bit
    image is also scaled. A flat image gives 0. Input: (C, H, W). Output: (1, H, W) float in 0 to 255.
    """
    x = _apply_voi_lut(_fix_missing_highbit(_float_for_dicom_steps(x))).float()
    if x.shape[0] in (2, 4):  # drop alpha
        x = x[:-1]
    if x.shape[0] == 3:
        x = 0.299 * x[0:1] + 0.587 * x[1:2] + 0.114 * x[2:3]
    lo, hi = np.percentile(np.asarray(x, dtype=np.float32), (0.5, 99.5))  # as MONAI ScaleIntensityRangePercentiles
    return ((x - lo) * (255.0 / (hi - lo))).clamp(0, 255) if hi > lo else torch.zeros_like(x)
