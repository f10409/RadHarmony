"""Shared tensor helpers for backbone transform recipes.

Internal to the ``backbones`` package — not part of the public API.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def squeeze_frame_dim(x: torch.Tensor) -> torch.Tensor:
    """Drop a trailing size-1 frame dimension produced by some MONAI readers.

    MONAI's ``LoadImage`` occasionally returns ``(C, W, H, 1)`` for 2-D
    images stored as 3-D volumes. This collapses it back to ``(C, W, H)``.
    """
    return x.squeeze(-1) if x.ndim == 4 and x.shape[-1] == 1 else x


def to_3channel(x: torch.Tensor) -> torch.Tensor:
    """Repeat a single-channel image to 3 channels ``(1, H, W) → (3, H, W)``.

    No-op if the tensor already has 3 channels.
    """
    return x.repeat(3, 1, 1) if x.ndim == 3 and x.shape[0] == 1 else x


def normalize_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Min-max rescale to [0, 255] and cast to ``torch.uint8``."""
    lo, hi = float(x.min()), float(x.max())
    return ((x - lo) / (hi - lo + 1e-8) * 255).to(torch.uint8)


def patches_to_spatial(x: torch.Tensor) -> torch.Tensor:
    """Reshape a patch sequence ``[B, N, D]`` to a spatial map ``[B, D, H, W]``.

    Assumes a square patch grid (N = H × W). Use this to convert the patch
    token sequence from a ViT into a dense feature map suitable for a
    convolutional segmentation head.
    """
    B, N, D = x.shape
    H = W = int(N ** 0.5)
    if H * W != N:
        raise ValueError(
            f"Patch count {N} is not a perfect square and cannot be reshaped to "
            f"a spatial map. Expected N = H×W for some integer H."
        )
    return x.permute(0, 2, 1).reshape(B, D, H, W)


def to_pil_rgb(x: torch.Tensor) -> Image.Image:
    """Convert a single-channel float tensor ``(1, H, W)`` to a PIL RGB image.

    Applies min-max normalization to [0, 255] before conversion.
    """
    hw = x.numpy()[0]
    hw = (hw - hw.min()) / (hw.max() - hw.min() + 1e-8) * 255
    return Image.fromarray(hw.astype(np.uint8)).convert("RGB")
