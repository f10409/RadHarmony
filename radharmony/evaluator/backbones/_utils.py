"""Shared tensor helpers for backbone transform recipes.

Internal to the ``backbones`` package. Not part of the public API.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import shutil
import tempfile

import monai.transforms as mt
import numpy as np
import torch
from PIL import Image

from radharmony.evaluator.transforms import _LoadImageAsShown, _display_255


def load_image() -> mt.LoadImage:
    """Return the image loader of the recipes (D-13, V-B-11).

    A palette or alpha PNG reads as the image it shows. The file keeps its own
    dtype, so that ``display_255`` keeps an 8-bit image as it is. Use
    ``mt.LoadImage`` for a mask, because a palette mask keeps its class numbers
    in the index.
    """
    return _LoadImageAsShown(image_only=True, ensure_channel_first=True, dtype=None)


def display_255(x: torch.Tensor) -> torch.Tensor:
    """Give the image as the dataset load shows it, float in 0 to 255 (D-13, D-27, V-B-12).

    A DICOM file gets its HighBit fix, and a MONOCHROME DICOM its VOI window. An 8-bit image stays as it
    is. Any other image is scaled with its minimum and maximum. Put it right after
    the transpose, on ``(C, H, W)``.
    """
    return _display_255(x)


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


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Round an image in 0 to 255 (the output of ``display_255``) and cast it to ``torch.uint8``.

    It does not stretch the image, so an 8-bit image keeps its values (D-27).
    """
    return x.float().round().to(torch.uint8)


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


def png_cache_dir(tmp_dir: str | None, prefix: str) -> str:
    """Return the folder for the PNG files of a generator, and make it.

    With ``tmp_dir=None``, it makes a new folder under the system temp folder and deletes it
    when Python exits. A folder that the user gives stays.
    """
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix=prefix)
        atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def png_path(tmp_dir: str, src: str) -> str:
    """Return the temp PNG path for the source image ``src``.

    The name has a hash of the full source path. Two images with the same
    file name (CheXpert: ``patientX/study1/view1_frontal.jpg``) then get two
    PNG files.
    """
    stem = os.path.splitext(os.path.basename(src))[0]
    digest = hashlib.sha1(os.path.abspath(src).encode()).hexdigest()[:16]
    return os.path.join(tmp_dir, f"{stem}_{digest}.png")


def to_pil_rgb(x: torch.Tensor) -> Image.Image:
    """Convert an image ``(C, H, W)`` in 0 to 255 (the output of ``display_255``) to a PIL RGB image.

    With 3 or more channels, the first 3 are R, G and B. With fewer, channel
    0 is gray. It rounds the values and does not stretch them (D-27, V-B-12).
    """
    arr = x.float().numpy()
    arr = arr[:3] if arr.shape[0] >= 3 else arr[:1]
    arr = np.round(arr)
    if arr.shape[0] == 3:
        return Image.fromarray(arr.transpose(1, 2, 0).astype(np.uint8), "RGB")
    return Image.fromarray(arr[0].astype(np.uint8)).convert("RGB")
