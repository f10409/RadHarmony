"""Mask encoding/decoding utilities for segmentation datasets."""

import numpy as np


def rle_to_mask(rle: str, width: int = 1024, height: int = 1024) -> np.ndarray:
    """Decode a SIIM-ACR RLE string to a binary mask of shape (H, W).

    Pixels are encoded column-major (index = col * height + row) with
    cumulative start positions. Returns zeros for '-1' (no finding).
    """
    mask = np.zeros(width * height, dtype=np.uint8)
    if rle.strip() == "-1":
        return mask.reshape(height, width)
    array = np.asarray([int(x) for x in rle.split()])
    starts, lengths = array[0::2], array[1::2]
    current = 0
    for start, length in zip(starts, lengths):
        current += start
        mask[current : current + length] = 1
        current += length
    return mask.reshape(width, height).T  # (W, H) → (H, W)


def mask_to_rle(mask: np.ndarray) -> str:
    """Encode a binary mask of shape (H, W) to a SIIM-ACR RLE string.

    Inverse of ``rle_to_mask``. Returns '-1' for an all-zero mask.
    """
    flat = (mask.T).reshape(-1)  # (H, W) → (W, H) → 1-D column-major
    flat = (flat > 0).astype(np.uint8)
    if flat.max() == 0:
        return "-1"
    rle = []
    current = 0
    last = 0
    run_start = -1
    run_length = 0
    for pixel in flat:
        if pixel != last:
            if pixel == 1:
                run_start = current
                run_length = 1
            else:
                rle += [str(run_start), str(run_length)]
                run_start = -1
                run_length = 0
                current = 0
        elif run_start > -1:
            run_length += 1
        last = pixel
        current += 1
    if run_start > -1:
        rle += [str(run_start), str(run_length)]
    return " ".join(rle)
