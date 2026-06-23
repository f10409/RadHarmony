"""Collation for multi-view LeJEPA batches."""

from __future__ import annotations

import math

import torch


def multicrop_collate_fn(batch: list[dict]) -> dict:
    """Stack multi-crop views from every sample into a flat batch.

    Input
        List of dicts, each with:
        - ``views``: list of *V* tensors, each ``(1, img_size, img_size)``
        - ``ptx_label``: float (may be NaN if unavailable)

    Output
        - ``views``:  ``(B * V, 1, img_size, img_size)``
        - ``batch_size``: int
        - ``num_views``: int  (V, constant across samples)
        - ``labels``: ``(B, 1)``  (NaN-filled where label is missing)
    """
    num_views = len(batch[0]["views"])
    batch_size = len(batch)

    all_views = []
    labels = []
    for sample in batch:
        all_views.extend(sample["views"])
        labels.append(sample["ptx_label"])

    return {
        "views": torch.stack(all_views),          # (B*V, 1, S, S)
        "batch_size": batch_size,
        "num_views": num_views,
        "labels": torch.tensor(labels, dtype=torch.float32).unsqueeze(1),  # (B, 1)
    }
