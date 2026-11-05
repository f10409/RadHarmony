"""Collate function for samples with boxes, or with a VQA ``answer_struct``.

The default DataLoader collate stacks each tensor key. A box tensor has one row
for each box. So the default collate fails when 2 images have a different
number of boxes. :func:`bbox_collate` keeps the per-box keys as lists. It uses
the default collate for the other keys.

Example::

    from torch.utils.data import DataLoader
    from radharmony.dataset import bbox_collate

    loader = DataLoader(train_ds, batch_size=8, collate_fn=bbox_collate)
    batch = next(iter(loader))
    batch["img"]       # Tensor(8, 1, 224, 224)
    batch["bbox"]      # list of 8 tensors, each (N_k, 4)
"""

from __future__ import annotations

from torch.utils.data import default_collate

#: Sample keys with one value for each box. Each image can have a different number of boxes.
BBOX_KEYS = ("bbox", "bbox_labels", "bbox_findings", "bbox_mask")

#: Keys that stay as a list of the sample values. They are the box keys and the VQA ``answer_struct``.
#: The keys of an ``answer_struct`` dict differ from sample to sample (V-D-21).
LIST_KEYS = BBOX_KEYS + ("answer_struct",)


def bbox_collate(batch: list[dict]) -> dict:
    """Collate samples with a different number of boxes into one batch.

    Args:
        batch: a list of samples (``ds[i]``).

    Returns:
        A dict. Each key in ``LIST_KEYS`` holds a list with one value for each
        sample. For example, ``out["bbox"][k]`` is the ``(N_k, 4)`` box tensor of
        sample ``k``. The other keys go through
        ``torch.utils.data.default_collate``.
    """
    out = default_collate([{k: v for k, v in s.items() if k not in LIST_KEYS} for s in batch])
    for key in LIST_KEYS:
        if key in batch[0]:
            out[key] = [s[key] for s in batch]
    return out
