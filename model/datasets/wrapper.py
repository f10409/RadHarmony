"""Wrap RadHarmony PersistentDatasets with LeJEPA multi-crop."""

import torch
from torch.utils.data import Dataset, Subset


class LeJEPAWrapper(Dataset):
    """Apply a multi-crop transform on top of a cached RadHarmony dataset.

    RadHarmony's PersistentDataset handles the expensive deterministic
    preprocessing (DICOM load, VOI LUT, normalisation, resize).  This
    wrapper adds the stochastic multi-crop augmentation at access time.

    Args:
        radharmony_dataset: A MONAI PersistentDataset (or Dataset) returned
            by ``BaseRadiologicalDataset.get_datasets()``.  The dataset's
            ``LABEL_COLS`` should already be overwritten to contain only
            the target label (e.g. pneumothorax) so that ``cls`` is ``(1,)``.
        multi_crop_transform: Callable that takes a ``(1, H, W)`` tensor and
            returns a list of ``(1, img_size, img_size)`` crop tensors.
    """

    def __init__(self, radharmony_dataset, multi_crop_transform):
        self.dataset = radharmony_dataset
        self.transform = multi_crop_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]
        img = sample["img"]  # (1, H, W), float in [-1, 1]
        views = self.transform(img)

        out = {"views": views}

        if "cls" in sample:
            out["ptx_label"] = float(sample["cls"][0])
        else:
            out["ptx_label"] = float("nan")

        return out


def subsample_dataset(
    dataset: Dataset,
    fraction: float | None = None,
    count: int | None = None,
    seed: int = 42,
) -> Subset:
    """Return a random subset of *dataset* for scaling studies.

    Exactly one of *fraction* or *count* must be provided.
    """
    n = len(dataset)
    if count is not None:
        k = min(count, n)
    elif fraction is not None:
        k = int(n * fraction)
    else:
        raise ValueError("Provide either fraction or count")

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=g)[:k].tolist()
    return Subset(dataset, indices)
