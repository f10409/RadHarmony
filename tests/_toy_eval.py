"""Toy datasets for the evaluator tests (CPU, no NAS, no image files).

``ToyClsDS`` and ``ToySegDS`` have only the parts of a RadHarmony dataset that
the evaluators use. ``ToyRealDS`` is a real ``BaseRadiologicalDataset``
subclass. Its transform maps each image name to a stored feature vector, so
the real ``get_datasets`` and ``get_data_dict`` code runs.
"""
from __future__ import annotations

import os

# The evaluators build DataLoaders with pin_memory=True. On a GPU machine this
# starts CUDA, which can take 20 s. The tests use the CPU only, so hide the GPUs
# (a value that the user sets stays).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import monai as mn  # noqa: E402

# Keyword arguments that make an evaluator run fast on the CPU.
CPU_KW = dict(device="cpu", autocast_dtype=None, num_workers=0)


class _ListDS(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class ToyClsDS:
    """Classification dataset. ``feats`` is the "image", so use an Identity encoder.

    Args:
        feats: [N, D] features.
        labels: [N, L] labels (0, 1 or NaN), in the order of ``label_cols``.
        label_cols: label names.
        patient_ids: one id per row. The default gives one patient per row.
    """

    def __init__(self, feats, labels, label_cols, patient_ids=None):
        self.feats = np.asarray(feats, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.LABEL_COLS = list(label_cols)
        n = len(self.feats)
        pids = list(patient_ids) if patient_ids is not None else [f"p{i}" for i in range(n)]
        self.df = pd.DataFrame({"image_path": [f"img_{i}.png" for i in range(n)], "patient_id": pids})

    def _get_harmonized_df(self):
        return self.df

    def get_datasets(self, n_splits=None, num_cores=2):
        return _ListDS([{"img": torch.tensor(f), "cls": torch.tensor(y)}
                        for f, y in zip(self.feats, self.labels, strict=True)])

    def subset(self, rows) -> "ToyClsDS":
        """Return a new dataset with only the given rows (bool mask or index array)."""
        rows = np.asarray(rows)
        return ToyClsDS(self.feats[rows], self.labels[rows], self.LABEL_COLS,
                        self.df["patient_id"].to_numpy()[rows])


class ToySegDS:
    """Segmentation dataset. ``imgs`` are dense feature maps [D, h, w], ``masks`` are [1, H, W]."""

    def __init__(self, imgs, masks, patient_ids=None):
        self.imgs = [torch.as_tensor(m, dtype=torch.float32) for m in imgs]
        self.masks = [torch.as_tensor(m, dtype=torch.float32) for m in masks]
        n = len(self.imgs)
        pids = list(patient_ids) if patient_ids is not None else [f"p{i}" for i in range(n)]
        self.df = pd.DataFrame({"image_path": [f"img_{i}.png" for i in range(n)], "patient_id": pids})

    def _get_harmonized_df(self):
        return self.df

    def get_datasets(self, n_splits=None, num_cores=2):
        return _ListDS([{"img": i, "mask": m} for i, m in zip(self.imgs, self.masks, strict=True)])


def toy_seg_data(n=24, d=4, h=4, out=8, seed=0, scale=255.0):
    """Make dense features and masks where the foreground is ``feature 0 > 0``."""
    rng = np.random.default_rng(seed)
    imgs, masks = [], []
    for _ in range(n):
        f = rng.normal(size=(d, h, h)).astype(np.float32)
        fg = (f[0] > 0).astype(np.float32)
        imgs.append(f)
        masks.append(np.kron(fg, np.ones((out // h, out // h), np.float32))[None] * scale)
    return imgs, masks


class _FeatTransform(mn.transforms.Transform):
    """Map a data dict to tensors: the image name gives a stored feature vector.

    A MONAI transform, so its stored features are in the cache key (D-15) and in the
    fingerprint of the embedding cache (V-E-25).
    """

    def __init__(self, feats, masks=None):
        self.feats = feats
        self.masks = masks or {}

    def __call__(self, d):
        out = {"img": torch.as_tensor(self.feats[os.path.basename(d["img"])])}
        if "cls" in d:
            out["cls"] = torch.as_tensor(np.asarray(d["cls"], dtype=np.float32))
        if "mask" in d:
            out["mask"] = torch.as_tensor(self.masks[d["mask"]])
        return out


def toy_real_cls_ds(feats, labels, label_cols, patient_ids=None):
    """Return a real ``BaseRadiologicalDataset`` subclass with ``output_cls=True``."""
    from radharmony.dataset.base import BaseRadiologicalDataset

    class _Real(BaseRadiologicalDataset):
        LABEL_COLS: list = []

        def _get_harmonized_df(self):
            return self._try_resolve_preset_harmonized()

    n = len(feats)
    names = [f"img_{i}.png" for i in range(n)]
    df = pd.DataFrame({"image_path": names,
                       "patient_id": list(patient_ids) if patient_ids is not None else names})
    for j, c in enumerate(label_cols):
        df[c] = np.asarray(labels)[:, j]
    t = _FeatTransform({nm: np.asarray(feats[i], np.float32) for i, nm in enumerate(names)})
    return _Real(base_image_dir="", transform=t, cache_dir=None, output_cls=True, harmonized_df=df)


def toy_real_seg_ds(imgs, masks, patient_ids=None, null_mask_rows=()):
    """Return a real ``BaseRadiologicalDataset`` subclass with ``output_mask=True``.

    The rows in ``null_mask_rows`` get an empty ``mask_path`` (not annotated, D-33).
    """
    from radharmony.dataset.base import BaseRadiologicalDataset

    class _Real(BaseRadiologicalDataset):
        LABEL_COLS: list = []

        def _get_harmonized_df(self):
            return self._try_resolve_preset_harmonized()

    n = len(imgs)
    names = [f"img_{i}.png" for i in range(n)]
    mpaths = [None if i in null_mask_rows else f"m_{nm}" for i, nm in enumerate(names)]
    df = pd.DataFrame({"image_path": names, "mask_path": mpaths,
                       "patient_id": list(patient_ids) if patient_ids is not None else names})
    t = _FeatTransform({nm: np.asarray(imgs[i], np.float32) for i, nm in enumerate(names)},
                       {f"m_{nm}": np.asarray(masks[i], np.float32) for i, nm in enumerate(names)})
    return _Real(base_image_dir="", transform=t, cache_dir=None, output_mask=True, harmonized_df=df)
