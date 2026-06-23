# Dataset API

## Constructor pattern

Every dataset class follows the same constructor signature:

```python
from radharmony.dataset import CheXpertTrainDataset
import torch

ds = CheXpertTrainDataset(
    base_image_dir="/data/CheXpert-v1.0/train/",   # required
    csv_path=None,                       # auto-discovered relative to base_image_dir
    output_cls=True,
    output_mask=False,
    output_report=False,
    output_bbox=False,
    transform=None,                      # None → default 2-D or 3-D pipeline; pass a custom MONAI Compose to override
    cache_dir="./cache",
    dtype=torch.float32,                 # override of default torch.bfloat16
)
```

Pass a custom MONAI `Compose` pipeline to `transform` to control resizing, normalization, and augmentation. See [Transforms](transforms.md) for details and ready-made examples.

### Common arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | — | Root directory containing images |
| `csv_path` | `str` | No | auto | Primary metadata CSV; auto-discovered if omitted |
| `output_cls` | `bool` | No | `False` | Include `"cls"` in data dict |
| `output_mask` | `bool` | No | `False` | Include `"mask"` in data dict |
| `output_report` | `bool` | No | `False` | Include `"report"` in data dict |
| `output_bbox` | `bool` | No | `False` | Include `"bbox"` and `"bbox_labels"` in data dict |
| `output_reg` | `bool` | No | `False` | Include `"reg"` in data dict (regression datasets only) |
| `transform` | MONAI transform | No | standard 2-D or 3-D pipeline | Override the default image transform pipeline |
| `cache_dir` | `str` | No | `"./cache"` | MONAI PersistentDataset cache directory; `None` disables caching |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame — skips CSV parsing |
| `harmonizer` | harmonizer instance | No | `None` | Pre-instantiated harmonizer — skips CSV parsing |
| `harmonizer_path` | `str` | No | `None` | Path to a saved harmonized CSV — skips CSV parsing |

Dataset-specific arguments (e.g. `drop_uncertain`, `bbox_csv_path`, `series_filter`) are documented on each [per-dataset page](datasets.md).

---

## Harmonizer: standalone usage

Run the harmonizer independently to inspect the standardised DataFrame or cache it to disk:

```python
from radharmony.harmonizer import CheXpertTrainHarmonizer

h = CheXpertTrainHarmonizer(
    csv_path="/data/CheXpert-v1.0/train/train.csv",
    base_image_dir="/data/CheXpert-v1.0/train/",
)
df = h.harmonize()

print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'atelectasis', ..., 'view_position']

df.to_csv("chexpert_harmonized.csv", index=False)
```

The harmonized DataFrame always contains at minimum: `patient_id`, `study_id`, `image_path`, and one column per label in `LABEL_COLS`. Depending on the dataset it may also include `view_position`, `series_id`, `mask_path`, `bbox`, `report`, and `split`.

---

## Load from a saved harmonized CSV

Passing a pre-built DataFrame skips CSV parsing and harmonization on every run. You can also filter or modify the DataFrame before passing it in — this is the easiest way to create a custom subset without subclassing anything:

```python
import pandas as pd
from radharmony.dataset import CheXpertTrainDataset

df = pd.read_csv("chexpert_harmonized.csv")

# Example: keep only frontal views with a confirmed pleural effusion
df = df[df["view_position"] == "PA"]
df = df[df["pleural_effusion"] == 1.0]

ds = CheXpertTrainDataset(
    base_image_dir="/data/CheXpert-v1.0/train/",
    harmonized_df=df,
    output_cls=True,
)
```

Three equivalent ways to skip re-harmonization:

| Method | When to use |
|--------|-------------|
| `harmonized_df=pd.read_csv(...)` | DataFrame already in memory |
| `harmonizer_path="path/to/harmonized.csv"` | CSV on disk — loaded automatically |
| `harmonizer=MyHarmonizer(...)` | Pass a pre-instantiated harmonizer object |

---

## Splitting data

### Train / val split

```python
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

Splits are **patient-level** (no patient appears in both train and val). `n_splits=5` puts 1/5 of patients in the val set. Pass `n_splits=None` to get a single dataset with all patients.

Full signature:

```python
ds.get_datasets(
    n_splits=None,           # int or None
    num_cores=2,             # parallel workers for building the data dict
    random_state=56,
    train_transform=None,    # override transform for train set
    val_transform=None,      # override transform for val set
)
```

### k-fold cross-validation

```python
for fold_idx, (train_ds, val_ds) in enumerate(ds.get_folds(n_splits=5)):
    # train on train_ds, evaluate on val_ds
    ...
```

Full signature:

```python
ds.get_folds(
    n_splits=5,
    num_cores=2,
    random_state=56,
    train_transform=None,
    val_transform=None,
)
```

---

## Verifying images on disk

```python
missing = ds.verify_images(drop_missing=True)
# Prints: "Dropped 3 rows with missing images (217 remaining)."
# missing is a DataFrame of the dropped rows
```

`verify_images()` checks that every `image_path` in the harmonized DataFrame exists on disk. With `drop_missing=True` (default) missing rows are removed before any splits are created.

---

## Data dict keys

Every sample returned by a dataset is a plain Python dict:

| Key | Shape | Dtype | Condition |
|-----|-------|-------|-----------|
| `"img"` | `(1, H, W)` or `(1, D, H, W)` | `dtype` | Always present |
| `"cls"` | `(n_labels,)` | `dtype` | `output_cls=True` |
| `"mask"` | `(1, H, W)` | `dtype` | `output_mask=True` |
| `"bbox"` | list of `[r0, r1, c0, c1]` | Python list | `output_bbox=True` |
| `"bbox_labels"` | list of str | Python list | `output_bbox=True` |
| `"report"` | str | — | `output_report=True` |
| `"reg"` | `(n_targets,)` | `dtype` | `output_reg=True` |

Bounding box coordinates are **fractional** in `[0, 1]` in `[dim0_min, dim0_max, dim1_min, dim1_max]` order (row-first, matching array indexing).

Image tensors are normalized to `[-1, 1]`.

---

## Caching

RadHarmony uses MONAI's `PersistentDataset`, which caches preprocessed (but not augmented) tensors to disk. The first run is slow; subsequent runs load from cache.

```python
# Default: cache in ./cache/
ds = CheXpertTrainDataset(base_image_dir="...", cache_dir="./cache")

# Custom cache location
ds = CheXpertTrainDataset(base_image_dir="...", cache_dir="/fast/ssd/radharmony_cache/")

# Disable caching (re-processes every sample at each access)
ds = CheXpertTrainDataset(base_image_dir="...", cache_dir=None)
```

The cache is keyed by the transform pipeline hash. Changing augmentation settings invalidates the cache automatically.

---

## Pre-warming the cache

`pre_cache()` iterates the entire dataset once so that every sample is written to disk before training starts. This avoids cache-miss slowdowns during the first epoch and is useful when you want a predictable training speed from epoch 1.

```python
train_ds, val_ds = ds.get_datasets(n_splits=5)

# Warm the cache for both splits before training
ds.pre_cache(train_ds, num_workers=4)
ds.pre_cache(val_ds,   num_workers=4)
```

`pre_cache` drives the dataset through a `DataLoader` with `batch_size=1` and shows a `tqdm` progress bar. `num_workers` controls how many parallel workers load and cache items simultaneously — set it to match the number of CPU cores available for I/O.
