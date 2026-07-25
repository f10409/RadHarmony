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
| `harmonizer_path` | `str` | No | `None` | Path to a pickle saved via `harmonizer.save()` — skips CSV parsing |

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

## Skip re-harmonization: save and reuse

Harmonization (CSV parsing, merging, path construction) can be slow for large
datasets. A dataset accepts three constructor arguments that bypass it, resolved
in this order:

| Argument | Loads | Use when |
|----------|-------|----------|
| `harmonized_df` | a DataFrame you pass directly (from a pickle, `pd.read_csv`, or built in memory) | you want to filter / inspect the table first |
| `harmonizer_path` | a **pickle** saved via `harmonizer.save()` | save once, reload every run |
| `harmonizer` | an already-harmonized harmonizer instance | you harmonized in memory and want to reuse it |

If none are given, the dataset runs its own harmonization.

### Save a harmonizer to a pickle

`harmonizer.save()` pickles the harmonized DataFrame, its `LABEL_COLS`, and an
init snapshot (constructor arguments) so the harmonizer can be reconstructed:

```python
from radharmony.harmonizer import CheXpertTrainHarmonizer

h = CheXpertTrainHarmonizer(csv_path="/data/CheXpert-v1.0/train/train.csv")
h.harmonize()
h.save("chexpert_harmonized.pkl")
```

Reload it without re-running any CSV work with the `load_from_saved()` class
method — pass init overrides if the data has moved:

```python
h = CheXpertTrainHarmonizer.load_from_saved("chexpert_harmonized.pkl")
print(h.harmonized_df.head())

# Override init params at load time (e.g. data moved to a new path)
h = CheXpertTrainHarmonizer.load_from_saved(
    "chexpert_harmonized.pkl", csv_path="/new/path/train.csv",
)
```

### `harmonizer_path` — load the pickle straight into a dataset

The most common pattern: save once, reload on every subsequent run. Each dataset
class sets `_HARMONIZER_CLS`, so it knows which harmonizer's `load_from_saved()`
to call.

```python
from radharmony.dataset import CheXpertTrainDataset

ds = CheXpertTrainDataset(
    base_image_dir="/data/CheXpert-v1.0/train/",
    harmonizer_path="chexpert_harmonized.pkl",   # pickle from harmonizer.save()
    output_cls=True,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

### `harmonized_df` — pass (or filter) a DataFrame directly

The easiest way to build a custom subset without subclassing. The DataFrame can
come from a saved pickle's `.harmonized_df`, or from a CSV you exported earlier
with `h.harmonize().to_csv(...)` and reload with `pd.read_csv`:

```python
import pandas as pd

# From a CSV export …
df = pd.read_csv("chexpert_harmonized.csv")
# … or from a pickle: df = CheXpertTrainHarmonizer.load_from_saved("...pkl").harmonized_df

# Keep only frontal views with a confirmed pleural effusion
df = df[(df["view_position"] == "PA") & (df["pleural_effusion"] == 1.0)]

ds = CheXpertTrainDataset(
    base_image_dir="/data/CheXpert-v1.0/train/",
    harmonized_df=df,
    output_cls=True,
)
```

A CSV carries only the table, so `LABEL_COLS` is re-inferred from the column
names; a pickle preserves the stored `LABEL_COLS` and init snapshot exactly.

### `harmonizer` — pass an in-memory instance

```python
h = CheXpertTrainHarmonizer(csv_path="/data/CheXpert-v1.0/train/train.csv")
h.harmonize()
ds = CheXpertTrainDataset(base_image_dir="/data/CheXpert-v1.0/train/", harmonizer=h, output_cls=True)
```

### Swap the DataFrame after construction

`set_harmonized_df()` pins a new DataFrame on an existing dataset; all later
`get_datasets()` / `get_folds()` calls use it:

```python
ds.set_harmonized_df(df)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

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
