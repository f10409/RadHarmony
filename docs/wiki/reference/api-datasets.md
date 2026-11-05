# Dataset API

Every image dataset class shares these facts: the constructor arguments, the keys of a sample and the
columns of the harmonized table. Each dataset page lists the arguments of its own class.

For the steps, see the how-to pages:

- [Load a dataset and pick outputs](../how-to/load-a-dataset.md)
- [Save and reuse a harmonized table](../how-to/save-and-reuse.md)
- [Split data](../how-to/split-data.md)
- [Cache the samples](../how-to/cache.md)

## Constructor arguments

Every image dataset class inherits these arguments from `BaseRadiologicalDataset`.

<!-- args: radharmony.dataset.base.BaseRadiologicalDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | | The folder of the images. `image_path` in the table is relative to it (D-04) |
| `transform` | MONAI transform | No | `None` in each dataset class | The image steps. `None` gives `RadiologyTransform2D` or `RadiologyTransform3D` ([Transforms](transforms.md)) |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache ([Cache the samples](../how-to/cache.md)) |
| `output_cls` | `bool` | No | `False` | Add `cls` to the sample |
| `output_mask` | `bool` | No | `False` | Add `mask` to the sample. The dataset drops the rows with no mask, and warns (D-33) |
| `output_report` | `bool` | No | `False` | Add `report` to the sample. The dataset drops the rows with no report, and warns (D-33) |
| `output_bbox` | `bool` | No | `False` | Add `bbox` and `bbox_labels` to the sample, and `bbox_findings` when the class has it |
| `output_reg` | `bool` | No | `False` | Add `reg` to the sample. A class with no regression target warns (D-11) |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Use this harmonized table. The class does not read the source files |
| `harmonizer` | harmonizer | No | `None` | Use the table of this harmonizer |
| `harmonizer_path` | `str` | No | `None` | Use the table of a file from `harmonizer.save()` |

- A flag that the class does not support gives a warning and no key (D-11). Every image class takes all 5
  flags.
- The dataset classes also take `dtype` (default `torch.bfloat16`), the type of `img`. The class gives
  it to its default transform ([Transforms: dtype](transforms.md#dtype)).
- Many classes also take `csv_path` and other file paths. When you leave out a file path, the class looks
  for the file in fixed places (D-35). The [dataset pages](datasets/index.md) list these arguments.
- A VQA or caption class (`BaseVQADataset`) has other arguments and other sample keys. Its dataset page
  lists them.

### Cache

`cache_dir` holds a MONAI `PersistentDataset` cache ([Cache the samples](../how-to/cache.md)).

- The cache keeps the tensors before the first random step. Augmentation runs again for each sample, so
  each epoch gets a new augmentation (D-15).
- Each split has its own folder: `train` and `val`, `fold_0/train` and so on. `get_datasets()` with no
  `n_splits` uses `cache_dir` itself.
- Inside this folder, the cache is in a folder named by a short key of the cached steps. A new load or
  preprocessing step, for example a new `img_size`, gives a new key and a new cache. A new augmentation
  setting, for example the flip probability, uses the same cache.
- A changed lambda or a changed RadHarmony step also gives a new key. If you change a helper function that
  a step calls, the key stays the same. Then delete the cache folder yourself.
- Old folders stay on the disk until you delete them.
- `torch.manual_seed(...)` repeats the random augmentation of a run, with any `num_workers` (D-15).

### Order of the table arguments

If you give more than one table argument, the class uses the first of this list
([Save and reuse a harmonized table](../how-to/save-and-reuse.md)):

| Order | Argument | Gives |
|---|---|---|
| 1 | `harmonized_df` | the table that you give |
| 2 | `harmonizer_path` | the table of a file from `save()` |
| 3 | `harmonizer` | `harmonized_df` of the harmonizer that you give |
| 4 | none of them | the class harmonizes the source files |

## Methods

| Method | What it does | How-to |
|---|---|---|
| `get_datasets(n_splits=None, num_cores=2, random_state=56, train_transform=None, val_transform=None)` | One dataset with all rows, or a `(train, val)` tuple grouped by patient | [Split data](../how-to/split-data.md) |
| `get_folds(n_splits=5, num_cores=2, random_state=56, train_transform=None, val_transform=None)` | Yields one `(train, val)` tuple for each fold | [Split data](../how-to/split-data.md) |
| `get_harmonized_df()` | Returns the harmonized table | [Load a dataset](../how-to/load-a-dataset.md) |
| `set_harmonized_df(df)` | Gives the dataset a new table for the next splits | [Save and reuse](../how-to/save-and-reuse.md) |
| `verify_images(drop_missing=True)` | Returns the rows with no image file, and drops them by default | [Load a dataset](../how-to/load-a-dataset.md) |
| `pre_cache(dataset, num_workers=2)` | Reads each sample of a split one time, to fill the cache | [Cache the samples](../how-to/cache.md) |

The arguments of `get_datasets()` and `get_folds()`:

| Name | Meaning |
|---|---|
| `n_splits` | `get_datasets()`: the val split gets about 1/`n_splits` of the patients. `None` gives one dataset with all rows. `get_folds()`: the number of folds, from 2 to the number of patients |
| `num_cores` | The number of processes that make the sample list |
| `random_state` | The seed of the patient draw |
| `train_transform` | The transform of the train split. `None` uses the transform of the dataset |
| `val_transform` | The transform of the val split. `None` uses the transform of the dataset, and warns if it has a random step (D-49) |

## Data dict keys

Each sample is a Python dict:

| Key | Shape | Type | Condition |
|---|---|---|---|
| `"img"` | `(C, H, W)` or `(1, D, H, W)` | `dtype` | Always |
| `"cls"` | `(n_labels,)` | float32 | `output_cls=True` |
| `"mask"` | `(1, H, W)` or `(1, D, H, W)` | float32 | `output_mask=True` |
| `"bbox"` | `(N, 4)` (2-D) or `(N, 6)` (3-D) | float32 | `output_bbox=True` |
| `"bbox_labels"` | list of `str`, one for each box | list | `output_bbox=True` |
| `"bbox_findings"` | list, one entry for each box | list | `output_bbox=True`, for a class that has it |
| `"report"` | `str` | `str` | `output_report=True` |
| `"reg"` | `(n_targets,)` | float32 | `output_reg=True` |

- `img`: values from -1 to 1 (D-14). `C` is 1 for a gray modality and 3 for the classes with color
  figures (D-42, [`channels`](transforms.md#radiologytransform2d)).
- `cls`: one value for each label, in the order of `sorted(ds.LABEL_COLS)` (the plain Python sort). After
  `get_datasets()` or `get_folds()`, `ds.LABEL_COLS` is in this order. So `ds.LABEL_COLS[i]` is the label
  of `cls[i]`. For example, `["edema", "atelectasis"]` gives `cls[0]` = atelectasis (D-05).
- `bbox`: fractions of the image size, from 0 to 1, in the order `[dim0_min, dim0_max, dim1_min,
  dim1_max]` (row first, as in array indexing). An image with no box gives shape `(0, 4)` or `(0, 6)`.
  The box matches the returned image (D-28). [Transforms: Boxes](transforms.md#boxes) says how each step
  moves it.
- The default `DataLoader` collate cannot stack boxes when the samples have a different number of boxes.
  Use `collate_fn=bbox_collate` (`from radharmony.dataset import bbox_collate`), for example
  `DataLoader(ds, batch_size=8, collate_fn=bbox_collate)`. It stacks `img`, `cls` and the other keys as
  usual. `bbox`, `bbox_labels`, `bbox_findings` and `bbox_mask` become lists with one entry for each
  sample.

## Harmonized table columns

The harmonized table has one row for each image (D-03). A few harmonizers give one row for each study,
series or volume, and their dataset page says so. The table always has `patient_id`, `study_id`,
`image_path` and one column for each label. `image_path` is relative to `base_image_dir` (D-04), so
`os.path.join(base_image_dir, image_path)` opens the file. Some datasets also have `series_id`,
`view_position`, `mask_path`, `bbox`, `bbox_labels`, `report` and `split`.

- `view_position` of a 2-D dataset uses the names `PA`, `AP` and `LATERAL`. An unknown view is NaN. Other
  source values are in upper case, for example `AP_HORIZONTAL` or `FRONTAL`. A 3-D dataset keeps its
  plane or series names, for example `axial` (CT-RATE) or `Sagittal T1` (RSNA 2024 lumbar).
- `age_years` and `sex` are in the table when the source has them. `age_years` is a float, in years
  (D-06). `sex` is `"M"`, `"F"` or NaN. NaN means unknown or other (D-26). `sex` comes after `age_years`.
- The "Harmonizer" section of each dataset page gives its extra columns.
