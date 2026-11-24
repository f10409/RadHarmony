# Load a dataset and pick outputs

<!-- blocks: run=local data=montgomery_cxr -->

This page shows how to make a dataset object, pick the keys of each sample and check the image files. The
examples use [Montgomery County CXR](../reference/datasets/cxr-boxes-masks/montgomery_cxr.md).

Before you start, [get the Montgomery data](get-the-data.md#montgomery-county-cxr).

## 1. Make the dataset

Give the class the image folder and the output flags. The class reads the source files and makes the
harmonized table.

```python
import torch
from radharmony.dataset import MontgomeryCXRDataset

ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    output_cls=True,
    output_report=True,
    transform=None,  # None gives the default image steps
    cache_dir=None,  # no disk cache
    dtype=torch.float32,
)
data = ds.get_datasets()  # one dataset with all the rows
# UserWarning: 1 of 138 rows have no 'report', so the dataset drops them (D-33).
sample = data[0]
print(sorted(sample))
# ['cls', 'img', 'report']
print(sample["img"].shape, sample["img"].dtype)
# torch.Size([1, 224, 224]) torch.float32
```

- `base_image_dir` is the image folder. The dataset page says which folder it is.
- Other file arguments, for example `csv_path`, are on the dataset page.
- `transform=None` gives the default transform. It keeps `img` and the keys of the flags that you set. To
  change the image size or to add augmentation, give your own transform
  ([Transforms](../reference/transforms.md)). Set its `output_keys` to all the keys that you need. If it
  drops the key of a flag, the dataset class warns (D-11).
- `dtype` is the type of `img` ([Transforms: dtype](../reference/transforms.md#dtype)).
- `cache_dir` is the cache folder ([Cache the samples](cache.md)).

## 2. Pick the outputs

`img` is always in the sample. Each `output_*` flag adds its own keys
([Data dict keys](../reference/api-datasets.md#data-dict-keys)). A flag that the class does not support
gives a warning and no key:

```python
ds_box = MontgomeryCXRDataset(base_image_dir="/data/MontgomerySet/CXR_png/", output_bbox=True)
# UserWarning: output_bbox=True is not supported by MontgomeryCXRDataset; the 'bbox' key will be absent
# from samples.
```

With `output_report=True` in the first block, the dataset drops the 1 Montgomery image with no report
(D-33). So `data` has 1 row less than the table:

```python
print(len(ds.get_harmonized_df()), len(data))
# 138 137
```

## 3. Check the image files

`verify_images()` checks that the file of each `image_path` is on the disk. It returns the rows with no
file.

```python
missing = ds.verify_images()
# All image files exist.
print(len(missing))
# 0
```

- With `drop_missing=True` (the default), the dataset drops these rows. The splits that you make after
  the call do not have them.
- With `drop_missing=False`, the dataset keeps all rows. You only get the list.
- A relative `image_path` with `base_image_dir=None` raises `ValueError` (D-04).

## 4. Look at the harmonized table

`get_harmonized_df()` returns the harmonized table of the dataset, one row for each image.

```python
df = ds.get_harmonized_df()
print(df.columns.tolist()[:7])
# ['patient_id', 'study_id', 'image_path', 'report', 'age_years', 'sex', 'tuberculosis']
```

Each dataset also has a harmonizer class that makes this table alone, with no dataset object. The
"Harmonizer" section of each dataset page shows it. To save the table, or to filter it before the dataset
uses it, see [Save and reuse a harmonized table](save-and-reuse.md).

## Check

- `sorted(sample)` gives only `img` and the keys of the flags that you set.
- `verify_images()` prints `All image files exist.`, or the number of rows with no file
  (`Dropped N rows ...` or `Found N rows ... (not dropped)`).

## See also

- [Dataset API](../reference/api-datasets.md): the shared arguments and the keys of a sample.
- [Dataset catalog](../reference/datasets/index.md): the class and the outputs of each dataset.
- [Split data](split-data.md): official splits, a train and val split, and folds.
- [Transforms](../reference/transforms.md): the default image steps and augmentation.
