# Split data

<!-- blocks: run=local -->

This page shows the 3 ways to split a dataset. Use the official split classes for a benchmark. Use a
random train and val split, or k folds, when a dataset has no official split. You can also use them to
split an official train split.

Before you start, [get the Montgomery County CXR data](get-the-data.md#montgomery-county-cxr) and the
[ChestX-Det10 data](get-the-data.md#other-datasets).

## 1. Use the official splits

When the dataset authors publish a split, each split has its own class and its own registry key (D-09).
Each class gives only the rows of its split. Example: ChestX-Det10 has a train class and a test class.

<!-- block: data=chestxdet10_train -->
```python
from radharmony.dataset import ChestXDet10TrainDataset
from radharmony.registry import resolve_dataset

train = ChestXDet10TrainDataset(base_image_dir="/data/ChestX-Det10/", output_cls=True, cache_dir=None)
test_cls = resolve_dataset("chestxdet10_test")  # the class of a registry key
test = test_cls(base_image_dir="/data/ChestX-Det10/", output_cls=True, cache_dir=None)
print(len(train.get_datasets()), len(test.get_datasets()))
# 3001 542
```

- The "Splits" section of each dataset page gives its split classes and registry keys.
- An old class that mixes the splits still works, for example `MIMICCXRJPGDataset` (key `mimic_cxr_jpg`).
  Do not use it for a benchmark.
- MR-RATE has split harmonizers, but no dataset class and no registry key yet.
- A split class keeps the split as the authors published it, also when 2 splits share patients (D-34).
  The dataset page says how many patients they share.

## 2. Make a train and val split

`get_datasets(n_splits=5)` puts about 1/5 of the patients in the val split. All images of one patient go
to the same split (D-10).

<!-- block: data=montgomery_cxr -->
```python
from radharmony.dataset import MontgomeryCXRDataset

ds = MontgomeryCXRDataset(base_image_dir="/data/MontgomerySet/CXR_png/", output_cls=True, cache_dir=None)
train_ds, val_ds = ds.get_datasets(n_splits=5)
print(len(train_ds), len(val_ds))
# 110 28
```

- `random_state=56` (the default) is the seed of the patient draw. The
  [Methods](../reference/api-datasets.md#methods) section of the Dataset API lists all arguments.
- The split does not keep the label ratio the same in train and val.
- If the train or the val split gets no patient, the call raises `ValueError`. Example: `n_splits=1`
  (D-10).

## 3. Make k folds

`get_folds()` shuffles the patients one time and puts them in `n_splits` groups of about the same size.
Each fold uses one group as the val split and the other groups as the train split. It takes the same
arguments as `get_datasets()`, but the default of `n_splits` is `5`.

<!-- block: data=montgomery_cxr -->
```python
for fold, (fold_train, fold_val) in enumerate(ds.get_folds(n_splits=5)):
    print(fold, len(fold_train), len(fold_val))
# 0 110 28
# 1 111 27
# 2 111 27
# 3 111 27
# 4 109 29
```

`n_splits` must be from 2 to the number of patients. Other values raise `ValueError` (D-10).

## 4. Augment only the train split

Give the augmentation to the train split only. With `val_transform=None`, the val split uses the transform
of the dataset. If that transform has a random step, `get_datasets(n_splits=...)` and `get_folds()` warn
(D-49).

<!-- block: data=montgomery_cxr -->
```python
from radharmony.dataset.transforms import RadiologyTransform2D

augment = RadiologyTransform2D(img_size=224).with_flip(spatial_axis=1, prob=0.5).get_transform()
plain = RadiologyTransform2D(img_size=224).get_transform()
train_ds, val_ds = ds.get_datasets(n_splits=5, train_transform=augment, val_transform=plain)
```

## Check

No patient is in both splits:

<!-- block: data=montgomery_cxr -->
```python
import os

table = ds.get_harmonized_df()
patient_of = dict(zip(table["image_path"], table["patient_id"]))


def patients(split):
    return {patient_of[os.path.relpath(d["img"], ds.base_image_dir)] for d in split.data}


print(patients(train_ds) & patients(val_ds))
# set()
```

## See also

- [Load a dataset and pick outputs](load-a-dataset.md)
- [Cache the samples](cache.md): each split has its own cache folder.
- [Transforms](../reference/transforms.md): the augmentation steps.
- [D-09](../design.md#d-09-official-splits-are-separate-classes),
  [D-10](../design.md#d-10-our-own-splits-group-by-patient) and
  [D-34](../design.md#d-34-official-splits-stay-as-published-also-when-they-share-patients) in Design
  decisions.
