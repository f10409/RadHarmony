# Cache the samples

<!-- blocks: run=local data=montgomery_cxr -->

Each sample needs a load step and preprocessing steps. These steps are slow. The cache keeps their result
on the disk, so the next epochs and the next runs read it back. Fill the cache before training to make the
first epoch as fast as the others. The examples use
[Montgomery County CXR](../reference/datasets/cxr-boxes-masks/montgomery_cxr.md).

Before you start, [get the Montgomery data](get-the-data.md#montgomery-county-cxr).

## 1. Pick the cache folder

`cache_dir` is the cache folder. The default is `"./cache"`. `cache_dir=None` gives no cache, so each
access loads the image again.

```python
from radharmony.dataset import MontgomeryCXRDataset

ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    output_cls=True,
    cache_dir="./cache",  # a fast local disk is best
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

The cache is a MONAI `PersistentDataset`. It writes a sample to the disk the first time that the code
reads the sample. The cache keeps the tensors before the first random step, so each epoch gets a new
augmentation. A new load or preprocessing step, for example a new `img_size`, gives a new cache folder.
[Dataset API: Cache](../reference/api-datasets.md#cache) gives the folder layout and when to delete it.

## 2. Fill the cache before training

`pre_cache()` reads each sample one time, before the training starts. It uses a `DataLoader` with
`batch_size=1` and shows a progress bar. `num_workers` is the number of processes that load the samples.

```python
ds.pre_cache(train_ds, num_workers=4)
ds.pre_cache(val_ds, num_workers=4)
# Pre-caching: 100%|██████████| 110/110
# Pre-caching: 100%|██████████| 28/28
```

## Check

The cache folder has one folder for each split, and each split folder holds one key folder:

```python
import os

print(sorted(os.listdir("./cache")))
# ['train', 'val']
print(len(os.listdir(os.path.join("./cache/train", os.listdir("./cache/train")[0]))))
# 110
```

## See also

- [Split data](split-data.md)
- [Transforms](../reference/transforms.md): which steps come before the first random step.
- [D-15](../design.md#d-15-cache-before-augmentation) and
  [D-49](../design.md#d-49-a-val-split-with-random-steps-warns) in Design decisions.
