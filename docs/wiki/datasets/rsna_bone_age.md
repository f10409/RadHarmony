# RSNA Pediatric Bone Age

**Modality:** Radiograph | **Format:** PNG | **Dim:** 2D | **Labels:** age in months (regression)

## Overview

The RSNA Pediatric Bone Age Challenge (2017) dataset contains 12,611 hand radiographs from patients aged 1 month to 228 months. The task is to predict skeletal age from the hand X-ray. This is a regression dataset — the target is `age_months`, not a classification label.

## Download

Available from the [RSNA website](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017).

Expected layout — train and val are separate downloads and each takes its
own `base_image_dir`. They do not have to share a parent directory.

Train (one ZIP):
```
<train base_image_dir>/                 # e.g. boneage-training-dataset/
  <id>.png                              # 12,611 PNGs
<train base_image_dir>/../train.csv     # sibling — auto-discovered
```

Val (separate ZIP):
```
<val base_image_dir>/                   # e.g. Bone Age Validation Set/
  Validation Dataset.csv
  boneage-validation-dataset-1/
    <id>.png                            # 800 PNGs (ids 1386–9708)
  boneage-validation-dataset-2/
    <id>.png                            # 625 PNGs (ids 10018–15612)
```

## Label columns

This dataset uses `REG_COLS` instead of `LABEL_COLS`:

| Column | Description |
|--------|-------------|
| `age_months` | Skeletal age in months |

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | `None` | Split-specific image directory. Train: `boneage-training-dataset/` (flat dir of `<id>.png` files). Val: `Bone Age Validation Set/` (contains `Validation Dataset.csv` plus the two `boneage-validation-dataset-{1,2}/` sub-subdirs — val is a structural exception with multi-sibling sub-subdirs). Train and val are separate downloads and may live anywhere on disk |
| `csv_path` | `str` | No | auto | `train.csv` or `Validation Dataset.csv`; auto-discovered |
| `output_cls` | `bool` | No | `False` | Include `"cls"` in data dict (no label cols — unused) |
| `output_mask` | `bool` | No | `False` | Include `"mask"` in data dict |
| `output_report` | `bool` | No | `False` | Include `"report"` in data dict |
| `output_bbox` | `bool` | No | `False` | Include `"bbox"` and `"bbox_labels"` in data dict |
| `output_reg` | `bool` | No | `False` | Include `"reg"` tensor with age in months |
| `cache_dir` | `str` | No | `"./cache"` | MONAI cache directory |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonized CSV |
| `split` | `str` | No | `'train'` | Which split to load |
| `transform` | MONAI transform | No | `None` | MONAI Compose transform; `None` uses the default pipeline |

## Dataset constructor

The constructor arguments above apply to both `RSNABoneAgeTrainDataset` and `RSNABoneAgeValDataset` (both omit the `split=` arg — split is fixed by the class).

### Train split

```python
import torch
from radharmony.dataset import RSNABoneAgeTrainDataset

ds_train = RSNABoneAgeTrainDataset(
    base_image_dir="/data/boneage-training-dataset/",
    output_reg=True,
    dtype=torch.float32,
)
train_ds, _ = ds_train.get_datasets(n_splits=5)
```

### Val split (separate download — different `base_image_dir`)

```python
from radharmony.dataset import RSNABoneAgeValDataset

ds_val = RSNABoneAgeValDataset(
    base_image_dir="/data/Bone Age Validation Set/",
    output_reg=True,
    dtype=torch.float32,
)
```

## Harmonizer: instantiate and inspect

```python
from radharmony.harmonizer import RSNABoneAgeTrainHarmonizer

h = RSNABoneAgeTrainHarmonizer(
    base_image_dir="/data/boneage-training-dataset/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'age_months']
df.to_csv("rsna_bone_age_harmonized.csv", index=False)
```

`RSNABoneAgeValHarmonizer` is the val-split equivalent.

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import RSNABoneAgeTrainDataset

ds = RSNABoneAgeTrainDataset(
    base_image_dir="/data/boneage-training-dataset/",
    harmonized_df=pd.read_csv("rsna_bone_age_harmonized.csv"),
    output_reg=True,
)
```

## Harmonizer notes

- Train and val splits use different CSV files and different image directories
- Use `RSNABoneAgeTrainDataset` for the train split and `RSNABoneAgeValDataset` for the val split — split is fixed by the class
- `REG_COLS = ["age_months"]`; use `output_reg=True` to get the regression target
- Age is in months (integer); model outputs should be in the same unit

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_reg=True` | `"reg"` | `(1,)` | Skeletal age in months |

## Example paths

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/path/to/rsna-boneage/boneage-training-dataset/` |
| `base_image_dir` (val) | `/path/to/rsna-boneage/Bone Age Validation Set/` |
| `csv_path` (train, `train.csv`) | `/path/to/rsna-boneage/train.csv` |
| `csv_path` (val, `Validation Dataset.csv`) | `/path/to/rsna-boneage/Bone Age Validation Set/Validation Dataset.csv` |
