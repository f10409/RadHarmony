# RSNA Pneumonia Detection (Kaggle Stage 2)

**Modality:** CXR | **Format:** DICOM | **Dim:** 2D | **Labels:** 3 classes + bboxes

## Overview

The RSNA Pneumonia Detection Challenge Kaggle Stage 2 dataset contains 26,684 chest X-rays distributed as two flat CSV files and a flat DICOM directory. Same underlying images as [RSNA Pneumonia](rsna_pneumonia.md), but uses the Kaggle CSV distribution format instead of the adjudicated JSON.

## Download

Available on [Kaggle: RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge). Requires Kaggle account.

## Expected layout

```
rsna-pneumonia-detection-challenge/
  stage_2_detailed_class_info.csv
  stage_2_train_labels.csv
  stage_2_train_images/
    <patientId>.dcm
```

## Label columns

| Column | Description |
|--------|-------------|
| `lung_opacity` | Lung opacity (pneumonia) present |
| `no_lung_opacity_/_not_normal` | Abnormal but no lung opacity |
| `normal` | Normal chest X-ray |

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | `None` | `stage_2_train_images/` (or `stage_2_test_images/` for the test split) — flat dir of `<patientId>.dcm` files |
| `csv_path` | `str` | No | auto | `stage_2_detailed_class_info.csv`; auto-discovered |
| `bbox_csv_path` | `str` | No | auto | `stage_2_train_labels.csv`; required for bounding boxes |

### Shared arguments (inherited from `BaseRadiologicalDataset`)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `output_cls` | `bool` | No | `False` | Include `"cls"` in data dict |
| `output_mask` | `bool` | No | `False` | Include `"mask"` in data dict |
| `output_report` | `bool` | No | `False` | Include `"report"` in data dict |
| `output_bbox` | `bool` | No | `False` | Include `"bbox"` and `"bbox_labels"` in data dict |
| `transform` | MONAI transform | No | `None` | MONAI Compose transform; `None` uses the default pipeline |
| `cache_dir` | `str` | No | `"./cache"` | MONAI cache directory |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonized CSV |

## Dataset constructor

```python
import torch
from radharmony.dataset import (
    RSNAPneumoniaKaggleTrainDataset,
    RSNAPneumoniaKaggleTestDataset,
)

# Train split (with class labels and bboxes)
ds_train = RSNAPneumoniaKaggleTrainDataset(
    base_image_dir="/data/rsna-pneumonia-detection-challenge/stage_2_train_images/",
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds_train.get_datasets(n_splits=5)

# Test split (images only — no labels)
ds_test = RSNAPneumoniaKaggleTestDataset(
    base_image_dir="/data/rsna-pneumonia-detection-challenge/stage_2_test_images/",
    dtype=torch.float32,
)
```

## Harmonizer

```python
from radharmony.harmonizer import RSNAPneumoniaKaggleHarmonizer

h = RSNAPneumoniaKaggleHarmonizer(
    csv_path="/data/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv",
    base_image_dir="/data/rsna-pneumonia-detection-challenge/stage_2_train_images/",
    bbox_csv_path="/data/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'lung_opacity',
#  'no_lung_opacity_/_not_normal', 'normal', 'bbox', 'bbox_labels']
df.to_csv("rsna_pneumonia_kaggle_harmonized.csv", index=False)
```

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import RSNAPneumoniaKaggleTrainDataset

ds = RSNAPneumoniaKaggleTrainDataset(
    base_image_dir="/data/rsna-pneumonia-detection-challenge/stage_2_train_images/",
    harmonized_df=pd.read_csv("rsna_pneumonia_kaggle_harmonized.csv"),
    output_cls=True,
)
```

## Harmonizer notes

- `stage_2_detailed_class_info.csv` provides class labels (`class` column: "Normal", "No Lung Opacity / Not Normal", "Lung Opacity")
- `stage_2_train_labels.csv` provides bounding boxes for `Target=1` rows; `Target=0` rows have NaN coordinates
- All images are 1024×1024 DICOM; bounding boxes are normalised by dividing by 1024
- Bboxes aggregated per patient: images with multiple opacities have a list of boxes
- Normal / No Lung Opacity images get `bbox=[]` and `bbox_labels=[]`

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_cls=True` | `"cls"` | `(3,)` | One-hot: lung_opacity, no_lung_opacity, normal |
| `output_bbox=True` | `"bbox"`, `"bbox_labels"` | list | Empty list for non-opacity cases |

## Example paths

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/path/to/rsna-pneumonia-detection-challenge/stage_2_train_images/` |
| `base_image_dir` (test) | `/path/to/rsna-pneumonia-detection-challenge/stage_2_test_images/` |
| `csv_path` (`stage_2_train_labels.csv`) | `/path/to/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv` |
