# ChestX-ray14

**Modality:** CXR | **Format:** PNG | **Dim:** 2D | **Labels:** 14 pathologies + No Finding

## Overview

ChestX-ray14 (NIH Chest X-ray Dataset) contains 112,120 frontal-view chest X-rays from 30,805 unique patients. Labels for 14 pathological findings plus "No Finding" were extracted using NLP from radiology reports. A subset of 880 images also has pixel-level bounding box annotations (984 boxes total across 8 classes).

RadHarmony provides three dataset classes for this dataset:
- `ChestXray14TrainDataset` — official train+val split (86,524 images); classification only
- `ChestXray14TestDataset` — official test split (25,596 images); classification only
- `ChestXray14BboxDataset` — only the 880 images with annotated bounding boxes; classification + bbox

## Download

Available on [Kaggle: NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data). Requires Kaggle account.

Expected layout:

```
CXR14/
  Data_Entry_2017.csv
  BBox_List_2017.csv
  images_001/images/
    00000001_000.png
    00000001_001.png
    ...
  images_002/images/
    ...
  ...
  images_012/images/
    ...
```

Images are split across twelve `images_001/` … `images_012/` sibling
directories (the canonical NIH layout). The harmonizer scans each at load
time and constructs `image_path = "images_00X/images/<filename>"`.

## Label columns

| Column | Description |
|--------|-------------|
| `atelectasis` | Partial lung collapse |
| `cardiomegaly` | Enlarged heart |
| `consolidation` | Airspace consolidation |
| `edema` | Pulmonary edema |
| `effusion` | Pleural effusion |
| `emphysema` | Emphysema |
| `fibrosis` | Pulmonary fibrosis |
| `hernia` | Hernia |
| `infiltration` | Infiltration |
| `mass` | Pulmonary mass |
| `no_finding` | No finding |
| `nodule` | Pulmonary nodule |
| `pleural_thickening` | Pleural thickening |
| `pneumonia` | Pneumonia |
| `pneumothorax` | Pneumothorax |

## Constructor arguments

All three classes (`ChestXray14TrainDataset`, `ChestXray14TestDataset`, `ChestXray14BboxDataset`) share the same signature:

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | — | CXR14 root containing `images_001/` … `images_012/` sibling dirs (e.g. `/data/NIH_CXR/CXR14/`) |
| `csv_path` | `str` | No | auto | `Data_Entry_2017.csv`; auto-discovered |
| `bbox_csv_path` | `str` | No | auto | `BBox_List_2017.csv`; required for BBox variant |
| `output_cls` | `bool` | No | `False` | Include `"cls"` in data dict |
| `output_mask` | `bool` | No | `False` | Include `"mask"` in data dict |
| `output_report` | `bool` | No | `False` | Include `"report"` in data dict |
| `output_bbox` | `bool` | No | `False` | Include `"bbox"` and `"bbox_labels"` in data dict |
| `cache_dir` | `str` | No | `"./cache"` | MONAI cache directory |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonized CSV |

## Dataset constructor

### Train split

```python
import torch
from radharmony.dataset import ChestXray14TrainDataset

ds = ChestXray14TrainDataset(
    base_image_dir="/data/NIH_CXR/CXR14/",
    output_cls=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

### Test split

```python
from radharmony.dataset import ChestXray14TestDataset

ds = ChestXray14TestDataset(
    base_image_dir="/data/NIH_CXR/CXR14/",
    output_cls=True,
    dtype=torch.float32,
)
```

### Bounding box subset only

```python
from radharmony.dataset import ChestXray14BboxDataset

ds = ChestXray14BboxDataset(
    base_image_dir="/data/NIH_CXR/CXR14/",
    bbox_csv_path="/data/NIH_CXR/CXR14/BBox_List_2017.csv",
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
)
```

## Harmonizer: instantiate and inspect

```python
from radharmony.harmonizer import ChestXray14TrainHarmonizer

h = ChestXray14TrainHarmonizer(
    csv_path="/data/NIH_CXR/CXR14/Data_Entry_2017.csv",
    base_image_dir="/data/NIH_CXR/CXR14/",
    bbox_csv_path="/data/NIH_CXR/CXR14/BBox_List_2017.csv",
)
df = h.harmonize()
print(df.columns.tolist())
df.to_csv("chestxray14_harmonized.csv", index=False)
```

`ChestXray14TestHarmonizer` and `ChestXray14BboxHarmonizer` share the same interface.

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import ChestXray14TrainDataset

ds = ChestXray14TrainDataset(
    base_image_dir="/data/NIH_CXR/CXR14/",
    harmonized_df=pd.read_csv("chestxray14_harmonized.csv"),
    output_cls=True,
)
```

## Harmonizer notes

- Labels are multi-label (pipe-separated in the original CSV); harmonizer one-hot encodes them
- The bbox CSV contains pixel-space boxes in `[x, y, width, height]` format; harmonizer converts to fractional `[dim0_min, dim0_max, dim1_min, dim1_max]`
- `ChestXray14Dataset` with `bbox_csv_path` excludes images that have bounding boxes (useful for classification-only training)
- `ChestXray14BboxDataset` only includes the 880 images with bbox annotations

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_cls=True` | `"cls"` | `(15,)` | Multi-label binary |
| `output_bbox=True` | `"bbox"`, `"bbox_labels"` | list | BBox variant only |

## Example paths

| Role | Path |
|---|---|
| `base_image_dir` | `/path/to/NIH_CXR/CXR14/` |
| `csv_path` (`Data_Entry_2017.csv`) | `/path/to/NIH_CXR/CXR14/Data_Entry_2017.csv` |
| `bbox_csv_path` (`BBox_List_2017.csv`) | `/path/to/NIH_CXR/CXR14/BBox_List_2017.csv` |

880 unique images have bounding boxes (984 boxes total across 8 classes). `ChestXray14BboxHarmonizer` requires `base_image_dir` to read image dimensions for bbox normalization.
