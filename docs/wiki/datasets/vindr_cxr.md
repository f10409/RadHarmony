# VinDr-CXR

**Modality:** CXR | **Format:** DICOM | **Dim:** 2D | **Labels:** 28 findings

## Overview

VinDr-CXR is a large-scale chest X-ray dataset from Vietnam containing 18,000 CXR scans annotated by a panel of experienced radiologists. It provides image-level classification labels and bounding box annotations for 28 critical findings. The dataset is split into train (15,000) and test (3,000) sets with separate annotation files.

## Download

Available from [PhysioNet: VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/). Requires CITI training and signed DUA.

Expected layout:

```
vindr-cxr/1.0.0/
  train/
    image_labels_train.csv
    annotations_train.csv
    <image_id>.dicom
  test/
    image_labels_test.csv
    annotations_test.csv
    <image_id>.dicom
```

## Label columns

28 findings (same for train and test):
`aortic_enlargement`, `atelectasis`, `calcification`, `cardiomegaly`, `clavicle_fracture`, `consolidation`, `copd`, `edema`, `emphysema`, `enlarged_pa`, `ild`, `infiltration`, `lung_cavity`, `lung_cyst`, `lung_opacity`, `lung_tumor`, `mediastinal_shift`, `no_finding`, `nodule/mass`, `other_diseases`, `other_lesion`, `pleural_effusion`, `pleural_thickening`, `pneumonia`, `pneumothorax`, `pulmonary_fibrosis`, `rib_fracture`, `tuberculosis`

## Constructor arguments

Both `VinDrCXRTrainDataset` and `VinDrCXRTestDataset` share the same signature:

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | — | Split-specific dir — `train/` or `test/` — flat directory of `<image_id>.dicom` files (e.g. `/data/VinDr-CXR/vindr-cxr/1.0.0/train/`) |
| `csv_path` | `str` | No | auto | Image-level labels CSV — `image_labels_train.csv` (train) / `image_labels_test.csv` (test); auto-discovered under `base_image_dir`, then sibling dirs |
| `bbox_csv_path` | `str` | No | auto | Bounding box annotations CSV; auto-discovered |
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
from radharmony.dataset import VinDrCXRTrainDataset

ds = VinDrCXRTrainDataset(
    base_image_dir="/data/VinDr-CXR/vindr-cxr/1.0.0/train/",
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

### Test split

```python
from radharmony.dataset import VinDrCXRTestDataset

ds = VinDrCXRTestDataset(
    base_image_dir="/data/VinDr-CXR/vindr-cxr/1.0.0/test/",
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
)
```

## Harmonizer: instantiate and inspect

```python
from radharmony.harmonizer import VinDrCXRTrainHarmonizer

h = VinDrCXRTrainHarmonizer(
    csv_path="/data/VinDr-CXR/vindr-cxr/1.0.0/train/image_labels_train.csv",
    base_image_dir="/data/VinDr-CXR/vindr-cxr/1.0.0/train/",
    bbox_csv_path="/data/VinDr-CXR/vindr-cxr/1.0.0/train/annotations_train.csv",
)
df = h.harmonize()
print(df.columns.tolist())
df.to_csv("vindr_cxr_train_harmonized.csv", index=False)
```

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import VinDrCXRTrainDataset

ds = VinDrCXRTrainDataset(
    base_image_dir="/data/VinDr-CXR/vindr-cxr/1.0.0/train/",
    harmonized_df=pd.read_csv("vindr_cxr_train_harmonized.csv"),
    output_cls=True,
)
```

## Harmonizer notes

- Image-level CSV provides multi-label classification; annotations CSV provides per-box labels
- Boxes are in pixel space; harmonizer normalises to fractional coordinates
- Multiple annotators may label the same image; labels are consolidated to consensus in the harmonizer

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_cls=True` | `"cls"` | `(28,)` | Multi-label binary |
| `output_bbox=True` | `"bbox"`, `"bbox_labels"` | list | One box per finding instance |

## Example paths

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/path/to/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/train/` |
| `base_image_dir` (test) | `/path/to/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/test/` |
| `annotations_csv_path` (train) | `/path/to/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/annotations/annotations_train.csv` |
| `annotations_csv_path` (test) | `/path/to/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/annotations/annotations_test.csv` |
| image-level labels (train) | `/path/to/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/annotations/image_labels_train.csv` |
| image-level labels (test) | `/path/to/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/annotations/image_labels_test.csv` |
