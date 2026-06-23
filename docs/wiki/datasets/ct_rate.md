# CT-RATE

**Modality:** CT | **Format:** NIfTI | **Dim:** 3D | **Labels:** 18 findings

## Overview

CT-RATE is a large-scale chest CT dataset containing 50,188 non-contrast chest CT volumes from 21,304 unique patients, paired with radiology reports and NLP-extracted labels for 18 pathological findings. It is one of the largest publicly available chest CT datasets with structured labels.

## Download

Available on [HuggingFace: ibrahimhamamci/CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE). Requires HuggingFace account.

Expected layout:

```
CT-RATE/dataset/
  train_fixed/
    train_predicted_labels.csv
    train_metadata.csv
    <volume_id>/
      <volume_id>.nii.gz
```

## Label columns

| Column | Description |
|--------|-------------|
| `arterial_wall_calcification` | Arterial wall calcification |
| `atelectasis` | Atelectasis |
| `bronchiectasis` | Bronchiectasis |
| `cardiomegaly` | Cardiomegaly |
| `consolidation` | Consolidation |
| `coronary_artery_wall_calcification` | Coronary artery calcification |
| `emphysema` | Emphysema |
| `hiatal_hernia` | Hiatal hernia |
| `interlobular_septal_thickening` | Interlobular septal thickening |
| `lung_nodule` | Lung nodule |
| `lung_opacity` | Lung opacity |
| `lymphadenopathy` | Lymphadenopathy |
| `medical_material` | Medical material / device |
| `mosaic_attenuation_pattern` | Mosaic attenuation pattern |
| `peribronchial_thickening` | Peribronchial thickening |
| `pericardial_effusion` | Pericardial effusion |
| `pleural_effusion` | Pleural effusion |
| `pulmonary_fibrotic_sequela` | Pulmonary fibrotic sequela |

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | — | `train_fixed/` (or `validation_fixed/`) — root containing the two-level volume hierarchy `<train_X>/<train_X_Y>/<volume>.nii.gz` (e.g. `/data/CT-RATE/dataset/train_fixed/`) |
| `csv_path` | `str` | No | auto | `train_predicted_labels.csv`; auto-discovered |
| `view_position_csv_path` | `str` | No | auto | `train_metadata.csv`; auto-discovered |
| `hu_window` | `tuple[float,float]\|None` | No | `(-1000, 1000)` | HU clip range |
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

```python
import torch
from radharmony.dataset import CTRATEDataset

ds = CTRATEDataset(
    base_image_dir="/data/CT-RATE/dataset/train_fixed/",
    output_cls=True,
    hu_window=(-1000, 400),   # lung window
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

## Harmonizer: instantiate and inspect

```python
from radharmony.harmonizer import CTRATEHarmonizer

h = CTRATEHarmonizer(
    csv_path="/data/CT-RATE/dataset/train_fixed/train_predicted_labels.csv",
    base_image_dir="/data/CT-RATE/dataset/train_fixed/",
    view_position_csv_path="/data/CT-RATE/dataset/train_fixed/train_metadata.csv",
)
df = h.harmonize()
print(df.columns.tolist())
df.to_csv("ct_rate_harmonized.csv", index=False)
```

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import CTRATEDataset

ds = CTRATEDataset(
    base_image_dir="/data/CT-RATE/dataset/train_fixed/",
    harmonized_df=pd.read_csv("ct_rate_harmonized.csv"),
    output_cls=True,
)
```

## Harmonizer notes

- Labels are NLP-extracted from radiology reports; binary (1 = present, 0 = absent)
- Metadata CSV provides view position and patient demographics
- Images are NIfTI (`.nii.gz`); MONAI loads them natively

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_cls=True` | `"cls"` | `(18,)` | Multi-label binary findings |

## Example paths

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/path/to/CT-RATE/dataset/train_fixed/` |
| `base_image_dir` (valid) | `/path/to/CT-RATE/dataset/valid_fixed/` |
| `csv_path` (train, `train_predicted_labels.csv`) | `/path/to/CT-RATE/dataset/tables/train_predicted_labels.csv` |
| `csv_path` (valid, `valid_predicted_labels.csv`) | `/path/to/CT-RATE/dataset/tables/valid_predicted_labels.csv` |
| `view_position_csv_path` (train, `train_metadata.csv`) | `/path/to/CT-RATE/dataset/tables/train_metadata.csv` |
| `view_position_csv_path` (valid, `validation_metadata.csv`) | `/path/to/CT-RATE/dataset/tables/validation_metadata.csv` |

CSVs live in `tables/` (sibling of `train_fixed/` and `valid_fixed/`), auto-discoverable via `infer_path`.
