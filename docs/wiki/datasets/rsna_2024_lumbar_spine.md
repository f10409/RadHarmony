# RSNA 2024 Lumbar Spine Degenerative Classification

**Modality:** MRI | **Format:** DICOM (per-series volumes) | **Dim:** 3D | **Labels:** 75 severity columns

## Overview

The RSNA 2024 Lumbar Spine Degenerative Classification challenge dataset contains MRI studies of the lumbar spine labelled for five degenerative conditions at five lumbar levels (L1/L2 through L5/S1), giving 25 condition–level combinations. Each is labelled Normal/Mild, Moderate, or Severe — encoded as 75 one-hot severity columns (3 per condition-level pair). Three MRI series types are available per study: Axial T2, Sagittal T1, and Sagittal T2/STIR.

## Download

Available on [Kaggle: RSNA 2024 Lumbar Spine](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification). Requires Kaggle account.

Expected layout:

```
rsna-2024-lumbar-spine-degenerative-classification/
  train.csv
  train_series_descriptions.csv
  train_label_coordinates.csv     # point annotations (optional)
  train_images/
    <study_id>/
      <series_id>/
        <instance_number>.dcm
```

## Label columns

75 columns in the format `<condition>_<level>_<severity>`:

Conditions: `spinal_canal_stenosis`, `left_neural_foraminal_narrowing`, `right_neural_foraminal_narrowing`, `left_subarticular_stenosis`, `right_subarticular_stenosis`

Levels: `l1_l2`, `l2_l3`, `l3_l4`, `l4_l5`, `l5_s1`

Severities: `normal_mild`, `moderate`, `severe`

Example: `spinal_canal_stenosis_l1_l2_normal_mild`, `spinal_canal_stenosis_l1_l2_moderate`, `spinal_canal_stenosis_l1_l2_severe`

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | — | `train_images/` (or `test_images/` for the test split) — direct parent of `<study_id>/<series_id>/` (e.g. `~/datasets/competitions/rsna-2024-lumbar-spine-degenerative-classification/train_images/`) |
| `csv_path` | `str` | No | auto | `train.csv`; auto-discovered |
| `series_description_csv_path` | `str` | No | auto | `train_series_descriptions.csv` |
| `coord_csv_path` | `str` | No | auto | `train_label_coordinates.csv` for point bbox annotations |
| `series_filter` | `str\|None` | No | `None` | Load only one series type: `"Axial T2"`, `"Sagittal T1"`, or `"Sagittal T2/STIR"` |
| `output_cls` | `bool` | No | `False` | Include `"cls"` tensor in data dict |
| `output_mask` | `bool` | No | `False` | Include `"mask"` in data dict |
| `output_report` | `bool` | No | `False` | Include `"report"` in data dict |
| `output_bbox` | `bool` | No | `False` | Include `"bbox"` and `"bbox_labels"` in data dict |
| `cache_dir` | `str` | No | `"./cache"` | MONAI cache directory |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonized CSV |

## Dataset constructor

The constructor arguments above apply to both `RSNA2024LumbarSpineTrainDataset` and `RSNA2024LumbarSpineTestDataset`. The test split has no labels — `output_cls` and `output_bbox` are silently ignored, and only `series_description_csv_path` is needed (no `train.csv` or coords CSV).

### Train split

```python
import torch
from radharmony.dataset import RSNA2024LumbarSpineTrainDataset

# Load Sagittal T2/STIR series only
ds = RSNA2024LumbarSpineTrainDataset(
    base_image_dir="/data/rsna-2024-lumbar-spine/train_images/",
    series_filter="Sagittal T2/STIR",
    output_cls=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

### Test split

```python
from radharmony.dataset import RSNA2024LumbarSpineTestDataset

ds = RSNA2024LumbarSpineTestDataset(
    base_image_dir="/data/rsna-2024-lumbar-spine/test_images/",
    series_filter="Sagittal T2/STIR",
    dtype=torch.float32,
)
```

## Harmonizer: instantiate and inspect

```python
from radharmony.harmonizer import RSNA2024LumbarSpineTrainHarmonizer

h = RSNA2024LumbarSpineTrainHarmonizer(
    csv_path="/data/rsna-2024-lumbar-spine/train.csv",
    base_image_dir="/data/rsna-2024-lumbar-spine/train_images/",
    series_description_csv_path="/data/rsna-2024-lumbar-spine/train_series_descriptions.csv",
)
df = h.harmonize()
print(df.columns.tolist())
df.to_csv("rsna_lumbar_harmonized.csv", index=False)
```

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import RSNA2024LumbarSpineTrainDataset

ds = RSNA2024LumbarSpineTrainDataset(
    base_image_dir="/data/rsna-2024-lumbar-spine/train_images/",
    harmonized_df=pd.read_csv("rsna_lumbar_harmonized.csv"),
    output_cls=True,
)
```

## Harmonizer notes

- `image_path` points to a series directory; ITKReader loads it as a 3D volume
- Labels are one-hot encoded across three severity levels per condition-level pair
- `series_filter` is applied at load time to select a subset of series types
- Point annotations from `train_label_coordinates.csv` are represented as small dot bounding boxes when `output_bbox=True`
- Each study may contribute up to 3 rows (one per series type) unless filtered

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_cls=True` | `"cls"` | `(75,)` | One-hot severity across 25 condition-level pairs |
| `output_bbox=True` | `"bbox"`, `"bbox_labels"` | list | Point annotations as dot boxes |

## Example paths

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/path/to/rsna-2024-lumbar-spine-degenerative-classification/train_images/` |
| `base_image_dir` (test) | `/path/to/rsna-2024-lumbar-spine-degenerative-classification/test_images/` |
| `csv_path` (`train.csv`) | `/path/to/rsna-2024-lumbar-spine-degenerative-classification/train.csv` |
| `series_desc_csv_path` (`train_series_descriptions.csv`) | `/path/to/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv` |
| `label_coord_csv_path` (`train_label_coordinates.csv`) | `/path/to/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv` |
