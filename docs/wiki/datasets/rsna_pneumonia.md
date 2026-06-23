# RSNA Pneumonia Detection

**Modality:** CXR | **Format:** DICOM | **Dim:** 2D | **Labels:** 3 classes + bboxes

## Overview

The RSNA Pneumonia Detection Challenge dataset (2018) contains 26,684 chest X-rays labelled as Normal, No Lung Opacity / Not Normal, or Lung Opacity. Lung Opacity cases carry bounding box annotations. This page covers the adjudicated JSON annotation distribution; for the Kaggle CSV distribution, see [RSNA Pneumonia (Kaggle)](rsna_pneumonia_kaggle.md).

## Download

Available from the [RSNA website](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018).

Expected layout:

```
rsna/
  pneumonia-challenge-annotations-adjudicated-kaggle_2018.json
  <StudyInstanceUID>/
    <SeriesInstanceUID>/
      <SOPInstanceUID>.dcm
```

This release uses a three-level UID hierarchy (study / series / SOP),
unlike the flat Kaggle layout.

## Label columns

| Column | Description |
|--------|-------------|
| `lung_opacity` | Lung opacity (pneumonia) present |
| `no_lung_opacity_/_not_normal` | Abnormal but not opacity |
| `normal` | Normal chest X-ray |

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | — | Root directory whose immediate subdirs are `<StudyInstanceUID>/` (e.g. `~/Downloads/rsna/`) |
| `csv_path` | `str` | No | auto | Adjudicated JSON annotation file; auto-discovered |
| `label_group` | `str` | No | `"Calculated"` | Which label group to use from the JSON annotations |
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
from radharmony.dataset import RSNAPneumoniaDataset

ds = RSNAPneumoniaDataset(
    base_image_dir="/data/rsna/",
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

## Harmonizer: instantiate and inspect

```python
from radharmony.harmonizer import RSNAPneumoniaHarmonizer

h = RSNAPneumoniaHarmonizer(
    csv_path="/data/rsna/pneumonia-challenge-annotations-adjudicated-kaggle_2018.json",
    base_image_dir="/data/rsna/",
)
df = h.harmonize()
print(df.columns.tolist())
df.to_csv("rsna_pneumonia_harmonized.csv", index=False)
```

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import RSNAPneumoniaDataset

ds = RSNAPneumoniaDataset(
    base_image_dir="/data/rsna/",
    harmonized_df=pd.read_csv("rsna_pneumonia_harmonized.csv"),
    output_cls=True,
)
```

## Harmonizer notes

- Uses the adjudicated JSON annotation format (not the Kaggle CSV)
- `label_group="Calculated"` selects the adjudicated consensus labels
- Bounding boxes are in pixel space in the JSON; harmonizer normalises to fractional `[dim0_min, dim0_max, dim1_min, dim1_max]`
- Each label is one-hot encoded from the three-class annotation

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_cls=True` | `"cls"` | `(3,)` | One-hot: lung_opacity, no_lung_opacity, normal |
| `output_bbox=True` | `"bbox"`, `"bbox_labels"` | list | Lung opacity bounding boxes; empty for non-opacity cases |

## Example paths

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/path/to/rsna-pneumonia-detection-challenge/stage_2_train_images/` |
| `base_image_dir` (test) | `/path/to/rsna-pneumonia-detection-challenge/stage_2_test_images/` |
| `csv_path` (adjudicated JSON) | **NOT ON NAS** — must be downloaded separately |

The harmonizer's `csv_path` points to `pneumonia-challenge-annotations-adjudicated-kaggle_2018.json`, not a CSV. NAS only has `stage_2_train_labels.csv` (used by the Kaggle variant).
