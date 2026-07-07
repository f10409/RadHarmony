# TAIX-Ray

**Modality:** CXR | **Format:** PNG | **Dim:** 2D | **Labels:** 8 findings (binary or ordinal)

## Overview

TAIX-Ray is a Thai chest X-ray dataset with annotations for 8 clinically relevant findings. It supports two label modes: **binary** (finding present/absent) and **ordinal** (severity grade). The dataset is distributed in two resolutions: 512×512 pixels and original resolution. RadHarmony provides a separate dataset class for each resolution variant.

## Download

Available from [HuggingFace: TLAIM/TAIX-Ray](https://huggingface.co/datasets/TLAIM/TAIX-Ray). Images are expected in a flat `images/` directory with an `annotation.csv`.

Expected layout:

```
TAIX-Ray/
  data_512/
    images/
      annotation.csv
      <image_id>.png
  data_original/
    images/
      annotation.csv
      <image_id>.png
```

## Label columns

| Column | Description |
|--------|-------------|
| `atelectasis_left` | Left atelectasis |
| `atelectasis_right` | Right atelectasis |
| `heart_size` | Heart size abnormality |
| `pleural_effusion_left` | Left pleural effusion |
| `pleural_effusion_right` | Right pleural effusion |
| `pulmonary_congestion` | Pulmonary congestion |
| `pulmonary_opacities_left` | Left pulmonary opacities |
| `pulmonary_opacities_right` | Right pulmonary opacities |

In **binary** mode: each column is 0 or 1.
In **ordinal** mode: each column holds a severity grade (0 = absent, higher = more severe).

## Constructor arguments

Both `TAIXRay512Dataset` (512px) and `TAIXRayDataset` (original resolution) share the same signature:

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | — | `images/` directory — flat dir of `<UID>.png` files (e.g. `/data/TAIX-Ray/data_512/images/` for the 512-px variant, or `data_original/images/` for the original-resolution variant) |
| `csv_path` | `str` | No | auto | `annotation.csv`; auto-discovered |
| `label_mode` | `str` | No | `"binary"` | `"binary"` or `"ordinal"` |
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

### 512px variant

```python
from radharmony.dataset import TAIXRay512Dataset

ds = TAIXRay512Dataset(
    base_image_dir="/data/TAIX-Ray/data_512/images/",
    label_mode="binary",
    output_cls=True,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

### Original resolution variant

```python
from radharmony.dataset import TAIXRayDataset

ds = TAIXRayDataset(
    base_image_dir="/data/TAIX-Ray/data_original/images/",
    label_mode="ordinal",
    output_cls=True,
)
```

## Harmonizer: instantiate and inspect

```python
from radharmony.harmonizer import TAIXRayHarmonizer

h = TAIXRayHarmonizer(
    csv_path="/data/TAIX-Ray/data_512/images/annotation.csv",
    base_image_dir="/data/TAIX-Ray/data_512/images/",
)
df = h.harmonize()
print(df.columns.tolist())
df.to_csv("taix_ray_harmonized.csv", index=False)
```

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import TAIXRay512Dataset

ds = TAIXRay512Dataset(
    base_image_dir="/data/TAIX-Ray/data_512/images/",
    harmonized_df=pd.read_csv("taix_ray_harmonized.csv"),
    output_cls=True,
)
```

## Harmonizer notes

- `label_mode="binary"` binarises ordinal grades (0 = absent, >0 = present)
- `label_mode="ordinal"` preserves the original severity grades
- Both `TAIXRay512Dataset` and `TAIXRayDataset` use the same harmonizer and annotation format; they differ only in expected image directory

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_cls=True` | `"cls"` | `(8,)` | Binary or ordinal labels depending on `label_mode` |

## Example paths

| Role | Path |
|---|---|
| `base_image_dir` (512 px) | `/path/to/TAIX-Ray/data_512/images/` |
| `base_image_dir` (original) | `/path/to/TAIX-Ray/data_original/` |
| `csv_path` (512 px, `annotation.csv`) | `/path/to/TAIX-Ray/data_512/annotation.csv` |
| `csv_path` (original, `annotation.csv`) | `/path/to/TAIX-Ray/data_original/annotation.csv` |
