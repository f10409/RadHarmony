# RANZCR CLiP

**Modality:** CXR | **Format:** JPEG | **Dim:** 2D | **Labels:** 11 catheter/line position labels

## Overview

RANZCR CLiP (Royal Australian and New Zealand College of Radiologists Catheter and Line in Position) is a Kaggle competition dataset containing ~30,000 frontal chest X-rays annotated for the presence and placement of central venous catheters (CVC), endotracheal tubes (ETT), nasogastric tubes (NGT), and Swan-Ganz catheters. The primary task is multi-label binary classification.

A subset of training images also includes polyline annotations tracing each catheter/line, which RadHarmony can rasterize into binary masks.

## Download

Available via Kaggle: [ranzcr-clip-catheter-line-classification](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/data). Accept the competition rules before downloading.

```python
import kagglehub
path = kagglehub.competition_download("ranzcr-clip-catheter-line-classification")
```

Expected layout:

```
ranzcr-clip-catheter-line-classification/
  train.csv
  train_annotations.csv
  sample_submission.csv
  train/
    <StudyInstanceUID>.jpg    # ~30k frontal CXRs with labels
  test/
    <StudyInstanceUID>.jpg    # competition holdout, no public labels
```

## Label columns

11 binary multi-label targets:

| Label | Description |
|---|---|
| `cvc_abnormal` | CVC in abnormal position |
| `cvc_borderline` | CVC in borderline position |
| `cvc_normal` | CVC in normal position |
| `ett_abnormal` | ETT in abnormal position |
| `ett_borderline` | ETT in borderline position |
| `ett_normal` | ETT in normal position |
| `ngt_abnormal` | NGT in abnormal position |
| `ngt_borderline` | NGT in borderline position |
| `ngt_incompletely_imaged` | NGT incompletely imaged |
| `ngt_normal` | NGT in normal position |
| `swan_ganz_catheter_present` | Swan-Ganz catheter present |

Test split rows have `NaN` for all label columns (no public labels released).

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | `None` | Kaggle download root (contains `train/`, `test/`, and CSVs) |
| `csv_path` | `str` | No | auto | Path to `train.csv`; auto-discovered |
| `annotations_csv_path` | `str` | No | auto | Path to `train_annotations.csv`; required only for `output_mask=True` |
| `mask_output_dir` | `str` | No | `None` | Directory for rasterized polyline PNGs; required for `output_mask=True` |
| `mask_line_thickness` | `int` | No | `15` | Pixel width for `cv2.polylines` when rasterizing |
| `include_test_split` | `bool` | No | `True` | Include test rows (NaN labels) in harmonized DataFrame |
| `mask_num_cores` | `int` | No | `1` | Worker threads for mask pre-decoding |

### Shared arguments (inherited from `BaseRadiologicalDataset`)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `output_cls` | `bool` | No | `False` | Include `"cls"` in data dict |
| `output_mask` | `bool` | No | `False` | Include rasterized polyline mask under `"mask"` |
| `output_report` | `bool` | No | `False` | Include `"report"` in data dict |
| `output_bbox` | `bool` | No | `False` | Include `"bbox"` and `"bbox_labels"` in data dict |
| `transform` | MONAI transform | No | `None` | MONAI Compose transform; `None` uses the default pipeline |
| `cache_dir` | `str` | No | `"./cache"` | MONAI cache directory |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonized CSV |

## Dataset constructor

### Classification only (cross-validation split)

```python
import torch
from radharmony.dataset import RANZCRClipDataset

ds = RANZCRClipDataset(
    base_image_dir="~/.cache/kagglehub/competitions/ranzcr-clip-catheter-line-classification/",
    output_cls=True,
    include_test_split=False,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=10)
```

### Official train / test split

```python
from radharmony.dataset import RANZCRClipDataset

ds = RANZCRClipDataset(
    base_image_dir="~/.cache/kagglehub/competitions/ranzcr-clip-catheter-line-classification/",
    output_cls=True,
    include_test_split=True,
)
train_ds, test_ds = ds.get_datasets_predefined()
# test_ds rows have NaN labels — use for inference/submission only
```

### With polyline masks

```python
from radharmony.dataset import RANZCRClipDataset

ds = RANZCRClipDataset(
    base_image_dir="~/.cache/kagglehub/competitions/ranzcr-clip-catheter-line-classification/",
    output_cls=True,
    output_mask=True,
    mask_output_dir="/tmp/ranzcr_masks/",   # rasterized PNGs written here
    mask_line_thickness=15,
)
train_ds, val_ds = ds.get_datasets(n_splits=10)
# sample["mask"] → torch.Tensor, shape (1, 224, 224)
```

!!! note
    Polyline annotations only cover a subset of training images. Images without annotations will have `NaN` for `mask_path` and no `"mask"` key in their data dict.

## Harmonizer: instantiate and inspect

```python
from radharmony.harmonizer import RANZCRClipHarmonizer

h = RANZCRClipHarmonizer(
    csv_path="~/.cache/kagglehub/competitions/ranzcr-clip-catheter-line-classification/train.csv",
    base_image_dir="~/.cache/kagglehub/competitions/ranzcr-clip-catheter-line-classification/",
    include_test_split=True,
)
df = h.harmonize()
print(df.columns.tolist())
print(df["split"].value_counts())
df.to_csv("ranzcr_clip_harmonized.csv", index=False)
```

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import RANZCRClipDataset

ds = RANZCRClipDataset(
    base_image_dir="~/.cache/kagglehub/competitions/ranzcr-clip-catheter-line-classification/",
    harmonized_df=pd.read_csv("ranzcr_clip_harmonized.csv"),
    output_cls=True,
)
```

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_cls=True` | `"cls"` | `(11,)` | Binary multi-label targets |
| `output_mask=True` | `"mask"` | `(1, 224, 224)` | Rasterized polyline mask; only for annotated images |

## Harmonizer notes

- Raw CSV column names use `"CVC - Abnormal"` style; these are renamed to clean snake_case (`cvc_abnormal`) during harmonization
- `image_path` is `train/<StudyInstanceUID>.jpg` or `test/<StudyInstanceUID>.jpg`, relative to `base_image_dir`
- Test rows are discovered by walking `<base_image_dir>/test/` for `.jpg` files; all label columns are `NaN`
- A `split` column (`"train"` / `"test"`) is always present in the harmonized DataFrame
- Polyline masks are aggregated per study and JSON-encoded in `mask_path`; `mask_output_dir` triggers rasterization to PNG before MONAI loads them
