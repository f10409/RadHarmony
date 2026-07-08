# SIIM-ACR Pneumothorax

**Modality:** CXR | **Format:** DICOM | **Dim:** 2D | **Labels:** pneumothorax (binary) + mask

## Overview

The SIIM-ACR Pneumothorax Segmentation dataset (2019 Kaggle challenge) contains ~12,047 training chest X-rays; the `train-rle.csv` has 12,954 RLE rows because some images carry multiple pneumothorax annotations. Each image is labelled for the presence of pneumothorax, and positive cases include one or more pixel-level segmentation masks encoded as run-length encoded (RLE) strings in the CSV.

## Download

Available on [Kaggle: SIIM-ACR Pneumothorax Segmentation](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation). Requires Kaggle account.

Expected layout:

```
SIIM_ACR_Pneumothorax/
  train-rle.csv
  dicom-images-train/                    # ← base_image_dir for the train split
    <StudyInstanceUID>/
      <SeriesInstanceUID>/
        <image_id>.dcm
  dicom-images-test/                     # ← base_image_dir for the test split
    <StudyInstanceUID>/
      <SeriesInstanceUID>/
        <image_id>.dcm
```

The harmonizer globs `**/*.dcm` under `dicom_dir` and keeps the last
three path components (`<study>/<series>/<image>.dcm`) as `image_path`.

## Label columns

| Column | Description |
|--------|-------------|
| `pneumothorax` | 1 = pneumothorax present, 0 = absent |

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | — | `dicom-images-train/` (or `dicom-images-test/` for the test split) — root of the DICOM tree (passed internally to the harmonizer as `dicom_dir`) |
| `csv_path` | `str` | No | auto | `train-rle.csv`; auto-discovered |
| `mask_output_dir` | `str` | No | `None` | Directory to save decoded PNG masks; required for `output_mask=True` |
| `mask_num_cores` | `int` | No | `1` | Parallel workers for RLE mask decoding |
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

The constructor arguments above apply to both `SIIMACRPTXTrainDataset` and `SIIMACRPTXTestDataset`. The test split has no labels — `output_cls`, `output_mask`, and `output_bbox` are silently ignored.

### Train split

```python
import torch
from radharmony.dataset import SIIMACRPTXTrainDataset

ds = SIIMACRPTXTrainDataset(
    base_image_dir="/data/SIIM_ACR_Pneumothorax/dicom-images-train/",
    mask_output_dir="/data/SIIM_ACR_Pneumothorax/masks/",
    output_cls=True,
    output_mask=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

### Test split

```python
from radharmony.dataset import SIIMACRPTXTestDataset

ds = SIIMACRPTXTestDataset(
    base_image_dir="/data/SIIM_ACR_Pneumothorax/dicom-images-test/",
    dtype=torch.float32,
)
```

## Harmonizer: instantiate and inspect

```python
from radharmony.harmonizer import SIIMACRPTXTrainHarmonizer

h = SIIMACRPTXTrainHarmonizer(
    csv_path="/data/SIIM_ACR_Pneumothorax/train-rle.csv",
    dicom_dir="/data/SIIM_ACR_Pneumothorax/dicom-images-train/",
)
df = h.harmonize()
print(df.columns.tolist())
df.to_csv("siim_ptx_harmonized.csv", index=False)
```

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import SIIMACRPTXTrainDataset

ds = SIIMACRPTXTrainDataset(
    base_image_dir="/data/SIIM_ACR_Pneumothorax/dicom-images-train/",
    harmonized_df=pd.read_csv("siim_ptx_harmonized.csv"),
    output_cls=True,
)
```

## Harmonizer notes

- RLE masks in `train-rle.csv` use the value `-1` to indicate no pneumothorax (no mask)
- When `output_mask=True`, the harmonizer decodes RLE strings to PNG files in `mask_output_dir` on first run
- Multiple RLE strings per image (multiple opacities) are merged into a single binary mask

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_cls=True` | `"cls"` | `(1,)` | Binary: pneumothorax present/absent |
| `output_mask=True` | `"mask"` | `(1, H, W)` | Decoded binary segmentation mask |

## Example paths

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/path/to/SIIM_ACR_Pneumothorax/dicom-images-train/` |
| `base_image_dir` (test) | `/path/to/SIIM_ACR_Pneumothorax/dicom-images-test/` |
| `csv_path` (`train-rle.csv`) | `/path/to/SIIM_ACR_Pneumothorax/train-rle.csv` |
| pre-decoded mask dir | `/path/to/SIIM_ACR_Pneumothorax/masks` |
