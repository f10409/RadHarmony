# BRAX

**Modality:** CXR | **Format:** DICOM + PNG | **Dim:** 2D | **Labels:** 14 pathologies

## Overview

BRAX (Brazilian Chest X-Ray) v1.1.0 is a PhysioNet credentialed-access release containing ~40,967 chest X-ray images from 19,351 patients across a Brazilian hospital network. Labels are CheXpert-aligned (14 classes), NLP-derived from Portuguese radiology reports using the original CheXpert labeller.

Both DICOM and PNG variants ship in the same download. RadHarmony provides two dataset classes sharing one harmonizer:

- `BRAXDataset` — DICOM images under `Anonymized_DICOMs/`
- `BRAXPNGDataset` — PNG images under `images/` (faster to load for large training runs)

## Download

Available at [PhysioNet: BRAX](https://physionet.org/content/brax/). Requires credentialed access and a signed DUA.

Expected layout:

```
brax/1.1.0/
  master_spreadsheet.csv          # or master_spreadsheet_update.csv
  Anonymized_DICOMs/
    id_<PatientID>/Study_<UID>/Series_<UID>/image-<UID>.dcm
  images/
    id_<PatientID>/Study_<UID>/Series_<UID>/image-<UID>.png
```

`base_image_dir` should point at the BRAX root (the directory containing `master_spreadsheet.csv` and both image subdirectories).

## Label columns

14 CheXpert-aligned findings:
`atelectasis`, `cardiomegaly`, `consolidation`, `edema`, `enlarged_cardiomediastinum`, `fracture`, `lung_lesion`, `lung_opacity`, `no_finding`, `pleural_effusion`, `pleural_other`, `pneumonia`, `pneumothorax`, `support_devices`

Label cells use the encoding: `1` = positive, `0` = negation, `-1` = uncertain, `NaN` = not mentioned.

## Uncertain label strategies

The `uncertain_strategy` parameter controls how `-1` (uncertain) cells are handled:

| Strategy | `-1` → | `NaN` → | Rows dropped |
|---|---|---|---|
| `raw` (default) | preserved | preserved | none |
| `u_zeros` | `0` | `0` | none |
| `u_ones` | `1` | `0` | none |
| `u_ignore` | `NaN` | `0` | none |
| `drop` | `NaN` | `0` | rows with any NaN label |

Use `u_zeros` for standard training-ready 0/1 tensors. Use `raw` when you need to faithfully reproduce the source encoding.

!!! note
    Reports are **not** distributed in BRAX 1.1.0. `output_report=True` is ignored with a warning.

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | `None` | BRAX root (contains `master_spreadsheet.csv`, `Anonymized_DICOMs/`, `images/`) |
| `csv_path` | `str` | No | auto | Path to `master_spreadsheet.csv`; auto-discovered |
| `uncertain_strategy` | `str` | No | `"raw"` | One of `raw`, `u_zeros`, `u_ones`, `u_ignore`, `drop` |
| `output_cls` | `bool` | No | `False` | Include `"cls"` in data dict |
| `cache_dir` | `str` | No | `"./cache"` | MONAI cache directory |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonized CSV |
| `output_bbox` | `bool` | No | `False` | Include `"bbox"` and `"bbox_labels"` in data dict |
| `output_mask` | `bool` | No | `False` | Include `"mask"` in data dict |
| `output_report` | `bool` | No | `False` | Include `"report"` in data dict |
| `transform` | MONAI transform | No | `None` | MONAI Compose transform; `None` uses the default pipeline |

## Dataset constructor

### DICOM variant

```python
import torch
from radharmony.dataset import BRAXDataset

ds = BRAXDataset(
    base_image_dir="/data/physionet.org/files/brax/1.1.0/",
    uncertain_strategy="u_zeros",
    output_cls=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=10)
```

### PNG variant

```python
from radharmony.dataset import BRAXPNGDataset

ds = BRAXPNGDataset(
    base_image_dir="/data/physionet.org/files/brax/1.1.0/",
    uncertain_strategy="u_zeros",
    output_cls=True,
)
train_ds, val_ds = ds.get_datasets(n_splits=10)
```

## Harmonizer: instantiate and inspect

```python
from radharmony.harmonizer import BRAXHarmonizer

h = BRAXHarmonizer(
    csv_path="/data/physionet.org/files/brax/1.1.0/master_spreadsheet.csv",
    base_image_dir="/data/physionet.org/files/brax/1.1.0/",
    image_format="dicom",       # or "png"
    uncertain_strategy="u_zeros",
)
df = h.harmonize()
print(df.columns.tolist())
df.to_csv("brax_harmonized.csv", index=False)
```

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import BRAXDataset

ds = BRAXDataset(
    base_image_dir="/data/physionet.org/files/brax/1.1.0/",
    harmonized_df=pd.read_csv("brax_harmonized.csv"),
    output_cls=True,
)
```

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_cls=True` | `"cls"` | `(14,)` | CheXpert-aligned labels |

## Harmonizer notes

- CSV auto-discovery checks for `master_spreadsheet.csv` then `master_spreadsheet_update.csv`
- `image_path` in the harmonized DataFrame is relative to `base_image_dir` and includes the `Anonymized_DICOMs/` or `images/` prefix as written in the source CSV
- `series_id` is extracted from the `Series_<UID>` segment of the image path via regex
- Extra demographic columns (`patient_sex`, `patient_age`, `manufacturer`, `study_date`) flow through as metadata; `patient_age` is binned into 5-year groups in the de-identified release (treat as ordinal, not continuous)
- Both frontal (AP/PA) and lateral views are retained; filter via the harmonized DataFrame if frontal-only is needed
