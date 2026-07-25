# MIMIC-CXR (DICOM)

**Modality:** CXR | **Format:** DICOM | **Dim:** 2D | **Labels:** 14 pathologies + reports

## Overview

MIMIC-CXR is a large publicly available dataset of chest radiographs in DICOM format from the Beth Israel Deaconess Medical Center, containing 227,835 imaging studies for 65,379 patients (377,110 images total). It includes free-text radiology reports and CheXpert-extracted labels. Access requires credentialing on PhysioNet.

## Download

Available at [PhysioNet: MIMIC-CXR](https://physionet.org/content/mimic-cxr/). Requires CITI training and signed DUA.

Expected layout:

```
mimic-cxr/2.1.0/
  cxr-record-list.csv.gz
  cxr-study-list.csv.gz
  files/
    p10/
      p10000032/
        s50414267/
          02aa804e-bde0afdd-...dcm
```

## Label columns

Same 14 labels as CheXpert (NLP-extracted from reports):
`atelectasis`, `cardiomegaly`, `consolidation`, `edema`, `enlarged_cardiomediastinum`, `fracture`, `lung_lesion`, `lung_opacity`, `no_finding`, `pleural_effusion`, `pleural_other`, `pneumonia`, `pneumothorax`, `support_devices`

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | `None` | `files/` subtree root (e.g. `/data/mimic-cxr/2.1.0/files/`) — direct parent of the `p10/`, `p11/`, … patient prefix dirs |
| `dicom_base_dir` | `str` | No | same as `base_image_dir` | Explicit DICOM root if different |
| `csv_path` | `str` | No | auto | `cxr-record-list.csv.gz`; auto-discovered |
| `label_csv_path` | `str` | No | auto | CheXpert labels CSV (from MIMIC-CXR-JPG) |
| `report_csv_path` | `str` | No | auto | `cxr-study-list.csv.gz` for report text |
| `output_cls` | `bool` | No | `False` | Include `"cls"` in data dict |
| `output_mask` | `bool` | No | `False` | Include `"mask"` in data dict |
| `output_report` | `bool` | No | `False` | Include `"report"` in data dict |
| `output_bbox` | `bool` | No | `False` | Include `"bbox"` and `"bbox_labels"` in data dict |
| `drop_uncertain` | `bool` | No | `True` | Drop rows with uncertain labels |
| `cache_dir` | `str` | No | `"./cache"` | MONAI cache directory |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonized CSV |
| `transform` | MONAI transform | No | `None` | MONAI Compose transform; `None` uses the default pipeline |

## Dataset constructor

```python
import torch
from radharmony.dataset import MIMICCXRDataset

ds = MIMICCXRDataset(
    base_image_dir="/data/mimic-cxr/2.1.0/files/",
    label_csv_path="/data/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv",
    output_cls=True,
    output_report=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

## Harmonizer: instantiate and inspect

```python
from radharmony.harmonizer import MIMICCXRHarmonizer

h = MIMICCXRHarmonizer(
    csv_path="/data/mimic-cxr/2.1.0/cxr-record-list.csv.gz",
    dicom_base_dir="/data/mimic-cxr/2.1.0/files/",
    label_csv_path="/data/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv",
    report_csv_path="/data/mimic-cxr/2.1.0/cxr-study-list.csv.gz",
    report_base_dir="/data/mimic-cxr/2.1.0/files/",
)
df = h.harmonize()
print(df.columns.tolist())
df.to_csv("mimic_cxr_harmonized.csv", index=False)
```

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import MIMICCXRDataset

ds = MIMICCXRDataset(
    base_image_dir="/data/mimic-cxr/2.1.0/files/",
    harmonized_df=pd.read_csv("mimic_cxr_harmonized.csv"),
    output_cls=True,
)
```

## Harmonizer notes

- Primary CSV is `cxr-record-list.csv.gz` (image–study mapping)
- Labels come from a separate CheXpert label CSV (available in MIMIC-CXR-JPG)
- Reports come from `cxr-study-list.csv.gz`
- Uncertain labels handled as in CheXpert

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_cls=True` | `"cls"` | `(14,)` | CheXpert-extracted labels |
| `output_report=True` | `"report"` | `str` | Free-text radiology report |

## Example paths

| Role | Path |
|---|---|
| `base_image_dir` | `/path/to/MIMIC-CXR-V2-AWS/files/` |
| `csv_path` (`mimic-cxr-2.0.0-metadata.csv`) | `/path/to/MIMIC_CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv` |
| `label_csv_path` (`mimic-cxr-2.0.0-chexpert.csv`) | `/path/to/MIMIC_CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv` |
| `report_csv_path` (`cxr-study-list.csv.gz`) | `/path/to/MIMIC-CXR-V2-AWS/cxr-study-list.csv.gz` |

Metadata and chexpert CSVs live in the JPG release tree, not in the DICOM tree. Pass them explicitly — auto-discovery may not find them when the two releases are stored under different mounts.
