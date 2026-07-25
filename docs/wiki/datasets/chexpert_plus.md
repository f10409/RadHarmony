# CheXpert-Plus

**Modality:** CXR | **Format:** DICOM | **Dim:** 2D | **Labels:** 14 pathologies + reports

## Overview

CheXpert-Plus is an extension of the original CheXpert dataset that pairs de-identified radiology reports (and patient demographics) with 223,228 chest X-rays across 187,711 studies from 64,725 patients. The images are pixel-identical to CheXpert v1.0 — 186 studies from the original release are absent because their reports could not be recovered — and the release additionally ships DICOM and PNG image formats.

## Download

Available from the [Stanford AIMI Shared Datasets](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1) page. Requires registration.

The metadata CSV's `path_to_dcm` column stores paths like
`train/patient42142/study5/view1_frontal.dcm`, so `base_image_dir` must be
the directory that has `train/` and `valid/` as immediate subdirectories
— i.e. `DICOM/Uncompressed/` for the official DICOM release.

## Expected layout

```
chexpertplus/
  df_chexpert_plus_240401.csv
  findings_fixed.json
  impression_fixed.json
  report_fixed.json                 # de-identified reports (optional)
  DICOM/
    Uncompressed/                   # ← base_image_dir points here
      train/
        patient00001/study1/view1_frontal.dcm
        ...
      valid/
        ...
    Compressed/
      ...
  PNG/
    png_chexpert_plus_chunk_*.zip   # PNG ships as zipped chunks
```

## Label columns

Same 14 labels as CheXpert:
`atelectasis`, `cardiomegaly`, `consolidation`, `edema`, `enlarged_cardiomediastinum`, `fracture`, `lung_lesion`, `lung_opacity`, `no_finding`, `pleural_effusion`, `pleural_other`, `pneumonia`, `pneumothorax`, `support_devices`

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | `None` | Direct parent of `train/` and `valid/` (the `path_to_dcm` column starts with `train/...` or `valid/...`). For the canonical DICOM release: `/data/chexpertplus/DICOM/Uncompressed/` |
| `csv_path` | `str` | No | auto | `df_chexpert_plus_240401.csv`; auto-discovered |
| `label_json_path` | `str` | No | auto | `report_fixed.json`; auto-discovered near `base_image_dir` |
| `drop_uncertain` | `bool` | No | `True` | Drop rows with uncertain labels |

### Shared arguments (inherited from `BaseRadiologicalDataset`)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `output_cls` | `bool` | No | `False` | Include `"cls"` in data dict |
| `output_mask` | `bool` | No | `False` | Not supported; ignored |
| `output_report` | `bool` | No | `False` | Include `"report"` (text) in data dict |
| `output_bbox` | `bool` | No | `False` | Not supported; ignored |
| `transform` | MONAI transform | No | `None` | MONAI Compose transform; `None` uses the default pipeline |
| `cache_dir` | `str` | No | `"./cache"` | MONAI cache directory |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonized CSV |

## Dataset constructor

```python
import torch
from radharmony.dataset import CheXpertPlusDataset

ds = CheXpertPlusDataset(
    base_image_dir="/data/chexpertplus/DICOM/Uncompressed/",   # contains train/ and valid/
    csv_path="/data/chexpertplus/df_chexpert_plus_240401.csv",
    label_json_path="/data/chexpertplus/report_fixed.json",
    output_cls=True,
    output_report=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

## Harmonizer

```python
from radharmony.harmonizer import CheXpertPlusHarmonizer

h = CheXpertPlusHarmonizer(
    csv_path="/data/chexpertplus/df_chexpert_plus_240401.csv",
    base_image_dir="/data/chexpertplus/DICOM/Uncompressed/",
    label_json_path="/data/chexpertplus/report_fixed.json",
)
df = h.harmonize()
print(df.columns.tolist())
df.to_csv("chexpert_plus_harmonized.csv", index=False)
```

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import CheXpertPlusDataset

ds = CheXpertPlusDataset(
    base_image_dir="/data/chexpertplus/DICOM/Uncompressed/",
    harmonized_df=pd.read_csv("chexpert_plus_harmonized.csv"),
    output_cls=True,
    output_report=True,
)
```

## Harmonizer notes

- Reads the main CSV and optionally `report_fixed.json` for de-identified report text
- Uncertain labels are handled the same way as CheXpert (`drop_uncertain`)
- Bounding boxes and masks (when present) are normalised to fractional coordinates

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_cls=True` | `"cls"` | `(14,)` | Binary labels |
| `output_report=True` | `"report"` | `str` | De-identified report text |

## Example paths

| Role | Path |
|---|---|
| `base_image_dir` | `/path/to/CheXpert_Plus/chexpertplus/DICOM/Uncompressed` |
| `csv_path` (`df_chexpert_plus_240401.csv`) | `/path/to/CheXpert_Plus/chexpertplus/df_chexpert_plus_240401.csv` |
| `label_json_path` (`report_fixed.json`) | `/path/to/CheXpert_Plus/chexpertplus/report_fixed.json` |

`impression_fixed.json` and `findings_fixed.json` also exist alongside `report_fixed.json`. CSVs and JSONs live 2 levels above `base_image_dir` and are auto-discoverable via `infer_path`.
