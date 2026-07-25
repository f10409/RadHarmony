# MIMIC-CXR-JPG

**Modality:** CXR | **Format:** JPEG | **Dim:** 2D | **Labels:** 14 pathologies

## Overview

MIMIC-CXR-JPG is the JPEG version of MIMIC-CXR. It contains 227,827 studies (a handful fewer than the DICOM release could not be converted) but stores images as compressed JPEG files, making it faster to load and more suitable for large-scale training. The CheXpert-extracted labels match MIMIC-CXR; the JPG release itself ships no free-text reports.

RadHarmony provides two dataset classes for this dataset:
- `MIMICCXRJPGDataset` — all studies, labelled by the standard CheXpert NLP labeller (`mimic-cxr-2.0.0-chexpert.csv`).
- `MIMICCXRJPGTestDataset` — strict 687-study labeled test subset, using the independently-relabeled `mimic-cxr-2.1.0-test-set-labeled.csv`. Useful for fair evaluation since these labels were not derived from the same NLP pipeline used to label the training set.

## Download

Available at [PhysioNet: MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/). Requires CITI training and signed DUA.

## Expected layout

```
mimic-cxr-jpg/2.0.0/
  mimic-cxr-2.0.0-metadata.csv.gz
  mimic-cxr-2.0.0-chexpert.csv             # full-set labels (all studies)
  mimic-cxr-2.1.0-test-set-labeled.csv     # 687-study test set (optional)
  files/
    p10/
      p10000032/
        s50414267/
          02aa804e-bde0afdd-....jpg
```

## Label columns

Same 14 labels as CheXpert:
`atelectasis`, `cardiomegaly`, `consolidation`, `edema`, `enlarged_cardiomediastinum`, `fracture`, `lung_lesion`, `lung_opacity`, `no_finding`, `pleural_effusion`, `pleural_other`, `pneumonia`, `pneumothorax`, `support_devices`

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | `None` | `files/` subtree root (e.g. `/data/mimic-cxr-jpg/2.0.0/files/`) — direct parent of the `p10/`, `p11/`, … patient prefix dirs |
| `csv_path` | `str` | No | auto | `mimic-cxr-2.0.0-metadata.csv.gz`; auto-discovered |
| `label_csv_path` | `str` | No | auto | `mimic-cxr-2.0.0-chexpert.csv` |
| `drop_uncertain` | `bool` | No | `True` | Drop rows with uncertain labels |

### Shared arguments (inherited from `BaseRadiologicalDataset`)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `output_cls` | `bool` | No | `False` | Include `"cls"` in data dict |
| `output_mask` | `bool` | No | `False` | Not supported — warns "no effect" if `True` |
| `output_report` | `bool` | No | `False` | Not supported (JPG dataset has no report text) — warns "no effect" if `True`. Use [MIMIC-CXR (DICOM)](mimic_cxr.md) for reports |
| `output_bbox` | `bool` | No | `False` | Not supported — warns "no effect" if `True` |
| `transform` | MONAI transform | No | `None` | MONAI Compose transform; `None` uses the default pipeline |
| `cache_dir` | `str` | No | `"./cache"` | MONAI cache directory |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonized CSV |

## Dataset constructor

### All studies

```python
import torch
from radharmony.dataset import MIMICCXRJPGDataset

ds = MIMICCXRJPGDataset(
    base_image_dir="/data/mimic-cxr-jpg/2.0.0/files/",
    output_cls=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

### Labeled test subset (687 studies)

```python
from radharmony.dataset import MIMICCXRJPGTestDataset

ds_test = MIMICCXRJPGTestDataset(
    base_image_dir="/data/mimic-cxr-jpg/2.0.0/files/",
    output_cls=True,
    dtype=torch.float32,
)
# Auto-discovers mimic-cxr-2.1.0-test-set-labeled.csv near base_image_dir.
# Constructor signature is identical to MIMICCXRJPGDataset; the only
# difference is which label CSV is loaded.
```

The test variant exposes the same 14 `LABEL_COLS` as the full dataset.
The test CSV's `"Airspace Opacity"` column is renamed to `"Lung Opacity"`
during preprocessing so the harmonized DataFrame is schema-compatible
with the train variant. Joining is on `study_id` only (the test CSV has
no `subject_id`), so non-test studies are dropped via inner-merge.

**Uncertain labels (`-1`)**: unlike the train variant, `MIMICCXRJPGTestDataset`
defaults to `drop_uncertain=False` so every labeled study is retained
(test metrics should match the published numbers). `-1` values are
**preserved as-is** in the harmonized DataFrame; downstream code can
apply U-Zeros (`-1 → 0`), U-Ones (`-1 → 1`), or U-Ignore depending on
the convention being reproduced. Pass `drop_uncertain=True` explicitly
to drop any study with at least one uncertain label (~226 of 687).

## Harmonizer

```python
from radharmony.harmonizer import MIMICCXRJPGHarmonizer

h = MIMICCXRJPGHarmonizer(
    csv_path="/data/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz",
    base_image_dir="/data/mimic-cxr-jpg/2.0.0/files/",
    label_csv_path="/data/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv",
)
df = h.harmonize()
print(df.columns.tolist())
df.to_csv("mimic_cxr_jpg_harmonized.csv", index=False)
```

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import MIMICCXRJPGDataset

ds = MIMICCXRJPGDataset(
    base_image_dir="/data/mimic-cxr-jpg/2.0.0/files/",
    harmonized_df=pd.read_csv("mimic_cxr_jpg_harmonized.csv"),
    output_cls=True,
)
```

## Harmonizer notes

- Primary CSV is the metadata file (`mimic-cxr-2.0.0-metadata.csv.gz`)
- Labels are merged from `mimic-cxr-2.0.0-chexpert.csv` via study ID
- Faster to load than MIMIC-CXR DICOM; preferred for large-scale training

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_cls=True` | `"cls"` | `(14,)` | CheXpert-extracted labels |

## Example paths

| Role | Path |
|---|---|
| `base_image_dir` | `/path/to/MIMIC_CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/files/` |
| `csv_path` (`mimic-cxr-2.0.0-metadata.csv`) | `/path/to/MIMIC_CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv` |
| `label_csv_path` (`mimic-cxr-2.0.0-chexpert.csv`) | `/path/to/MIMIC_CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv` |
| `split_csv_path` (`mimic-cxr-2.0.0-split.csv`) | `/path/to/MIMIC_CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv` |

All CSVs live in the parent dir of `files/` and are auto-discoverable from `base_image_dir`.
