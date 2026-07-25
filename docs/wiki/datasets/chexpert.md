# CheXpert

**Modality:** CXR | **Format:** JPEG | **Dim:** 2D | **Labels:** 14 pathologies

## Overview

CheXpert is a large chest X-ray dataset from Stanford Medicine containing 224,316 chest radiographs from 65,240 patients. Labels for 14 radiological observations were extracted automatically from radiology reports using an NLP labeller. Labels can be positive (1), negative (0), or uncertain (-1); RadHarmony drops uncertain labels by default (`drop_uncertain=True`).

## Download

Available from the [Stanford AIMI Shared Datasets](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2) page. Requires registration.

Expected layout after extraction:

```
CheXpert-v1.0/
  train/
    train.csv
    patient00001/
      study1/
        view1_frontal.jpg
        ...
  valid/
    valid.csv
    ...
```

## Label columns

| Column | Description |
|--------|-------------|
| `atelectasis` | Partial lung collapse |
| `cardiomegaly` | Enlarged heart |
| `consolidation` | Airspace consolidation |
| `edema` | Pulmonary edema |
| `enlarged_cardiomediastinum` | Widened mediastinum |
| `fracture` | Rib/bone fracture |
| `lung_lesion` | Lung lesion |
| `lung_opacity` | Lung opacity |
| `no_finding` | No pathology detected |
| `pleural_effusion` | Pleural effusion |
| `pleural_other` | Other pleural abnormality |
| `pneumonia` | Pneumonia |
| `pneumothorax` | Pneumothorax |
| `support_devices` | Support devices present |

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | `None` | `train/` (or `valid/`) directory — direct parent of `patient.../study.../view*.jpg` (e.g. `/data/CheXpert-v1.0/train/`). Use `CheXpertValidDataset` for the valid split |
| `csv_path` | `str` | No | auto | Path to `train.csv`; auto-discovered if omitted |
| `drop_uncertain` | `bool` | No | `True` | Drop rows with uncertain (-1) labels |

### Shared arguments (inherited from `BaseRadiologicalDataset`)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `output_cls` | `bool` | No | `False` | Include `"cls"` tensor in data dict |
| `transform` | MONAI transform | No | `None` | MONAI Compose transform; `None` uses the default pipeline |
| `cache_dir` | `str` | No | `"./cache"` | MONAI cache directory; `None` = no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonized CSV |

## Dataset constructor

The constructor arguments above apply to both `CheXpertTrainDataset` and `CheXpertValidDataset`.

### Train split

```python
import torch
from radharmony.dataset import CheXpertTrainDataset

ds = CheXpertTrainDataset(
    base_image_dir="/data/CheXpert-v1.0/train/",
    output_cls=True,
    drop_uncertain=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
sample = train_ds[0]
# sample["img"]  → Tensor(1, 224, 224)
# sample["cls"]  → Tensor(14,)
```

### Valid split

```python
from radharmony.dataset import CheXpertValidDataset

ds = CheXpertValidDataset(
    base_image_dir="/data/CheXpert-v1.0/valid/",
    output_cls=True,
    dtype=torch.float32,
)
```

## Harmonizer: instantiate and inspect

```python
from radharmony.harmonizer import CheXpertTrainHarmonizer

h = CheXpertTrainHarmonizer(
    csv_path="/data/CheXpert-v1.0/train/train.csv",
    base_image_dir="/data/CheXpert-v1.0/train/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'atelectasis', ..., 'view_position']
df.to_csv("chexpert_harmonized.csv", index=False)
```

`CheXpertValidHarmonizer` has the same interface; point it at the `valid/` directory and `valid.csv`.

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import CheXpertTrainDataset

ds = CheXpertTrainDataset(
    base_image_dir="/data/CheXpert-v1.0/train/",
    harmonized_df=pd.read_csv("chexpert_harmonized.csv"),
    output_cls=True,
)
```

## Harmonizer notes

- Reads `train.csv` (or `valid.csv`); the CSV contains one row per image with one column per label
- Uncertain labels (-1) are either dropped (default) or kept depending on `drop_uncertain`
- `image_path` is a relative path under `base_image_dir`
- `view_position` is read from the `Frontal/Lateral` column

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_cls=True` | `"cls"` | `(14,)` | Binary labels; uncertain rows dropped by default |

## Example paths

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/path/to/CheXpert-v1.0/train/` |
| `base_image_dir` (valid) | `/path/to/CheXpert-v1.0/valid/` |
| `csv_path` (`train.csv`) | `/path/to/CheXpert-v1.0/train.csv` |
| `csv_path` (`valid.csv`) | `/path/to/CheXpert-v1.0/valid.csv` |
