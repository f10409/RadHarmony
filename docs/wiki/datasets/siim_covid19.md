# SIIM COVID-19 Detection

**Modality:** CXR | **Format:** DICOM | **Dim:** 2D | **Labels:** 4 appearance classes + bboxes

## Overview

The SIIM-FISABIO-RSNA COVID-19 Detection dataset from the 2021 Kaggle challenge contains 6,334 chest X-rays labelled for COVID-19 lung opacity appearance. Each study has a study-level label and images may have bounding box annotations for opacities. The dataset uses two-level annotations: study-level labels and image-level bounding boxes.

## Download

Available on [Kaggle: SIIM COVID-19 Detection](https://www.kaggle.com/c/siim-covid19-detection). Requires Kaggle account.

Expected layout:

```
siim-covid19-detection/
  train/
    train_study_level.csv
    train_image_level.csv
    <studyInstanceUID>/
      <seriesInstanceUID>/
        <sopInstanceUID>.dcm
```

## Label columns

| Column | Description |
|--------|-------------|
| `atypical_appearance` | COVID-19 atypical appearance |
| `indeterminate_appearance` | Indeterminate appearance |
| `negative_for_pneumonia` | No pneumonia findings |
| `typical_appearance` | COVID-19 typical appearance |

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | `None` | `train/` (or `test/` for the test split) — root walked for `<study>/<series>/<sop>.dcm` files (e.g. `~/datasets/competitions/siim-covid19-detection/train/`) |
| `csv_path` | `str` | No | auto | `train_study_level.csv`; auto-discovered |
| `image_csv_path` | `str` | No | auto | `train_image_level.csv` for bbox annotations |
| `output_cls` | `bool` | No | `False` | Include `"cls"` in data dict |
| `output_mask` | `bool` | No | `False` | Include `"mask"` in data dict |
| `output_report` | `bool` | No | `False` | Include `"report"` in data dict |
| `output_bbox` | `bool` | No | `False` | Include `"bbox"` and `"bbox_labels"` in data dict |
| `cache_dir` | `str` | No | `"./cache"` | MONAI cache directory |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonized CSV |
| `transform` | MONAI transform | No | `None` | MONAI Compose transform; `None` uses the default pipeline |

## Dataset constructor

The constructor arguments above apply to `SIIMCOVID19TrainDataset`. `SIIMCOVID19TestDataset` accepts the same args **except** `csv_path` and `image_csv_path` (the test split has no annotation CSVs — those are hardcoded to `None` internally). The test split also silently ignores `output_cls` and `output_bbox`.

### Train split

```python
import torch
from radharmony.dataset import SIIMCOVID19TrainDataset

ds = SIIMCOVID19TrainDataset(
    base_image_dir="/data/siim-covid19-detection/train/",
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

### Test split

```python
from radharmony.dataset import SIIMCOVID19TestDataset

ds = SIIMCOVID19TestDataset(
    base_image_dir="/data/siim-covid19-detection/test/",
    dtype=torch.float32,
)
```

## Harmonizer: instantiate and inspect

```python
from radharmony.harmonizer import SIIMCOVID19TrainHarmonizer

h = SIIMCOVID19TrainHarmonizer(
    csv_path="/data/siim-covid19-detection/train_study_level.csv",
    base_image_dir="/data/siim-covid19-detection/train/",
    image_csv_path="/data/siim-covid19-detection/train_image_level.csv",
)
df = h.harmonize()
print(df.columns.tolist())
df.to_csv("siim_covid19_harmonized.csv", index=False)
```

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import SIIMCOVID19TrainDataset

ds = SIIMCOVID19TrainDataset(
    base_image_dir="/data/siim-covid19-detection/train/",
    harmonized_df=pd.read_csv("siim_covid19_harmonized.csv"),
    output_cls=True,
)
```

## Harmonizer notes

- Study-level CSV provides the appearance class labels (one per study)
- Image-level CSV provides bounding boxes (opacity locations, one per box instance)
- A study may have multiple images; each image inherits the study-level label
- Bounding boxes are normalised to fractional coordinates

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_cls=True` | `"cls"` | `(4,)` | One-hot appearance class |
| `output_bbox=True` | `"bbox"`, `"bbox_labels"` | list | Opacity bounding boxes |

## Example paths

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/path/to/siim-covid19-detection/train/` |
| `base_image_dir` (test) | `/path/to/siim-covid19-detection/test/` |
| `csv_path` (`train_study_level.csv`) | `/path/to/siim-covid19-detection/train_study_level.csv` |
| `image_csv_path` (`train_image_level.csv`) | `/path/to/siim-covid19-detection/train_image_level.csv` |

CSVs live at the competition root (parent of `train/`), auto-discoverable via `infer_path`. `SIIMCOVID19TestDataset` does NOT accept `csv_path` or `image_csv_path`.
