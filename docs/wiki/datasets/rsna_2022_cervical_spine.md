# RSNA 2022 Cervical Spine

**Modality:** CT | **Format:** DICOM | **Dim:** 3D | **Labels:** 8 fracture labels

## Overview

RSNA 2022 Cervical Spine Fracture Detection is a Kaggle challenge dataset containing 2,019 cervical-spine CT studies (one per patient). Each study is a directory of per-slice DICOM files loaded as a 3-D volume by MONAI's `ITKReader`. Eight study-level binary labels encode presence/absence of fracture at each vertebra (`C1`–`C7`) and an overall flag (`patient_overall`).

A subset of 87 studies ships with whole-volume NIfTI segmentation masks. A separate bbox variant (`RSNA2022CervicalSpineBboxDataset`) provides per-volume 3-D fracture bounding boxes for the 235 studies that have at least one fracture-localizing annotation.

## Download

Available at [Kaggle — RSNA 2022 Cervical Spine Fracture Detection](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection).

Expected layout:

```
rsna-2022-cervical-spine-fracture-detection/
  train.csv
  train_bounding_boxes.csv
  segmentations/
    <StudyInstanceUID>.nii
    ...
  train_images/
    <StudyInstanceUID>/
      100.dcm
      101.dcm
      ...
```

## Label columns

| Column | Description |
|--------|-------------|
| `c1` | C1 vertebra fracture |
| `c2` | C2 vertebra fracture |
| `c3` | C3 vertebra fracture |
| `c4` | C4 vertebra fracture |
| `c5` | C5 vertebra fracture |
| `c6` | C6 vertebra fracture |
| `c7` | C7 vertebra fracture |
| `patient_overall` | Any cervical fracture present |

The `cls` tensor has values in sorted (alphabetical) order: `c1, c2, c3, c4, c5, c6, c7, patient_overall`.

---

## 3D Volume variant — `RSNA2022CervicalSpineDataset`

### Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes* | `None` | `train_images/` (or `test_images/` for the test split) — direct parent of `<StudyInstanceUID>/` DICOM dirs (e.g. `~/datasets/external/rsna-2022-cervical-spine-fracture-detection/train_images/`) |
| `csv_path` | `str` | No | auto | `train.csv`; auto-discovered in parent of `base_image_dir` |
| `segmentation_dir` | `str` | No | auto | Directory of `<StudyUID>.nii` masks; auto-discovered as `../segmentations/` |
| `hu_window` | `tuple` or `None` | No | `(-200, 1800)` | HU clipping window (bone-centric default) |

### Shared arguments (inherited from `BaseRadiologicalDataset`)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `output_cls` | `bool` | No | `False` | Include `"cls"` in data dict |
| `output_mask` | `bool` | No | `False` | Include `"mask"`; restricts dataset to 87 studies with NIfTI masks |
| `output_report` | `bool` | No | `False` | Not supported; silently ignored |
| `output_bbox` | `bool` | No | `False` | Not supported; use `RSNA2022CervicalSpineBboxDataset` for bboxes |
| `transform` | MONAI Compose | No | 3D pipeline | Custom MONAI transform |
| `cache_dir` | `str` | No | `"./cache"` | MONAI PersistentDataset cache; `None` disables caching |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype; use `torch.float32` on CPU |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonized CSV |

*`base_image_dir` can be omitted when `harmonizer_path`, `harmonizer`, or `harmonized_df` is provided.

### Dataset constructor

The constructor arguments above apply to both `RSNA2022CervicalSpineTrainDataset` and `RSNA2022CervicalSpineTestDataset`. The test split has no labels or segmentations — `output_cls`, `output_mask`, and `output_bbox` are silently ignored.

#### Train split

```python
import torch
from radharmony.dataset import RSNA2022CervicalSpineTrainDataset

ds = RSNA2022CervicalSpineTrainDataset(
    base_image_dir="/data/rsna-2022-cervical-spine-fracture-detection/train_images/",
    csv_path="/data/rsna-2022-cervical-spine-fracture-detection/train.csv",
    output_cls=True,
    hu_window=(-200, 1800),
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

With segmentation masks (restricts to 87 studies):

```python
ds = RSNA2022CervicalSpineTrainDataset(
    base_image_dir="/data/rsna-2022-cervical-spine-fracture-detection/train_images/",
    segmentation_dir="/data/rsna-2022-cervical-spine-fracture-detection/segmentations/",
    output_cls=True,
    output_mask=True,
)
```

#### Test split

```python
from radharmony.dataset import RSNA2022CervicalSpineTestDataset

ds = RSNA2022CervicalSpineTestDataset(
    base_image_dir="/data/rsna-2022-cervical-spine-fracture-detection/test_images/",
    hu_window=(-200, 1800),
)
```

### Harmonizer: instantiate and inspect

```python
from radharmony.harmonizer import RSNA2022CervicalSpineTrainHarmonizer

h = RSNA2022CervicalSpineTrainHarmonizer(
    csv_path="/data/rsna-2022-cervical-spine-fracture-detection/train.csv",
    base_image_dir="/data/rsna-2022-cervical-spine-fracture-detection/train_images/",
    segmentation_dir="/data/rsna-2022-cervical-spine-fracture-detection/segmentations/",
)
df = h.harmonize()
print(df.columns.tolist())
df.to_csv("rsna_cervical_spine_harmonized.csv", index=False)
```

### Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import RSNA2022CervicalSpineTrainDataset

ds = RSNA2022CervicalSpineTrainDataset(
    base_image_dir="/data/rsna-2022-cervical-spine-fracture-detection/train_images/",
    harmonized_df=pd.read_csv("rsna_cervical_spine_harmonized.csv"),
    output_cls=True,
)
```

### Harmonizer notes

- One row per `StudyInstanceUID` (= one CT volume)
- `image_path` is a DICOM study directory (single level of nesting under `train_images/`)
- MONAI's `ITKReader` assembles the per-slice `.dcm` files into a 3-D volume
- `mask_path` is populated for the 87 studies that have a matching `.nii` file; `NaN` for the rest
- `output_mask=True` naturally restricts the dataset to the 87 mask-having studies via the framework's `dropna` on `mask_path`
- HU window `(-200, 1800)` is bone-centric; use `(-1000, 1000)` for general soft-tissue
- CSV columns `C1`..`C7` (uppercase) are renamed to lowercase `c1`..`c7` by the harmonizer

### Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_cls=True` | `"cls"` | `(8,)` | Binary fracture labels |
| `output_mask=True` | `"mask"` | `(1, D, H, W)` | NIfTI segmentation (87 studies only) |

---

## 3D Bbox variant — `RSNA2022CervicalSpineBboxDataset`

### Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes* | `None` | `train_images/` directory — direct parent of `<StudyInstanceUID>/` DICOM dirs (e.g. `~/datasets/external/rsna-2022-cervical-spine-fracture-detection/train_images/`) |
| `csv_path` | `str` | No | auto | `train.csv`; auto-discovered |
| `bbox_csv_path` | `str` | No | auto | `train_bounding_boxes.csv`; auto-discovered |

### Shared arguments (inherited from `BaseRadiologicalDataset`)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `output_cls` | `bool` | No | `False` | Include study-level `"cls"` (8-D) in data dict |
| `output_mask` | `bool` | No | `False` | Not supported; silently ignored |
| `output_report` | `bool` | No | `False` | Not supported; silently ignored |
| `output_bbox` | `bool` | No | `False` | Include per-volume `"bbox"` and `"bbox_labels"` |
| `transform` | MONAI Compose | No | 3D pipeline | Custom MONAI transform |
| `cache_dir` | `str` | No | `"./cache"` | MONAI PersistentDataset cache; `None` disables |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype; use `torch.float32` on CPU |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonized CSV |

*`base_image_dir` can be omitted when `harmonizer_path`, `harmonizer`, or `harmonized_df` is provided.

### Dataset constructor

```python
from radharmony.dataset import RSNA2022CervicalSpineBboxDataset

ds = RSNA2022CervicalSpineBboxDataset(
    base_image_dir="/data/rsna-2022-cervical-spine-fracture-detection/train_images/",
    csv_path="/data/rsna-2022-cervical-spine-fracture-detection/train.csv",
    bbox_csv_path="/data/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv",
    output_cls=True,
    output_bbox=True,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

### Harmonizer: instantiate and inspect

```python
from radharmony.harmonizer import RSNA2022CervicalSpineBboxHarmonizer

h = RSNA2022CervicalSpineBboxHarmonizer(
    csv_path="/data/rsna-2022-cervical-spine-fracture-detection/train.csv",
    bbox_csv_path="/data/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv",
    base_image_dir="/data/rsna-2022-cervical-spine-fracture-detection/train_images/",
)
df = h.harmonize()
print(df.columns.tolist())
df.to_csv("rsna_cervical_spine_bbox_harmonized.csv", index=False)
```

### Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import RSNA2022CervicalSpineBboxDataset

ds = RSNA2022CervicalSpineBboxDataset(
    base_image_dir="/data/rsna-2022-cervical-spine-fracture-detection/train_images/",
    harmonized_df=pd.read_csv("rsna_cervical_spine_bbox_harmonized.csv"),
    output_cls=True,
    output_bbox=True,
)
```

### Harmonizer notes

- One row per `StudyInstanceUID` (= one CT volume) with at least one fracture bbox
- `image_path` is the `<StudyUID>/` DICOM directory; ITKReader assembles it into a 3-D volume
- Multiple fracture annotations are aggregated into a list of `[d_min, d_max, y_min, y_max, x_min, x_max]` 6-tuples (post-transpose `(D, H, W)` axis order)
- Coordinates are normalised to `[0, 1]` against the loaded volume dimensions
- Study-level `cls` labels are joined from `train.csv` by `StudyInstanceUID`

### Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_cls=True` | `"cls"` | `(8,)` | Study-level binary fracture labels |
| `output_bbox=True` | `"bbox"` | list of `[d_min, d_max, y_min, y_max, x_min, x_max]` | Fractional coords (post-transpose `(D, H, W)` order); one entry per fracture annotation in the study |
| `output_bbox=True` | `"bbox_labels"` | list of `str` | Label name per bounding box |

## Example paths

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/path/to/rsna-2022-cervical-spine-fracture-detection/train_images/` |
| `base_image_dir` (test) | `/path/to/rsna-2022-cervical-spine-fracture-detection/test_images/` |
| `csv_path` (`train.csv`) | `/path/to/rsna-2022-cervical-spine-fracture-detection/train.csv` |
| `bbox_csv_path` (`train_bounding_boxes.csv`) | `/path/to/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv` |
