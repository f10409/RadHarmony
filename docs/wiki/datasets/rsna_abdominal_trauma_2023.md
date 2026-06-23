# RSNA 2023 Abdominal Trauma

**Modality:** CT | **Format:** DICOM | **Dim:** 3D | **Labels:** 14 injury labels

## Overview

RSNA 2023 Abdominal Trauma Detection is a Kaggle challenge dataset of 4,711 abdominal CT series from 3,147 patients (mean ~1.5 series per patient). Each series is a DICOM directory loaded as a 3-D volume. Fourteen binary labels cover bowel and extravasation injury (binary) plus kidney, liver, and spleen severity (3-class one-hot per organ) and an overall `any_injury` flag.

Patient-level labels are replicated across all series belonging to the same patient, matching the challenge framing ("predict patient-level injury from any of the patient's series").

## Download

Available at [Kaggle — RSNA 2023 Abdominal Trauma Detection](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection).

Expected layout:

```
rsna-2023-abdominal-trauma-detection/
  train_2024.csv
  train_series_meta.csv
  train_images/
    <patient_id>/
      <series_id>/
        <instance_number>.dcm
        ...
```

## Label columns

| Column | Description |
|--------|-------------|
| `any_injury` | Any abdominal injury present |
| `bowel_healthy` | Bowel healthy |
| `bowel_injury` | Bowel injury |
| `extravasation_healthy` | No active extravasation |
| `extravasation_injury` | Active extravasation |
| `kidney_healthy` | Kidney healthy |
| `kidney_high` | Kidney high-grade injury |
| `kidney_low` | Kidney low-grade injury |
| `liver_healthy` | Liver healthy |
| `liver_high` | Liver high-grade injury |
| `liver_low` | Liver low-grade injury |
| `spleen_healthy` | Spleen healthy |
| `spleen_high` | Spleen high-grade injury |
| `spleen_low` | Spleen low-grade injury |

The `cls` tensor has 14 values in alphabetical order.

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes* | — | `train_images/` (or `test_images/` for the test split) — direct parent of `<patient_id>/<series_id>/` DICOM dirs (e.g. `~/datasets/external/rsna-2023-abdominal-trauma-detection/train_images/`) |
| `csv_path` | `str` | No | auto | `train_2024.csv`; auto-discovered in parent of `base_image_dir` |
| `series_meta_csv_path` | `str` | No | auto | `train_series_meta.csv`; auto-discovered |
| `transform` | MONAI Compose | No | 3D pipeline | Custom MONAI transform |
| `cache_dir` | `str` | No | `"./cache"` | MONAI PersistentDataset cache; `None` disables caching |
| `hu_window` | `tuple` or `None` | No | `(-150, 250)` | HU clipping window (soft-tissue / contrast abdomen) |
| `output_cls` | `bool` | No | `False` | Include `"cls"` in data dict |
| `output_mask` | `bool` | No | `False` | Not supported in this release; silently ignored |
| `output_report` | `bool` | No | `False` | Not supported; silently ignored |
| `output_bbox` | `bool` | No | `False` | Not supported in this release; silently ignored |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonized CSV |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype; use `torch.float32` on CPU |

*`base_image_dir` can be omitted when `harmonizer_path`, `harmonizer`, or `harmonized_df` is provided.

## Dataset constructor

```python
from radharmony.dataset import RSNAAbdominalTrauma2023Dataset

ds = RSNAAbdominalTrauma2023Dataset(
    base_image_dir="/data/rsna-2023-abdominal-trauma-detection/train_images/",
    csv_path="/data/rsna-2023-abdominal-trauma-detection/train_2024.csv",
    series_meta_csv_path="/data/rsna-2023-abdominal-trauma-detection/train_series_meta.csv",
    output_cls=True,
    hu_window=(-150, 250),
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

There is no separate test-split dataset class — the competition test images have no public labels. To load test volumes for inference, point `base_image_dir` at `test_images/` and omit `csv_path`.

## Harmonizer: instantiate and inspect

```python
from radharmony.harmonizer import RSNAAbdominalTrauma2023Harmonizer

h = RSNAAbdominalTrauma2023Harmonizer(
    csv_path="/data/rsna-2023-abdominal-trauma-detection/train_2024.csv",
    series_meta_csv_path="/data/rsna-2023-abdominal-trauma-detection/train_series_meta.csv",
    base_image_dir="/data/rsna-2023-abdominal-trauma-detection/train_images/",
)
df = h.harmonize()
print(df.columns.tolist())
df.to_csv("rsna_abdominal_trauma_harmonized.csv", index=False)
```

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import RSNAAbdominalTrauma2023Dataset

ds = RSNAAbdominalTrauma2023Dataset(
    base_image_dir="/data/rsna-2023-abdominal-trauma-detection/train_images/",
    harmonized_df=pd.read_csv("rsna_abdominal_trauma_harmonized.csv"),
    output_cls=True,
)
```

## Harmonizer notes

- One row per series; patient-level labels from `train_2024.csv` are joined onto each series by `patient_id`
- Series-level metadata `aortic_hu` (contrast-phase indicator) and `incomplete_organ` are carried through as extra columns in the harmonized DataFrame
- `image_path` is `<patient_id>/<series_id>` (relative DICOM series directory); MONAI's `ITKReader` assembles per-slice `.dcm` files into a 3-D volume
- HU window `(-150, 250)` covers soft-tissue / contrast-enhanced abdomen
- `bowel` and `extravasation` are binary (healthy vs. injury); `kidney`, `liver`, `spleen` use 3-class one-hot columns
- Organ segmentation masks and per-slice `Active_Extravasation` point annotations exist in the release but are not wired into this harmonizer

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_cls=True` | `"cls"` | `(14,)` | Binary injury labels |

## Example paths

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/path/to/rsna-2023-abdominal-trauma-detection/train_images/` |
| `base_image_dir` (test) | `/path/to/rsna-2023-abdominal-trauma-detection/test_images/` |
| `csv_path` (`train_2024.csv`) | `/path/to/rsna-2023-abdominal-trauma-detection/train_2024.csv` |
| `series_meta_csv_path` (`train_series_meta.csv`) | `/path/to/rsna-2023-abdominal-trauma-detection/train_series_meta.csv` |
| test series meta (`test_series_meta.csv`) | `/path/to/rsna-2023-abdominal-trauma-detection/test_series_meta.csv` |

Only one dataset class: `RSNAAbdominalTrauma2023Dataset` (no separate Train/Test class).
