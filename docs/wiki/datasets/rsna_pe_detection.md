# RSNA PE Detection

**Modality:** CT | **Format:** DICOM (per-series volumes) | **Dim:** 3D | **Labels:** 13 PE labels

## Overview

The RSNA-STR Pulmonary Embolism Detection Challenge (2020) dataset contains CT pulmonary angiography (CTPA) studies labelled for the presence and characteristics of pulmonary embolism. Each study has image-level (per-slice) and exam-level labels. The dataset contains approximately 7,279 training exams.

## Download

Available on [Kaggle: RSNA STR Pulmonary Embolism Detection](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection). Requires Kaggle account.

## Expected layout

```
rsna_pe_dataset/
  train.csv
  train/
    <StudyInstanceUID>/
      <SeriesInstanceUID>/
        <SOPInstanceUID>.dcm
        ...
```

## Label columns

| Column | Description |
|--------|-------------|
| `acute_and_chronic_pe` | Acute and chronic PE |
| `central_pe` | Central PE |
| `chronic_pe` | Chronic PE only |
| `flow_artifact` | Flow artifact |
| `indeterminate` | Indeterminate |
| `leftsided_pe` | Left-sided PE |
| `negative_exam_for_pe` | No PE |
| `qa_contrast` | QA: contrast issue |
| `qa_motion` | QA: motion artifact |
| `rightsided_pe` | Right-sided PE |
| `rv_lv_ratio_gte_1` | RV/LV ratio ≥ 1 |
| `rv_lv_ratio_lt_1` | RV/LV ratio < 1 |
| `true_filling_defect_not_pe` | Filling defect, not PE |

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | `None` | `train/` (or `test/` for the test split) — direct parent of `<StudyInstanceUID>/<SeriesInstanceUID>/` (e.g. `/data/rsna_pe_dataset/train/`) |
| `csv_path` | `str` | No | auto | `train.csv`; auto-discovered |
| `hu_window` | `tuple[float,float]\|None` | No | `(-1000, 1000)` | HU clip range for CT intensity normalisation |

### Shared arguments (inherited from `BaseRadiologicalDataset`)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `output_cls` | `bool` | No | `False` | Include `"cls"` in data dict |
| `output_mask` | `bool` | No | `False` | Not supported; silently ignored with a warning |
| `output_report` | `bool` | No | `False` | Not supported; silently ignored with a warning |
| `output_bbox` | `bool` | No | `False` | Not supported; silently ignored with a warning |
| `transform` | MONAI transform | No | `None` | MONAI Compose transform; `None` uses the default pipeline |
| `cache_dir` | `str` | No | `"./cache"` | MONAI cache directory |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype; use `torch.float32` on CPU |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonized CSV |

## Dataset constructor

The constructor arguments above apply to both `RSNAPEDetectionTrainDataset` and `RSNAPEDetectionTestDataset`. The test split has no labels — `output_cls` is silently ignored.

### Train split

```python
from radharmony.dataset import RSNAPEDetectionTrainDataset

ds = RSNAPEDetectionTrainDataset(
    base_image_dir="/data/rsna_pe_dataset/train/",
    hu_window=(-1000, 1000),
    output_cls=True,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

### Test split

```python
from radharmony.dataset import RSNAPEDetectionTestDataset

ds = RSNAPEDetectionTestDataset(
    base_image_dir="/data/rsna_pe_dataset/test/",
    hu_window=(-1000, 1000),
)
```

## Harmonizer

```python
from radharmony.harmonizer import RSNAPEDetectionTrainHarmonizer

h = RSNAPEDetectionTrainHarmonizer(
    csv_path="/data/rsna_pe_dataset/train.csv",
    base_image_dir="/data/rsna_pe_dataset/train/",
)
df = h.harmonize()
print(df.columns.tolist())
df.to_csv("rsna_pe_harmonized.csv", index=False)
```

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import RSNAPEDetectionTrainDataset

ds = RSNAPEDetectionTrainDataset(
    base_image_dir="/data/rsna_pe_dataset/train/",
    harmonized_df=pd.read_csv("rsna_pe_harmonized.csv"),
    output_cls=True,
)
```

## Harmonizer notes

- Each row in `train.csv` corresponds to one DICOM series (a full CT volume)
- `image_path` is a relative path to the series directory, not an individual DICOM file
- MONAI's `ITKReader` loads the entire series directory as a single 3D volume
- Labels are exam-level (same label for all slices in a study)
- HU window `(-1000, 1000)` is suitable for CTPA; adjust with `hu_window=` if needed

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_cls=True` | `"cls"` | `(13,)` | Multi-label binary PE labels |

## Example paths

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/path/to/rsna_pe_dataset/train/` |
| `base_image_dir` (test) | `/path/to/rsna_pe_dataset/test/` |
| `csv_path` (`train.csv`) | `/path/to/rsna_pe_dataset/train.csv` |
| test CSV (`test.csv`) | `/path/to/rsna_pe_dataset/test.csv` |
