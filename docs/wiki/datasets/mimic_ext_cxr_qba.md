# MIMIC-Ext-CXR-QBA

**Modality:** CXR | **Format:** DICOM / JPEG | **Dim:** 2D | **Task:** Visual Question Answering

## Overview

MIMIC-Ext-CXR-QBA is a visual question-answering (VQA) dataset built on top of MIMIC-CXR. Each sample pairs a chest X-ray image with a clinical question and an answer. The dataset is distributed as parquet metadata exports plus a `qa.zip` archive containing per-study JSON files with question/answer text.

This dataset uses `BaseVQADataset` rather than `BaseRadiologicalDataset`. The data dict contains mixed types (Tensor + strings + optional dict), so the standard `output_cls` / `output_bbox` mechanism does not apply.

## Download

Available from [PhysioNet: MIMIC-Ext-CXR-QBA](https://physionet.org/content/mimic-ext-cxr-qba/1.0.0/). Requires CITI training and signed DUA (also covers MIMIC-CXR access for the underlying images).

Expected layout:

```
mimic-cxr/
  2.1.0/
    files/
      p10/p10000032/s50414267/  # DICOM study dirs
        ...
exports/
  A_frontal/
    metadata/
      q1M/
        question_metadata.parquet
        question_image_metadata.parquet
        image_metadata.parquet
qa.zip                          # per-study QA JSON files
```

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes* | `None` | MIMIC-CXR `files/` subtree root (e.g. `/data/mimic-cxr/2.1.0/files/`) — direct parent of `p10/`, `p11/`, … patient prefix dirs. Image path resolves to `<patient_id[:3]>/<patient_id>/<study_id>/<image_id>.dcm` relative to this |
| `metadata_dir` | `str` | Yes** | `None` | Directory containing the parquet metadata files |
| `qa_zip_path` | `str` | No | `None` | Path to `qa.zip`; when `None`, `question` and `answer` are empty strings |
| `image_ext` | `str` | No | `".dcm"` | `".dcm"` for MIMIC-CXR DICOM or `".jpg"` for MIMIC-CXR-JPG |
| `max_studies` | `int` or `None` | No | `None` | Cap on studies to process; useful during development |

### Shared arguments (inherited from `BaseVQADataset`)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `output_struct` | `bool` | No | `False` | Include `"answer_struct"` (raw answer list) in data dict |
| `output_question_type` | `bool` | No | `False` | Include `"question_type"` string in data dict |
| `transform` | MONAI Compose | No | 2D pipeline | Custom MONAI transform (must not use `SelectItemsD`) |
| `cache_dir` | `str` | No | `"./cache"` | MONAI PersistentDataset cache; `None` disables caching |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype for `"img"`; use `torch.float32` on CPU |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonized CSV |

*`base_image_dir` can be omitted when `harmonizer_path`, `harmonizer`, or `harmonized_df` is provided.

**`metadata_dir` is required when constructing the harmonizer from scratch (not using a preset).

## Dataset constructor

```python
from radharmony.dataset import MIMICExtCXRQBADataset

ds = MIMICExtCXRQBADataset(
    base_image_dir="/data/mimic-cxr/2.1.0/files",
    metadata_dir="/data/exports/A_frontal/metadata/q1M",
    qa_zip_path="/data/qa.zip",
    image_ext=".dcm",
    output_struct=True,
    output_question_type=True,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

For fast iteration during development:

```python
ds = MIMICExtCXRQBADataset(
    base_image_dir="/data/mimic-cxr/2.1.0/files",
    metadata_dir="/data/exports/A_frontal/metadata/q1M",
    max_studies=100,
)
```

## Harmonizer: instantiate and inspect

```python
from radharmony.harmonizer import MIMICExtCXRQBAHarmonizer

h = MIMICExtCXRQBAHarmonizer(
    metadata_dir="/data/exports/A_frontal/metadata/q1M",
    qa_zip_path="/data/qa.zip",
    base_image_dir="/data/mimic-cxr/2.1.0/files",
    image_ext=".dcm",
)
df = h.harmonize()
print(df.columns.tolist())
df.to_csv("mimic_ext_cxr_qba_harmonized.csv", index=False)
```

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import MIMICExtCXRQBADataset

ds = MIMICExtCXRQBADataset(
    base_image_dir="/data/mimic-cxr/2.1.0/files",
    harmonized_df=pd.read_csv("mimic_ext_cxr_qba_harmonized.csv"),
    output_struct=True,
)
```

## Harmonizer notes

- Reads `question_metadata.parquet` and `question_image_metadata.parquet`; one row per `(patient_id, study_id, question_id)` after picking one image per question
- `qa.zip` is read lazily — each unique study's `qa.json` is read once to extract question and answer text
- `image_path` is relative to `base_image_dir`: `p<xx>/<patient_id>/<study_id>/<image_id><ext>`
- `answer_struct` holds the raw list of answer dicts from the JSON (multiple annotators may provide answers)
- `question_type` comes from the parquet `question.question_type` column
- When `qa_zip_path=None`, `question` and `answer` are empty strings (useful for image-only pre-processing)

## VQA data dict

Each sample contains:

| Key | Type | Notes |
|-----|------|-------|
| `"img"` | `Tensor (1, H, W)` | Loaded and normalised CXR image |
| `"question"` | `str` | Clinical question text |
| `"answer"` | `str` | Answer text (concatenated from annotators) |
| `"patient_id"` | `str` | MIMIC-CXR patient ID |
| `"study_id"` | `str` | MIMIC-CXR study ID |
| `"question_id"` | `str` | QBA question identifier |
| `"answer_struct"` | `list` | Raw answer objects (only when `output_struct=True`) |
| `"question_type"` | `str` | Question category (only when `output_question_type=True`) |

## Notes on BaseVQADataset

MIMIC-Ext-CXR-QBA inherits `BaseVQADataset` instead of `BaseRadiologicalDataset` for three reasons:

1. The standard 2-D transform uses `SelectItemsD` to drop extra keys — that would silently discard `question`, `answer`, and `question_id`
2. The `output_cls` / `output_bbox` flag system and `LABEL_COLS` concept do not apply to VQA
3. Data dicts contain mixed types (`Tensor` + `str` + `dict`) that cannot pass through a standard MONAI tensor pipeline unchanged
