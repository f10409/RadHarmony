# MS-CXR (Local Alignment / Phrase Grounding)

**Modality:** CXR | **Format:** JPG (MIMIC-CXR-JPG) | **Dim:** 2D | **Labels:** 8 binary findings + phrase–bbox pairs

## Overview

Phrase grounding benchmark for chest X-rays with explicit phrase-to-bounding-box annotations.
1,047 images from MIMIC-CXR-JPG with 1,448 annotated (phrase, bounding box) pairs across 8 pathology categories.

- **1,047 images** — a subset of MIMIC-CXR-JPG 2.0.0 (credentialed access required)
- **1,448 phrase–bbox annotations** — each links a natural-language description to a bounding box
- **8 pathology categories** — binary flags derived from category annotations

**Source (PhysioNet — requires MIMIC credentialed access):**
- https://physionet.org/content/ms-cxr/

**Images:** MIMIC-CXR-JPG 2.0.0 — `base_image_dir` must point at the root containing `files/`.

## Download

```bash
# 1. Download the annotation CSV from PhysioNet
wget -r -N --no-parent -np \
  https://physionet.org/content/ms-cxr/0.1/ \
  -P ./ms-cxr/

# 2. You must already have MIMIC-CXR-JPG 2.0.0 downloaded.
#    The base_image_dir is the root containing files/ (e.g. mimic-cxr-jpg/2.0.0/)
```

## Expected layout

```
<base_image_dir>/           ← MIMIC-CXR-JPG 2.0.0 root
  files/
    p10/
      p10233088/
        s54276838/
          675d792f-a3521e48-5eec8573-1e81d644-e60c34f8.jpg
        ...

MS_CXR_Local_Alignment_v1.1.0.csv   ← csv_path
```

## Label columns

| Column | Description |
|--------|-------------|
| `atelectasis` | 1 if any annotation in the image is Atelectasis |
| `cardiomegaly` | 1 if any annotation is Cardiomegaly |
| `consolidation` | 1 if any annotation is Consolidation |
| `edema` | 1 if any annotation is Edema |
| `lung_opacity` | 1 if any annotation is Lung Opacity |
| `pleural_effusion` | 1 if any annotation is Pleural Effusion |
| `pneumonia` | 1 if any annotation is Pneumonia |
| `pneumothorax` | 1 if any annotation is Pneumothorax |

## Extra metadata columns

| Column | Type | Description |
|--------|------|-------------|
| `split` | `str` | `"train"` or `"test"` |
| `bbox` | `str` (JSON) | List of `[y_min, y_max, x_min, x_max]` normalized boxes (one per annotation) |
| `bbox_labels` | `str` (JSON) | Category name for each box (parallel to `bbox`) |
| `label_text` | `str` (JSON) | Natural-language phrase for each annotation |
| `image_width` | `int` | Image width in pixels |
| `image_height` | `int` | Image height in pixels |

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes* | `None` | MIMIC-CXR-JPG 2.0.0 root (contains `files/`) |
| `csv_path` | `str` | Yes* | `None` | Path to `MS_CXR_Local_Alignment_v1.1.0.csv` |

### Shared arguments (inherited from `BaseRadiologicalDataset`)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `output_cls` | `bool` | No | `False` | Include 8 binary finding labels under `"cls"` |
| `output_mask` | `bool` | No | `False` | Include `"mask"` in data dict |
| `output_report` | `bool` | No | `False` | Include `"report"` in data dict |
| `output_bbox` | `bool` | No | `False` | Include phrase-grounding boxes under `"bbox"` / `"bbox_labels"` |
| `transform` | Compose | No | standard 2-D 224 px | MONAI Compose transform |
| `cache_dir` | `str` | No | `"./cache"` | MONAI cache directory. `None` disables |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonizer pickle |

\* Required unless `harmonizer_path` or `harmonized_df` is provided.

## Usage

```python
from radharmony.dataset import MSCXRDataset

ds = MSCXRDataset(
    base_image_dir="/data/mimic-cxr-jpg/2.0.0/",
    csv_path="/data/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv",
    output_cls=True,
    output_bbox=True,
    cache_dir="./cache",
)

sample = ds.get_datasets()[0]
print(sample["img"].shape)         # torch.Size([3, 224, 224])
print(sample["cls"])               # tensor([0., 0., 0., 1., ...])  — 8 binary labels
print(sample["bbox"])              # [[y_min, y_max, x_min, x_max], ...]  — normalized [0, 1]
print(sample["bbox_labels"])       # ["Edema", ...]
```

Access annotations via the harmonized DataFrame:

```python
df = ds.get_harmonized_df()
import json
first = df.iloc[0]
print(first["image_path"])
print(json.loads(first["bbox"]))        # [[0.31, 0.62, 0.23, 0.51], ...]  — normalized [0, 1]
print(json.loads(first["label_text"]))  # ["Fluffy bilateral opacities..."]
```

## Harmonizer

```python
from radharmony.harmonizer import MSCXRHarmonizer

h = MSCXRHarmonizer(
    csv_path="/data/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv",
    mimic_base_dir="/data/mimic-cxr-jpg/2.0.0/",
)
df = h.harmonize()
print(df.shape)           # (1047, 18)
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'split', 'bbox', 'bbox_labels',
#  'label_text', 'image_width', 'image_height',
#  'atelectasis', 'cardiomegaly', 'consolidation', 'edema',
#  'lung_opacity', 'pleural_effusion', 'pneumonia', 'pneumothorax']
```

## Harmonizer notes

- **One row per image** — multiple annotations per image are aggregated into JSON lists (`bbox`, `bbox_labels`, `label_text`).
- **Binary flags** — derived from `category_name` column via snake_case mapping (`"Lung Opacity"` → `lung_opacity`). An image with two Edema annotations gets `edema=1`.
- **image_path** — taken directly from the `path` column in the CSV, already relative to the MIMIC-CXR-JPG root (e.g. `files/p10/p10233088/s54276838/675d792f-....jpg`).
- **Bounding boxes** — stored as `[y_min, y_max, x_min, x_max]` normalized coordinates (`[0, 1]`), converted from the original CSV `[x, y, w, h]` pixel format to match the RadHarmony bbox convention expected by the transform pipeline.
