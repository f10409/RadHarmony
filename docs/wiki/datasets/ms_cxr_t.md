# MS-CXR-T (Temporal Sentences and Image Pairs)

**Modality:** CXR | **Format:** JPG (MIMIC-CXR-JPG) | **Dim:** 2D | **Labels:** temporal progression (stable/improving/worsening) for 5 findings

## Overview

Temporal benchmark built on top of MS-CXR. 1,045 longitudinal image pairs from MIMIC-CXR-JPG,
where each row links a current CXR to a prior CXR from the same patient.
Each pair is annotated with a progression label (stable / improving / worsening) for up to 5 pathology findings.

- **1,045 image pairs** — one current image + one prior image per row
- **5 findings** — consolidation, edema, pleural effusion, pneumonia, pneumothorax
- **Progression encoded as integer** — `-1` = improving, `0` = stable, `1` = worsening, `NaN` = not annotated
- **Label quality** — each finding also has a quality column: `one_expert` / `multiple_experts` / `disagreement` / NaN

**Source (PhysioNet — requires MIMIC credentialed access):**
- https://physionet.org/content/ms-cxr-t/

**Images:** MIMIC-CXR-JPG 2.0.0 — same image root as MS-CXR.

## Download

```bash
# Download from PhysioNet
wget -r -N --no-parent -np \
  https://physionet.org/content/ms-cxr-t/1.0.0/ \
  -P ./ms-cxr-t/

# Images are from MIMIC-CXR-JPG 2.0.0 (same base_image_dir as MS-CXR)
```

## Expected layout

```
<base_image_dir>/           ← MIMIC-CXR-JPG 2.0.0 root
  files/
    p10/
      p10002428/
        s55758034/
          3bea0373-....jpg
        ...

MS_CXR_T_temporal_image_classification_v1.0.0.csv   ← csv_path
```

## Progression columns

| Column | Values | Description |
|--------|--------|-------------|
| `consolidation_progression` | -1, 0, 1, NaN | Improving / stable / worsening |
| `edema_progression` | -1, 0, 1, NaN | |
| `pleural_effusion_progression` | -1, 0, 1, NaN | |
| `pneumonia_progression` | -1, 0, 1, NaN | |
| `pneumothorax_progression` | -1, 0, 1, NaN | |

## Quality columns

| Column | Values | Description |
|--------|--------|-------------|
| `consolidation_label_quality` | `one_expert`, `multiple_experts`, `disagreement`, NaN | |
| `edema_label_quality` | same | |
| `pleural_effusion_label_quality` | same | |
| `pneumonia_label_quality` | same | |
| `pneumothorax_label_quality` | same | |

## Extra metadata columns

| Column | Type | Description |
|--------|------|-------------|
| `previous_image_path` | `str` | Relative path to the prior-visit image |
| `previous_study_id` | `str` | Study ID of the prior visit |

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes* | — | MIMIC-CXR-JPG 2.0.0 root (contains `files/`) |
| `csv_path` | `str` | Yes* | — | Path to `MS_CXR_T_temporal_image_classification_v1.0.0.csv` |
| `output_previous` | `bool` | No | `False` | Include prior-visit image under `"previous_img"` |
| `transform` | Compose | No | standard 2-D 224 px | MONAI Compose transform |
| `cache_dir` | `str` | No | `"./cache"` | MONAI cache directory. `None` disables |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonizer pickle |

\* Required unless `harmonizer_path` or `harmonized_df` is provided.

## Usage

```python
from radharmony.dataset import MSCXRTDataset

ds = MSCXRTDataset(
    base_image_dir="/data/mimic-cxr-jpg/2.0.0/",
    csv_path="/data/ms-cxr-t/MS_CXR_T_temporal_image_classification_v1.0.0.csv",
    cache_dir="./cache",
)

sample = ds.get_datasets()[0]
print(sample["img"].shape)   # torch.Size([3, 224, 224]) — current visit
```

Access progression labels and prior images via the harmonized DataFrame:

```python
df = ds.get_harmonized_df()
print(df[["study_id", "previous_study_id",
          "pleural_effusion_progression",
          "pleural_effusion_label_quality"]].head(4))
#    study_id  previous_study_id  pleural_effusion_progression  pleural_effusion_label_quality
# 0  55758034          50414267                          0.0                    multiple_experts
# 1  57375967          54276838                          1.0                    one_expert
# ...
```

## Harmonizer

```python
from radharmony.harmonizer import MSCXRTHarmonizer

h = MSCXRTHarmonizer(
    csv_path="/data/ms-cxr-t/MS_CXR_T_temporal_image_classification_v1.0.0.csv",
)
df = h.harmonize()
print(df.shape)           # (1045, 15)
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'previous_image_path', 'previous_study_id',
#  'consolidation_progression', 'edema_progression', 'pleural_effusion_progression',
#  'pneumonia_progression', 'pneumothorax_progression',
#  'consolidation_label_quality', 'edema_label_quality', 'pleural_effusion_label_quality',
#  'pneumonia_label_quality', 'pneumothorax_label_quality']
```

## Harmonizer notes

- **No `LABEL_COLS`** — this dataset has no binary classification targets; progression labels live in `_progression` columns.
- **Progression encoding** — raw strings (`"improving"` / `"stable"` / `"worsening"`) are mapped to integers (`-1` / `0` / `1`). Missing annotations remain `NaN`.
- **`image_path`** — derived from `dicom_id` in the CSV via `"files/" + dicom_id + ".jpg"`. Format: `files/p10/pXXXXXXXX/sYYYYYYYY/<hash>.jpg`.
- **`previous_image_path`** — same derivation applied to `previous_dicom_id`.
- **Images are shared with MIMIC-CXR-JPG** — both current and previous images live under the same `base_image_dir`.
