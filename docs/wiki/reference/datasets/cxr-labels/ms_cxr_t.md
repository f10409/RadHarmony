# MS-CXR-T

<!-- claim: id=ms_cxr_t.pairs tier=ci cmd="python scripts/evalkit/dataset_stat.py ms_cxr_t ms_cxr_t.rows" expect="1,045" -->
<!-- claim: id=ms_cxr_t.images tier=ci cmd="python scripts/evalkit/dataset_stat.py ms_cxr_t ms_cxr_t.images" expect="1,036" -->
<!-- claim: id=ms_cxr_t.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py ms_cxr_t ms_cxr_t.patients" expect="800" -->
| | |
|---|---|
| Registry keys | `ms_cxr_t` |
| Task and outputs | Change of 5 findings between 2 visits, in the harmonized table only. Image pairs (`img` and `previous_img`) |
| Images in the table | 1,045 image pairs, with 1,036 current images of 800 patients |
| Modality | Chest X-ray, 2-D, JPEG from MIMIC-CXR-JPG 2.0.0 |
| License | PhysioNet credentialed access, as for MIMIC-CXR |
| Download | [Get the data: other datasets](../../../how-to/get-the-data.md#other-datasets). The images come from [MIMIC-CXR-JPG](../../../how-to/get-the-data.md#mimic-cxr-jpg) |

MS-CXR-T is a temporal benchmark (Bannur et al., CVPR 2023). Its image task extends the progression labels
of Chest ImaGenome. Each row links a current chest X-ray to a prior chest X-ray of the same patient. Experts
labeled the change of up to 5 findings: improving, stable or worsening. The labels are in the harmonized
table, not in `cls`. RadHarmony reads only the image classification CSV of the source, not the sentence
similarity CSV.

## Quick use

`base_image_dir` is the `files/` folder of MIMIC-CXR-JPG 2.0.0, the same folder as for MS-CXR. The CSV
is in its own folder, so pass `csv_path`.

```
mimic-cxr-jpg/2.0.0/files/        <- base_image_dir
  p10/p10002428/s55758034/3bea0373-0d10dd77-1cac5b90-651be924-d343b184.jpg
ms-cxr-t/
  MS_CXR_T_temporal_image_classification_v1.0.0.csv   <- csv_path
```

```python
import torch
from radharmony.dataset import MSCXRTDataset

ds = MSCXRTDataset(
    base_image_dir="/data/mimic-cxr-jpg/2.0.0/files/",
    csv_path="/data/ms-cxr-t/MS_CXR_T_temporal_image_classification_v1.0.0.csv",
    output_previous=True,
    dtype=torch.float32,
)
pairs = ds.get_datasets()
sample = pairs[0]
# sample["img"]: Tensor(1, 224, 224), the current image
# sample["previous_img"]: Tensor(1, 224, 224), the prior image
```

## How it is harmonized

- Source: `MS_CXR_T_temporal_image_classification_v1.0.0.csv`. One row is one image pair, not one
  image (D-03).
- IDs: `patient_id` is `subject_id`. `study_id` and `previous_study_id` come from the CSV. `image_path`
  is `dicom_id` plus `.jpg`, relative to the `files/` folder (D-04). Example: the `dicom_id`
  `p10/p10002428/s55758034/3bea0373-0d10dd77-1cac5b90-651be924-d343b184` gives the `image_path`
  `p10/p10002428/s55758034/3bea0373-0d10dd77-1cac5b90-651be924-d343b184.jpg`, the `patient_id`
  `10002428` and the `study_id` `55758034`. `previous_image_path` comes from `previous_dicom_id` in the
  same way.
- Labels: the class has no `LABEL_COLS`, so it gives no `cls`. In each `<finding>_progression` column,
  `improving` gives -1, `stable` gives 0 and `worsening` gives 1. Here -1 means "improving", not
  "uncertain" as in D-05.
- Blank labels: a finding with no label gets NaN in its progression column and in its quality column.
- View, age and sex: the CSV has none of them. So the harmonized table has no `view_position`,
  `age_years` or `sex`.
- Dropped rows: none from the CSV. With `output_previous=True`, the class drops a row with no
  `previous_image_path`. Only a table that you make yourself can have such a row (D-11).

## Labels

| Column | Values | Meaning |
|---|---|---|
| `consolidation_progression` | -1, 0, 1, NaN | Change of consolidation: improving, stable, worsening, no label |
| `edema_progression` | -1, 0, 1, NaN | Change of edema |
| `pleural_effusion_progression` | -1, 0, 1, NaN | Change of pleural effusion |
| `pneumonia_progression` | -1, 0, 1, NaN | Change of pneumonia |
| `pneumothorax_progression` | -1, 0, 1, NaN | Change of pneumothorax |

Each finding also has a column `<finding>_label_quality`, for example `edema_label_quality`. Its values
are `one_expert`, `multiple_experts`, `disagreement` or NaN, as in the source.

```python
df = ds.get_harmonized_df()
print(df[["study_id", "previous_study_id", "pleural_effusion_progression",
          "pleural_effusion_label_quality"]].head(2))
#    study_id previous_study_id  pleural_effusion_progression pleural_effusion_label_quality
# 0  55758034          57887188                           NaN                            NaN
# 1  54394630          57157809                          -1.0               multiple_experts
```

## Prior image

With `output_previous=True`, the sample also has `previous_img`, the prior image. It goes through the
same transform as `img`, in a second call. So a random step, for example a flip, can give a different
result for the prior image (D-11).

## Splits

<!-- claim: id=ms_cxr_t.splits.pairs tier=ci cmd="python scripts/evalkit/dataset_stat.py ms_cxr_t ms_cxr_t.rows" expect="1,045" -->
<!-- claim: id=ms_cxr_t.splits.images tier=ci cmd="python scripts/evalkit/dataset_stat.py ms_cxr_t ms_cxr_t.images" expect="1,036" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `MSCXRTDataset` | `ms_cxr_t` | 1,036 current images in 1,045 pairs | `MS_CXR_T_temporal_image_classification_v1.0.0.csv` |

MS-CXR-T has no official split. `get_datasets(n_splits=5)` makes a train and val split. It groups by
patient (D-10). `get_datasets()` gives all pairs.

## Arguments

<!-- args: radharmony.dataset.MSCXRTDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `files/` folder of MIMIC-CXR-JPG 2.0.0. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `MS_CXR_T_temporal_image_classification_v1.0.0.csv`. Without it, the class looks in `base_image_dir` and in the folder above it, and raises `FileNotFoundError` if the file is not there |
| `output_previous` | `bool` | No | `False` | Add the prior image as `previous_img` (D-11) |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11). The labels are in the harmonized table only |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_report` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img` and `previous_img` |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import MSCXRTHarmonizer

h = MSCXRTHarmonizer(
    csv_path="/data/ms-cxr-t/MS_CXR_T_temporal_image_classification_v1.0.0.csv",
    mimic_base_dir="/data/mimic-cxr-jpg/2.0.0/files/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'previous_image_path', 'previous_study_id',
#  'consolidation_progression', ..., 'pneumothorax_progression',
#  'consolidation_label_quality', ..., 'pneumothorax_label_quality']
```

A harmonized table saved before 2026-10-28 has `image_path` values that start with `files/`. Harmonize
again, or use the folder above `files/` with it (D-04).

<!-- args: radharmony.harmonizer.MSCXRTHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `MS_CXR_T_temporal_image_classification_v1.0.0.csv` |
| `mimic_base_dir` | `str` | No | `None` | The `files/` folder of MIMIC-CXR-JPG 2.0.0. The harmonizer keeps it for save and load, but does not read it |

## Known data problems

- Disagreement: for some labels, the experts did not agree. The quality column is then `disagreement`,
  but the progression column still has a value. Filter on the quality column if you need clean labels.
- Repeated images: 9 current images are in 2 rows each, with 2 different prior images (D-03).
