# TAIX-Ray

<!-- claim: id=taix_ray.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py taix_ray_512 taix_ray_512_train.images" expect="137,593" -->
<!-- claim: id=taix_ray.train.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py taix_ray_512 taix_ray_512_train.patients" expect="30,543" -->
<!-- claim: id=taix_ray.val.images tier=ci cmd="python scripts/evalkit/dataset_stat.py taix_ray_512 taix_ray_512_val.images" expect="34,860" -->
<!-- claim: id=taix_ray.val.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py taix_ray_512 taix_ray_512_val.patients" expect="7,636" -->
<!-- claim: id=taix_ray.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py taix_ray_512 taix_ray_512_test.images" expect="42,928" -->
<!-- claim: id=taix_ray.test.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py taix_ray_512 taix_ray_512_test.patients" expect="9,545" -->
| | |
|---|---|
| Registry keys | `taix_ray_512_train`, `taix_ray_512_val`, `taix_ray_512_test`, `taix_ray_train`, `taix_ray_val`, `taix_ray_test` |
| Task and outputs | Classification, 8 labels (`cls`), binary or ordinal |
| Images in the table | train 137,593 (30,543 patients), val 34,860 (7,636 patients), test 42,928 (9,545 patients) |
| Modality | Chest X-ray, 2-D, 16-bit PNG. One source with 512 pixels on the long side, and one at the original size |
| License | CC BY 4.0 |
| Download | [Get the data: TAIX-Ray](../../../how-to/get-the-data.md#taix-ray) |

<!-- claim: id=taix_ray.total.images tier=ci cmd="python scripts/evalkit/dataset_stat.py taix_ray_512 taix_ray_512_train.images+taix_ray_512_val.images+taix_ray_512_test.images" expect="215,381" -->
<!-- claim: id=taix_ray.total.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py taix_ray_512 taix_ray_512_train.patients+taix_ray_512_val.patients+taix_ray_512_test.patients" expect="47,724" -->
TAIX-Ray is a bedside chest X-ray dataset from the intensive care units of University Hospital RWTH
Aachen, Germany. It has 215,381 images of 47,724 patients. During routine work, radiologists graded 8
findings in a structured report. The 512-pixel and the original source have the same `annotation.csv`
content, so the counts above are the same for both. Paper: Truhn et al., Scientific Data 2026,
[DOI 10.1038/s41597-026-07271-7](https://doi.org/10.1038/s41597-026-07271-7).

## Quick use

`base_image_dir` is the `images/` folder. The class finds `annotation.csv` in that folder or in the
folder above it. The Hugging Face source has parquet files. The notebook
`notebooks/downloads/download_huggingface.ipynb` writes the PNG files and `annotation.csv` from them.

```
TAIX-Ray/
  data_512/
    annotation.csv
    images/<UID>.png
  data_original/
    annotation.csv
    images/<UID>.png
```

```python
import torch
from radharmony.dataset import TAIXRay512TestDataset, TAIXRay512TrainDataset, TAIXRay512ValDataset

kw = dict(
    base_image_dir="/data/TAIX-Ray/data_512/images/",
    label_mode="binary",
    output_cls=True,
    dtype=torch.float32,
)
train_ds = TAIXRay512TrainDataset(**kw).get_datasets()
val_ds = TAIXRay512ValDataset(**kw).get_datasets()
test_ds = TAIXRay512TestDataset(**kw).get_datasets()
sample = val_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(8,)
```

For the original-size images, use `TAIXRayTrainDataset`, `TAIXRayValDataset` and `TAIXRayTestDataset`
with `data_original/images/`.

## How it is harmonized

- Source: `annotation.csv`. One row is one image (D-03).
- IDs: `patient_id` is `PatientID`. Each `UID` has one image, so `study_id` is the `UID`. `image_path` is
  `<UID>.png`, relative to `images/` (D-04). Example: the `UID`
  `b6c2215aac6e552fa54e483b8e2a66c1d6ef3b481c75125e345ad6c27807bf4f` gives the `image_path`
  `b6c2215aac6e552fa54e483b8e2a66c1d6ef3b481c75125e345ad6c27807bf4f.png`. `PatientID` is also a
  64-character code, for example `fe59e9dd10af944bac7a83ccfcc1491f40216c5f1c5cf2c6b3eb8d5b5c9c9a56`.
- Labels (D-05): the harmonizer keeps the raw grades, as float32. The dataset class applies
  `label_mode`. In `"binary"` mode, grade 1 or more gives 1, and grade 0 gives 0. `"ordinal"` keeps the
  grade.
- Uncertain and blank labels: the source has no `-1` and no blank label. Grade 1 means "Questionable"
  (or "Borderline Enlarged" for `heart_size`). In binary mode, grade 1 gives 1.
- View (D-01): the source has no view column, so the harmonized table has no `view_position`. All images
  are bedside AP (Truhn et al. 2026).
- Age and sex: `age_years` comes from `Age`, in days, divided by 365.25 (D-06). `sex` is `M` or `F`,
  from `Sex` (D-26).
- Extra columns: `age` is the raw `Age` in days. `study_date` is the `StudyDate` (for example
  `2012-06-25`). `fold` is the source `Fold` (0 on every row). `split` is `train`, `val` or `test`.
  `physician_id` is the source `PhysicianID`, a 64-character code. The source does not document it.
- Dropped rows: none. A split class keeps only the rows of its own `Split` value (D-09).

## Labels

The 8 label columns, in the order of `cls`:

| Column | Source column | Meaning | Grades |
|---|---|---|---|
| `atelectasis_left` | `Atelectasis_Left` | Atelectasis in the left lung | 0 to 4 |
| `atelectasis_right` | `Atelectasis_Right` | Atelectasis in the right lung | 0 to 4 |
| `heart_size` | `HeartSize` | The heart is large | 0 to 3 |
| `pleural_effusion_left` | `PleuralEffusion_Left` | Left pleural effusion | 0 to 4 |
| `pleural_effusion_right` | `PleuralEffusion_Right` | Right pleural effusion | 0 to 4 |
| `pulmonary_congestion` | `PulmonaryCongestion` | Pulmonary congestion | 0 to 4 |
| `pulmonary_opacities_left` | `PulmonaryOpacities_Left` | Opacities in the left lung | 0 to 4 |
| `pulmonary_opacities_right` | `PulmonaryOpacities_Right` | Opacities in the right lung | 0 to 4 |

The grades of `heart_size` are 0 Normal, 1 Borderline Enlarged, 2 Enlarged and 3 Massively Enlarged.
The grades of the other labels are 0 None, 1 Questionable, 2 Mild, 3 Moderate and 4 Severe.

## Splits

<!-- claim: id=taix_ray.splits.train tier=ci cmd="python scripts/evalkit/dataset_stat.py taix_ray_512 taix_ray_512_train.images" expect="137,593" -->
<!-- claim: id=taix_ray.splits.val tier=ci cmd="python scripts/evalkit/dataset_stat.py taix_ray_512 taix_ray_512_val.images" expect="34,860" -->
<!-- claim: id=taix_ray.splits.test tier=ci cmd="python scripts/evalkit/dataset_stat.py taix_ray_512 taix_ray_512_test.images" expect="42,928" -->
<!-- claim: id=taix_ray.splits.all tier=ci cmd="python scripts/evalkit/dataset_stat.py taix_ray_512 taix_ray_512_train.images+taix_ray_512_val.images+taix_ray_512_test.images" expect="215,381" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `TAIXRay512TrainDataset`, `TAIXRayTrainDataset` | `taix_ray_512_train`, `taix_ray_train` | 137,593 | `Split` is `train` |
| `TAIXRay512ValDataset`, `TAIXRayValDataset` | `taix_ray_512_val`, `taix_ray_val` | 34,860 | `Split` is `val` |
| `TAIXRay512TestDataset`, `TAIXRayTestDataset` | `taix_ray_512_test`, `taix_ray_test` | 42,928 | `Split` is `test`. Use it as the test split |
| `TAIXRay512Dataset`, `TAIXRayDataset` | `taix_ray_512`, `taix_ray` | 215,381 | All images. It mixes the 3 splits, so do not use it for a benchmark (D-09) |

The `512` classes read the 512-pixel PNG files. The other classes read the original PNG files. No
patient is in 2 splits.

`TAIXRay512Dataset` and `TAIXRayDataset` stay so that old code still runs. Their
`get_datasets(n_splits=...)` ignores the `split` column. Their `get_datasets_predefined()` gives the
official train, val and test splits. `get_datasets_predefined()` warns when the val or test transform has
a random step (D-49). On a split class, `get_datasets_predefined()` raises `TypeError`.

## Arguments

The arguments are the same for all 8 dataset classes.

<!-- args: radharmony.dataset.TAIXRay512TrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `images/` folder. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `annotation.csv`. Without it, the class looks in `base_image_dir` and in the folder above it |
| `label_mode` | `str` | No | `"binary"` | `"binary"` gives 0 or 1. `"ordinal"` keeps the grades 0 to 4 (0 to 3 for `heart_size`) |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_report` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The labels stay float32 |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import TAIXRayValHarmonizer

h = TAIXRayValHarmonizer(
    csv_path="/data/TAIX-Ray/data_512/annotation.csv",
    base_image_dir="/data/TAIX-Ray/data_512/images/",
)
df = h.harmonize()  # the raw grades. The dataset class applies label_mode.
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'age_years', 'sex', 'atelectasis_left', ...,
#  'pulmonary_opacities_right', 'age', 'study_date', 'fold', 'split', 'physician_id']
```

`TAIXRayTrainHarmonizer` and `TAIXRayTestHarmonizer` take the same arguments. `TAIXRayHarmonizer`
gives all 3 splits.

<!-- args: radharmony.harmonizer.TAIXRayHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `annotation.csv` |
| `base_image_dir` | `str` | No | `None` | The `images/` folder |

## Known data problems

- Age: `Age` is in days, not in years. Read as years, almost every age is above 110 (D-06).
- Age 0: 55 images have `Age` 0. These are real newborns, not a code for a missing age (D-06).
