# RSNA Pediatric Bone Age

<!-- claim: id=rsna_bone_age.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_bone_age rsna_bone_age_train.images" expect="12,611" -->
<!-- claim: id=rsna_bone_age.train.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_bone_age rsna_bone_age_train.patients" expect="12,611" -->
<!-- claim: id=rsna_bone_age.val.images tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_bone_age rsna_bone_age_val.images" expect="1,425" -->
<!-- claim: id=rsna_bone_age.val.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_bone_age rsna_bone_age_val.patients" expect="1,425" -->
| | |
|---|---|
| Registry keys | `rsna_bone_age_train`, `rsna_bone_age_val`, `rsna_bone_age_test` |
| Task and outputs | Regression of the bone age in months (`reg`). The test split has images only |
| Images in the table | train 12,611, val 1,425, test 200. One image is one patient |
| Modality | Hand X-ray, 2-D, PNG |
| License | The terms of the RSNA download page |
| Download | [Get the data: RSNA Pediatric Bone Age](../../../how-to/get-the-data.md#rsna-pediatric-bone-age) |

<!-- claim: id=rsna_bone_age.train.images.again tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_bone_age rsna_bone_age_train.images" expect="12,611" -->
<!-- claim: id=rsna_bone_age.val.images.again tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_bone_age rsna_bone_age_val.images" expect="1,425" -->
The RSNA Pediatric Bone Age Challenge (2017) has hand X-rays of children
([Halabi et al., Radiology 2019](https://doi.org/10.1148/radiol.2018180736)). The task is to predict the bone
age (the age of the skeleton) from the image. The target is the bone age in months, a regression target, not
a label. The train split has 12,611 images and the val split has 1,425. The 200 test images have no bone age.

## Quick use

Train and val are 2 downloads, and each has its own `base_image_dir`. The train class finds `train.csv` in
the folder above `base_image_dir`. The val class finds `Validation Dataset.csv` in `base_image_dir`.

```
rsna-boneage/
  train.csv
  boneage-training-dataset/<id>.png
  Bone Age Validation Set/
    Validation Dataset.csv
    boneage-validation-dataset-1/<id>.png   # 800 images, ids 1386 to 9708
    boneage-validation-dataset-2/<id>.png   # 625 images, ids 9727 to 15612
```

```python
import torch
from radharmony.dataset import RSNABoneAgeTrainDataset, RSNABoneAgeValDataset

train = RSNABoneAgeTrainDataset(
    base_image_dir="/data/rsna-boneage/boneage-training-dataset/",
    output_reg=True,
    dtype=torch.float32,
)
train_ds, val_ds = train.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["reg"]: Tensor(1,), the bone age in months, for example tensor([180.])

test_ds = RSNABoneAgeValDataset(
    base_image_dir="/data/rsna-boneage/Bone Age Validation Set/",
    output_reg=True,
    dtype=torch.float32,
).get_datasets()
```

## How it is harmonized

- Source: `train.csv` has the columns `id`, `boneage` and `male`. `Validation Dataset.csv` has `Image ID`,
  `Bone Age (months)` and `male`. The harmonizer renames the val columns to the train names. One row is one
  image.
- IDs: the train `id` `1377` gives `patient_id` `1377`, `study_id` `1377` and `image_path` `1377.png`. A val
  image path has its subfolder, for example `boneage-validation-dataset-1/1386.png`. `image_path` is relative
  to `base_image_dir` (D-04). The source has no patient ID, so one image is one patient.
- Target (D-05): `age_months` comes from `boneage` or `Bone Age (months)`, as a whole number. It is in
  `REG_COLS`, not in `LABEL_COLS`.
- Sex: `sex` is `M` or `F`, from the True or False column `male` (D-26). The extra column `sex_female` (1 for
  a girl and 0 for a boy) stays, because bone-age models use it as a number.
- Age: the source has no age of the child, so the harmonized table has no `age_years` (D-06). `age_months`
  is the bone age, not the age of the child.
- View (D-01): the source has no view, so the harmonized table has no `view_position`.
- Dropped rows: the harmonizer drops no row. Every id in the 2 CSV files has a PNG.

## Labels

The class has no label columns, so `cls` is not supported. `reg` holds the one regression column:

| Column | Meaning |
|---|---|
| `age_months` | The bone age in months, from the source CSV. It is from 1 to 228 in the train split |

## Splits

<!-- claim: id=rsna_bone_age.splits.train tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_bone_age rsna_bone_age_train.images" expect="12,611" -->
<!-- claim: id=rsna_bone_age.splits.val tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_bone_age rsna_bone_age_val.images" expect="1,425" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `RSNABoneAgeTrainDataset` | `rsna_bone_age_train` | 12,611 | `train.csv` and `boneage-training-dataset/` |
| `RSNABoneAgeValDataset` | `rsna_bone_age_val` | 1,425 | `Validation Dataset.csv` and its 2 image folders. Use it as the test split |
| `RSNABoneAgeTestDataset` | `rsna_bone_age_test` | 200 | The PNG files in `base_image_dir`, with no bone age (D-25). Use it only to make predictions |
| `RSNABoneAgeDataset` | `rsna_bone_age` | | The old key. It reads the split that `split` names (`"train"` by default, or `"val"`). Use the split classes for a benchmark (D-09) |

`get_datasets(n_splits=5)` makes a train and val split from the train class. It groups by patient (D-10). The
source has no patient ID, so one image is one patient, and nobody can check that a patient is in only 1 split.
No id is in both the train and the val CSV.

The test class lists the `<id>.png` files in `base_image_dir`, for example a folder `boneage-test-dataset/`.
Its sample has only `img`. No RadHarmony test reads the real test files. A test checks the harmonized test
table on 2 fake PNG files.

## Arguments

The arguments are the same for `RSNABoneAgeTrainDataset` and `RSNABoneAgeValDataset`.
`RSNABoneAgeTestDataset` takes no `csv_path` and no `split`.

<!-- args: radharmony.dataset.RSNABoneAgeTrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | `boneage-training-dataset/` or `Bone Age Validation Set/`. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `train.csv` or `Validation Dataset.csv`. Without it, the train class looks in the folder above `base_image_dir`, and the val class looks in `base_image_dir` |
| `split` | `str` | No | The split of the class (`"train"` or `"val"`) | The class always reads its own split. Another value raises `ValueError` |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11). The class has no label columns |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_report` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_reg` | `bool` | No | `False` | Add the `reg` tensor (the bone age in months) to the sample. In `RSNABoneAgeTestDataset`: not supported. The class warns and adds no key (D-25) |
| `output_bbox` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. `reg` stays float32 |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import RSNABoneAgeTrainHarmonizer

h = RSNABoneAgeTrainHarmonizer(
    base_image_dir="/data/rsna-boneage/boneage-training-dataset/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'sex', 'age_months', 'sex_female']
```

`RSNABoneAgeValHarmonizer` takes the same arguments, with the val folder. `RSNABoneAgeTestHarmonizer` takes
only `base_image_dir`. Its harmonized table has `patient_id`, `study_id` and `image_path`.

<!-- args: radharmony.harmonizer.RSNABoneAgeTrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | | The image folder of the split. A folder that does not exist raises `ValueError` |
| `csv_path` | `str` | No | `None` | `train.csv` or `Validation Dataset.csv`. Without it, the harmonizer looks next to `base_image_dir` (train) or in it (val), and raises `FileNotFoundError` when the file is not there |

## Known data problems

None known.
