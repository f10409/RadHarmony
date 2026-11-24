# QaTa-COV19

<!-- claim: id=qata_cov19.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py qata_cov19 qata_cov19_train.images" expect="7,145" -->
<!-- claim: id=qata_cov19.train.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py qata_cov19 qata_cov19_train.patients" expect="4,607" -->
<!-- claim: id=qata_cov19.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py qata_cov19 qata_cov19_test.images" expect="2,113" -->
| | |
|---|---|
| Registry keys | `qata_cov19_train`, `qata_cov19_test` |
| Task and outputs | Segmentation of the COVID-19 infection (`mask`). 1 label, always 1 (`cls`) |
| Images in the table | train 7,145 (4,607 patient IDs), test 2,113 (474 patients) |
| Modality | Chest X-ray, 2-D, PNG, 8-bit gray, 224 x 224 (H x W) |
| License | The source files have no license file. See the Kaggle page of the dataset |
| Download | [Get the data: other datasets](../../../how-to/get-the-data.md#other-datasets) |

QaTa-COV19 is a COVID-19 chest X-ray dataset from Qatar University and Tampere University. The papers are
[Degerli et al., Health Inf Sci Syst 2021](https://doi.org/10.1007/s13755-021-00146-8) and
[Yamac et al., Comput Biol Med 2021](https://doi.org/10.1016/j.compbiomed.2021.104834).

<!-- claim: id=qata_cov19.all.images tier=ci cmd="python scripts/evalkit/dataset_stat.py qata_cov19 qata_cov19_train.images+qata_cov19_test.images" expect="9,258" -->
RadHarmony reads version 2 (v2). It has 9,258 COVID-19 images, and each image has a mask of the infected
lung region. The first author put the data on Kaggle (`aysendegerli/qatacov19-dataset`).

The download also has `QaTa-COV19-v1/`, `Control_Group/` and `Early-QaTa-COV19/`. No class reads them. The
v1 images with a mask are the `covid_` images of the v2 train split. The control groups hold ChestX-ray14
images (a RadHarmony dataset with its own classes), pediatric pneumonia images and a set of normal images.
`Early-QaTa-COV19/` is a part of v1.

## Quick use

`base_image_dir` is the `Images/` folder of one split. The class finds the masks in the `Ground-truths/`
folder next to it.

```
QaTa-COV19/
  readme.txt
  QaTa-COV19-v2/
    Train Set/Images/covid_1000.png
    Train Set/Ground-truths/mask_covid_1000.png
    Test Set/Images/sub-S08047_ses-E16280_run-1_bp-chest_vp-ap_dx.png
    Test Set/Ground-truths/mask_sub-S08047_ses-E16280_run-1_bp-chest_vp-ap_dx.png
```

```python
import torch
from radharmony.dataset import QaTaCOV19TrainDataset, QaTaCOV19TestDataset

train = QaTaCOV19TrainDataset(
    base_image_dir="/data/QaTa-COV19/QaTa-COV19-v2/Train Set/Images/",
    output_cls=True,
    output_mask=True,
    dtype=torch.float32,
)
train_ds, val_ds = train.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(1,), always 1
# sample["mask"]: Tensor(1, 224, 224), values 0 and 1

test_ds = QaTaCOV19TestDataset(
    base_image_dir="/data/QaTa-COV19/QaTa-COV19-v2/Test Set/Images/",
    output_cls=True,
    output_mask=True,
    dtype=torch.float32,
).get_datasets()
```

## How it is harmonized

- Source: the PNG files in `base_image_dir`. One file is one image. The source has no CSV. All columns come
  from the file name and the `Ground-truths/` folder.
- IDs: the file `sub-S08047_ses-E16280_run-1_bp-chest_vp-ap_dx.png` gives `patient_id` `S08047`, `study_id`
  `E16280` and `image_path` `sub-S08047_ses-E16280_run-1_bp-chest_vp-ap_dx.png`. A `covid_` file has no
  subject, so `covid_1000.png` gives `patient_id` and `study_id` `covid_1000`. `image_path` is relative to
  `base_image_dir` (D-04).
- Name scheme: `filename_scheme` is `bids` for a `sub-` file and `qata` for a `covid_` file. All `covid_`
  files are in the train split.
- Labels (D-05): `covid` is 1 on every row, because every v2 image is COVID-19 positive. It is not a
  classification task alone. Use it with a dataset of negative images.
- View (D-01): from the `vp-` part of a `sub-` name. `vp-ap` gives `AP`, `vp-pa` gives `PA` and `vp-ll` gives
  `LATERAL`. A name with no `vp-` part, and every `covid_` name, gives NaN.
- `acquisition`: `CR` (computed radiography) or `DX` (digital radiography), from the end of a `sub-` name.
  A `covid_` name gives NaN.
- Age and sex: the source has neither, so the harmonized table has no `age_years` or `sex`.
- Dropped rows: the harmonizer drops no row.

## Labels

The 1 label column of `cls`:

| Column | Meaning |
|---|---|
| `covid` | COVID-19 pneumonia. It is 1 on every image |

## Masks

- Source: the `Ground-truths/` folder of each split has one PNG mask for each image, named
  `mask_<image file name>`. A mask is 224 x 224 (H x W), with 0 for the background and 255 for the infected region.
- `mask_path` is the full path of the mask file, for example
  `/data/QaTa-COV19/QaTa-COV19-v2/Test Set/Ground-truths/mask_sub-S08047_ses-E16280_run-1_bp-chest_vp-ap_dx.png`.
  It is not relative to `base_image_dir`, because the class adds `base_image_dir` only to `image_path`.
- The masks are PNG files, so they need no decode step. So the class needs no `mask_output_dir`.
- An image with no mask file gets an empty `mask_path`. An empty `mask_path` means "not annotated" (D-33). With
  `output_mask=True`, the class drops these rows and warns. In the source files, every image has a mask.
- The sample: `mask` is a float32 tensor `(1, 224, 224)` with the values 0 and 1 (D-14). The transform resizes
  it with the image.

## Splits

<!-- claim: id=qata_cov19.splits.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py qata_cov19 qata_cov19_train.images" expect="7,145" -->
<!-- claim: id=qata_cov19.splits.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py qata_cov19 qata_cov19_test.images" expect="2,113" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `QaTaCOV19TrainDataset` | `qata_cov19_train` | 7,145 | `Train Set/Images/` |
| `QaTaCOV19TestDataset` | `qata_cov19_test` | 2,113 | `Test Set/Images/`. Use it as the test split |
| `QaTaCOV19Dataset` | `qata_cov19` | | The old key. It reads the split of the folder in `base_image_dir`. Use the split classes for a benchmark (D-09) |

A split class needs a `base_image_dir` in its own `Train Set/` or `Test Set/` folder. With the folder of the
other split, `get_datasets()` raises `ValueError`. The constructor does not check the folder.

`get_datasets(n_splits=5)` makes a train and val split from the train class. It groups by patient (D-10).
Each `covid_` image is its own patient, because its name has no subject. No patient is in 2 splits. But 1
test image is also a train image (see [Known data problems](#known-data-problems)). The split stays as
published (D-34).

## Arguments

The arguments are the same for `QaTaCOV19TrainDataset` and `QaTaCOV19TestDataset`.

<!-- args: radharmony.dataset.QaTaCOV19TrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `Images/` folder of the split. Not needed with `harmonizer_path`, which holds it |
| `masks_dir` | `str` | No | `None` | The `Ground-truths/` folder. `None` gives the `Ground-truths/` folder next to `base_image_dir` |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample |
| `output_mask` | `bool` | No | `False` | Add the `mask` tensor to the sample. The class drops the rows with no mask, and warns (D-33) |
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
from radharmony.harmonizer import QaTaCOV19TrainHarmonizer

h = QaTaCOV19TrainHarmonizer(base_dir="/data/QaTa-COV19/QaTa-COV19-v2/Train Set/Images/")
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'view_position', 'mask_path', 'covid', 'filename_scheme', 'acquisition']
```

`QaTaCOV19TestHarmonizer` takes the same arguments, with the `Test Set/Images/` folder.

<!-- args: radharmony.harmonizer.QaTaCOV19TrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_dir` | `str` | Yes | | The `Images/` folder of the split. A folder of the other split raises `ValueError` |
| `masks_dir` | `str` | No | `None` | The `Ground-truths/` folder. `None` gives the `Ground-truths/` folder next to `base_dir` |

## Known data problems

- Image source: the source does not document where the `sub-` images came from. So `filename_scheme` gives
  only the name scheme.
- No subject: a `covid_` name has no subject, session or view. So 2 `covid_` images of one patient get 2
  patient IDs, and our own split can put them in train and val (D-10).
- Copies in train: 37 pairs of train files look like one picture under 2 names (correlation above 0.99 at
  32 x 32 pixels). Example: `covid_104.png` and `covid_2791.png`. Each copy has its own `patient_id`, so our own
  split can put the 2 copies in train and val (D-10).
- Copy in test: the test image `sub-S09512_ses-E21506_run-1_bp-chest_vp-pa_dx.png` is also in train, as
  `covid_4134.png` (correlation 0.9999). The split stays as published (D-34).
- Image size: the v2 images are 224 x 224 (H x W). The source has no larger copy of them.
