# VinDr-PCXR

<!-- claim: id=vindr_pcxr.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py vindr_pcxr vindr_pcxr_train.images" expect="7,728" -->
<!-- claim: id=vindr_pcxr.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py vindr_pcxr vindr_pcxr_test.images" expect="1,397" -->
| | |
|---|---|
| Registry keys | `vindr_pcxr_train`, `vindr_pcxr_test` |
| Task and outputs | Classification, 15 labels (`cls`). Boxes of 36 findings (`bbox`) |
| Images in the table | train 7,728, test 1,397. One image is one patient |
| Modality | Chest X-ray of children, 2-D, DICOM |
| License | PhysioNet Restricted Health Data License 1.5.0. Restricted access: a PhysioNet account and a signed data use agreement |
| Download | [Get the data: other datasets](../../../how-to/get-the-data.md#other-datasets) |

VinDr-PCXR is a pediatric chest X-ray dataset from Vietnam
([Pham et al., arXiv:2203.10612](https://arxiv.org/abs/2203.10612), and the
[PhysioNet page](https://physionet.org/content/vindr-pcxr/1.0.0/)). The PhysioNet page says that every patient
is under 10 years old. Radiologists gave each image 15
labels and drew boxes of 36 findings. One radiologist read each image.

## Quick use

`base_image_dir` is the dataset root. The class finds the CSV files in this folder.

```
VINDR-PCXR/
  image_labels_train.csv
  image_labels_test.csv
  annotations_train.csv
  annotations_test.csv
  train/6cb53aff85c71b98ad13d67a131708c6.dicom
  test/d7e71a052a753c3f2f3e317d60177bec.dicom
```

```python
import torch
from radharmony.dataset import VinDrPCXRTrainDataset, VinDrPCXRTestDataset

train = VinDrPCXRTrainDataset(
    base_image_dir="/data/VINDR-PCXR/",
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
)
train_ds, val_ds = train.get_datasets(n_splits=5)  # 1/5 of the images go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(15,)
# sample["bbox"]: Tensor(N, 4), one row for each box. N is 0 for an image with no box
# sample["bbox_labels"]: a list of N names, for example ['reticulonodular_opacity']

test_ds = VinDrPCXRTestDataset(
    base_image_dir="/data/VINDR-PCXR/",
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
).get_datasets()
```

## How it is harmonized

- Source: `image_labels_train.csv` or `image_labels_test.csv`. One row is one image. The boxes come from
  `annotations_train.csv` or `annotations_test.csv`. The image size, age and sex come from the DICOM header.
- IDs: the source and the DICOM headers have no patient ID and no study ID. The `image_id`
  `d7e71a052a753c3f2f3e317d60177bec` of `image_labels_test.csv` gives the same `patient_id` and `study_id`,
  and `image_path` `test/d7e71a052a753c3f2f3e317d60177bec.dicom`. `image_path` is relative to
  `base_image_dir` (D-04).
- Labels (D-05): the 15 label columns of the source hold 0 or 1. The source has no uncertain or blank label.
- View (D-01): the source has no view, so the harmonized table has no `view_position`.
- Age and sex: `age_years` comes from the DICOM tag `PatientAge`, in years (D-06). The tag has a unit:
  `005Y` gives 5, `006M` gives 0.5 and `000D` gives 0. `sex` is `M`, `F` or NaN, from `PatientSex` (D-26).
- Extra columns: `split` (`"train"` or `"test"`), `image_width` and `image_height` (from the DICOM header).
- Dropped rows: the harmonizer drops no row.

## Labels

The 15 label columns, in the order of `cls`:

| Column | Meaning |
|---|---|
| `brocho_pneumonia` | Bronchopneumonia (the source spelling) |
| `bronchiolitis` | Bronchiolitis |
| `bronchitis` | Bronchitis |
| `congenital_emphysema` | Congenital emphysema |
| `cpam` | Congenital pulmonary airway malformation |
| `diagphramatic_hernia` | Diaphragmatic hernia (the source spelling) |
| `hyaline_membrane_disease` | Hyaline membrane disease |
| `lung_tumor` | A lung tumor |
| `mediastinal_tumor` | A mediastinal tumor |
| `no_finding` | No finding |
| `other_disease` | Another disease |
| `pleuro_pneumonia` | Pleuropneumonia |
| `pneumonia` | Pneumonia |
| `situs_inversus` | Situs inversus |
| `tuberculosis` | Tuberculosis |

## Boxes

- Source: `annotations_train.csv` and `annotations_test.csv` have one row for each box, with `image_id`,
  `rad_ID` (the radiologist), `class_name` and `x_min`, `y_min`, `x_max`, `y_max` in pixels. A
  `No finding` row has empty coordinates, so it gives no box.
- The harmonizer reads the size of each image file (`Rows` and `Columns` of the DICOM header), so it needs
  `base_image_dir`. It divides `y_min` and `y_max` by the height, and `x_min` and `x_max` by the width.
- The box is `[y_min, y_max, x_min, x_max]`, in fractions of the image size (D-07).
- Example: the test image `d7e71a052a753c3f2f3e317d60177bec` is 1692 x 1255 (H x W). Its first row
  is `Reticulonodular opacity` with `x_min` 157.3, `y_min` 426.9, `x_max` 487.9 and `y_max` 914.6. It gives
  the box `[0.2523, 0.5405, 0.1254, 0.3887]` (rounded) with the label `reticulonodular_opacity`.
- `bbox_labels`: the snake_case name of `class_name` (D-07). The train split has 36 box labels (the list
  below), and the test split has 23 of them.
- Readers (D-36): one radiologist read each image, so the harmonizer does not group boxes.
- `bbox` and `bbox_labels` are JSON text in the harmonized table. An image with only a `No finding` row gets
  `[]`.
- The sample: `bbox` is a float32 tensor `(N, 4)`, in fractions of the returned image (D-28). `bbox_labels` is
  a list of N names. Example: `(2, 4)` and `['reticulonodular_opacity', 'reticulonodular_opacity']`. An image
  with no box gives `(0, 4)` and `[]`. A train image has up to 10 boxes, and a test image has up to 6.
- Dropped boxes: the harmonizer clips each box to the image. It drops a box with no area after the clip, and
  warns (D-07). The source files have no such box. Also, an image with boxes and a DICOM file that the
  harmonizer cannot read gets no box, and a warning gives the count.

The 36 box labels:

```
anterior_mediastinal_mass, aortic_enlargement, atelectasis, boot-shaped_heart, bronchectasis,
bronchial_thickening, calcification, cardiomegaly, chest_wall_mass, clavicle_fracture, consolidation,
dextro_cardia, diffuse_aveolar_opacity, edema, egg_on_string_sign, emphysema, enlarged_pa,
expanded_edges_of_the_anterior_ribs, infiltration, interstitial_lung_disease_-_ild,
intrathoracic_digestive_structure, lung_cavity, lung_cyst, lung_hyperinflation, mediastinal_shift,
other_lesion, other_nodule/mass, other_opacity, paraveterbral_mass, peribronchovascular_interstitial_opacity,
pleural_effusion, pleural_thickening, pneumothorax, pulmonary_fibrosis, reticulonodular_opacity,
stomach_on_the_right_side
```

## Splits

<!-- claim: id=vindr_pcxr.splits.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py vindr_pcxr vindr_pcxr_train.images" expect="7,728" -->
<!-- claim: id=vindr_pcxr.splits.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py vindr_pcxr vindr_pcxr_test.images" expect="1,397" -->
<!-- claim: id=vindr_pcxr.splits.all.images tier=ci cmd="python scripts/evalkit/dataset_stat.py vindr_pcxr vindr_pcxr_train.images+vindr_pcxr_test.images" expect="9,125" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `VinDrPCXRTrainDataset` | `vindr_pcxr_train` | 7,728 | `image_labels_train.csv` and `train/` |
| `VinDrPCXRTestDataset` | `vindr_pcxr_test` | 1,397 | `image_labels_test.csv` and `test/`. Use it as the test split |
| `VinDrPCXRDataset` | `vindr_pcxr` | 9,125 | All images. It mixes the 2 splits, so do not use it for a benchmark (D-09) |

`get_datasets(n_splits=5)` makes a train and val split from the train class. It groups by patient (D-10). The
source has no patient ID, so one image is one patient, and nobody can check that a patient is in only 1 split.

## Arguments

The arguments are the same for `VinDrPCXRTrainDataset` and `VinDrPCXRTestDataset`.

<!-- args: radharmony.dataset.VinDrPCXRTrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The dataset root with `train/`, `test/` and the 4 CSV files. Not needed with `harmonizer_path`, which holds it |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_report` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Add `bbox` and `bbox_labels` to the sample |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The labels stay float32 |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import VinDrPCXRTrainHarmonizer

h = VinDrPCXRTrainHarmonizer(base_dir="/data/VINDR-PCXR/")
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'bbox', 'bbox_labels', 'age_years', 'sex', 'brocho_pneumonia', ...]
```

`VinDrPCXRTestHarmonizer` takes the same argument.

<!-- args: radharmony.harmonizer.VinDrPCXRTrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_dir` | `str` | Yes | | The dataset root with `train/`, `test/` and the 4 CSV files |

## Known data problems

- Age over 10: the PhysioNet page says that every patient is under 10. But `PatientAge` gives 72 train and 14
  test images an age over 10, for example `011Y`, `019Y` and `035Y`. The harmonized table keeps the tag value
  (D-06).
- No age: 294 train and 59 test images have no age. The tag is missing (185 images), empty (167) or `Y` with no
  number (1).
- Age 0: 150 images have `000D`, which gives 0. A newborn is a real patient, so the harmonized table keeps it
  (D-06).
- Sex `O`: 46 train and 16 test images have the sex `O`, which gives NaN (D-26).
- Spelling: the source spells some names wrong, for example `Brocho-pneumonia`, `Diagphramatic hernia`,
  `Bronchectasis` and `Paraveterbral mass`. The harmonized names keep the source spelling.
