# VinDr-CXR

<!-- claim: id=vindr_cxr.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py vindr-cxr-train vindr_cxr_train.images" expect="15,000" -->
<!-- claim: id=vindr_cxr.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py vindr-cxr-test vindr_cxr_test.images" expect="3,000" -->
<!-- claim: id=vindr_cxr.train.boxes tier=ci cmd="python scripts/evalkit/dataset_stat.py vindr-cxr-train vindr_cxr_train.boxes" expect="8,328" -->
<!-- claim: id=vindr_cxr.test.boxes tier=ci cmd="python scripts/evalkit/dataset_stat.py vindr-cxr-test vindr_cxr_test.boxes" expect="2,697" -->
| | |
|---|---|
| Registry keys | `vindr_cxr_train`, `vindr_cxr_test` |
| Task and outputs | Classification, 28 labels (`cls`). Boxes of 22 findings (`bbox`) |
| Images in the table | train 15,000 (8,328 boxes), test 3,000 (2,697 boxes). One image is one patient |
| Modality | Chest X-ray, 2-D, DICOM |
| License | PhysioNet credentialed access and data use agreement |
| Download | [Get the data: VinDr-CXR](../../../how-to/get-the-data.md#vindr-cxr) |

<!-- claim: id=vindr_cxr.intro.train tier=ci cmd="python scripts/evalkit/dataset_stat.py vindr-cxr-train vindr_cxr_train.images" expect="15,000" -->
<!-- claim: id=vindr_cxr.intro.test tier=ci cmd="python scripts/evalkit/dataset_stat.py vindr-cxr-test vindr_cxr_test.images" expect="3,000" -->
VinDr-CXR is a chest X-ray dataset from Vietnam, with 15,000 train and 3,000 test images. All images are
PA. In train, 3 radiologists labeled each image on their own and drew boxes. In test, 5 radiologists agreed
on one set of labels and boxes. The source has no patient ID.

## Quick use

`base_image_dir` is the `train/` or `test/` folder of `<image_id>.dicom` files. The class finds the label
CSV and the box CSV in `annotations/` of the folder above it (D-35).

```
vindr-cxr/1.0.0/
  annotations/
    image_labels_train.csv
    annotations_train.csv
    image_labels_test.csv
    annotations_test.csv
  train/000434271f63a053c4128a0ba6352c7f.dicom
  test/002a34c58c5b758217ed1f584ccbcfe9.dicom
```

```python
import torch
from radharmony.dataset import VinDrCXRTrainDataset, VinDrCXRTestDataset

train = VinDrCXRTrainDataset(
    base_image_dir="/data/vindr-cxr/1.0.0/train/",
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
)
train_ds, val_ds = train.get_datasets(n_splits=5)  # 1/5 of the images go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(28,)
# sample["bbox"]: Tensor(N, 4), and Tensor(0, 4) for an image with no box
# sample["bbox_labels"]: a list of N names

test_ds = VinDrCXRTestDataset(
    base_image_dir="/data/vindr-cxr/1.0.0/test/",
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
).get_datasets()
```

## How it is harmonized

- Source: `image_labels_train.csv` or `image_labels_test.csv`. The train CSV has 3 rows for each image, one
  for each radiologist (`rad_id`). The test CSV has one row for each image. The harmonized table has one row
  for each image.
- IDs: the source has no patient or study. The `image_id` `e0dc2e79105ad93532484e956ef8a71a` gives
  `patient_id` and `study_id` `e0dc2e79105ad93532484e956ef8a71a`, and `image_path`
  `e0dc2e79105ad93532484e956ef8a71a.dicom`. `image_path` is relative to `base_image_dir` (D-04).
- Labels (28 columns, D-05): train uses a majority vote. A label is 1 when 2 or more of the 3 radiologists
  gave 1. The test labels are 0 or 1 as in the CSV. The harmonizer renames the test column `Other disease`
  to `other_diseases`, as in train.
- View (D-01): all images are PA, so the harmonized table has no `view_position`.
- Age and sex: from the DICOM headers, only with `base_image_dir`. `age_years` comes from `PatientAge`, in
  years (D-06). An age of 0 gives NaN (see Known data problems). `sex` is `M`, `F` or NaN, from
  `PatientSex` (D-26).
- Boxes: see [Boxes](#boxes).
- Dropped rows: the harmonizer drops no row.

## Labels

The 28 label columns, in the order of `cls`. The source has boxes for the 22 findings with "Yes" in the
column Box.

| Column | Meaning | Box |
|---|---|---|
| `aortic_enlargement` | The aorta is wide | Yes |
| `atelectasis` | Part of a lung is collapsed | Yes |
| `calcification` | A calcification | Yes |
| `cardiomegaly` | The heart is large | Yes |
| `clavicle_fracture` | A clavicle fracture | Yes |
| `consolidation` | Airspace consolidation | Yes |
| `copd` | Chronic obstructive pulmonary disease | No |
| `edema` | Pulmonary edema | Yes |
| `emphysema` | Emphysema | Yes |
| `enlarged_pa` | The pulmonary artery is wide | Yes |
| `ild` | Interstitial lung disease | Yes |
| `infiltration` | Infiltration | Yes |
| `lung_cavity` | A cavity in a lung | Yes |
| `lung_cyst` | A cyst in a lung | Yes |
| `lung_opacity` | A lung opacity | Yes |
| `lung_tumor` | A lung tumor | No |
| `mediastinal_shift` | The mediastinum moved to one side | Yes |
| `no_finding` | No finding | No |
| `nodule/mass` | A nodule or a mass | Yes |
| `other_diseases` | Another disease | No |
| `other_lesion` | Another lesion | Yes |
| `pleural_effusion` | Pleural effusion | Yes |
| `pleural_thickening` | Pleural thickening | Yes |
| `pneumonia` | Pneumonia | No |
| `pneumothorax` | Pneumothorax | Yes |
| `pulmonary_fibrosis` | Pulmonary fibrosis | Yes |
| `rib_fracture` | A rib fracture | Yes |
| `tuberculosis` | Tuberculosis | No |

## Boxes

<!-- claim: id=vindr_cxr.boxes.train tier=ci cmd="python scripts/evalkit/dataset_stat.py vindr-cxr-train vindr_cxr_train.boxes" expect="8,328" -->
<!-- claim: id=vindr_cxr.boxes.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py vindr-cxr-train vindr_cxr_train.rows_with_bbox" expect="4,167" -->
<!-- claim: id=vindr_cxr.boxes.test tier=ci cmd="python scripts/evalkit/dataset_stat.py vindr-cxr-test vindr_cxr_test.boxes" expect="2,697" -->
- Source: `annotations_train.csv` and `annotations_test.csv`. One row is one box: `image_id`, `class_name`
  and `x_min`, `y_min`, `x_max`, `y_max` in the pixels of the DICOM image. The train file also has
  `rad_id`. A `No finding` row has no box.
- The harmonizer reads the size of each image file (the DICOM tags `Rows` and `Columns`), so it needs
  `base_image_dir`. It divides `y_min` and `y_max` by the height, and `x_min` and `x_max` by the width.
- The box is `[y_min, y_max, x_min, x_max]`, in fractions of the image size (D-07). Example: the test image
  `e0dc2e79105ad93532484e956ef8a71a` is 2420 x 2148 (H x W). Its `Calcification` box `x_min` 1221.6,
  `y_min` 661.6, `x_max` 1343.3, `y_max` 783.3 gives `[0.2734, 0.3237, 0.5687, 0.6254]`.
- `bbox_labels`: the snake_case name of `class_name`. For example, `Nodule/Mass` gives `nodule/mass` (D-07).
- Readers (D-36), train only: 3 radiologists drew boxes on each image. The harmonizer groups the boxes of one
  image and one finding that overlap (IoU > 0.5). A group with boxes of 2 or more radiologists is one
  lesion, and its box is the mean of the boxes in the group. The harmonizer drops a group with boxes of only
  1 radiologist. So train has 8,328 boxes on 4,167 images.
- Readers (D-36), test: the test file has one agreed set of boxes, so the harmonizer does not group boxes.
  Test has 2,697 boxes on 949 images, and no `Edema` box.
- The sample: `bbox` is a float32 tensor `(N, 4)`, in fractions of the returned image (D-28). `bbox_labels`
  is a list of N names. Example: `(8, 4)` and `['calcification', 'cardiomegaly', 'ild', ...]`. An image with
  no box gives `(0, 4)` and `[]`.
- Dropped boxes: the harmonizer clips each box to the image. It drops a box with no area after the clip, and
  warns (D-07). The source files have no such box. When the harmonizer cannot read the DICOM file of an
  image, the image gets no box, and the harmonizer warns. Without `base_image_dir`, every image gets no box,
  and `harmonize()` warns.

## Splits

<!-- claim: id=vindr_cxr.splits.train tier=ci cmd="python scripts/evalkit/dataset_stat.py vindr-cxr-train vindr_cxr_train.images" expect="15,000" -->
<!-- claim: id=vindr_cxr.splits.test tier=ci cmd="python scripts/evalkit/dataset_stat.py vindr-cxr-test vindr_cxr_test.images" expect="3,000" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `VinDrCXRTrainDataset` | `vindr_cxr_train` | 15,000 | `image_labels_train.csv` |
| `VinDrCXRTestDataset` | `vindr_cxr_test` | 3,000 | `image_labels_test.csv`. Use it as the test split |

`get_datasets(n_splits=5)` makes a train and val split from the train class. It groups by patient (D-10). The
source has no patient ID, so one image is one patient, and nobody can check that a patient is in only 1
split.

## Arguments

The arguments are the same for `VinDrCXRTrainDataset` and `VinDrCXRTestDataset`.

<!-- args: radharmony.dataset.VinDrCXRTrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `train/` or `test/` folder. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `image_labels_train.csv` or `image_labels_test.csv`. Without it, the class looks in `base_image_dir`, the folder above it, and their `annotations/` folders |
| `bbox_csv_path` | `str` | No | `None` | `annotations_train.csv` or `annotations_test.csv`. Without it, the class looks in the same folders. If it finds no file, the harmonized table has no boxes |

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
from radharmony.harmonizer import VinDrCXRTrainHarmonizer

h = VinDrCXRTrainHarmonizer(
    csv_path="/data/vindr-cxr/1.0.0/annotations/image_labels_train.csv",
    bbox_csv_path="/data/vindr-cxr/1.0.0/annotations/annotations_train.csv",
    base_image_dir="/data/vindr-cxr/1.0.0/train/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'bbox', 'bbox_labels', 'age_years', 'sex',
#  'aortic_enlargement', ...]
```

`VinDrCXRTestHarmonizer` takes the same arguments, with the test files and the `test/` folder.

<!-- args: radharmony.harmonizer.VinDrCXRTrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `image_labels_train.csv` or `image_labels_test.csv` |
| `bbox_csv_path` | `str` | No | `None` | `annotations_train.csv` or `annotations_test.csv`. Without it, the harmonized table has no boxes |
| `base_image_dir` | `str` | No | `None` | The `train/` or `test/` folder. The boxes, `age_years` and `sex` come from the DICOM headers in it. Without it, the harmonized table has no `age_years` and no `sex`, and every image gets an empty box list, with a warning |

## Known data problems

<!-- claim: id=vindr_cxr.train.sex_known tier=ci cmd="python scripts/evalkit/dataset_stat.py vindr-cxr-train vindr_cxr_train.sex.F+vindr_cxr_train.sex.M" expect="7,354" -->
<!-- claim: id=vindr_cxr.test.sex_known tier=ci cmd="python scripts/evalkit/dataset_stat.py vindr-cxr-test vindr_cxr_test.sex.F+vindr_cxr_test.sex.M" expect="1,254" -->
- Sex: many headers have `PatientSex` `O` or an empty tag. These give NaN (D-26). Only 7,354 of the train
  images and 1,254 of the test images have `M` or `F`.
- Age: many headers have no `PatientAge` tag, an empty tag, or a bad value such as `Y`. These give NaN
  (D-06). 905 train and 269 test images have the age 0 (`000Y` or `000D`), which means unknown. One train
  age is `238Y`, above 130, so it gives NaN. 4 test ages have no unit letter (for example `39`), so they give
  NaN.
- Column name: the test CSV writes `Other disease`, but the train CSV writes `Other diseases`.
- Box counts: some train radiologists drew several boxes of one finding on one image (D-36). Example: on
  the train image `03e6ecfa6f6fb33dfeac6ca4f9b459c9`, 3 radiologists drew 55 `Nodule/Mass` boxes, and the
  harmonized table keeps 14.
