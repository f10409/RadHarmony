# RSNA Pneumonia (Kaggle)

<!-- claim: id=rsna_pneumonia_kaggle.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_pneumonia_kaggle rsna_pneumonia_kaggle_train.images" expect="26,684" -->
<!-- claim: id=rsna_pneumonia_kaggle.train.boxes tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_pneumonia_kaggle rsna_pneumonia_kaggle_train.boxes" expect="9,555" -->
<!-- claim: id=rsna_pneumonia_kaggle.train.rows_with_bbox tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_pneumonia_kaggle rsna_pneumonia_kaggle_train.rows_with_bbox" expect="6,012" -->
<!-- claim: id=rsna_pneumonia_kaggle.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_pneumonia_kaggle rsna_pneumonia_kaggle_test.images" expect="3,000" -->
| | |
|---|---|
| Registry keys | `rsna_pneumonia_kaggle_train`, `rsna_pneumonia_kaggle_test` |
| Task and outputs | Classification, 3 labels (`cls`). Boxes (`bbox`) of lung opacities |
| Images in the table | train 26,684 (9,555 boxes on 6,012 images), test 3,000 (no labels) |
| Modality | Chest X-ray, 2-D, DICOM |
| License | The rules of the Kaggle competition. Accept them before the download |
| Download | [Get the data: RSNA Pneumonia (Kaggle release)](../../../how-to/get-the-data.md#rsna-pneumonia-kaggle-release) |

The RSNA Pneumonia Detection Challenge (2018) is a Kaggle competition of the Radiological Society of North
America. The images come from [ChestX-ray14](../cxr-labels/chestxray14.md). Each train image has 1 of the
3 labels, and radiologists drew boxes on the lung opacities.

<!-- claim: id=rsna_pneumonia_kaggle.source.rows tier=nas cmd="python scripts/evalkit/dataset_stat.py --source rsna_pneumonia_kaggle_train csv_path rows" expect="30,227" -->
This page is about the Kaggle stage 2 files: 2 CSV files and 2 folders of DICOM files, with no subfolders.
Each CSV file has one row for each box, and one row for each image with no box. So each has 30,227 rows.
The adjudicated JSON release of the same challenge has its own page: [RSNA Pneumonia](rsna_pneumonia.md).

## Quick use

`base_image_dir` is the `stage_2_train_images/` or `stage_2_test_images/` folder. The train class finds the
2 CSV files in the folder above it.

```
rsna-pneumonia-detection-challenge/
  stage_2_detailed_class_info.csv
  stage_2_train_labels.csv
  stage_2_train_images/<patientId>.dcm
  stage_2_test_images/<patientId>.dcm
```

```python
import torch
from radharmony.dataset import RSNAPneumoniaKaggleTrainDataset, RSNAPneumoniaKaggleTestDataset

train = RSNAPneumoniaKaggleTrainDataset(
    base_image_dir="/data/rsna-pneumonia-detection-challenge/stage_2_train_images/",
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
)
train_ds, val_ds = train.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(3,)
# sample["bbox"]: Tensor(N, 4), N is 0 for an image with no box
# sample["bbox_labels"]: a list of N names

test_ds = RSNAPneumoniaKaggleTestDataset(
    base_image_dir="/data/rsna-pneumonia-detection-challenge/stage_2_test_images/",
    dtype=torch.float32,
).get_datasets()
# test_ds[0] has only "img", for a Kaggle submission
```

## How it is harmonized

- Source: `stage_2_detailed_class_info.csv`, with `patientId` and `class`. It has one row for each box. The
  harmonized table has one row for each image (D-03).
- Test source: the test harmonizer lists the `.dcm` files in `base_image_dir`.
- IDs: the `patientId` `0004cfab-14fd-4e49-80ba-63a80b6bddd6` gives `patient_id` and `study_id` (the same
  text), and `image_path` `0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm`. `image_path` is relative to
  `base_image_dir` (D-04). So one patient is one study and one image.
- Labels (3 columns, D-05): `class` gives 1 in its own column and 0 in the 2 other columns. For example,
  `Lung Opacity` gives `lung_opacity` 1. The source has no blank and no uncertain label.
- View (D-01): from the DICOM tag `ViewPosition`, because the CSV files have no view. It is `PA` or `AP`.
  The D-01 notes have the counts.
- Age and sex: `age_years` comes from the DICOM tag `PatientAge`, in years (D-06). `sex` is `M` or `F`, from
  the DICOM tag `PatientSex` (D-26). The harmonizer reads the DICOM headers only when `base_image_dir` is set.
- Boxes: from `stage_2_train_labels.csv`. See [Boxes](#boxes).
- Dropped rows: the harmonizer drops only the extra rows of an image with 2 or more boxes. It drops no image.

## Labels

The 3 label columns, in the order of `cls`:

| Column | Meaning |
|---|---|
| `lung_opacity` | A lung opacity that can be pneumonia. Only these images have boxes |
| `no_lung_opacity_/_not_normal` | No lung opacity, but the image is not normal |
| `normal` | A normal chest X-ray |

Each image has 1 in exactly 1 column. The test images have no labels (D-25).

## Boxes

- Source: `stage_2_train_labels.csv` has `patientId`, `x`, `y`, `width`, `height` and `Target`. A box is
  in the pixels of the DICOM image. A row with `Target` 0 has no box.
- The harmonizer divides each value by 1024, because every image is 1024 x 1024 (H x W). Example:
  `patientId` `00436515-870c-4b36-a041-de91049b9ab4` has `x` 264, `y` 152, `width` 213 and `height` 379.
  This gives `[0.1484375, 0.5185546875, 0.2578125, 0.4658203125]`.
- The box is `[y_min, y_max, x_min, x_max]`, in fractions of the image size (D-07).
- `bbox_labels` has the name `lung_opacity` for each box. An image with no box has the empty list `[]` in
  both `bbox` and `bbox_labels` (D-07).
- Readers (D-36): the source has one set of boxes for each image, so the harmonizer does not group boxes.
- The sample: `bbox` is a float32 tensor `(N, 4)`, in fractions of the returned image (D-28).
  `bbox_labels` is a list of N names. Example: `(2, 4)` and `['lung_opacity', 'lung_opacity']`. An image
  with no box gives `(0, 4)` and `[]`.
- Dropped boxes: the harmonizer clips each box to the image. It drops a box with no area after the clip, and
  warns (D-07). The source files have no such box.

## Splits

<!-- claim: id=rsna_pneumonia_kaggle.splits.train tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_pneumonia_kaggle rsna_pneumonia_kaggle_train.images" expect="26,684" -->
<!-- claim: id=rsna_pneumonia_kaggle.splits.test tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_pneumonia_kaggle rsna_pneumonia_kaggle_test.images" expect="3,000" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `RSNAPneumoniaKaggleTrainDataset` | `rsna_pneumonia_kaggle_train` | 26,684 | `stage_2_detailed_class_info.csv` |
| `RSNAPneumoniaKaggleTestDataset` | `rsna_pneumonia_kaggle_test` | 3,000 | `stage_2_test_images/*.dcm`, with no labels and no boxes (D-25). Use it only for a Kaggle submission |
| `RSNAPneumoniaKaggleDataset` | `rsna_pneumonia_kaggle` | | The old key. It reads the train files, so it gives the train images. Use the split classes for a benchmark (D-09) |

`get_datasets(n_splits=5)` makes a train and val split from the train class. It groups by patient (D-10). No
patient is in 2 splits.

## Arguments

`RSNAPneumoniaKaggleTestDataset` takes the same arguments, but it ignores `csv_path` and `bbox_csv_path`.
With `harmonizer_path`, the train class also loads a harmonizer file saved under the old name
`RSNAPneumoniaKaggleHarmonizer`.

<!-- args: radharmony.dataset.RSNAPneumoniaKaggleTrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `stage_2_train_images/` or `stage_2_test_images/` folder. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `stage_2_detailed_class_info.csv`. Without it, the class looks in `base_image_dir` and the folder above it. The test class does not use it |
| `bbox_csv_path` | `str` | No | `None` | `stage_2_train_labels.csv`. Without it, the class looks in `base_image_dir` and the folder above it. `output_bbox=True` needs the file. The test class does not use it |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample. In `RSNAPneumoniaKaggleTestDataset`: not supported. The class warns and adds no key (D-25) |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_report` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Add `bbox` and `bbox_labels` to the sample. In `RSNAPneumoniaKaggleTestDataset`: not supported. The class warns and adds no key (D-25) |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The labels stay float32 |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import RSNAPneumoniaKaggleTrainHarmonizer

h = RSNAPneumoniaKaggleTrainHarmonizer(
    csv_path="/data/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv",
    base_image_dir="/data/rsna-pneumonia-detection-challenge/stage_2_train_images/",
    bbox_csv_path="/data/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'view_position', 'bbox', 'bbox_labels', 'age_years', 'sex',
#  'lung_opacity', 'no_lung_opacity_/_not_normal', 'normal']
```

`RSNAPneumoniaKaggleTestHarmonizer` takes only `base_image_dir`. With no `.dcm` file in the folder,
`harmonize()` raises `FileNotFoundError` (D-35).

<!-- args: radharmony.harmonizer.RSNAPneumoniaKaggleTrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `stage_2_detailed_class_info.csv` |
| `base_image_dir` | `str` | No | `None` | The `stage_2_train_images/` folder. Without it, the harmonized table has no `view_position`, `age_years` or `sex` |
| `bbox_csv_path` | `str` | No | `None` | `stage_2_train_labels.csv`. Without it, the harmonized table has no `bbox` |

## Known data problems

- Age: 5 train ages (148 to 155) and 1 test age (412) are above 130, so they are errors. They give NaN (D-06).
