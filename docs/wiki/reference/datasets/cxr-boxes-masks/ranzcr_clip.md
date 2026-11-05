# RANZCR CLiP

<!-- claim: id=ranzcr_clip.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py ranzcr_clip ranzcr_clip_train.images" expect="30,083" -->
<!-- claim: id=ranzcr_clip.train.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py ranzcr_clip ranzcr_clip_train.patients" expect="3,255" -->
<!-- claim: id=ranzcr_clip.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py ranzcr_clip ranzcr_clip_test.images" expect="3,582" -->
<!-- claim: id=ranzcr_clip.train.masks tier=ci cmd="python scripts/evalkit/dataset_stat.py ranzcr_clip ranzcr_clip_train.rows_with_mask" expect="9,095" -->
| | |
|---|---|
| Registry keys | `ranzcr_clip_train`, `ranzcr_clip_test` |
| Task and outputs | Classification, 11 labels (`cls`). Line masks (`mask`) with the train class |
| Images in the table | train 30,083 (3,255 patients, 9,095 with line masks), test 3,582 (no labels) |
| Modality | Chest X-ray, 2-D, JPEG |
| License | The rules of the Kaggle competition. Accept them before the download |
| Download | [Get the data: RANZCR CLiP](../../../how-to/get-the-data.md#ranzcr-clip) |

<!-- claim: id=ranzcr_clip.annotations.rows tier=nas cmd="python scripts/evalkit/dataset_stat.py --source ranzcr_clip_train annotations_csv_path rows" expect="17,999" -->
RANZCR CLiP (Catheter and Line Position) is a Kaggle competition of the Royal Australian and New Zealand
College of Radiologists. The images show endotracheal tubes (ETT), nasogastric tubes (NGT), central venous
catheters (CVC) and Swan-Ganz catheters. Most of the 11 labels say if the position of a tube or line is
normal, borderline or abnormal. For some train images, `train_annotations.csv` has 17,999 polylines that trace
the tubes and lines. The harmonizer can draw them as a binary mask. The Kaggle test images have no labels.

## Quick use

`base_image_dir` is the Kaggle folder. The class finds `train.csv` and `train_annotations.csv` in it.

```
ranzcr-clip-catheter-line-classification/
  train.csv
  train_annotations.csv
  sample_submission.csv
  train/<StudyInstanceUID>.jpg
  test/<StudyInstanceUID>.jpg
```

```python
import torch
from radharmony.dataset import RANZCRClipTrainDataset, RANZCRClipTestDataset

train = RANZCRClipTrainDataset(
    base_image_dir="/data/ranzcr-clip-catheter-line-classification/",
    output_cls=True,
    dtype=torch.float32,
)
train_ds, val_ds = train.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(11,)

test_ds = RANZCRClipTestDataset(
    base_image_dir="/data/ranzcr-clip-catheter-line-classification/",
    dtype=torch.float32,
).get_datasets()
# test_ds[0] has only "img", for a Kaggle submission
```

## How it is harmonized

- Source: `train.csv`. One row is one image. The test harmonizer lists the `.jpg` files in `test/`.
- IDs: the row with `PatientID` `ec89415d1` gives `patient_id` `ec89415d1`. Its
  `StudyInstanceUID` `1.2.826.0.1.3680043.8.498.26697628953273228189375557799582420561` gives
  `study_id` (the same text). So one study is one image. The `StudyInstanceUID` also gives
  `image_path` `train/1.2.826.0.1.3680043.8.498.26697628953273228189375557799582420561.jpg`,
  relative to `base_image_dir` (D-04).
- Test IDs: `patient_id` is the `StudyInstanceUID`, because the test rows have no patient ID.
- Labels (11 columns, D-05): the harmonizer renames each column, for example `CVC - Abnormal` gives
  `cvc_abnormal`. The values stay 0 and 1. The source has no blank and no uncertain label.
- View, age and sex: the source has none of them, so the harmonized table has no `view_position`,
  `age_years` or `sex` (D-01).
- Masks: see [Masks](#masks).
- Dropped rows: the harmonizer drops no row.

## Labels

The 11 label columns, in the order of `cls`:

| Column | Meaning |
|---|---|
| `cvc_abnormal` | A CVC in an abnormal position |
| `cvc_borderline` | A CVC in a borderline position |
| `cvc_normal` | A CVC in a normal position |
| `ett_abnormal` | An ETT in an abnormal position |
| `ett_borderline` | An ETT in a borderline position |
| `ett_normal` | An ETT in a normal position |
| `ngt_abnormal` | An NGT in an abnormal position |
| `ngt_borderline` | An NGT in a borderline position |
| `ngt_incompletely_imaged` | An NGT that the image does not fully show |
| `ngt_normal` | An NGT in a normal position |
| `swan_ganz_catheter_present` | A Swan-Ganz catheter is in the image |

An image can have more than 1 label, for example `cvc_normal` and `ett_normal`.

## Masks

With `output_mask=True`, `RANZCRClipTrainDataset` adds a line mask to the sample. It needs
`mask_output_dir`, a folder for the mask PNG files.

```python
import torch
from radharmony.dataset import RANZCRClipTrainDataset

mask_ds = RANZCRClipTrainDataset(
    base_image_dir="/data/ranzcr-clip-catheter-line-classification/",
    output_cls=True,
    output_mask=True,
    mask_output_dir="/data/ranzcr_masks/",  # any folder you can write to
    dtype=torch.float32,
).get_datasets()
sample = mask_ds[0]
# sample["mask"]: Tensor(1, 224, 224), values 0 and 1
```

- Source: `train_annotations.csv` has one row for each polyline, with `StudyInstanceUID`, `label` and
  `data`. `data` is a list of `[x, y]` points in the pixels of the JPEG, for example
  `[[1487, 1279], [1477, 1168], ...]`.
- The harmonizer puts all polylines of an image into one JSON text in `mask_path`. The line labels are
  not in the mask. Without `mask_output_dir`, `mask_path` keeps this JSON text.
- `harmonize(mask_output_dir=...)` draws each polyline with `cv2.polylines` as a line of
  `mask_line_thickness` pixels (default 15) on an image of the JPEG size. It saves one PNG for each
  image and puts the PNG path in `mask_path`.
- An image with no polyline has an empty `mask_path`. An empty `mask_path` means "not annotated" (D-33).
  With `output_mask=True`, the class drops these rows and warns. So 9,095 images stay.
- The sample: `mask` is a float32 tensor `(1, 224, 224)` with the values 0 and 1 (D-14). The transform
  resizes it with the image.

## Splits

<!-- claim: id=ranzcr_clip.splits.train tier=ci cmd="python scripts/evalkit/dataset_stat.py ranzcr_clip ranzcr_clip_train.images" expect="30,083" -->
<!-- claim: id=ranzcr_clip.splits.test tier=ci cmd="python scripts/evalkit/dataset_stat.py ranzcr_clip ranzcr_clip_test.images" expect="3,582" -->
<!-- claim: id=ranzcr_clip.splits.all tier=ci cmd="python scripts/evalkit/dataset_stat.py ranzcr_clip ranzcr_clip_train.images+ranzcr_clip_test.images" expect="33,665" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `RANZCRClipTrainDataset` | `ranzcr_clip_train` | 30,083 | `train.csv` |
| `RANZCRClipTestDataset` | `ranzcr_clip_test` | 3,582 | `test/*.jpg`, with no labels and no masks (D-25). Use it only for a Kaggle submission |
| `RANZCRClipDataset` | `ranzcr_clip` | 33,665 | All images. It mixes the 2 splits, so do not use it for a benchmark (D-09) |

`get_datasets(n_splits=5)` makes a train and val split from the train class. It groups by patient (D-10).

`RANZCRClipDataset` adds the test images by default (`include_test_split=True`), with NaN labels and a
`split` column. With `output_cls=True`, a test sample raises `ValueError` when it loads.
`RANZCRClipDataset(...).get_datasets_predefined()` returns the train and test datasets. The split classes
hold one split, so their `get_datasets_predefined()` raises `TypeError`.

## Arguments

`RANZCRClipTestDataset` takes only `base_image_dir` and the shared arguments.

<!-- args: radharmony.dataset.RANZCRClipTrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The Kaggle folder with `train/`, `test/` and the CSV files. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `train.csv`. Without it, the class looks in `base_image_dir` and the folder above it |
| `annotations_csv_path` | `str` | No | `None` | `train_annotations.csv`. Without it, the class looks in `base_image_dir` and the folder above it. `output_mask=True` needs the file |
| `mask_output_dir` | `str` | No | `None` | The folder for the mask PNG files. Needed with `output_mask=True`. Without it, the class raises `ValueError` |
| `mask_num_cores` | `int` | No | `1` | The number of threads that write the mask PNG files |
| `mask_line_thickness` | `int` | No | `15` | The width of a drawn line, in pixels of the JPEG |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample. In `RANZCRClipTestDataset`: not supported. The class warns and adds no key (D-25) |
| `output_mask` | `bool` | No | `False` | Add the `mask` tensor to the sample. It needs `mask_output_dir`. The class drops the rows with an empty `mask_path`, and warns (D-33). In `RANZCRClipTestDataset`: not supported. The class warns and adds no key (D-25) |
| `output_report` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11). The polylines are lines, not boxes |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The labels stay float32 |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import RANZCRClipTrainHarmonizer

h = RANZCRClipTrainHarmonizer(
    csv_path="/data/ranzcr-clip-catheter-line-classification/train.csv",
    base_image_dir="/data/ranzcr-clip-catheter-line-classification/",
    annotations_csv_path="/data/ranzcr-clip-catheter-line-classification/train_annotations.csv",
)
df = h.harmonize()  # harmonize(mask_output_dir=...) also writes the mask PNG files
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'mask_path', 'cvc_abnormal', ...]
```

`RANZCRClipTestHarmonizer` takes only `base_image_dir`. With no `.jpg` file in `test/`, `harmonize()`
raises `FileNotFoundError` (D-35).

<!-- args: radharmony.harmonizer.RANZCRClipTrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `train.csv` |
| `base_image_dir` | `str` | No | `None` | The Kaggle folder. The mask step reads the JPEG sizes from it |
| `annotations_csv_path` | `str` | No | `None` | `train_annotations.csv`. Without it, the harmonized table has no `mask_path` |
| `mask_line_thickness` | `int` | No | `15` | The width of a drawn line, in pixels of the JPEG |

## Known data problems

- Test patients: the test images have no `PatientID`. So nobody can check if a test patient is also in train.
- Lines with no polyline: 24 images have polylines, but one of their positive labels has no polyline. For
  example, an image has 1 in `CVC - Normal`, but none of its polylines has the label `CVC - Normal`. So the
  mask does not show that line.
