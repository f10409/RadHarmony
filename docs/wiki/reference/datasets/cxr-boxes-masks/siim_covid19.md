# SIIM COVID-19 Detection

<!-- claim: id=siim_covid19.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py siim_covid19 siim_covid19_train.images" expect="6,334" -->
<!-- claim: id=siim_covid19.train.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py siim_covid19 siim_covid19_train.patients" expect="3,261" -->
<!-- claim: id=siim_covid19.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py siim_covid19 siim_covid19_test.images" expect="1,263" -->
<!-- claim: id=siim_covid19.test.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py siim_covid19 siim_covid19_test.patients" expect="652" -->
| | |
|---|---|
| Registry keys | `siim_covid19_train`, `siim_covid19_test` |
| Task and outputs | Classification, 4 labels for each study (`cls`). Opacity boxes (`bbox`). The test split has images only |
| Images in the table | train 6,334 (3,261 patients), test 1,263 (652 patients) |
| Modality | Chest X-ray, 2-D, DICOM |
| License | The rules of the Kaggle competition. Accept them before the download |
| Download | [Get the data: SIIM COVID-19](../../../how-to/get-the-data.md#siim-covid-19) |

<!-- claim: id=siim_covid19.train.images.again tier=ci cmd="python scripts/evalkit/dataset_stat.py siim_covid19 siim_covid19_train.images" expect="6,334" -->
<!-- claim: id=siim_covid19.train.studies tier=ci cmd="python scripts/evalkit/dataset_stat.py siim_covid19 siim_covid19_train.studies" expect="6,054" -->
<!-- claim: id=siim_covid19.test.images.again tier=ci cmd="python scripts/evalkit/dataset_stat.py siim_covid19 siim_covid19_test.images" expect="1,263" -->
SIIM-FISABIO-RSNA COVID-19 Detection is a 2021 Kaggle challenge. The train split has 6,334 images of
6,054 studies. The 1,263 test images have no labels and no boxes.

<!-- claim: id=siim_covid19.train.rows_with_bbox tier=ci cmd="python scripts/evalkit/dataset_stat.py siim_covid19 siim_covid19_train.rows_with_bbox" expect="4,294" -->
<!-- claim: id=siim_covid19.train.boxes tier=ci cmd="python scripts/evalkit/dataset_stat.py siim_covid19 siim_covid19_train.boxes" expect="7,853" -->
The source gives labels at 2 levels. Each study has 1 of 4 study labels, for example "Typical
Appearance". Each image has an image label: `opacity` with 1 or more boxes, or `none`. 4,294 train images
have 7,853 opacity boxes.

## Quick use

`base_image_dir` is the `train/` or `test/` folder. The train class finds `train_study_level.csv` and
`train_image_level.csv` in `base_image_dir` or in the folder above it.

```
siim-covid19-detection/
  train_study_level.csv
  train_image_level.csv
  train/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
  test/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
```

```python
import torch
from radharmony.dataset import SIIMCOVID19TrainDataset, SIIMCOVID19TestDataset

train = SIIMCOVID19TrainDataset(
    base_image_dir="/data/siim-covid19-detection/train/",
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
)
train_ds, val_ds = train.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(4,), 1 for the study label and 0 for the other 3
# sample["bbox"]: Tensor(2, 4), one row for each box
# sample["bbox_labels"]: a list of N names, here ['opacity', 'opacity']

test_ds = SIIMCOVID19TestDataset(
    base_image_dir="/data/siim-covid19-detection/test/",
    dtype=torch.float32,
).get_datasets()
# test_ds[0] has only "img", for a Kaggle submission
```

## How it is harmonized

- Source: `train_image_level.csv` (one row for each image) and `train_study_level.csv` (one row for each
  study). One row is one image (D-03). Each image gets the labels of its study.
- IDs: `study_id` is the `StudyInstanceUID` of the image CSV. The harmonizer finds the file of each image
  in the folders of `base_image_dir`. `patient_id` is the DICOM tag `PatientID`, or `study_id` when the
  tag is empty. For example, `patient_id` `dcc3fdf6426a`, `study_id` `5776db0cec75` and `image_path`
  `5776db0cec75/81456c9c5423/000a312787f2.dcm`. `image_path` is relative to `base_image_dir` (D-04).
- Labels (D-05): the study CSV has 4 columns with 0 or 1, and each study has exactly one 1. The study `id`
  `5776db0cec75_study` gives the `StudyInstanceUID` `5776db0cec75`. The column names become snake_case,
  for example `Typical Appearance` gives `typical_appearance`.
- Image label: the harmonizer does not keep the column `label`. An image with the label `none` has no box,
  so its `bbox` is an empty list.
- View (D-01): the source has no view, so the harmonized table has no `view_position`.
- Age and sex: `sex` is `M` or `F`, from the DICOM tag `PatientSex` (D-26). The DICOM files have no
  `PatientAge`, so the harmonized table has no `age_years` (D-06).
- Dropped rows: the harmonizer drops a row of the image CSV with no DICOM file, and prints the count. With
  the source files, it drops no row.

## Labels

The 4 label columns, in the order of `cls`:

| Column | Meaning |
|---|---|
| `atypical_appearance` | Atypical appearance for COVID-19 pneumonia |
| `indeterminate_appearance` | Indeterminate appearance for COVID-19 pneumonia |
| `negative_for_pneumonia` | No pneumonia |
| `typical_appearance` | Typical appearance for COVID-19 pneumonia |

The labels are study labels. All images of one study have the same `cls`. No image of a
`negative_for_pneumonia` study has a box. 304 images of the other studies have no box either.

## Boxes

- Source: the column `boxes` of `train_image_level.csv`. It is the text of a Python list of
  `{x, y, width, height}` dicts, in the pixels of the DICOM image. It is empty for an image with no box.
- The harmonizer reads `Rows` and `Columns` of each DICOM header, so it needs `base_image_dir`. It divides
  `y` and `height` by `Rows` (the height), and `x` and `width` by `Columns` (the width).
- The box is `[y_min, y_max, x_min, x_max]`, in fractions of the image size (D-07).
- For example, image `000a312787f2` is 3488 x 4256 pixels (H x W). The source box
  `{'x': 789.28836, 'y': 582.43035, 'width': 1026.65662, 'height': 1917.30292}` gives
  `[0.167, 0.717, 0.185, 0.427]`.
- `bbox_labels` has `opacity` for each box (D-07).
- Readers (D-36): the source has one set of boxes for each image, so the harmonizer does not group boxes.
- The sample: `bbox` is a float32 tensor `(N, 4)`, in fractions of the returned image (D-28). `bbox_labels`
  is a list of N names. Example: `(2, 4)` and `['opacity', 'opacity']`. An image with no box gives `(0, 4)`
  and `[]`.
- Dropped boxes: the harmonizer clips each box to the image. It drops a box with no area after the clip,
  and warns (D-07). The source files have no such box.

## Splits

<!-- claim: id=siim_covid19.splits.train tier=ci cmd="python scripts/evalkit/dataset_stat.py siim_covid19 siim_covid19_train.images" expect="6,334" -->
<!-- claim: id=siim_covid19.splits.test tier=ci cmd="python scripts/evalkit/dataset_stat.py siim_covid19 siim_covid19_test.images" expect="1,263" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `SIIMCOVID19TrainDataset` | `siim_covid19_train` | 6,334 | `train_image_level.csv`, `train_study_level.csv` and `train/` |
| `SIIMCOVID19TestDataset` | `siim_covid19_test` | 1,263 | `test/`, with no labels (D-25). Use it only for a Kaggle submission |
| `SIIMCOVID19Dataset` | `siim_covid19` | | The old key. It reads only the train files, so it gives the train images. Use the split classes for a benchmark (D-09) |

The source has no labeled test split. `get_datasets(n_splits=5)` makes a train and val split from the train
class. It groups by patient (D-10), so all studies of one patient stay together. No patient is in 2 splits.

## Arguments

`SIIMCOVID19TestDataset` takes the same arguments, except `csv_path` and `image_csv_path`.

<!-- args: radharmony.dataset.SIIMCOVID19TrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `train/` or `test/` folder. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `train_study_level.csv`. Without it, the class looks in `base_image_dir` and the folder above it. The test class does not have it |
| `image_csv_path` | `str` | No | `None` | `train_image_level.csv`. Without it, the class looks in `base_image_dir` and the folder above it. The test class does not have it |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample. In `SIIMCOVID19TestDataset`: not supported. The class warns and adds no key (D-25) |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_report` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Add `bbox` and `bbox_labels` to the sample. In `SIIMCOVID19TestDataset`: not supported. The class warns and adds no key (D-25) |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The labels stay float32 |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import SIIMCOVID19TrainHarmonizer

h = SIIMCOVID19TrainHarmonizer(
    csv_path="/data/siim-covid19-detection/train_study_level.csv",
    image_csv_path="/data/siim-covid19-detection/train_image_level.csv",
    base_image_dir="/data/siim-covid19-detection/train/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'bbox', 'bbox_labels', 'sex', 'atypical_appearance', ...]
```

`SIIMCOVID19TestHarmonizer` takes only `base_image_dir`. Its harmonized table has `patient_id`, `study_id`,
`image_path` and `sex`. With `harmonizer_path`, `SIIMCOVID19TrainDataset` also loads a file that the old
`SIIMCOVID19Harmonizer` saved.

<!-- args: radharmony.harmonizer.SIIMCOVID19TrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `train_study_level.csv` |
| `image_csv_path` | `str` | No | `None` | `train_image_level.csv`. Needed by `harmonize()`. Without it, `harmonize()` raises `ValueError` |
| `base_image_dir` | `str` | No | `None` | The `train/` folder. Needed by `harmonize()`. Without it, `harmonize()` raises `ValueError` |
| `num_workers` | `int` | No | `8` | The number of threads that read the DICOM headers |

## Known data problems

- Box outside the image: one source box starts at `x` = -3.56 pixels. The clip sets its `x_min` to 0
  (D-07).
