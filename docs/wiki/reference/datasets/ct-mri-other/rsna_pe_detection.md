# RSNA PE Detection

<!-- claim: id=rsna_pe_detection.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_pe_detection rsna_pe_detection_train.images" expect="7,279" -->
| | |
|---|---|
| Registry keys | `rsna_pe_detection_train`, `rsna_pe_detection_test` |
| Task and outputs | Classification, 13 study labels (`cls`). The test split has images only |
| Images in the table | train 7,279 volumes, test 650 volumes. One study is one volume and one patient |
| Modality | Chest CT pulmonary angiography (CTPA), 3-D, DICOM series. One row is one study, which is one series and one volume |
| License | The rules of the Kaggle competition. Accept them before the download |
| Download | [Get the data: RSNA PE Detection](../../../how-to/get-the-data.md#rsna-pe-detection) |

<!-- claim: id=rsna_pe_detection.train.pe_studies tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_pe_detection rsna_pe_detection_train.positives.rv_lv_ratio_gte_1+rsna_pe_detection_train.positives.rv_lv_ratio_lt_1" expect="2,211" -->
The RSNA-STR Pulmonary Embolism (PE) Detection Challenge (2020) is a Kaggle competition. The Radiological
Society of North America (RSNA) and the Society of Thoracic Radiology (STR) made it. Each study is one CTPA
scan, with one series of slices. 2,211 train studies have PE.

<!-- claim: id=rsna_pe_detection.train.csv_rows tier=nas cmd="python scripts/evalkit/dataset_stat.py --source rsna_pe_detection_train csv_path rows" expect="1,790,594" -->
<!-- claim: id=rsna_pe_detection.train.studies tier=nas cmd="python scripts/evalkit/dataset_stat.py --source rsna_pe_detection_train csv_path unique:StudyInstanceUID" expect="7,279" -->
`train.csv` has one row for each slice: 1,790,594 rows for 7,279 studies. It has labels at 2 levels. The 13
study labels have the same value on every slice of a study. The image label `pe_present_on_image` says if
one slice shows PE.

## Quick use

`base_image_dir` is the `train/` or `test/` folder. The class finds `train.csv` or `test.csv` in
`base_image_dir` or in the folder above it.

```
rsna_pe_dataset/
  train.csv
  test.csv
  train/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
  test/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
```

```python
import torch
from radharmony.dataset import RSNAPEDetectionTrainDataset, RSNAPEDetectionTestDataset

train = RSNAPEDetectionTrainDataset(
    base_image_dir="/data/rsna_pe_dataset/train/",
    output_cls=True,
    dtype=torch.float32,
)
train_ds, val_ds = train.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 112, 112, 112), 1 channel, then (D, H, W)
# sample["cls"]: Tensor(13,)

test_ds = RSNAPEDetectionTestDataset(
    base_image_dir="/data/rsna_pe_dataset/test/",
    dtype=torch.float32,
).get_datasets()
# test_ds[0] has only "img", for a Kaggle submission
```

The image steps: MONAI `ITKReader` reads all DICOM files of the series folder as one volume. The class keeps
the default `base_transpose=True`. So the transform turns each volume to the axis code `IPL` and then to
`(D, H, W)` (D-12). It clips the values to `hu_window` (default `(-1000, 1000)`, for CTPA soft tissue and
lung) (D-14). Then it scales the 0.5 and 99.5 percentiles to -1 and 1. It resizes the longest side to 112,
and pads the volume with -1 to 112 x 112 x 112 (D x H x W).

The first run writes a MONAI cache to `./cache`. `cache_dir=None` gives no cache.

## How it is harmonized

- Source: `train.csv` or `test.csv`, with one row for each slice. The harmonizer keeps the first row of
  each `StudyInstanceUID`. So one row is one study, with one series and one 3-D volume (D-03).
- IDs: `StudyInstanceUID` `6897fa9de148` and `SeriesInstanceUID` `2bfbb7fd2e8b` give `patient_id` and
  `study_id` `6897fa9de148`, and `image_path` `6897fa9de148/2bfbb7fd2e8b`. `image_path` is the series
  folder, relative to `base_image_dir` (D-04).
- Labels (D-05): the 13 study columns are already snake_case, with 0 and 1. The harmonizer keeps them. The
  source has no uncertain or blank label. The harmonizer drops the image label `pe_present_on_image`,
  because one row of the harmonized table is a whole volume.
- View (D-01): the DICOM headers have `ImageOrientationPatient`, but the harmonizer does not read it. So the
  harmonized table has no `view_position`. D-01 lists this as a known gap.
- Age and sex: the CSV files have no age or sex. The 5 DICOM headers we checked have no `PatientAge` and no
  `PatientSex` tag. So the harmonized table has no `age_years` or `sex` (D-06, D-26).
- Test source: `test.csv` has only the 3 UID columns, so the harmonized test table has no label columns
  (D-25).
- Dropped rows: the harmonizer drops only the extra slice rows of each study. It drops no study.

## Labels

The 13 label columns, in the order of `cls`:

| Column | Meaning |
|---|---|
| `acute_and_chronic_pe` | The PE is both acute and chronic |
| `central_pe` | PE in the central pulmonary arteries |
| `chronic_pe` | The PE is chronic, not acute |
| `flow_artifact` | A flow artifact in the arteries |
| `indeterminate` | The study quality is too low to say if PE is present |
| `leftsided_pe` | PE on the left side |
| `negative_exam_for_pe` | No PE in the study |
| `qa_contrast` | The contrast is not good enough |
| `qa_motion` | Motion makes the images unclear |
| `rightsided_pe` | PE on the right side |
| `rv_lv_ratio_gte_1` | The right to left ventricle ratio is 1 or more |
| `rv_lv_ratio_lt_1` | The right to left ventricle ratio is less than 1 |
| `true_filling_defect_not_pe` | A filling defect that is not PE |

A study with PE has 0 in `negative_exam_for_pe` and in `indeterminate`. In such a study, exactly 1 of
`rv_lv_ratio_gte_1` and `rv_lv_ratio_lt_1` is 1. In the other studies, both are 0.

Acute PE has no column. A study with PE and 0 in `chronic_pe` and `acute_and_chronic_pe` has acute PE. Each
indeterminate study has 1 in `qa_motion` or `qa_contrast`. The test studies have no labels (D-25).

## Splits

<!-- claim: id=rsna_pe_detection.splits.train tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_pe_detection rsna_pe_detection_train.images" expect="7,279" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `RSNAPEDetectionTrainDataset` | `rsna_pe_detection_train` | 7,279 | `train.csv` |
| `RSNAPEDetectionTestDataset` | `rsna_pe_detection_test` | 650 | `test.csv`, with no labels (D-25). Use it only for a Kaggle submission |
| `RSNAPEDetectionDataset` | `rsna_pe_detection` | | The old key. It reads `train.csv`, so it has the train studies. Use the split classes for a benchmark (D-09) |

`get_datasets(n_splits=5)` makes a train and val split from the train class. It groups by patient (D-10).
The source has no patient ID, so one study is one patient, and nobody can check that a patient is in only 1
split. `train.csv` and `test.csv` share no study.

## Arguments

The arguments are the same for `RSNAPEDetectionTrainDataset` and `RSNAPEDetectionTestDataset`. The test
class looks for `test.csv`, not `train.csv`.

<!-- args: radharmony.dataset.RSNAPEDetectionTrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `train/` or `test/` folder. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `train.csv` or `test.csv`. Without it, the class looks in `base_image_dir` or in the folder above it |
| `hu_window` | `tuple` or `None` | No | `(-1000, 1000)` | The HU window. The transform clips each volume to it before the percentile scale (D-14). `None` gives no clip. It has no effect with your own `transform` |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform3D` with `img_size=112` and the class argument `hu_window` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample. In `RSNAPEDetectionTestDataset`: not supported. The class warns and adds no key (D-25) |
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
from radharmony.harmonizer import RSNAPEDetectionTrainHarmonizer

h = RSNAPEDetectionTrainHarmonizer(
    csv_path="/data/rsna_pe_dataset/train.csv",
    base_image_dir="/data/rsna_pe_dataset/train/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'acute_and_chronic_pe', 'central_pe', ...]
```

`RSNAPEDetectionTestHarmonizer` takes the same arguments, with `test.csv` and the `test/` folder. The
harmonized test table has only `patient_id`, `study_id` and `image_path`. `image_path` is a folder, so
`verify_images()` checks that each series folder exists.

<!-- args: radharmony.harmonizer.RSNAPEDetectionTrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `train.csv` or `test.csv` |
| `base_image_dir` | `str` | No | `None` | The `train/` or `test/` folder. `save()` keeps it, and `verify_images()` uses it |

## Known data problems

None known.
