# RSNA 2024 Lumbar Spine

<!-- claim: id=rsna_2024_lumbar_spine.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_2024_lumbar_spine rsna_2024_lumbar_spine_train.images" expect="6,294" -->
<!-- claim: id=rsna_2024_lumbar_spine.train.studies tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_2024_lumbar_spine rsna_2024_lumbar_spine_train.studies" expect="1,975" -->
| | |
|---|---|
| Registry keys | `rsna_2024_lumbar_spine_train`, `rsna_2024_lumbar_spine_test` |
| Task and outputs | Classification, 75 labels (`cls`): 25 targets with 3 severities each. Small boxes around points (`bbox`). The test split has images only |
| Images in the table | train 6,294 volumes (1,975 studies), test 3 volumes (1 study, sample files only). One study is one patient |
| Modality | Lower back MRI, 3-D, DICOM series. One row is one series, which is one volume |
| License | The rules of the Kaggle competition. Accept them before the download |
| Download | [Get the data: RSNA 2024 Lumbar Spine](../../../how-to/get-the-data.md#rsna-2024-lumbar-spine) |

<!-- claim: id=rsna_2024_lumbar_spine.train.studies.again tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_2024_lumbar_spine rsna_2024_lumbar_spine_train.studies" expect="1,975" -->
<!-- claim: id=rsna_2024_lumbar_spine.train.images.again tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_2024_lumbar_spine rsna_2024_lumbar_spine_train.images" expect="6,294" -->
RSNA 2024 Lumbar Spine Degenerative Classification is a 2024 Kaggle challenge. The train split has 1,975 MRI
studies of the lower back. Radiologists graded 5 conditions at 5 disc levels (L1/L2 to L5/S1), so a study has
25 targets ([LumbarDISC dataset paper, arXiv:2506.09162](https://arxiv.org/abs/2506.09162)). Each target is
Normal/Mild, Moderate or Severe, or blank in `train.csv`. A study has about 3 series (Sagittal T1, Sagittal
T2/STIR and Axial T2), and RadHarmony makes one row for each series: 6,294 rows.

A point file marks where each target is on one slice. The test images are hidden (D-25).

## Quick use

`base_image_dir` is `train_images/` or `test_images/`. The classes find the CSV files in `base_image_dir` or
in the folder above it.

```
rsna-2024-lumbar-spine-degenerative-classification/
  train.csv
  train_series_descriptions.csv
  train_label_coordinates.csv
  test_series_descriptions.csv
  train_images/<study_id>/<series_id>/<instance_number>.dcm
  test_images/<study_id>/<series_id>/<instance_number>.dcm
```

<!-- claim: id=rsna_2024_lumbar_spine.train.images.quick tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_2024_lumbar_spine rsna_2024_lumbar_spine_train.images" expect="6,294" -->
When the train class finds `train_label_coordinates.csv`, it makes the boxes. For this, it reads the header of
each slice file. This took about 15 minutes for the 6,294 series in a test run.

```python
import torch
from radharmony.dataset import RSNA2024LumbarSpineTestDataset, RSNA2024LumbarSpineTrainDataset

train = RSNA2024LumbarSpineTrainDataset(
    base_image_dir="/data/rsna-2024-lumbar-spine-degenerative-classification/train_images/",
    series_filter="Sagittal T2/STIR",  # keep one series type
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
)
train_ds, val_ds = train.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 112, 112, 112), 1 channel, then (D, H, W)
# sample["cls"]: Tensor(75,)
# sample["bbox"]: Tensor(N, 6), one small box for each point
# sample["bbox_labels"]: a list of N names, for example ['spinal_canal_stenosis_l1_l2']

test_ds = RSNA2024LumbarSpineTestDataset(
    base_image_dir="/data/rsna-2024-lumbar-spine-degenerative-classification/test_images/",
    series_filter="Sagittal T2/STIR",
    dtype=torch.float32,
).get_datasets()
# test_ds[0] has only "img", for a Kaggle submission
```

The image steps: MONAI `ITKReader` reads all DICOM files of the series folder as one volume. The class keeps
the default `base_transpose=True`. So the transform turns each volume to the axis code `IPL` and then to
`(D, H, W)` (D-12). MRI has no fixed value scale, so the class uses `hu_window=None` and clips nothing. Then
the transform scales the 0.5 and 99.5 percentiles to -1 and 1. It resizes the longest side to 112, and pads
the volume with -1 to 112 x 112 x 112 (D x H x W).

Example: a sagittal series of 15 slices of 640 x 640 (H x W) becomes a volume about 3 voxels thick.

## How it is harmonized

- Source: `train.csv` has one row for each study, with 25 target columns. `train_series_descriptions.csv` has
  one row for each series. The harmonizer joins them on `study_id`, so one row is one series.
- IDs: the `study_id` `4003253` and `series_id` `702807833` give `patient_id` `4003253`, `study_id`
  `4003253`, `series_id` `702807833` and `image_path` `4003253/702807833`. `image_path` is the series folder,
  relative to `base_image_dir` (D-04). The source has no patient ID column. The DICOM `PatientID` equals
  `study_id`, so one study is one patient.
- Labels (D-05): each target becomes 3 columns, one for each severity. `Moderate` gives 1 in `..._moderate`
  and 0 in the other 2. A blank target gives NaN in all 3 columns, not 0. The labels of a study repeat on
  each of its series.
- View (D-01): `view_position` holds the series description: `Axial T2`, `Sagittal T1` or
  `Sagittal T2/STIR`. D-01 asks for the plane (`axial` or `sagittal`) in `view_position` and the description
  in a column `series_description`. D-01 lists this as a known gap. `series_filter` keeps the rows of one
  series description.
- Age and sex: the source files and the DICOM headers have neither, so the harmonized table has no
  `age_years` or `sex` (D-06, D-26).
- Dropped rows: the harmonizer drops a series whose study is not in `train.csv`. The source files have no
  such series.

## Labels

The 75 label columns are `<condition>_<level>_<severity>`, sorted by name, in the order of `cls`. Example:
`spinal_canal_stenosis_l1_l2_normal_mild`, `spinal_canal_stenosis_l1_l2_moderate` and
`spinal_canal_stenosis_l1_l2_severe`.

| Part | Values | Meaning |
|---|---|---|
| Condition | `spinal_canal_stenosis` | The spinal canal is narrow |
| | `left_neural_foraminal_narrowing`, `right_neural_foraminal_narrowing` | The opening for a nerve root, on the left or right side, is narrow |
| | `left_subarticular_stenosis`, `right_subarticular_stenosis` | The space below a facet joint, on the left or right side, is narrow |
| Level | `l1_l2`, `l2_l3`, `l3_l4`, `l4_l5`, `l5_s1` | The disc level |
| Severity | `normal_mild`, `moderate`, `severe` | The grade of the condition |

572 study targets are blank, in 185 studies. Their 3 columns are NaN in `cls`, so the probes skip them (D-05).

## Boxes

<!-- claim: id=rsna_2024_lumbar_spine.boxes.points tier=nas cmd="python scripts/evalkit/dataset_stat.py --source rsna_2024_lumbar_spine_train coord_csv_path rows" expect="48,692" -->
<!-- claim: id=rsna_2024_lumbar_spine.boxes.series tier=nas cmd="python scripts/evalkit/dataset_stat.py --source rsna_2024_lumbar_spine_train coord_csv_path unique:series_id" expect="6,291" -->
- Source: `train_label_coordinates.csv` has 48,692 points on 6,291 series. Each row is one point of one
  target: `instance_number` (the slice), `condition`, `level`, and `x` and `y` in pixels.
- The harmonizer lists the slice files of each series in the order of the 3-D loader, with
  `itk.GDCMSeriesFileNames`, the same call as `ITKReader`. It reads the header of each slice file, so it
  needs `base_image_dir`. The rank of the point slice in this order gives the depth: from `rank / n_slices`
  to `(rank + 1) / n_slices`. It divides `y` by `Rows` (the height) and `x` by `Columns` (the width). The
  box goes 0.015 (`bbox_epsilon`) below and above each of these 2 values.
- The box is `[dim0_min, dim0_max, dim1_min, dim1_max, dim2_min, dim2_max]`, in fractions of the volume
  size (D-07). In the harmonized table, the order is slice, `y`, `x`, in the slice order of the 3-D loader.
  The class turns the box with the volume (D-12, D-28). So in `ds[i]` of a sagittal series, the slice is the
  last axis (W).
- Example: series `4003253/702807833` (Sagittal T2/STIR) has 15 slices, and each slice is 640 x 640 (H x W).
  Its first point has `instance_number` 8, `x` 322.83185840707966 and `y` 227.9646017699115, for
  Spinal Canal Stenosis at L1/L2. Slice 8 has the rank 7. The box is `[0.4666666666666667,
  0.5333333333333333, 0.3411946902654867, 0.37119469026548674, 0.48942477876106194, 0.519424778761062]`.
- `bbox_labels`: `<condition>_<level>` in snake_case, for example `spinal_canal_stenosis_l1_l2` (D-07).
- Readers (D-36): the source has one point for each target on a series, so the harmonizer does not
  group boxes.
- The sample: `bbox` is a float32 tensor `(N, 6)`, in fractions of the returned volume (D-28). `bbox_labels`
  is a list of N names. Example: `(5, 6)` and
  `['spinal_canal_stenosis_l1_l2', ..., 'spinal_canal_stenosis_l5_s1']` for the series above. A series with
  no point gives `(0, 6)` and `[]`.
- Points with no box: some series folders hold 2 GDCM sub-series, and the 3-D loader reads only the first. A
  point on the second sub-series is not on the loaded volume, so the harmonizer makes no box for it. A point
  on a series folder or a slice file that cannot be read also has no box.
  `harmonize()` warns with the count (D-07). In a test run, 151 points had no box.
- Dropped boxes: the harmonizer clips each box to the volume. It drops a box with no area after the clip,
  and warns (D-07). The source files have no such box.
- Without a point file, the harmonized table has no `bbox` column. Then `output_bbox=True` warns and adds no
  box keys (D-11).

## Splits

<!-- claim: id=rsna_2024_lumbar_spine.splits.train tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_2024_lumbar_spine rsna_2024_lumbar_spine_train.images" expect="6,294" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `RSNA2024LumbarSpineTrainDataset` | `rsna_2024_lumbar_spine_train` | 6,294 | `train.csv`, `train_series_descriptions.csv` and `train_images/` |
| `RSNA2024LumbarSpineTestDataset` | `rsna_2024_lumbar_spine_test` | 3 | `test_series_descriptions.csv` and `test_images/`, with no labels (D-25). Use it only for a Kaggle submission |
| `RSNA2024LumbarSpineDataset` | `rsna_2024_lumbar_spine` | | The old key. It reads `train.csv`, so it has the train series. Use the split classes for a benchmark (D-09) |

The source has no labeled test split. `get_datasets(n_splits=5)` makes a train and val split from the train
class. It groups by patient (D-10). The source has no patient ID, so one study is one patient, and nobody can
check that a patient is in only 1 split. All series of one study stay in one split.

This is a Kaggle code competition (D-25). The public download has only sample test files. Use the test class
in a Kaggle notebook of the competition, where the hidden test set has the same layout.

## Arguments

The arguments are the same for `RSNA2024LumbarSpineTrainDataset` and `RSNA2024LumbarSpineTestDataset`. The
test class takes no `csv_path` and no `coord_csv_path`, and its `series_description_csv_path` is
`test_series_descriptions.csv`.

<!-- args: radharmony.dataset.RSNA2024LumbarSpineTrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | `train_images/` or `test_images/`. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `train.csv`. Without it, the class looks in `base_image_dir` and the folder above it. A missing file raises `FileNotFoundError` (D-35) |
| `series_description_csv_path` | `str` | No | `None` | `train_series_descriptions.csv`. Without it, the class looks in `base_image_dir` and the folder above it. A missing file raises `FileNotFoundError` (D-35) |
| `coord_csv_path` | `str` | No | `None` | `train_label_coordinates.csv`, for the boxes. Without it, the class looks in `base_image_dir` and the folder above it. With no file, the harmonized table has no boxes |
| `series_filter` | `str` | No | `None` | Keep only one series type: `"Axial T2"`, `"Sagittal T1"` or `"Sagittal T2/STIR"`. Another value raises `ValueError`. It also filters a harmonized table from `harmonized_df`, `harmonizer` or `harmonizer_path` |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform3D` with `img_size=112` and `hu_window=None` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample. In `RSNA2024LumbarSpineTestDataset`: not supported. The class warns and adds no key (D-25) |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_report` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Add `bbox` and `bbox_labels` to the sample. With no point file, the class warns and adds no box keys (D-11). In `RSNA2024LumbarSpineTestDataset`: not supported. The class warns and adds no key (D-25) |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The labels stay float32 |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import RSNA2024LumbarSpineTrainHarmonizer

h = RSNA2024LumbarSpineTrainHarmonizer(
    csv_path="/data/rsna-2024-lumbar-spine-degenerative-classification/train.csv",
    series_description_csv_path="/data/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv",
    base_image_dir="/data/rsna-2024-lumbar-spine-degenerative-classification/train_images/",
)
df = h.harmonize()  # with coord_csv_path=..., it also makes the boxes
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'series_id', 'view_position', 'left_neural_foraminal_narrowing_l1_l2_moderate', ...]
```

`RSNA2024LumbarSpineTestHarmonizer` takes `series_description_csv_path` (`test_series_descriptions.csv`) and
`base_image_dir`. Its harmonized table has `patient_id`, `study_id`, `image_path`, `series_id` and
`view_position`.

<!-- args: radharmony.harmonizer.RSNA2024LumbarSpineTrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `train.csv` |
| `series_description_csv_path` | `str` | No | `None` | `train_series_descriptions.csv`. `harmonize()` raises `ValueError` without it |
| `coord_csv_path` | `str` | No | `None` | `train_label_coordinates.csv`. It adds `bbox` and `bbox_labels`. It needs `base_image_dir`, or `harmonize()` raises `ValueError` |
| `base_image_dir` | `str` | No | `None` | `train_images/`. The harmonizer reads the slice order and size from it |
| `bbox_epsilon` | `float` | No | `0.015` | Half the height and width of a box, as a fraction |
| `bbox_z_epsilon` | `float` | No | `None` | Half the depth of a box, as a fraction. `None` gives `0.5 / n_slices`, so the box is one slice deep |
| `num_workers` | `int` | No | `8` | The number of threads that read the slice headers |

## Known data problems

- Series per study: most studies have 3 series, but not all. 335 studies have 2 or more Axial T2 series,
  and 7 have 2 Sagittal T1 series. 3 studies have only 2 series: 2 have no Sagittal T1, and 1 has no
  Sagittal T2/STIR.
- Sub-series: series `1973833645/1328374636` (Axial T2) has 2 GDCM sub-series of 25 and 26 files. 6 of its
  10 points are on the second one, so they have no box (see Boxes).
- Slice order: in about half of the series, the slice order of the 3-D loader is not the order of
  `ImagePositionPatient[2]` (see Boxes).
