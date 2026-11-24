# RSNA 2023 Abdominal Trauma

<!-- claim: id=rsna_abdominal_trauma_2023.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_abdominal_trauma_2023 rsna_abdominal_trauma_2023_train.images" expect="4,711" -->
<!-- claim: id=rsna_abdominal_trauma_2023.train.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_abdominal_trauma_2023 rsna_abdominal_trauma_2023_train.patients" expect="3,147" -->
| | |
|---|---|
| Registry keys | `rsna_abdominal_trauma_2023_train`, `rsna_abdominal_trauma_2023_test` |
| Task and outputs | Classification, 14 injury labels (`cls`). The test split has images only |
| Images in the table | train 4,711 volumes (3,147 patients), test 6 volumes (sample files only) |
| Modality | Abdominal CT, 3-D, DICOM series. One row is one series, which is one volume |
| License | The rules of the Kaggle competition. Accept them before the download |
| Download | [Get the data: RSNA 2023 Abdominal Trauma](../../../how-to/get-the-data.md#rsna-2023-abdominal-trauma) |

<!-- claim: id=rsna_abdominal_trauma_2023.train.label_rows tier=nas cmd="python scripts/evalkit/dataset_stat.py --source rsna_abdominal_trauma_2023_train csv_path rows" expect="3,147" -->
<!-- claim: id=rsna_abdominal_trauma_2023.train.series_rows tier=nas cmd="python scripts/evalkit/dataset_stat.py --source rsna_abdominal_trauma_2023_train series_meta_csv_path rows" expect="4,711" -->
The RSNA 2023 Abdominal Trauma Detection challenge is a Kaggle competition of the Radiological Society of
North America (RSNA). Each patient has 1 or 2 abdominal CT series. `train_2024.csv` has 3,147 rows, one for
each patient, with injury labels for 5 organs. `train_series_meta.csv` has 4,711 rows, one for each series.
The harmonized table has one row for each series, and the labels of the patient repeat on each series of
the patient. So a label belongs to the patient, not to one series.

<!-- claim: id=rsna_abdominal_trauma_2023.source.slice_labels tier=nas cmd="python scripts/evalkit/dataset_stat.py --source rsna_abdominal_trauma_2023_train base_image_dir/../image_level_labels_2024.csv rows" expect="15,632" -->
<!-- claim: id=rsna_abdominal_trauma_2023.source.slice_labels_extravasation tier=nas cmd="python scripts/evalkit/dataset_stat.py --source rsna_abdominal_trauma_2023_train base_image_dir/../image_level_labels_2024.csv rows@injury_name=Active_Extravasation" expect="8,400" -->
<!-- claim: id=rsna_abdominal_trauma_2023.source.slice_labels_bowel tier=nas cmd="python scripts/evalkit/dataset_stat.py --source rsna_abdominal_trauma_2023_train base_image_dir/../image_level_labels_2024.csv rows@injury_name=Bowel" expect="7,232" -->
The source also has organ masks for 206 series in `segmentations/`. It has 15,632 slice labels in
`image_level_labels_2024.csv`: 8,400 of active extravasation and 7,232 of bowel injury. The harmonizer reads
neither of them.

## Quick use

`base_image_dir` is the `train_images/` folder. The class finds `train_2024.csv` and
`train_series_meta.csv` in `base_image_dir` or in the folder above it. The harmonizer finds `train_demographics_2024.csv` next to
`train_2024.csv`.

```
rsna-2023-abdominal-trauma-detection/
  train_2024.csv
  train_series_meta.csv
  train_demographics_2024.csv
  test_series_meta.csv
  train_images/<patient_id>/<series_id>/<instance_number>.dcm
  test_images/<patient_id>/<series_id>/<instance_number>.dcm
```

```python
import torch
from radharmony.dataset import RSNAAbdominalTrauma2023TrainDataset

train = RSNAAbdominalTrauma2023TrainDataset(
    base_image_dir="/data/rsna-2023-abdominal-trauma-detection/train_images/",
    output_cls=True,
    dtype=torch.float32,
)
train_ds, val_ds = train.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 112, 112, 112), 1 channel, then (D, H, W)
# sample["cls"]: Tensor(14,)
```

The image steps: MONAI `ITKReader` reads all DICOM files of the series folder as one volume. The class keeps
the default `base_transpose=True`. So the transform turns each volume to the axis code `IPL` and then to
`(D, H, W)` (D-12). It clips the values to `hu_window` (default `(-150, 250)`, for the soft tissue of the
abdomen) (D-14). Then it scales the 0.5 and 99.5 percentiles to -1 and 1. It resizes the longest side to
112, and pads the volume with -1 to 112 x 112 x 112 (D x H x W).

A series can have hundreds of DICOM files, so the first load of a sample can take more than 30 seconds. The
first run writes a MONAI cache to `./cache`. `cache_dir=None` gives no cache.

## How it is harmonized

- Source: `train_series_meta.csv` (one row for each series) and `train_2024.csv` (one row for each
  patient). The harmonizer joins them on `patient_id`. So one row is one series, a 3-D volume (D-03).
- IDs: `patient_id` `10004` and `series_id` `21057` give `study_id` `10004` and `image_path`
  `10004/21057`. `image_path` is the series folder, relative to `base_image_dir` (D-04). The source has no
  study ID, so `study_id` is `patient_id`. `series_id` is a column of the harmonized table.
- Labels (D-05): the 14 source columns have 0 and 1, and the harmonizer keeps them. Each series of a patient
  gets the labels of the patient. The source has no uncertain or blank label.
- Age and sex: they come from `train_demographics_2024.csv`, next to `csv_path`. `age_years` comes from
  `Age`, in years (D-06). `90+` gives 90, and `Unknown` gives NaN. `sex` is `M`, `F` or NaN, from `Sex`
  (D-26). With no such file, the harmonizer warns, and the harmonized table has no `age_years` or `sex`.
- View (D-01): the DICOM headers have `ImageOrientationPatient`, but the harmonizer does not read it. So the
  harmonized table has no `view_position`. D-01 lists this as a known gap.
- Extra columns (D-01): `aortic_hu` and `incomplete_organ` come from `train_series_meta.csv`. `aortic_hu`
  is the HU value in the aorta, a sign of the contrast phase. `incomplete_organ` is 1 when the scan does not
  hold an organ fully.
- Test source: `test_series_meta.csv`, with `patient_id`, `series_id` and `aortic_hu`. The harmonized test
  table has no labels, no age and no sex (D-25).
- Dropped rows: the join drops a series whose patient has no label row. The source files have no such
  series.

## Labels

The 14 label columns, in the order of `cls`:

| Column | Meaning |
|---|---|
| `any_injury` | An injury of 1 or more of the 5 organs |
| `bowel_healthy` | No bowel injury |
| `bowel_injury` | A bowel injury |
| `extravasation_healthy` | No active extravasation (bleeding) |
| `extravasation_injury` | Active extravasation |
| `kidney_healthy` | No kidney injury |
| `kidney_high` | A high-grade kidney injury |
| `kidney_low` | A low-grade kidney injury |
| `liver_healthy` | No liver injury |
| `liver_high` | A high-grade liver injury |
| `liver_low` | A low-grade liver injury |
| `spleen_healthy` | No spleen injury |
| `spleen_high` | A high-grade spleen injury |
| `spleen_low` | A low-grade spleen injury |

<!-- claim: id=rsna_abdominal_trauma_2023.labels.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_abdominal_trauma_2023 rsna_abdominal_trauma_2023_train.patients" expect="3,147" -->
Each organ has 1 in exactly 1 of its columns. For example, `kidney_low` 1 comes with 0 in `kidney_healthy`
and `kidney_high`. `any_injury` is 1 when an injury column is 1. In the source, 910 of the 3,147 patients
have an injury. The test series have no labels (D-25).

## Splits

<!-- claim: id=rsna_abdominal_trauma_2023.splits.train tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_abdominal_trauma_2023 rsna_abdominal_trauma_2023_train.images" expect="4,711" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `RSNAAbdominalTrauma2023TrainDataset` | `rsna_abdominal_trauma_2023_train` | 4,711 | `train_series_meta.csv` and `train_2024.csv` |
| `RSNAAbdominalTrauma2023TestDataset` | `rsna_abdominal_trauma_2023_test` | 6 | `test_series_meta.csv`, with no labels (D-25). Use it only for a Kaggle submission |
| `RSNAAbdominalTrauma2023Dataset` | `rsna_abdominal_trauma_2023` | | The old key. It reads the train files, so it has the train series. Use the split classes for a benchmark (D-09) |

`get_datasets(n_splits=5)` makes a train and val split from the train class. It groups by patient (D-10),
so all series of a patient go to the same split.

This is a Kaggle code competition (D-25). The public download has only sample test files.
`test_series_meta.csv` lists 6 series, but `test_images/` has only 3 of them, with 1 slice each. So on the
public download, no test sample loads. 3 series folders are not there, and the 3 others have 1 slice, which
the resize to 112 cannot use. Both give `RuntimeError`.

Use the test class in a Kaggle notebook of the competition, where the hidden test set has the same layout.

## Arguments

The arguments are the same for `RSNAAbdominalTrauma2023TrainDataset` and
`RSNAAbdominalTrauma2023TestDataset`. The test class does not use `csv_path`, and it looks for
`test_series_meta.csv`.

<!-- args: radharmony.dataset.RSNAAbdominalTrauma2023TrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `train_images/` or `test_images/` folder. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `train_2024.csv`. Without it, the class looks in `base_image_dir` or in the folder above it. The test class does not use it |
| `series_meta_csv_path` | `str` | No | `None` | `train_series_meta.csv` or `test_series_meta.csv`. Without it, the class looks in `base_image_dir` or in the folder above it |
| `hu_window` | `tuple` or `None` | No | `(-150, 250)` | The HU window. The transform clips each volume to it before the percentile scale (D-14). `None` gives no clip. It has no effect with your own `transform` |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform3D` with `img_size=112` and the class argument `hu_window` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample. In `RSNAAbdominalTrauma2023TestDataset`: not supported. The class warns and adds no key (D-25) |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11). The harmonizer does not read the masks in `segmentations/` |
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
from radharmony.harmonizer import RSNAAbdominalTrauma2023TrainHarmonizer

h = RSNAAbdominalTrauma2023TrainHarmonizer(
    csv_path="/data/rsna-2023-abdominal-trauma-detection/train_2024.csv",
    series_meta_csv_path="/data/rsna-2023-abdominal-trauma-detection/train_series_meta.csv",
    base_image_dir="/data/rsna-2023-abdominal-trauma-detection/train_images/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'series_id', 'age_years', 'sex', 'any_injury', ...,
#  'spleen_low', 'aortic_hu', 'incomplete_organ']
```

`RSNAAbdominalTrauma2023TestHarmonizer` takes only `series_meta_csv_path` and `base_image_dir`.
`image_path` is a folder, so `verify_images()` checks that each series folder exists.

<!-- args: radharmony.harmonizer.RSNAAbdominalTrauma2023TrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `train_2024.csv`. The harmonizer also reads `train_demographics_2024.csv` in the same folder |
| `series_meta_csv_path` | `str` | Yes | | `train_series_meta.csv`. It lists the series |
| `base_image_dir` | `str` | No | `None` | The `train_images/` folder. `save()` keeps it, and `verify_images()` uses it |

## Known data problems

- Incomplete organs: 313 train series have `incomplete_organ` 1, but they keep all labels of the patient.
- Age and sex: in `train_demographics_2024.csv`, 40 patients have `Age` `Unknown`, and 34 have `Sex` `Unknown`.
