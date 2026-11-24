# CT-RATE

<!-- claim: id=ct_rate.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py ct_rate ct_rate_train.images" expect="47,149" -->
<!-- claim: id=ct_rate.train.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py ct_rate ct_rate_train.patients" expect="20,000" -->
<!-- claim: id=ct_rate.valid.images tier=ci cmd="python scripts/evalkit/dataset_stat.py ct_rate ct_rate_valid.images" expect="3,039" -->
<!-- claim: id=ct_rate.valid.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py ct_rate ct_rate_valid.patients" expect="1,304" -->
| | |
|---|---|
| Registry keys | `ct_rate_train`, `ct_rate_valid` |
| Task and outputs | Classification, 18 labels (`cls`) |
| Images in the table | train 47,149 volumes (20,000 patients), valid 3,039 volumes (1,304 patients) |
| Modality | Chest CT, 3-D, NIfTI. One row is one volume |
| License | CC BY-NC-SA 4.0. Accept the terms on Hugging Face before the download |
| Download | [Get the data: CT-RATE](../../../how-to/get-the-data.md#ct-rate) |

<!-- claim: id=ct_rate.all.images tier=ci cmd="python scripts/evalkit/dataset_stat.py ct_rate ct_rate_train.images+ct_rate_valid.images" expect="50,188" -->
<!-- claim: id=ct_rate.all.studies tier=ci cmd="python scripts/evalkit/dataset_stat.py ct_rate ct_rate_train.studies+ct_rate_valid.studies" expect="25,692" -->
<!-- claim: id=ct_rate.all.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py ct_rate ct_rate_train.patients+ct_rate_valid.patients" expect="21,304" -->
CT-RATE is a chest CT dataset with radiology reports
([Hamamci et al., arXiv:2403.17834](https://arxiv.org/abs/2403.17834)). The CT scans have no contrast.
A scan can have more than 1 volume, one for each reconstruction. A text model read each report and gave
18 labels. The 2 splits have 50,188 volumes of 25,692 scans from 21,304 patients, as the paper says. The
source also has the reports, but the harmonizer does not read them.

## Quick use

`base_image_dir` is the `train_fixed/` or `valid_fixed/` folder. The class finds the label CSV and the
metadata CSV in `tables/`, next to that folder. The Hugging Face download has these CSVs in
`multi_abnormality_labels/` and `metadata/`. Move them to `tables/`, or pass `csv_path` and
`view_position_csv_path`.

```
CT-RATE/dataset/
  tables/train_predicted_labels.csv
  tables/train_metadata.csv
  tables/valid_predicted_labels.csv
  tables/validation_metadata.csv
  train_fixed/train_1/train_1_a/train_1_a_1.nii.gz
  valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz
```

```python
import torch
from radharmony.dataset import CTRATETrainDataset, CTRATEValidDataset

train = CTRATETrainDataset(
    base_image_dir="/data/CT-RATE/dataset/train_fixed/",
    output_cls=True,
    dtype=torch.float32,
)
train_ds, val_ds = train.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 112, 112, 112), 1 channel, then (D, H, W)
# sample["cls"]: Tensor(18,)

test_ds = CTRATEValidDataset(
    base_image_dir="/data/CT-RATE/dataset/valid_fixed/",
    output_cls=True,
    dtype=torch.float32,
).get_datasets()
```

The image steps: the class keeps the default `base_transpose=True`. So the transform turns each volume to
the axis code `IPL` and then to `(D, H, W)` (D-12). It clips the values to `hu_window` (default
`(-1000, 1000)`) (D-14). Then it scales the 0.5 and 99.5 percentiles to -1 and 1. It resizes the longest
side to 112, and pads the volume with -1 to 112 x 112 x 112 (D x H x W).

A volume file is large, so the first run is slow. The first run also writes a MONAI cache to `./cache`.
`cache_dir=None` gives no cache.

## How it is harmonized

- Source: `train_predicted_labels.csv` or `valid_predicted_labels.csv`. One row is one volume.
- IDs: the `VolumeName` `valid_1_a_1.nii.gz` gives `patient_id` `valid_1`, `study_id` `valid_1_a`, and
  `image_path` `valid_1/valid_1_a/valid_1_a_1.nii.gz`. `image_path` is relative to `base_image_dir` (D-04).
- Labels (D-05): the source has only 0 and 1. It has no uncertain or blank label.
- View (D-01): `view_position` is the nearest plane ("axial", "coronal" or "sagittal") of
  `ImageOrientationPatient` in the metadata CSV. The vector goes to the extra column `image_orientation`, in
  one form: a whole number becomes an int, and the other numbers keep all their digits. The train split has
  19 coronal and 14 sagittal volumes, and the others are axial. All valid volumes are axial.
- Age and sex: they also come from the metadata CSV. `age_years` comes from `PatientAge`, in years (D-06).
  For example, `036Y` gives 36. `sex` is `M`, `F` or NaN, from `PatientSex` (D-26). Without the metadata
  CSV, the harmonized table has no `view_position`, `image_orientation`, `age_years` or `sex`.
- Split: the train class keeps only the volumes whose name starts with `train_`, and the valid class keeps
  the `valid_` volumes. A label CSV with no such volume raises `ValueError`.
- Dropped rows: the harmonizer drops no row.

## Labels

The 18 label columns, in the order of `cls`:

| Column | Meaning |
|---|---|
| `arterial_wall_calcification` | Calcification in the wall of an artery |
| `atelectasis` | Part of a lung is collapsed |
| `bronchiectasis` | The bronchi are wide |
| `cardiomegaly` | The heart is large |
| `consolidation` | Airspace consolidation |
| `coronary_artery_wall_calcification` | Calcification in the wall of a coronary artery |
| `emphysema` | Emphysema |
| `hiatal_hernia` | Hiatal hernia |
| `interlobular_septal_thickening` | The interlobular septa are thick |
| `lung_nodule` | A lung nodule |
| `lung_opacity` | A lung opacity |
| `lymphadenopathy` | Large lymph nodes |
| `medical_material` | A tube, line or other device |
| `mosaic_attenuation_pattern` | Mosaic attenuation pattern |
| `peribronchial_thickening` | The walls around the bronchi are thick |
| `pericardial_effusion` | Pericardial effusion |
| `pleural_effusion` | Pleural effusion |
| `pulmonary_fibrotic_sequela` | Fibrosis after an earlier lung disease |

## Splits

<!-- claim: id=ct_rate.splits.train tier=ci cmd="python scripts/evalkit/dataset_stat.py ct_rate ct_rate_train.images" expect="47,149" -->
<!-- claim: id=ct_rate.splits.valid tier=ci cmd="python scripts/evalkit/dataset_stat.py ct_rate ct_rate_valid.images" expect="3,039" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `CTRATETrainDataset` | `ct_rate_train` | 47,149 | `train_predicted_labels.csv` and `train_fixed/` |
| `CTRATEValidDataset` | `ct_rate_valid` | 3,039 | `valid_predicted_labels.csv` and `valid_fixed/`. Use it as the test split |
| `CTRATEDataset` | `ct_rate` | | The old key. The image folder picks the tables. A folder name that starts with `valid` reads the valid tables. Other names read the train tables. Use the split classes for a benchmark (D-09) |

`get_datasets(n_splits=5)` makes a train and val split from the train class. It groups by patient (D-10).
No patient is in 2 splits.

## Arguments

The arguments are the same for `CTRATETrainDataset` and `CTRATEValidDataset`.

<!-- args: radharmony.dataset.CTRATETrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `train_fixed/` or `valid_fixed/` folder. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `train_predicted_labels.csv` or `valid_predicted_labels.csv`. Without it, the class looks in `base_image_dir` or in the folder above it, also in their `tables/` folders |
| `view_position_csv_path` | `str` | No | `None` | `train_metadata.csv` or `validation_metadata.csv`. It holds the view, the age and the sex. Without it, the class looks in `base_image_dir` or in the folder above it, also in their `tables/` folders |
| `hu_window` | `tuple` or `None` | No | `(-1000, 1000)` | The HU window. The transform clips each volume to it before the percentile scale (D-14). `None` gives no clip. It has no effect with your own `transform` |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform3D` with `img_size=112` and the class argument `hu_window` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_report` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11). The harmonizer does not read the reports |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The labels stay float32 |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import CTRATETrainHarmonizer

h = CTRATETrainHarmonizer(
    csv_path="/data/CT-RATE/dataset/tables/train_predicted_labels.csv",
    view_position_csv_path="/data/CT-RATE/dataset/tables/train_metadata.csv",
    base_image_dir="/data/CT-RATE/dataset/train_fixed/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'view_position', 'age_years', 'sex', ..., 'image_orientation']
```

`CTRATEValidHarmonizer` takes the same arguments, with the valid files. The extra column is
`image_orientation`, the `ImageOrientationPatient` vector.

<!-- args: radharmony.harmonizer.CTRATETrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `train_predicted_labels.csv` or `valid_predicted_labels.csv` |
| `view_position_csv_path` | `str` | No | `None` | `train_metadata.csv` or `validation_metadata.csv`. It adds `view_position`, `image_orientation`, `age_years` and `sex` |
| `base_image_dir` | `str` | No | `None` | The `train_fixed/` or `valid_fixed/` folder. `harmonize()` does not read it. `verify_images()` uses it (D-04), and a saved harmonizer passes it to the dataset class |

## Known data problems

<!-- claim: id=ct_rate.train.orientation_int tier=nas cmd="python scripts/evalkit/dataset_stat.py --source ct_rate_train view_position_csv_path 'rows@ImageOrientationPatient=[1, 0, 0, 0, 1, 0]'" expect="45,013" -->
<!-- claim: id=ct_rate.train.orientation_float tier=nas cmd="python scripts/evalkit/dataset_stat.py --source ct_rate_train view_position_csv_path 'rows@ImageOrientationPatient=[1.000000, 0.000000, 0.000000, 0.000000, 1.000000, -0.000000]'" expect="2,084" -->
- Orientation text: the metadata writes the same axial orientation in 2 forms. In the train metadata,
  45,013 volumes have `[1, 0, 0, 0, 1, 0]` and 2,084 have
  `[1.000000, 0.000000, 0.000000, 0.000000, 1.000000, -0.000000]` (D-01).
- Tilted volumes: 12 train volumes are tilted more than 8° from the nearest plane (D-01).
- Age in months: 7 train volumes have an age in months, for example `018M` (1.5 years) (D-06).
- Missing values: in the train metadata, 6 volumes have no sex and 12 have no age. In the valid metadata, 2
  volumes have no age.
- Volumes with no chest: some copies of CT-RATE have the lists `no_chest_train.txt` (752 volumes) and
  `no_chest_valid.txt` (37 volumes) in `tables/`. The harmonizer does not read them, so the harmonized table
  keeps these volumes (D-02). To remove them, filter the harmonized table by the file name:

<!-- block: run=skip reason="example only, needs the CT-RATE no-chest lists" -->
```python
with open("/data/CT-RATE/dataset/tables/no_chest_train.txt") as f:
    no_chest = {line.strip().rsplit("/", 1)[-1] for line in f if line.strip()}
df = df[~df["image_path"].str.rsplit("/", n=1).str[-1].isin(no_chest)]
```
