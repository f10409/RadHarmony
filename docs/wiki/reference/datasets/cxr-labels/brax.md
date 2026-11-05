# BRAX

<!-- claim: id=brax.images tier=ci cmd="python scripts/evalkit/dataset_stat.py brax brax.images" expect="40,967" -->
<!-- claim: id=brax.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py brax brax.patients" expect="18,442" -->
<!-- claim: id=brax.studies tier=ci cmd="python scripts/evalkit/dataset_stat.py brax brax.studies" expect="23,276" -->
| | |
|---|---|
| Registry keys | `brax` (DICOM), `brax_png` (PNG) |
| Task and outputs | Classification, 14 labels (`cls`) |
| Images in the table | 40,967 (18,442 patients, 23,276 studies) in each key |
| Modality | Chest X-ray, 2-D, DICOM or PNG |
| License | PhysioNet Credentialed Health Data License 1.5.0 |
| Download | [Get the data: BRAX](../../../how-to/get-the-data.md#brax) |

BRAX (Brazilian Chest X-Ray) v1.1.0 is a chest X-ray dataset from a hospital network in Brazil. The
original CheXpert labeler read the Portuguese reports and gave the 14 CheXpert labels. The source has
no reports. One download has the DICOM and the PNG images. `BRAXDataset` reads the DICOM files, and
`BRAXPNGDataset` reads the PNG files. Both use one harmonizer, so both keys have the same rows.

## Quick use

`base_image_dir` is the BRAX root. The class finds `master_spreadsheet.csv` or
`master_spreadsheet_update.csv` in this folder or in the folder above it.

```
brax/1.1.0/
  master_spreadsheet_update.csv
  Anonymized_DICOMs/id_<PatientID>/Study_<UID>/Series_<UID>/image-<UID>.dcm
  images/id_<PatientID>/Study_<UID>/Series_<UID>/image-<UID>.png
```

```python
import torch
from radharmony.dataset import BRAXDataset

ds = BRAXDataset(
    base_image_dir="/data/brax/1.1.0/",
    uncertain_strategy="u_zeros",  # -1 and blank give 0
    output_cls=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(14,)
```

`BRAXPNGDataset` takes the same arguments.

## How it is harmonized

- Source: `master_spreadsheet.csv` or `master_spreadsheet_update.csv`. One row is one image (D-03).
- IDs: `patient_id` is `PatientID`, for example `id_55c25711-75a21898-593e2eea-46007548-5afa539c`.
  `study_id` is `AccessionNumber`, for example `79565147` (D-01). `series_id` is the `Series_<UID>` folder
  name in the path.
- `image_path` is `DicomPath` (`brax`) or `PngPath` (`brax_png`), as in the source. It is relative to the
  BRAX root (D-04). Example (PNG):
  `images/id_55c25711-75a21898-593e2eea-46007548-5afa539c/Study_79565147.82013478.83925914.72078900.07189967/Series_40604671.16381531.63216171.19463495.14134010/image-61667501-86352188-29978958-28578248-00095672.png`.
- Labels (D-05): the source codes are `1` (positive), `0` (negative), `-1` (uncertain) and blank (not
  mentioned). `uncertain_strategy="raw"` (the default) keeps all 4 codes, so `cls` can hold -1 and NaN.
  The other strategies are in [Labels](#labels).
- View (D-01): from `ViewPosition`. `L` and `RL` give `LATERAL`. A blank gives NaN. Other values, such
  as `LT-DECUB` or `RLO`, stay as in the source.
- Age and sex: `age_years` comes from `PatientAge`, in years (D-06). It is the lowest age of the 5-year
  group. For example, `40` (40 to 44 years) gives 40, and `85 or more` gives 85. `sex` is `M` or `F`,
  from `PatientSex` (D-26).
- Size (D-01): `rows` and `columns` give the DICOM size from the source, also for `brax_png`. The PNG
  files are smaller. Example: 2752 x 2232 in the harmonized table, but the PNG is 1261 x 1024.
- Dropped rows: none with `"raw"`. With `uncertain_strategy="drop"`, the harmonizer drops each row with a
  -1 label.

## Labels

The 14 label columns, in the order of `cls`:

| Column | Meaning |
|---|---|
| `atelectasis` | Part of a lung is collapsed |
| `cardiomegaly` | The heart is large |
| `consolidation` | Airspace consolidation |
| `edema` | Pulmonary edema |
| `enlarged_cardiomediastinum` | The mediastinum is wide |
| `fracture` | A rib or other bone fracture |
| `lung_lesion` | A lung lesion |
| `lung_opacity` | A lung opacity |
| `no_finding` | No finding |
| `pleural_effusion` | Pleural effusion |
| `pleural_other` | Another pleural finding |
| `pneumonia` | Pneumonia |
| `pneumothorax` | Pneumothorax |
| `support_devices` | A tube, line or other device |

`uncertain_strategy` sets the values in the harmonized table and in `cls`:

| Strategy | `-1` gives | Blank gives | Dropped rows |
|---|---|---|---|
| `"raw"` (default) | -1 | NaN | None |
| `"u_zeros"` | 0 | 0 | None |
| `"u_ones"` | 1 | 0 | None |
| `"u_ignore"` | NaN | 0 | None |
| `"drop"` | Not in the harmonized table | 0 | The rows with a -1 label |

With `"raw"` and `output_cls=True`, the dataset warns that `cls` can hold -1. The probes skip -1 and NaN
(D-05).

## Splits

<!-- claim: id=brax.splits.dicom tier=ci cmd="python scripts/evalkit/dataset_stat.py brax brax.images" expect="40,967" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `BRAXDataset` | `brax` | 40,967 | The master CSV, DICOM files |
| `BRAXPNGDataset` | `brax_png` | 40,967 | The master CSV, PNG files |

BRAX has no official split. `get_datasets(n_splits=5)` makes a train and val split. It groups by patient
(D-10).

## Arguments

The arguments are the same for `BRAXDataset` and `BRAXPNGDataset`.

<!-- args: radharmony.dataset.BRAXDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The BRAX root, with `Anonymized_DICOMs/` and `images/`. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | The master CSV. Without it, the class looks in `base_image_dir` and the folder above it |
| `uncertain_strategy` | `str` | No | `"raw"` | `"raw"`, `"u_zeros"`, `"u_ones"`, `"u_ignore"` or `"drop"`. See [Labels](#labels) |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_report` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11). The source has no reports |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The labels stay float32 |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import BRAXHarmonizer

h = BRAXHarmonizer(
    csv_path="/data/brax/1.1.0/master_spreadsheet_update.csv",
    base_image_dir="/data/brax/1.1.0/",
    image_format="png",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'series_id', 'view_position', 'age_years', 'sex',
#  'atelectasis', ..., 'support_devices', 'patient_sex', 'patient_age', 'manufacturer',
#  'study_date', 'rows', 'columns']
```

The extra columns keep the source values. `patient_sex` is `PatientSex`, `patient_age` is the age group
in `PatientAge`, and `manufacturer` and `study_date` are `Manufacturer` and `StudyDate`. `rows` and
`columns` are the DICOM size.

<!-- args: radharmony.harmonizer.BRAXHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | The master CSV |
| `base_image_dir` | `str` | No | `None` | The BRAX root |
| `image_format` | `str` | No | `"dicom"` | `"dicom"` reads `DicomPath`, and `"png"` reads `PngPath` |
| `uncertain_strategy` | `str` | No | `"raw"` | See [Labels](#labels) |

## Known data problems

- Shared study IDs: 2 `AccessionNumber` values (21498229 and 26826491) belong to 2 patients each, with
  different study dates. The harmonizer warns. Group by `patient_id` and `study_id` together (D-01).
- Age: the source gives only 5-year age groups, not the exact age (D-06).
- View: many rows of the source have a blank `ViewPosition` (D-01).
- Size: the source gives only the DICOM size. The PNG files are smaller (D-01).

<!-- claim: id=brax.source.patients tier=nas cmd="python scripts/evalkit/dataset_stat.py --source brax csv_path unique:PatientID" expect="18,442" -->
<!-- claim: id=brax.source.studies tier=nas cmd="python scripts/evalkit/dataset_stat.py --source brax csv_path unique:AccessionNumber" expect="23,276" -->
<!-- claim: id=brax.paper.patients source="Reis et al., Scientific Data 2022" expect="19,351" -->
<!-- claim: id=brax.paper.studies source="Reis et al., Scientific Data 2022" expect="24,959" -->
- Counts: the source CSV has 18,442 `PatientID` values and 23,276 `AccessionNumber` values. The BRAX
  paper (Reis et al., Scientific Data 2022) gives 19,351 patients and 24,959 studies. The image count is
  the same.
