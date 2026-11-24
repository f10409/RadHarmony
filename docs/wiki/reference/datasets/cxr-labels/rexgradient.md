# ReXGradient-160K

<!-- claim: id=rexgradient.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py rexgradient rexgradient_train.images" expect="238,968" -->
<!-- claim: id=rexgradient.train.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py rexgradient rexgradient_train.patients" expect="95,716" -->
<!-- claim: id=rexgradient.valid.images tier=ci cmd="python scripts/evalkit/dataset_stat.py rexgradient rexgradient_valid.images" expect="17,007" -->
<!-- claim: id=rexgradient.valid.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py rexgradient rexgradient_valid.patients" expect="6,964" -->
<!-- claim: id=rexgradient.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py rexgradient rexgradient_test.images" expect="17,029" -->
<!-- claim: id=rexgradient.test.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py rexgradient rexgradient_test.patients" expect="6,807" -->
| | |
|---|---|
| Registry keys | `rexgradient_train`, `rexgradient_valid`, `rexgradient_test` |
| Task and outputs | The report text (`report`). Classification, 14 labels (`cls`), only with `labels_csv_path` |
| Images in the table | train 238,968 (95,716 patients), valid 17,007 (6,964 patients), test 17,029 (6,807 patients) |
| Modality | Chest X-ray, 2-D, PNG |
| License | ReXGradient-160K Non-Commercial Data Access and Use Agreement |
| Download | [Get the data: ReXGradient-160K](../../../how-to/get-the-data.md#rexgradient-160k) |

<!-- claim: id=rexgradient.all.images tier=ci cmd="python scripts/evalkit/dataset_stat.py rexgradient rexgradient_train.images+rexgradient_valid.images+rexgradient_test.images" expect="273,004" -->
<!-- claim: id=rexgradient.all.studies tier=ci cmd="python scripts/evalkit/dataset_stat.py rexgradient rexgradient_train.studies+rexgradient_valid.studies+rexgradient_test.studies" expect="160,000" -->
<!-- claim: id=rexgradient.all.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py rexgradient rexgradient_train.patients+rexgradient_valid.patients+rexgradient_test.patients" expect="109,487" -->
ReXGradient-160K is a chest X-ray dataset from 79 sites in the U.S. The Rajpurkar Lab released it on
Hugging Face. It has 273,004 images of 160,000 studies and 109,487 patients. Each study has a report in 4
sections: Indication, Comparison, Findings and Impression. The source has no labels.

The PNG width and height are 25% of the original. The ReXrank benchmark keeps a private test set, and this
page does not cover it. The source also has `interstitial_pattern_bbox.json`, with boxes on 398 images.
The harmonizer does not read it.

## Quick use

`base_image_dir` is the `deid_png/` folder. The class finds the JSON file of its split in this folder, or
in `download/metadata/` of the folder above it (D-35).

```
ReXGradient-160K/
  deid_png/<PatientID>/<AccessionNumber>/studies/<StudyUID>/series/<SeriesUID>/instances/<SopUID>.png
  download/metadata/
    train_metadata_view_position.json
    valid_metadata_view_position.json
    test_metadata_view_position.json
```

```python
import torch
from radharmony.dataset import ReXGradientTrainDataset, ReXGradientTestDataset

train = ReXGradientTrainDataset(
    base_image_dir="/data/ReXGradient-160K/deid_png/",
    output_report=True,
    dtype=torch.float32,
)
train_ds, val_ds = train.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["report"]: 'INDICATION:\n...\n\nCOMPARISON:\n...\n\nFINDINGS:\n...\n\nIMPRESSION:\n...'

test_ds = ReXGradientTestDataset(
    base_image_dir="/data/ReXGradient-160K/deid_png/",
    output_report=True,
    dtype=torch.float32,
).get_datasets()
```

## How it is harmonized

- Source: `<split>_metadata_view_position.json`. One JSON entry is one study, with a list of its images in
  `ImagePath`. One row is one image (D-03). The plain `<split>_metadata.csv` has no image paths, so a `.csv`
  path raises `ValueError`.
- IDs: `patient_id` is `PatientID`, for example `GRDNRZ6LE5QCTQ5A`. `study_id` is `StudyInstanceUid`, for
  example `1.2.826.0.1.3680043.8.498.96466585901041020901411755035473531374`. `AccessionNumber` goes in
  the extra column `accession_number` (D-01).
- `image_path`: `ImagePath` with the `../deid_png/` at the start removed. It is relative to
  `base_image_dir` (D-04). Example:
  `GRDNRZ6LE5QCTQ5A/GRDNZ9W6GFJMI7SO/studies/1.2.826.0.1.3680043.8.498.96466585901041020901411755035473531374/series/1.2.826.0.1.3680043.8.498.59425159848405213029298826561879810186/instances/1.2.826.0.1.3680043.8.498.51759003029321912113312962240139898600.png`.
- Report: the 4 sections in one text. Each section starts with its name, for example `FINDINGS:`. All
  images of one study get the same report. See [Reports](#reports).
- Labels (D-05): with `labels_csv_path`, the harmonizer joins 14 CheXpert labels on `accession_number`.
  `Present` gives 1, `Absent` gives 0, and any other value gives NaN. Without it, the harmonized table has
  no labels.
- View (D-01): from `ImageViewPosition`, one value for each image. `LL`, `RL` and `LAT` give `LATERAL`,
  `POSTERO_ANTERIOR` gives `PA`, and `ANTERO_POSTERIOR` gives `AP`. `UNKNOWN`, `N/A`, null and the text
  `None` give NaN. Other values, such as `LAO`, stay as in the source.
- Age and sex: `age_years` comes from `PatientAge`, in years (D-06). `PatientAge` is a DICOM age text
  such as `029Y` or `002D`. `sex` is `M`, `F` or NaN, from `PatientSex`. The value `O` gives NaN (D-26).
- Dropped rows: the harmonizer drops a study with no `ImagePath`. The source has no such study.

## Labels

The source has no labels. `labels_csv_path` adds 14 CheXpert labels that the model GPT-OSS-120B gave
from the reports. This label CSV is not part of the source, and the code and configs name no download for
it. Without it, `output_cls=True` warns and adds no `cls` key (D-11).

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
| `no_finding` | No finding. The label CSV gives `Abnormal_response` (is the study abnormal?). So the harmonizer flips it: `Absent` gives 1, and `Present` gives 0 |
| `pleural_effusion` | Pleural effusion |
| `pleural_other` | Another pleural finding |
| `pneumonia` | Pneumonia |
| `pneumothorax` | Pneumothorax |
| `support_devices` | A tube, line or other device |

## Reports

`report` holds the 4 sections in this order. Each section starts with its name in capital letters:

```
INDICATION:
Adm

COMPARISON:
Comparison exams: 2/20/2013

FINDINGS:
The lungs are poorly expanded with volume loss at both lung bases. ...

IMPRESSION:
...
```

- The harmonizer leaves out a section that is empty, NaN or the text `None`.
- A study with all 4 sections empty gets `report` None. With `output_report=True`, the dataset drops such
  rows and warns (D-33).

## Splits

<!-- claim: id=rexgradient.splits.train tier=ci cmd="python scripts/evalkit/dataset_stat.py rexgradient rexgradient_train.images" expect="238,968" -->
<!-- claim: id=rexgradient.splits.valid tier=ci cmd="python scripts/evalkit/dataset_stat.py rexgradient rexgradient_valid.images" expect="17,007" -->
<!-- claim: id=rexgradient.splits.test tier=ci cmd="python scripts/evalkit/dataset_stat.py rexgradient rexgradient_test.images" expect="17,029" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `ReXGradientTrainDataset` | `rexgradient_train` | 238,968 | `train_metadata_view_position.json` |
| `ReXGradientValidDataset` | `rexgradient_valid` | 17,007 | `valid_metadata_view_position.json` |
| `ReXGradientTestDataset` | `rexgradient_test` | 17,029 | `test_metadata_view_position.json`, the public test split. Use it as the test split |
| `ReXGradientDataset` | `rexgradient` | One split | One JSON file: `csv_path`, or the first file it finds (train, valid, then test). Use the split keys |

A split class raises `ValueError` for the JSON file of another split (D-09). `get_datasets(n_splits=5)`
makes a train and val split from one class. It groups by patient (D-10). No patient is in 2 splits.

## Arguments

The arguments are the same for the 3 split classes.

<!-- args: radharmony.dataset.ReXGradientTrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `deid_png/` folder. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | The JSON file of the split, not the CSV. Without it, the class looks in `base_image_dir` and in `download/metadata/` of the folder above it |
| `labels_csv_path` | `str` | No | `None` | The CSV of GPT-OSS-120B CheXpert labels. `output_cls=True` needs it |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor, with `labels_csv_path` or a saved harmonized table with the labels. Otherwise: not supported. The class warns and adds no key (D-11) |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_report` | `bool` | No | `False` | Add the `report` text to the sample. The class drops the rows with no report, and warns (D-33) |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The labels stay float32 |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import ReXGradientValidHarmonizer

h = ReXGradientValidHarmonizer(
    csv_path="/data/ReXGradient-160K/download/metadata/valid_metadata_view_position.json",
    base_image_dir="/data/ReXGradient-160K/deid_png/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'view_position', 'report', 'age_years', 'sex',
#  'accession_number', 'patient_sex', 'patient_age', 'patient_ethnic_group', 'institution', 'study_date']
```

`ReXGradientTrainHarmonizer` and `ReXGradientTestHarmonizer` take the same arguments. The extra columns
keep the source values: `patient_sex` keeps `O`, `patient_age` keeps the DICOM age text, and `institution` is
`InstitutionName`, the site.

<!-- args: radharmony.harmonizer.ReXGradientValidHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | The JSON file of the split. A `.csv` path raises `ValueError` |
| `base_image_dir` | `str` | No | `None` | The `deid_png/` folder |
| `labels_csv_path` | `str` | No | `None` | The CSV of GPT-OSS-120B CheXpert labels, joined on `accession_number` |

## Known data problems

- View: many images have the view `UNKNOWN`, `N/A` or no view. The train JSON also has values that are not
  views, such as `PICC LINE`, `ERECT` and `KUB` (D-01).
- Age: some studies have no `PatientAge`. The age unit can be days, weeks, months or years (D-06).
- Sex: `PatientSex` can be `O` or `U`, and some studies have no `PatientSex` (D-26).
