# MIMIC-CXR (DICOM)

<!-- claim: id=mimic_cxr.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr mimic_cxr_train.images" expect="291,240" -->
<!-- claim: id=mimic_cxr.train.studies tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr mimic_cxr_train.studies" expect="171,373" -->
<!-- claim: id=mimic_cxr.train.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr mimic_cxr_train.patients" expect="60,024" -->
<!-- claim: id=mimic_cxr.validate.images tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr mimic_cxr_validate.images" expect="2,320" -->
<!-- claim: id=mimic_cxr.validate.studies tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr mimic_cxr_validate.studies" expect="1,355" -->
<!-- claim: id=mimic_cxr.validate.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr mimic_cxr_validate.patients" expect="459" -->
<!-- claim: id=mimic_cxr.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr mimic_cxr_test.images" expect="3,556" -->
<!-- claim: id=mimic_cxr.test.studies tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr mimic_cxr_test.studies" expect="2,216" -->
<!-- claim: id=mimic_cxr.test.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr mimic_cxr_test.patients" expect="287" -->
| | |
|---|---|
| Registry keys | `mimic_cxr_train`, `mimic_cxr_validate`, `mimic_cxr_test` |
| Task and outputs | Classification, 14 labels (`cls`), and the report text (`report`) |
| Images in the table | train 291,240 (171,373 studies, 60,024 patients), validate 2,320 (1,355 studies, 459 patients), test 3,556 (2,216 studies, 287 patients) |
| Modality | Chest X-ray, 2-D, DICOM |
| License | PhysioNet credentialed access (CITI training and a signed DUA) |
| Download | [Get the data: MIMIC-CXR](../../../how-to/get-the-data.md#mimic-cxr-dicom) |

<!-- claim: id=mimic_cxr.source.images tier=nas cmd="python scripts/evalkit/dataset_stat.py --source mimic_cxr_train csv_path rows" expect="377,110" -->
<!-- claim: id=mimic_cxr.source.studies tier=nas cmd="python scripts/evalkit/dataset_stat.py --source mimic_cxr_train csv_path unique:study_id" expect="227,835" -->
<!-- claim: id=mimic_cxr.source.patients tier=nas cmd="python scripts/evalkit/dataset_stat.py --source mimic_cxr_train csv_path unique:subject_id" expect="65,379" -->
<!-- claim: id=mimic_cxr.train.images.again tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr mimic_cxr_train.images" expect="291,240" -->
MIMIC-CXR has chest X-rays and their reports from the Beth Israel Deaconess Medical Center.
`cxr-record-list.csv.gz` has 377,110 images in 227,835 studies of 65,379 patients. The 14 labels come
from the CheXpert labeler, which read each report. The default `drop_uncertain=True` drops each study with
a `-1` label, so 291,240 train images stay. For the JPEG copy of the same images, see
[MIMIC-CXR-JPG](mimic_cxr_jpg.md).

## Quick use

`base_image_dir` is the `files/` folder. The class finds `cxr-record-list.csv.gz` and
`cxr-study-list.csv.gz` in `base_image_dir` or in the folder above it. The label file and the split file
are in the MIMIC-CXR-JPG release, so give their paths.

```
mimic-cxr/2.1.0/
  cxr-record-list.csv.gz
  cxr-study-list.csv.gz
  files/p10/p10000032/s50414267.txt
  files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.dcm
mimic-cxr-jpg/2.0.0/
  mimic-cxr-2.0.0-chexpert.csv
  mimic-cxr-2.0.0-split.csv
```

```python
import torch
from radharmony.dataset import MIMICCXRTestDataset, MIMICCXRTrainDataset, MIMICCXRValidateDataset

kw = dict(
    base_image_dir="/data/mimic-cxr/2.1.0/files/",
    label_csv_path="/data/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv",
    split_csv_path="/data/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv",
    output_cls=True,
    output_report=True,
    dtype=torch.float32,
)
train_ds = MIMICCXRTrainDataset(**kw).get_datasets()
val_ds = MIMICCXRValidateDataset(**kw).get_datasets()
test_ds = MIMICCXRTestDataset(**kw).get_datasets()
sample = val_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(14,)
# sample["report"]: str
```

## How it is harmonized

- Source: `cxr-record-list.csv.gz`. One row is one DICOM image. A split class keeps the images of its
  split in `mimic-cxr-2.0.0-split.csv`, by `dicom_id` (D-09).
- IDs: `subject_id` 10000032 gives `patient_id` `10000032`, and `study_id` 50414267 gives `50414267`. The
  `path` `files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.dcm` gives
  `image_path` `p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.dcm`, relative to
  `base_image_dir` (D-04).
- Labels (D-05): `mimic-cxr-2.0.0-chexpert.csv` has one row for each study. The harmonizer joins it on
  `patient_id` and `study_id`, so all images of one study get the same labels. `1` gives 1 and `0` gives
  0. A blank gives 0 (`not_mentioned="negative"`) or NaN (`not_mentioned="nan"`). A study with every label
  blank gets NaN in every label.
- View (D-01): only with `dicom_base_dir`. The harmonizer reads the DICOM tag `ViewPosition` of each
  file. A shared step maps the values to `AP`, `PA` and `LATERAL`. A rare value keeps its name in upper
  case. Without `dicom_base_dir`, the harmonized table has no `view_position`.
- Age and sex: the source has no age and no sex.
- Dropped rows: with `drop_uncertain=True` (the default), the harmonizer drops a study with `-1` in any
  label, with all its images. With `False`, `-1` stays in `cls`. The harmonizer also drops an image whose
  study has no row in the label file, and warns with the count (D-25).

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

## Reports

<!-- claim: id=mimic_cxr.source.reports tier=nas cmd="python scripts/evalkit/dataset_stat.py --source mimic_cxr_train report_csv_path rows" expect="227,835" -->
With `output_report=True`, `sample["report"]` is the text of the report file of the study.
`cxr-study-list.csv.gz` lists 227,835 report files. Its `path`, for example
`files/p10/p10000032/s50414267.txt`, gives the file `p10/p10000032/s50414267.txt` in `base_image_dir`. All
images of one study get the same report. The report generation evaluator finds the FINDINGS and
IMPRESSION sections from the headers in the text (D-38).

## Splits

<!-- claim: id=mimic_cxr.splits.train tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr mimic_cxr_train.images" expect="291,240" -->
<!-- claim: id=mimic_cxr.splits.validate tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr mimic_cxr_validate.images" expect="2,320" -->
<!-- claim: id=mimic_cxr.splits.test tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr mimic_cxr_test.images" expect="3,556" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `MIMICCXRTrainDataset` | `mimic_cxr_train` | 291,240 | The `train` images of `mimic-cxr-2.0.0-split.csv` |
| `MIMICCXRValidateDataset` | `mimic_cxr_validate` | 2,320 | The `validate` images of the split file |
| `MIMICCXRTestDataset` | `mimic_cxr_test` | 3,556 | The `test` images of the split file, with the labels from the CheXpert labeler. Use it as the test split |
| `MIMICCXRDataset` | `mimic_cxr` | | All images. It mixes the 3 splits, so do not use it for a benchmark (D-09) |

No patient is in 2 splits. For the radiologist labels of the test set, use `mimic_cxr_jpg_test` on the
[MIMIC-CXR-JPG](mimic_cxr_jpg.md) page.

## Arguments

The arguments are the same for `MIMICCXRTrainDataset`, `MIMICCXRValidateDataset` and
`MIMICCXRTestDataset`. `MIMICCXRDataset` has no `split_csv_path`.

<!-- args: radharmony.dataset.MIMICCXRTrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `files/` folder. Not needed with `harmonizer_path`, which holds it |
| `dicom_base_dir` | `str` | No | `None` | The DICOM `files/` folder, usually the same as `base_image_dir`. With it, the harmonizer reads `ViewPosition` from each file. With `None`, the harmonized table has no `view_position` |
| `csv_path` | `str` | No | `None` | `cxr-record-list.csv.gz`. Without it, the class looks in `base_image_dir` and the folder above it |
| `label_csv_path` | `str` | No | `None` | `mimic-cxr-2.0.0-chexpert.csv`. Without it, the class looks in the same places. With `output_cls=True`, a missing file raises `FileNotFoundError` |
| `report_csv_path` | `str` | No | `None` | `cxr-study-list.csv.gz`. Without it, the class looks in the same places. `output_report=True` needs it |
| `split_csv_path` | `str` | No | `None` | `mimic-cxr-2.0.0-split.csv`. Without it, the class looks next to `csv_path` and `label_csv_path`. A missing file raises `FileNotFoundError` |
| `drop_uncertain` | `bool` | No | `True` | Drop the studies with an uncertain label (`-1`). With `False`, `-1` stays in `cls` |
| `not_mentioned` | `str` | No | `"negative"` | A blank label. `"negative"` gives 0. `"nan"` keeps NaN, so the probes skip it (D-05) |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample |
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
from radharmony.harmonizer import MIMICCXRValidateHarmonizer

h = MIMICCXRValidateHarmonizer(
    csv_path="/data/mimic-cxr/2.1.0/cxr-record-list.csv.gz",
    label_csv_path="/data/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv",
    dicom_base_dir="/data/mimic-cxr/2.1.0/files/",
    report_csv_path="/data/mimic-cxr/2.1.0/cxr-study-list.csv.gz",
    report_base_dir="/data/mimic-cxr/2.1.0/files/",
    split_csv_path="/data/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv",
)
df = h.harmonize()  # harmonize() takes drop_uncertain and not_mentioned, as the dataset does
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'view_position', 'report', 'atelectasis', ...,
#  'support_devices']
```

`MIMICCXRTrainHarmonizer` and `MIMICCXRTestHarmonizer` take the same arguments. With `dicom_base_dir`,
the harmonizer opens every DICOM file of the split, so the train split takes a long time.

<!-- args: radharmony.harmonizer.MIMICCXRValidateHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `cxr-record-list.csv.gz` |
| `label_csv_path` | `str` | No | `None` | `mimic-cxr-2.0.0-chexpert.csv`. With `None`, the harmonized table has no label columns |
| `dicom_base_dir` | `str` | No | `None` | The DICOM `files/` folder, to read `ViewPosition` |
| `report_csv_path` | `str` | No | `None` | `cxr-study-list.csv.gz`. With `None`, the harmonized table has no `report` |
| `report_base_dir` | `str` | No | `None` | The `files/` folder. The harmonizer removes `files/` from the start of each report `path` and reads the file from this folder. The dataset classes pass `base_image_dir` as this argument |
| `split_csv_path` | `str` | No | `None` | `mimic-cxr-2.0.0-split.csv`. Without it, the class looks next to `csv_path` and `label_csv_path` |

## Known data problems

- 15 train images in 8 studies have no row in `mimic-cxr-2.0.0-chexpert.csv` (D-25).
- Some DICOM files have an empty or missing `ViewPosition` tag (D-01).
