# Emory CXR v2

<!-- claim: id=emory_cxr.images tier=ci cmd="python scripts/evalkit/dataset_stat.py emory_cxr emory_cxr.images" expect="2,430,209" -->
<!-- claim: id=emory_cxr.studies tier=ci cmd="python scripts/evalkit/dataset_stat.py emory_cxr emory_cxr.studies" expect="1,214,535" -->
<!-- claim: id=emory_cxr.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py emory_cxr emory_cxr.patients" expect="402,809" -->
| | |
|---|---|
| Registry keys | `emory_cxr` (PNG), `emory_cxr_dicom` (DICOM) |
| Task and outputs | Classification, 14 labels (`cls`), and the report text (`report`) |
| Images in the table | 2,430,209 images, 1,214,535 studies and 402,809 patients (manifest `08152025`) |
| Modality | Chest X-ray, 2-D. 16-bit PNG (`emory_cxr`) or 8-bit DICOM (`emory_cxr_dicom`) |
| License | Internal Emory data. It is not public, and you must not share it |
| Download | No download. The data paths are in `configs/data_paths.json` |

Emory CXR v2 is an internal chest X-ray dataset of Emory. The images are from 2008 to 2021, from 11
institutions. The institution names are de-identified.

Almost every study has a de-identified free-text report. The reports are the main reason to use this
dataset, for example as a report-generation target. The 14 labels follow CheXpert. Most label cells are
blank, and a blank means "not mentioned", not "absent".

## Quick use

`base_image_dir` is `DEID_PNG/`. The class finds the 3 tables in `TABLES/`, next to `DEID_PNG/`. It
reads the label table only with `output_cls=True`, and the report table only with `output_report=True`.

```
EmoryCXRv2/
  DEID_PNG/<empi_anon>/<AccessionNumber_anon>/<SOP>.png
  DEID_DICOM/B1/ ... DEID_DICOM/B5/
  TABLES/
    EmoryCXR_v2_Metadata_08152025.csv
    EmoryCXR_v2_FindingLabel_10162024.csv
    EmoryCXR_v2_Report_05102026.csv
    mapping_deid_dcm_sopid.csv
```

```python
import torch
from radharmony.dataset import EmoryCXRDataset, EmoryCXRDicomDataset

ds = EmoryCXRDataset(
    base_image_dir="/data/EmoryCXRv2/DEID_PNG/",
    output_cls=True,
    output_report=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(14,), with 1, 0, -1 and NaN
# sample["report"]: str

# The DICOM tree. The class finds TABLES/mapping_deid_dcm_sopid.csv.
dicom_ds = EmoryCXRDicomDataset(
    base_image_dir="/data/EmoryCXRv2/DEID_DICOM/",
    dtype=torch.float32,
).get_datasets()
# dicom_ds[0]["img"]: Tensor(1, 224, 224)
```

## How it is harmonized

- Source: 3 tables, joined on `AccessionNumber_anon` as a string. The metadata manifest has one row for
  each image. One row of the harmonized table is one image (D-03).
- IDs: `patient_id` is `empi_anon`. `study_id` and `accession_number` are `AccessionNumber_anon`.
  `image_path` is `ImagePath`, relative to `DEID_PNG/` (D-04). Example: `image_path`
  `84269073/3739633430788944/1.2.826.0.1.3680043.8.498.11539031758371223425771024448750240761.png`
  has `patient_id` `84269073` and `study_id` `3739633430788944`.
- Labels (D-05): the labels are for each study. A left join gives them to each image of the study. The
  14 columns get snake_case names. An image whose study has no label row gets NaN in every label.
- Uncertain and blank labels: `-1` stays in `cls`, and a blank stays NaN. There is no `drop_uncertain`
  and no `not_mentioned` argument. A study can have 2 label rows. If one row has only `No Finding` = 1,
  the harmonizer keeps the other row (D-03).
- View (D-01): the PNG class uses `Frontal` first, and `ViewPosition` second:
  - `Frontal` 1 with `AP` or `PA` keeps `AP` or `PA`.
  - `Frontal` 1 with `Lateral` or a blank gives NaN, because AP or PA is unknown.
  - `Frontal` 1 with `Other` gives `OTHER`.
  - `Frontal` 0 with `AP` gives NaN.
  - `Frontal` 0 with any other value (`PA`, `Lateral`, `Other` or a blank) gives `LATERAL`.
  - The DICOM class keeps `ViewPosition` as in the source.
- Age and sex: `age_years` comes from `Age`, in years (D-06). `sex` is `M`, `F` or NaN, from `Sex`
  (D-26). The raw values stay in `patient_age` and `patient_sex`.
- Dropped rows: none in the PNG class. The DICOM class joins `mapping_deid_dcm_sopid.csv` on the `SOP`
  column, and drops an image with no mapping row. Its `image_path` starts at the batch folder, for
  example `B2/mdai_emory_project_..._part01of10/<UID>/<UID>/<UID>.dcm`. `<UID>` is a new MD.ai UID, not
  the `SOP`. Only the mapping CSV links the two.

## Labels

The 14 label columns, in the order of `cls`:

| Column | Source column | Meaning |
|---|---|---|
| `atelectasis` | `Atelectasis` | Part of a lung is collapsed |
| `cardiomegaly` | `Cardiomegaly` | The heart is large |
| `consolidation` | `Consolidation` | Airspace consolidation |
| `edema` | `Edema` | Pulmonary edema |
| `enlarged_cardiomediastinum` | `Enlarged Cardiomediastinum` | The mediastinum is wide |
| `fracture` | `Fracture` | A rib or other bone fracture |
| `lung_lesion` | `Lung Lesion` | A lung lesion |
| `lung_opacity` | `Lung Opacity` | A lung opacity |
| `no_finding` | `No Finding` | No finding |
| `pleural_effusion` | `Pleural Effusion` | Pleural effusion |
| `pleural_other` | `Pleural Other` | Another pleural finding |
| `pneumonia` | `Pneumonia` | Pneumonia |
| `pneumothorax` | `Pneumothorax` | Pneumothorax |
| `support_devices` | `Support Devices` | A tube, line or other device |

## Reports

The report table has the text in the column `deid_report`. The harmonized table gives it in `report`. A
report has 66 words on average. 98.9% of the reports have an IMPRESSION section, and 86.5% have a
FINDINGS section.

A blank template (only "REPORT", spaces and underscores) counts as no report. When a study has a blank
template and a real report, the harmonizer keeps the real report (D-03). With `output_report=True`, the
dataset drops the images with no report (D-33).

## Splits

<!-- claim: id=emory_cxr.splits.png tier=ci cmd="python scripts/evalkit/dataset_stat.py emory_cxr emory_cxr.images" expect="2,430,209" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `EmoryCXRDataset` | `emory_cxr` | 2,430,209 | `DEID_PNG/`, 16-bit PNG. Use it for training |
| `EmoryCXRDicomDataset` | `emory_cxr_dicom` | The same images | `DEID_DICOM/`, 8-bit DICOM. Use it only for the DICOM headers, or to send annotations to MD.ai and read them back |

Emory CXR v2 has no official split. `get_datasets(n_splits=5)` makes a train and val split. It groups by
patient (D-10). Do not split by study: 42.7% of the patients have more than one study (up to 296). The
app lists only `emory_cxr`, because `emory_cxr_dicom` shows the same images (D-09).

## Arguments

<!-- args: radharmony.dataset.EmoryCXRDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | `DEID_PNG/`, or `DEID_DICOM/` for the DICOM class. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `EmoryCXR_v2_Metadata_<date>.csv`. Without it, the class tries the known manifests in `TABLES/`, newest first. Pass it to always use the same manifest |
| `label_csv_path` | `str` | No | `None` | `EmoryCXR_v2_FindingLabel_<date>.csv`. The class reads it only with `output_cls=True`. Without it, the class looks in `TABLES/` |
| `report_csv_path` | `str` | No | `None` | `EmoryCXR_v2_Report_<date>.csv`. The class reads it only with `output_report=True`. Without it, the class looks in `TABLES/` |

`EmoryCXRDicomDataset` takes the same arguments, plus `mapping_csv_path` (default `None`). This is
`mapping_deid_dcm_sopid.csv`. Without it, the class looks in `TABLES/`. If it finds no file, it raises
`ValueError`.

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
from radharmony.harmonizer import EmoryCXRHarmonizer

h = EmoryCXRHarmonizer(
    csv_path="/data/EmoryCXRv2/TABLES/EmoryCXR_v2_Metadata_08152025.csv",
    base_image_dir="/data/EmoryCXRv2/DEID_PNG/",
    label_csv_path="/data/EmoryCXRv2/TABLES/EmoryCXR_v2_FindingLabel_10162024.csv",
    report_csv_path="/data/EmoryCXRv2/TABLES/EmoryCXR_v2_Report_05102026.csv",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'view_position', 'report', 'age_years', 'sex',
#  'atelectasis', ..., 'support_devices', 'accession_number', 'patient_sex', 'patient_age',
#  'study_date', 'institution', 'frontal']
```

`frontal` is 1 for a frontal image and 0 for the other images. `institution` and `study_date` are
de-identified. `EmoryCXRDicomHarmonizer` takes the same arguments, plus `mapping_csv_path` (required).
`EmoryCXRDeidHarmonizer` is the same as `EmoryCXRHarmonizer`, but it raises `ValueError` for an
`ORIGINAL_PNG` folder.

<!-- args: radharmony.harmonizer.EmoryCXRHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `EmoryCXR_v2_Metadata_<date>.csv` |
| `base_image_dir` | `str` | No | `None` | `DEID_PNG/` |
| `label_csv_path` | `str` | No | `None` | `EmoryCXR_v2_FindingLabel_<date>.csv`. `None` gives no labels |
| `report_csv_path` | `str` | No | `None` | `EmoryCXR_v2_Report_<date>.csv`. `None` gives no reports |

## Known data problems

<!-- claim: id=emory_cxr.manifest_08112025.images tier=nas cmd="python scripts/evalkit/dataset_stat.py --source emory_cxr base_image_dir/../TABLES/EmoryCXR_v2_Metadata_08112025.csv rows" expect="2,486,502" -->
<!-- claim: id=emory_cxr.manifest_08112025.patients tier=nas cmd="python scripts/evalkit/dataset_stat.py --source emory_cxr base_image_dir/../TABLES/EmoryCXR_v2_Metadata_08112025.csv unique:empi_anon" expect="410,449" -->
<!-- claim: id=emory_cxr.manifest_08152025.images tier=ci cmd="python scripts/evalkit/dataset_stat.py emory_cxr emory_cxr.images" expect="2,430,209" -->
<!-- claim: id=emory_cxr.manifest_08152025.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py emory_cxr emory_cxr.patients" expect="402,809" -->
- View: for some images, `Frontal` and `ViewPosition` do not agree. In a manual check of some images,
  `Frontal` was right (D-01).
- View: most images with `Frontal` 0 and `ViewPosition` AP are upright frontal or decubitus images, not
  lateral images (D-01).
- Repeated rows: some studies have 2 or more rows in the label or report table. The report table has 151
  such studies, and the label table has 75. Most of them have a blank report template and a real
  report. 8 label and 17 report conflicts have 2 real rows that do not agree (D-03).
- Reports: 3 studies (7 images) have only a blank report template, so they have no report (D-03, D-33).
- Labels: most label cells are blank, and `No Finding` has no 0. So a blank as 0 gives false negatives.
- Manifests: the dated metadata manifests are not the same cohort. `08112025` has 2,486,502 images of
  410,449 patients. `08152025` keeps only an age of 18 or more, and has 2,430,209 images of 402,809
  patients.
- Pixels: each PNG is min-max scaled to the full 16-bit range, with no VOI LUT. So pixel values do not
  compare across images. `ORIGINAL_PNG/` has a different scaling. Do not mix it with `DEID_PNG/`.
- DICOM tree: the files are an 8-bit MD.ai export, not the source DICOM files. `AccessionNumber` is
  blank, and `PatientID` is different in each file. Only `mapping_deid_dcm_sopid.csv` links a file to a
  patient. The MD.ai manifests name 3 files that were never delivered.
