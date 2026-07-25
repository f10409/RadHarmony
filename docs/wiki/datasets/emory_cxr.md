# EmoryCXR v2

**Modality:** CXR | **Format:** PNG | **Dim:** 2D | **Labels:** 14 pathologies | **Access:** Internal (Emory)

## Overview

EmoryCXR v2 is an internal Emory de-identified chest radiography dataset containing ~2.4M images from ~360K patients across Emory Healthcare institutions. It includes:

- **Image-level metadata**: patient demographics (age, sex, BMI, race, ethnicity), acquisition parameters (view position, frontal flag), institution, and study date.
- **Study-level finding labels**: 14 CheXpert-aligned pathology labels (binary 0/1/NaN; no uncertain `-1`).
- **Study-level de-identified reports**: free-text radiology reports stored inline in a CSV (not as separate files).

Labels and reports are stored at the accession (study) level and are automatically broadcast to all images within each study by the harmonizer.

## Data paths

| Resource | Path |
|---|---|
| Images (PNG) | `/path/to/EmoryCXRv2/DEID_PNG/` |
| Metadata CSV | `/path/to/EmoryCXRv2/TABLES/metadata.csv` |
| Label CSV | `/path/to/EmoryCXRv2/TABLES/finding_labels.csv` |
| Report CSV | `/path/to/EmoryCXRv2/TABLES/reports.csv` |

Image paths follow the structure: `<empi_anon>/<AccessionNumber_anon>/<SOP>.png` relative to the PNG root.

## Label columns

14 CheXpert-aligned findings (binary 0/1/NaN, no `-1` uncertain):

`atelectasis`, `cardiomegaly`, `consolidation`, `edema`, `enlarged_cardiomediastinum`, `fracture`, `lung_lesion`, `lung_opacity`, `no_finding`, `pleural_effusion`, `pleural_other`, `pneumonia`, `pneumothorax`, `support_devices`

## Extra metadata columns

The following demographic and acquisition columns are passed through to the harmonized DataFrame:

| Column | Description |
|---|---|
| `sex` | Patient sex |
| `age` | Patient age |
| `bmi` | Body mass index |
| `race` | Patient race |
| `ethnicity` | Patient ethnicity |
| `frontal` | Whether the view is frontal (bool) |
| `institution` | De-identified institution name |
| `study_date` | De-identified study date |

## Harmonizer

```python
from radharmony.harmonizer import EmoryCXRHarmonizer

h = EmoryCXRHarmonizer(
    csv_path="/path/to/EmoryCXRv2/TABLES/metadata.csv",
    base_image_dir="/path/to/EmoryCXRv2/DEID_PNG",
    label_csv_path="/path/to/EmoryCXRv2/TABLES/finding_labels.csv",
    report_csv_path="/path/to/EmoryCXRv2/TABLES/reports.csv",
)
df = h.harmonize()
# df columns: patient_id, study_id, image_path,
#             view_position, report,
#             atelectasis, cardiomegaly, ..., support_devices,
#             sex, age, bmi, race, ethnicity, frontal, institution, study_date
```

## Dataset (MONAI)

```python
from radharmony.dataset import EmoryCXRDataset

ds = EmoryCXRDataset(
    base_image_dir="/path/to/EmoryCXRv2/DEID_PNG",
    csv_path="/path/to/EmoryCXRv2/TABLES/metadata.csv",
    label_csv_path="/path/to/EmoryCXRv2/TABLES/finding_labels.csv",
    report_csv_path="/path/to/EmoryCXRv2/TABLES/reports.csv",
    output_cls=True,
    output_report=True,
    cache_dir="./cache",
)
sample = ds[0]
# sample["img"]    → image tensor (C, H, W)
# sample["cls"]    → 14-class label tensor
# sample["report"] → de-identified report string
```

## Constructor arguments

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes* | `None` | PNG root (contains patient sub-folders) |
| `csv_path` | `str` | No | auto | Path to metadata CSV |
| `label_csv_path` | `str` | No | `None` | Path to finding label CSV; required for `output_cls=True` |
| `report_csv_path` | `str` | No | `None` | Path to report CSV; required for `output_report=True` |
| `output_cls` | `bool` | No | `False` | Yield 14-class label tensor under `cls` |
| `output_report` | `bool` | No | `False` | Yield de-identified report text under `report` |
| `output_mask` | `bool` | No | `False` | Not supported; silently ignored |
| `output_bbox` | `bool` | No | `False` | Not supported; silently ignored |
| `transform` | `Compose` | No | 2D default | MONAI transform pipeline |
| `cache_dir` | `str` | No | `./cache` | PersistentDataset cache root; `None` disables caching |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Image tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonized CSV |

*Required unless `harmonizer_path`, `harmonized_df`, or `harmonizer` is provided.

## Notes

- Labels are at study level; they are broadcast to all SOP images in the same accession via a left merge. Images whose accession has no label entry will have `NaN` for all label columns.
- Reports are stored as inline text in the report CSV (not as file paths). The `report` column in the harmonized DataFrame contains the de-identified text directly.
- Masks and bounding boxes are not available in this release.
