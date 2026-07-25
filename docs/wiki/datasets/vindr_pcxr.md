# VinDr-PCXR Pediatric Chest X-ray

**Modality:** CXR | **Format:** DICOM | **Dim:** 2D | **Labels:** 15 binary conditions + bounding boxes

## Overview

9,125 pediatric chest X-ray DICOMs (7,728 train + 1,397 test) from VinMec International Hospital, Vietnam. Annotated by experienced radiologists with image-level binary labels and bounding boxes for 37 thoracic conditions.

- **9,125 DICOMs** — split into `train/` (7,728) and `test/` (1,397)
- **15 binary image-level labels** — one annotation per image
- **~12,222 bounding boxes** — pixel-space coords converted to normalized `[y_min, y_max, x_min, x_max]`

## Download

Access requires credentialed PhysioNet account (free registration + DUA).

```bash
# Requires physionet.org account
wget -r -N -c -np --user <username> --ask-password \
  https://physionet.org/files/vindr-pcxr/1.0.0/ \
  -P ./VINDR-PCXR
```

Or via the PhysioNet web interface at https://physionet.org/content/vindr-pcxr/

## Expected layout

```
VINDR-PCXR/                     ← base_image_dir points here
  train/
    6cb53aff85c71b98ad13d67a131708c6.dicom
    ...                          ← 7,728 DICOMs
  test/
    d7e71a052a753c3f2f3e317d60177bec.dicom
    ...                          ← 1,397 DICOMs
  image_labels_train.csv
  image_labels_test.csv
  annotations_train.csv
  annotations_test.csv
```

## Label columns

15 binary conditions (image-level, one radiologist per image).

| Column | Description |
|--------|-------------|
| `no_finding` | No abnormality detected |
| `bronchitis` | Bronchitis |
| `brocho_pneumonia` | Broncho-pneumonia (as in original CSV) |
| `other_disease` | Other disease not listed |
| `bronchiolitis` | Bronchiolitis |
| `situs_inversus` | Situs inversus |
| `pneumonia` | Pneumonia |
| `pleuro_pneumonia` | Pleuro-pneumonia |
| `diagphramatic_hernia` | Diaphragmatic hernia (as in original CSV) |
| `tuberculosis` | Tuberculosis |
| `congenital_emphysema` | Congenital emphysema |
| `cpam` | Congenital pulmonary airway malformation |
| `hyaline_membrane_disease` | Hyaline membrane disease |
| `mediastinal_tumor` | Mediastinal tumor |
| `lung_tumor` | Lung tumor |

## Bounding box classes

37 classes from `annotations_{split}.csv` (stored as `bbox_labels`):

`Anterior mediastinal mass`, `Aortic enlargement`, `Atelectasis`,
`Boot-shaped heart`, `Bronchectasis`, `Bronchial thickening`,
`Calcification`, `Cardiomegaly`, `Chest wall mass`, `Clavicle fracture`,
`Consolidation`, `Dextro cardia`, `Diffuse aveolar opacity`, `Edema`,
`Egg on string sign`, `Emphysema`, `Enlarged PA`, `Expanded edges of the anterior ribs`,
`Infiltration`, `Interstitial lung disease - ILD`, `Intrathoracic digestive structure`,
`Lung cavity`, `Lung cyst`, `Lung hyperinflation`, `Mediastinal shift`,
`No finding`, `Other lesion`, `Other nodule/mass`, `Other opacity`,
`Paraveterbral mass`, `Peribronchovascular interstitial opacity`,
`Pleural effusion`, `Pleural thickening`, `Pneumothorax`,
`Pulmonary fibrosis`, `Reticulonodular opacity`, `Stomach on the right side`

## Extra metadata columns

| Column | Type | Description |
|--------|------|-------------|
| `split` | `str` | `"train"` or `"test"` |
| `image_width` | `int` | DICOM image width in pixels |
| `image_height` | `int` | DICOM image height in pixels |

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes* | `None` | Dataset root (contains `train/`, `test/`, CSV files) |

### Shared arguments (inherited from `BaseRadiologicalDataset`)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `output_cls` | `bool` | No | `False` | Include `"cls"` tensor (15 labels) in data dict |
| `output_mask` | `bool` | No | `False` | Not supported; ignored |
| `output_report` | `bool` | No | `False` | Not supported; ignored |
| `output_bbox` | `bool` | No | `False` | Include `"bbox"` and `"bbox_labels"` in data dict |
| `transform` | Compose | No | standard 2-D 224 px | MONAI Compose transform |
| `cache_dir` | `str` | No | `"./cache"` | MONAI cache directory. `None` disables |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonizer pickle |

\* Required unless `harmonizer_path` or `harmonized_df` is provided.

## Dataset constructor

```python
from radharmony.dataset import VinDrPCXRDataset

ds = VinDrPCXRDataset(
    base_image_dir="/data/VINDR-PCXR/",
    output_cls=True,
    output_bbox=True,
    cache_dir="./cache",
)

sample = ds.get_datasets()[0]
print(sample["img"].shape)        # torch.Size([1, 224, 224])
print(sample["cls"].shape)        # torch.Size([15])
print(sample["bbox"])             # [[y_min, y_max, x_min, x_max], ...]
print(sample["bbox_labels"])      # ["Bronchial thickening", ...]
```

Filter to train split only:

```python
h = VinDrPCXRHarmonizer(base_dir="/data/VINDR-PCXR/")
df = h.harmonize()
train_df = df[df["split"] == "train"].reset_index(drop=True)

ds = VinDrPCXRDataset(
    base_image_dir="/data/VINDR-PCXR/",
    harmonized_df=train_df,
    output_cls=True,
)
```

## Harmonizer

```python
from radharmony.harmonizer import VinDrPCXRHarmonizer

h = VinDrPCXRHarmonizer(base_dir="/data/VINDR-PCXR/")
df = h.harmonize()
print(df.shape)           # (9125, 23)
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'bbox', 'bbox_labels',
#  'no_finding', 'bronchitis', ..., 'split', 'image_width', 'image_height']
```

## Harmonizer notes

- **No PatientID/StudyUID in DICOM** — `patient_id` and `study_id` are both set to `image_id` (the filename stem).
- **One annotation per image** — each image was annotated by a single radiologist (identified by `rad_ID`).
- **Bbox normalization** — DICOM dimensions are read from each image during `harmonize()` (header only, fast); bbox coords converted from pixel `[x_min, y_min, x_max, y_max]` to normalized `[y_min, y_max, x_min, x_max]`.
- **All images have bboxes** — all 9,125 images have at least one annotation in the annotations CSV.
- **Typos preserved** — `brocho_pneumonia` and `diagphramatic_hernia` follow the original CSV column names verbatim.
