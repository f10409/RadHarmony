# RadHarmony — Dataset API Guide

## Overview

RadHarmony provides two ways to work with radiology datasets:

- **Dataset classes** — MONAI `PersistentDataset`-backed objects that load, transform, and split data, ready for use in a PyTorch training loop.
- **Harmonizer classes** — lower-level objects that parse CSVs and return a standardised `pd.DataFrame`. Use these when you need direct DataFrame access without the full dataset pipeline.

Both layers share the same CSV discovery logic: pass only `base_image_dir` and RadHarmony will find the required CSV files automatically.

---

## Quick start

```python
from radharmony.dataset import CheXpertDataset

ds = CheXpertDataset(base_image_dir="/data/CheXpert-v1.0/train/")
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

That is all. No CSV paths required when the files live in a standard location near the image directory.

---

## Dataset classes

### Available datasets

| Class | Registry key | Modality |
|---|---|---|
| `BRAXDataset` | `brax` | 2-D CXR (DICOM) |
| `BRAXPNGDataset` | `brax_png` | 2-D CXR (PNG) |
| `CheXpertDataset` | `chexpert` | 2-D CXR (JPEG) |
| `CheXpertPlusDataset` | `chexpert_plus` | 2-D CXR (DICOM) |
| `ChestXray14Dataset` | `chestxray14` | 2-D CXR (PNG) |
| `ChestXray14BboxDataset` | `chestxray14_bbox` | 2-D CXR (PNG) |
| `MIMICCXRDataset` | `mimic_cxr` | 2-D CXR (DICOM) |
| `MIMICCXRJPGDataset` | `mimic_cxr_jpg` | 2-D CXR (JPEG) |
| `RANZCRClipDataset` | `ranzcr_clip` | 2-D CXR (JPEG) |
| `RSNAPneumoniaDataset` | `rsna_pneumonia` | 2-D CXR (DICOM) |
| `SIIMACRPTXDataset` | `siim_acr_ptx` | 2-D CXR (DICOM) |
| `VinDrCXRTrainDataset` | `vindr_cxr_train` | 2-D CXR (DICOM) |
| `VinDrCXRTestDataset` | `vindr_cxr_test` | 2-D CXR (DICOM) |
| `TAIXRay512Dataset` | `taix_ray_512` | 2-D CXR (PNG, 512 px) |
| `TAIXRayDataset` | `taix_ray` | 2-D CXR (PNG, native res) |
| `RSNA2022CervicalSpineDataset` | `rsna_2022_cervical_spine` | 3-D CT (DICOM volumes) |
| `RSNA2022CervicalSpineBboxDataset` | `rsna_2022_cervical_spine_bbox` | 2-D CT axial slices (DICOM) |
| `CTRATEDataset` | `ct_rate` | 3-D CT (NIfTI) |
| `RadChestCTDataset` | `radchestct` | 3-D CT (NumPy) |

Import a class directly or look it up by registry key:

```python
from radharmony.dataset import MIMICCXRJPGDataset

# or via registry
from radharmony.registry import resolve_dataset
MIMICCXRJPGDataset = resolve_dataset("mimic_cxr_jpg")
```

### Common constructor arguments

| Argument | Type | Default | All datasets | Notes |
|---|---|---|---|---|
| `base_image_dir` | `str` | required | yes | Root directory of the image files |
| `transform` | MONAI Compose | auto | yes | Transform pipeline; auto-built when `None` |
| `cache_dir` | `str` | `"./cache"` | yes | MONAI `PersistentDataset` cache root; `None` disables caching |
| `output_cls` | `bool` | `False` | yes | Include label vector under key `cls` |
| `output_mask` | `bool` | `False` | most¹ | Include segmentation mask under key `mask` |
| `output_report` | `bool` | `False` | most¹ | Include radiology report text under key `report` |
| `output_bbox` | `bool` | `False` | most¹ | Include bounding box under key `bbox` |
| `drop_uncertain` | `bool` | `True` | CheXpert, MIMIC-CXR | Drop rows where any label is uncertain (`-1` → `NaN`) |

¹ Not available in `CheXpertDataset` (only `output_cls` is supported).

Dataset-specific CSV path arguments are all optional — see the per-dataset sections below.

### Getting datasets

```python
# Full dataset (no split)
full_ds = ds.get_datasets()

# Train/val split — 1/n_splits fraction goes to val
train_ds, val_ds = ds.get_datasets(n_splits=5)

# k-fold cross-validation
for fold_idx, (train_ds, val_ds) in enumerate(ds.get_folds(n_splits=5)):
    ...
```

`get_datasets` and `get_folds` accept `num_cores` (default `2`) for parallel data dict construction and `random_state` (default `56`) for reproducible splits.

### Verifying image files

Before building datasets, you can check that all image paths in the harmonized table point to files that exist on disk:

```python
# At dataset level — uses base_image_dir automatically
missing = ds.verify_images()              # drops missing rows
missing = ds.verify_images(drop_missing=False)  # inspect only

# At harmonizer level — pass base_image_dir explicitly
missing = h.verify_images(base_image_dir="/data/CheXpert-v1.0/train/")
```

Both return a DataFrame of the rows with missing files. With `drop_missing=True` (default), the harmonized DataFrame is updated in place so subsequent `get_datasets()` calls skip the missing rows.

Typical workflow:

```python
ds = CheXpertDataset(base_image_dir="/data/CheXpert-v1.0/train/", output_cls=True)
missing = ds.verify_images()                    # check & clean
train_ds, val_ds = ds.get_datasets(n_splits=5)  # only valid rows
```

### Sample output

Each sample is a dict. Keys present depend on which `output_*` flags are enabled:

| Key | Shape | dtype | Always present |
|---|---|---|---|
| `img` | `(1, H, W)` or `(1, D, H, W)` | bfloat16, `[-1, 1]` | yes |
| `cls` | `(n_labels,)` | bfloat16 | only when `output_cls=True` |
| `mask` | same as `img` | bfloat16 | only when `output_mask=True` |
| `report` | `str` | — | only when `output_report=True` |
| `bbox` | `(4,)` or `(6,)` | bfloat16, normalised `[0, 1]` | only when `output_bbox=True` |

### Label columns

Dataset `LABEL_COLS` use snake_case (matching the harmonized DataFrame). Harmonizer `LABEL_COLS` keep the original raw column names.

```python
print(CheXpertDataset.LABEL_COLS)
# ['atelectasis', 'cardiomegaly', 'consolidation', 'edema', ...]

print(CheXpertHarmonizer.LABEL_COLS)
# ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', ...]
```

### Pre-caching

For long training runs, warm the cache before training begins:

```python
train_ds, val_ds = ds.get_datasets(n_splits=5)
ds.pre_cache(train_ds, num_workers=4)
ds.pre_cache(val_ds,   num_workers=4)
```

### Visualisation

```python
import matplotlib.pyplot as plt

for fig in ds.visualize_samples(train_ds, n=4):
    display(fig)
    plt.close(fig)

# One representative image per label
for fig in ds.visualize_samples(train_ds, per_label=True):
    display(fig)
    plt.close(fig)
```

---

## CSV path auto-discovery

All CSV path arguments default to `None`. When `None`, RadHarmony searches for the file automatically using a four-step resolution strategy anchored at `base_image_dir`:

1. **User-provided path** — if passed and exists on disk, used immediately.
2. **Direct child** — looks for the file directly inside `base_image_dir`.
3. **Parent / siblings** — checks the parent directory, then recursively searches each sibling directory of `base_image_dir`.
4. **Uncle directories** — if step 3 found nothing, recursively searches siblings of the parent.

The search never descends into `base_image_dir` itself (the image tree), avoiding false matches. If discovery fails and no path was provided, the harmonizer raises an error.

---

## Per-dataset reference

### BRAX (Brazilian Chest X-Ray)

[BRAX v1.1.0](https://physionet.org/content/brax/1.1.0/) — ~24,959 studies, ~40,967 images, 19,351 patients from a Brazilian hospital network. Labels are CheXpert-aligned (14 findings) and were generated by the original CheXpert NLP labeller back-end with a NegEx-Portuguese front-end translated and validated by Brazilian radiologists. Both DICOM and PNG variants ship in the same PhysioNet release. Credentialed access required.

Two thin dataset classes share a single `BRAXHarmonizer`:

| Class | base_image_dir target | Image format |
|---|---|---|
| `BRAXDataset` | BRAX root (e.g. `…/brax/1.1.0/`) | DICOM under `Anonymized_DICOMs/` |
| `BRAXPNGDataset` | BRAX root | PNG under `images/` |

```python
from radharmony.dataset import BRAXDataset, BRAXPNGDataset

# DICOM, faithful label preservation (default)
ds = BRAXDataset(
    base_image_dir="/data/physionet.org/files/brax/1.1.0/",
    csv_path=None,                # auto: master_spreadsheet.csv
    uncertain_strategy="raw",     # preserves source 1/0/-1/NaN
    output_cls=True,
)

# PNG, training-ready binary labels
ds = BRAXPNGDataset(
    base_image_dir="/data/physionet.org/files/brax/1.1.0/",
    uncertain_strategy="u_zeros", # -1 → 0; retain every row
    output_cls=True,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

**Constructor arguments** (dataset-specific; both classes)

| Argument | Type | Default | Description |
|---|---|---|---|
| `csv_path` | `str` | auto | Path to `master_spreadsheet.csv` |
| `uncertain_strategy` | `str` | `"raw"` | One of `"raw"`, `"u_zeros"`, `"u_ones"`, `"u_ignore"`, `"drop"`. See uncertainty handling below |

**Required files**

| Argument | File | Required |
|---|---|---|
| `csv_path` | `master_spreadsheet.csv` | Yes |

**Path construction**
```
base_image_dir + DicomPath = .../brax/1.1.0/Anonymized_DICOMs/id_<PID>/Study_<UID>/Series_<UID>/image-<UID>.dcm
base_image_dir + PngPath   = .../brax/1.1.0/images/id_<PID>/Study_<UID>/Series_<UID>/image-<UID>.png
```

**Labels** (14, snake_cased; CheXpert-aligned exactly):
`atelectasis`, `cardiomegaly`, `consolidation`, `edema`, `enlarged_cardiomediastinum`, `fracture`, `lung_lesion`, `lung_opacity`, `no_finding`, `pleural_effusion`, `pleural_other`, `pneumonia`, `pneumothorax`, `support_devices`.

**Cell encoding** (per the BRAX paper): `1` = positive, `0` = negation, `-1` = uncertainty, blank = labeller did not extract a mention.

**Uncertain handling.** Five strategies; default `"raw"` preserves the encoding exactly.

| Strategy | NaN → | -1 → | Rows dropped? |
|---|---|---|---|
| `"raw"` (default) | NaN | -1 | no |
| `"u_zeros"` | 0 | 0 | no |
| `"u_ones"` | 0 | 1 | no |
| `"u_ignore"` | 0 | NaN | no |
| `"drop"` | 0 | NaN, then drop rows with any remaining NaN | yes |

`"raw"` + `output_cls=True` will yield `cls` tensors containing `-1` — your loss must handle this. See `docs/uncertainty.md` for the broader project discussion on uncertainty handling.

**Extra metadata** (carried in the harmonized DataFrame): `view_position` (AP / PA / LL / RL / RLD / LLD / RLO / LLO), `patient_sex`, `patient_age` (5-year bins, capped at "85 or more"), `manufacturer` (integer-coded for de-bias work), `study_date` (fictitious), `rows`, `columns`, plus `series_id` extracted from the path's `Series_<UID>` segment.

**Note:** Reports themselves are *not* distributed in BRAX 1.1.0 — only the labels. `output_report` warns and is ignored. `output_mask` and `output_bbox` likewise unsupported.

---

### CheXpert

```python
from radharmony.dataset import CheXpertDataset

ds = CheXpertDataset(
    base_image_dir="/data/CheXpert-v1.0/train/",
    csv_path=None,          # auto-discovered: train.csv / valid.csv
    output_cls=True,
)
```

**Constructor arguments** (dataset-specific)

| Argument | Type | Default | Description |
|---|---|---|---|
| `csv_path` | `str` | auto | Path to `train.csv` / `valid.csv` |
| `drop_uncertain` | `bool` | `True` | Drop rows where any label is `-1` (uncertain) |

**Required CSV files**

| Argument | File | Required |
|---|---|---|
| `csv_path` | `train.csv` or `valid.csv` | Yes |

**Expected columns** — `Path`, `AP/PA`, `No Finding`, `Enlarged Cardiomediastinum`, `Cardiomegaly`, `Lung Opacity`, `Lung Lesion`, `Edema`, `Consolidation`, `Pneumonia`, `Atelectasis`, `Pneumothorax`, `Pleural Effusion`, `Pleural Other`, `Fracture`, `Support Devices`

**Note:** `CheXpertDataset` supports only `output_cls`. The flags `output_mask`, `output_report`, and `output_bbox` are not available.

**Path construction**
```
Path: "CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg"
→ patient_id : "patient00001"
→ study_id   : "patient00001_study1"
→ image_path : "patient00001/study1/view1_frontal.jpg"   (relative to base_image_dir)
```

**Label processing** — `NaN` → `0.0`; `-1` (uncertain) → `NaN` and row dropped by default.

---

### CheXpert-Plus (DICOM)

```python
from radharmony.dataset import CheXpertPlusDataset

ds = CheXpertPlusDataset(
    base_image_dir="/data/chexpertplus/DICOM/Uncompressed/",
    csv_path=None,            # auto-discovered: df_chexpert_plus_240401.csv
    label_json_path=None,     # auto-discovered: report_fixed.json
    output_cls=True,
    output_report=True,
)
```

**Constructor arguments** (dataset-specific)

| Argument | Type | Default | Description |
|---|---|---|---|
| `csv_path` | `str` | auto | Path to `df_chexpert_plus_240401.csv` |
| `label_json_path` | `str` | auto | Path to `report_fixed.json` (JSONL label file) |
| `drop_uncertain` | `bool` | `True` | Drop rows where any label is `null` (not annotated) |

**Required files**

| Argument | File | Required |
|---|---|---|
| `csv_path` | `df_chexpert_plus_240401.csv` | Yes |
| `label_json_path` | `report_fixed.json` | No (required for `output_cls=True`) |

**Note:** `base_image_dir` must point to the `DICOM/Uncompressed/` directory inside the CheXpert-Plus release (e.g. `.../chexpertplus/DICOM/Uncompressed/`).  `df_chexpert_plus_240401.csv` and `report_fixed.json` are auto-discovered near this directory.  `output_mask` and `output_bbox` are not supported.

**Path construction**
```
path_to_dcm: "train/patient42142/study5/view1_frontal.dcm"
→ patient_id : "patient42142"
→ study_id   : "patient42142_study5"
→ image_path : "train/patient42142/study5/view1_frontal.dcm"
```

**Label source** — `report_fixed.json` (one JSON object per line, keyed by `path_to_image`).  `null` entries are treated as NaN and dropped when `drop_uncertain=True`.

**View position** — derived from the `frontal_lateral` and `ap_pa` columns: frontal images use the `ap_pa` value (`AP` / `PA`); lateral images are labelled `Lateral`.

**Report** — inline text from the `report` column of `df_chexpert_plus_240401.csv`.

---

### MIMIC-CXR (DICOM)

```python
from radharmony.dataset import MIMICCXRDataset

ds = MIMICCXRDataset(
    base_image_dir="/data/mimic-cxr/2.1.0/files/",
    dicom_base_dir=None,    # optional: defaults to base_image_dir; reads ViewPosition from DICOM headers
    csv_path=None,          # auto-discovered: cxr-record-list.csv.gz
    label_csv_path=None,    # from MIMIC-CXR-JPG: mimic-cxr-2.0.0-chexpert.csv
    report_csv_path=None,   # auto-discovered: cxr-study-list.csv.gz
    output_cls=True,
    output_report=True,
)
```

**Constructor arguments** (dataset-specific)

| Argument | Type | Default | Description |
|---|---|---|---|
| `dicom_base_dir` | `str` | `None` | Root of the DICOM file tree for reading `ViewPosition` from headers. Defaults to `base_image_dir` when `None`. |
| `csv_path` | `str` | auto | Path to `cxr-record-list.csv.gz` |
| `label_csv_path` | `str` | `None` | Path to `mimic-cxr-2.0.0-chexpert.csv` (from MIMIC-CXR-JPG) |
| `report_csv_path` | `str` | auto | Path to `cxr-study-list.csv.gz` |
| `drop_uncertain` | `bool` | `True` | Drop rows where any label is `-1` (uncertain) |

**Required CSV files**

| Argument | File | Required |
|---|---|---|
| `csv_path` | `cxr-record-list.csv.gz` | Yes |
| `label_csv_path` | `mimic-cxr-2.0.0-chexpert.csv` | No (sourced from MIMIC-CXR-JPG distribution) |
| `report_csv_path` | `cxr-study-list.csv.gz` | No (required for `output_report=True`) |

**Note:** `base_image_dir` should point to the `files/` subdirectory (e.g. `…/2.1.0/files/`), not the version root. The label CSV (`mimic-cxr-2.0.0-chexpert.csv`) is distributed with MIMIC-CXR-JPG, not with the DICOM version — provide the path explicitly if the JPG files are not colocated.

**Path construction**
```
patient_id: "12345678", study_id: "11223344", dicom_id: "abcdef01"
→ image_path: "p12/p12345678/s11223344/abcdef01.dcm"   (relative to base_image_dir)
```

---

### MIMIC-CXR-JPG

```python
from radharmony.dataset import MIMICCXRJPGDataset

ds = MIMICCXRJPGDataset(
    base_image_dir="/data/mimic-cxr-jpg/2.0.0/files/",
    csv_path=None,          # auto-discovered: mimic-cxr-2.0.0-metadata.csv
    label_csv_path=None,    # auto-discovered: mimic-cxr-2.0.0-chexpert.csv
    output_cls=True,
)
```

**Constructor arguments** (dataset-specific)

| Argument | Type | Default | Description |
|---|---|---|---|
| `csv_path` | `str` | auto | Path to `mimic-cxr-2.0.0-metadata.csv` |
| `label_csv_path` | `str` | auto | Path to `mimic-cxr-2.0.0-chexpert.csv` |

**Required CSV files**

| Argument | File | Required |
|---|---|---|
| `csv_path` | `mimic-cxr-2.0.0-metadata.csv` | Yes |
| `label_csv_path` | `mimic-cxr-2.0.0-chexpert.csv` | No |

**Path construction** — same as MIMIC-CXR (DICOM) but produces `.jpg` paths.

---

### SIIM-ACR Pneumothorax

```python
from radharmony.dataset import SIIMACRPTXDataset

ds = SIIMACRPTXDataset(
    base_image_dir="/data/SIIM_ACR_Pneumothorax/dicom-images-train/",
    csv_path=None,              # auto-discovered: train-rle.csv
    mask_output_dir="./masks",  # required when output_mask=True
    output_cls=True,
    output_mask=True,
)
```

**Constructor arguments** (dataset-specific)

| Argument | Type | Default | Description |
|---|---|---|---|
| `csv_path` | `str` | auto | Path to `train-rle.csv` |
| `mask_output_dir` | `str` | `None` | Directory to write decoded PNG masks; required for `output_mask=True` |
| `mask_num_cores` | `int` | `1` | Parallel threads for RLE → PNG decoding on first use |

**Required CSV files**

| Argument | File | Required |
|---|---|---|
| `csv_path` | `train-rle.csv` | Yes |

**Mask generation** — pass `mask_output_dir` to decode RLE strings to PNG files on first use. The decoded PNGs are cached; subsequent runs skip re-decoding.

**Directory structure expected**
```
dicom-images-train/
└── <patient_id>/
    └── <study_id>/
        └── <image_id>.dcm
```

---

### ChestX-ray14

```python
from radharmony.dataset import ChestXray14Dataset

ds = ChestXray14Dataset(
    base_image_dir="/data/NIH_CXR/CXR14/",
    csv_path=None,          # auto-discovered: Data_Entry_2017.csv
    bbox_csv_path=None,     # auto-discovered: BBox_List_2017.csv (used to exclude bbox images)
    output_cls=True,
)
```

**Constructor arguments** (dataset-specific)

| Argument | Type | Default | Description |
|---|---|---|---|
| `csv_path` | `str` | auto | Path to `Data_Entry_2017.csv` |
| `bbox_csv_path` | `str` | auto | Path to `BBox_List_2017.csv` (images listed here are excluded) |

**Required CSV files**

| Argument | File | Required |
|---|---|---|
| `csv_path` | `Data_Entry_2017.csv` | Yes |
| `bbox_csv_path` | `BBox_List_2017.csv` | No (when provided, bbox images are excluded) |

**Note:** This split contains only the ~111,240 images that do **not** have bounding-box annotations. For the ~880 images with bboxes, use `ChestXray14BboxDataset`. Only `output_cls` is supported; `output_mask`, `output_report`, and `output_bbox` are not available.

**Path construction**
```
Image Index: "00000001_000.png"
→ patient_id : "00000001"
→ study_id   : "00000001_000"
→ image_path : "images_001/images/00000001_000.png"   (relative to base_image_dir)
```

**Labels:** 15 NLP-derived findings — `atelectasis`, `cardiomegaly`, `consolidation`, `edema`, `effusion`, `emphysema`, `fibrosis`, `hernia`, `infiltration`, `mass`, `no_finding`, `nodule`, `pleural_thickening`, `pneumonia`, `pneumothorax`.

---

### ChestX-ray14 (BBox)

```python
from radharmony.dataset import ChestXray14BboxDataset

ds = ChestXray14BboxDataset(
    base_image_dir="/data/NIH_CXR/CXR14/",
    csv_path=None,          # auto-discovered: Data_Entry_2017.csv
    bbox_csv_path=None,     # auto-discovered: BBox_List_2017.csv
    output_cls=True,
    output_bbox=True,
)
```

**Constructor arguments** (dataset-specific)

| Argument | Type | Default | Description |
|---|---|---|---|
| `csv_path` | `str` | auto | Path to `Data_Entry_2017.csv` |
| `bbox_csv_path` | `str` | auto | Path to `BBox_List_2017.csv` |

**Required CSV files**

| Argument | File | Required |
|---|---|---|
| `csv_path` | `Data_Entry_2017.csv` | Yes |
| `bbox_csv_path` | `BBox_List_2017.csv` | Yes |

**Note:** This split contains only the ~880 images with radiologist-drawn bounding boxes (984 boxes across 8 pathology classes). Supports `output_cls` and `output_bbox`. Bounding boxes are normalized to `[0, 1]` using the actual PNG dimensions.

**Bounding-box format** — each image may have multiple boxes. Bbox coordinates in the CSV (`Bbox [x, y, w, h]`) are pixel positions on the rescaled PNG. They are normalized by reading the actual image dimensions.

**Labels:** Same 15 findings as ChestX-ray14.

---

### RSNA Pneumonia

```python
from radharmony.dataset import RSNAPneumoniaDataset

ds = RSNAPneumoniaDataset(
    base_image_dir="/data/rsna/",
    csv_path=None,          # auto-discovered: pneumonia-challenge-annotations-adjudicated-kaggle_2018.json
    output_cls=True,
    output_bbox=True,
)
```

**Constructor arguments** (dataset-specific)

| Argument | Type | Default | Description |
|---|---|---|---|
| `csv_path` | `str` | auto | Path to `pneumonia-challenge-annotations-adjudicated-kaggle_2018.json` (adjudicated JSON export) |
| `label_group` | `str` | `"Calculated"` | Label group name in the JSON |

**Required annotation files**

| Argument | File | Required |
|---|---|---|
| `csv_path` | `pneumonia-challenge-annotations-adjudicated-kaggle_2018.json` | Yes |

**Note:** ~29,684 frontal chest X-rays from the RSNA Pneumonia Detection Challenge. DICOMs live in a three-level UID nested layout (`<StudyUID>/<SeriesUID>/<SOPUID>.dcm`) under `base_image_dir`. Supports `output_cls` and `output_bbox`; `output_mask` and `output_report` are ignored with a warning. Bounding boxes are provided only for the `lung_opacity` class and normalized by DICOM header dimensions.

**Labels:** 3 classes — `lung_opacity`, `no_lung_opacity_/_not_normal`, `normal`.

---

### RANZCR CLiP

~30,000 frontal chest X-rays from the [RANZCR CLiP — Catheter and Line Position Challenge](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification) (Royal Australian and New Zealand College of Radiologists, 2021 Kaggle competition). Multi-label classification of catheter / line positioning correctness, plus optional polyline annotations tracing each catheter for a subset of training images.

Kaggle is the **primary distribution channel** for this dataset; fetch via `kagglehub`:

```python
import kagglehub
base = kagglehub.competition_download("ranzcr-clip-catheter-line-classification")
```

```python
from radharmony.dataset import RANZCRClipDataset

# Classification only
ds = RANZCRClipDataset(
    base_image_dir=base,
    csv_path=None,                  # auto-discovered: train.csv
    include_test_split=False,       # exclude test/<id>.jpg rows (no labels)
    output_cls=True,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)

# Classification + rasterized polyline masks
ds = RANZCRClipDataset(
    base_image_dir=base,
    annotations_csv_path=None,      # auto-discovered: train_annotations.csv
    mask_output_dir="/path/to/ranzcr_masks/",  # required for output_mask=True
    include_test_split=False,
    output_cls=True,
    output_mask=True,
)
```

**Constructor arguments** (dataset-specific)

| Argument | Type | Default | Description |
|---|---|---|---|
| `csv_path` | `str` | auto | Path to `train.csv` |
| `annotations_csv_path` | `str` | auto | Path to `train_annotations.csv` (polylines). Required for `output_mask=True` |
| `mask_output_dir` | `str` | `None` | Directory where rasterized polyline PNGs are written. Required for `output_mask=True` |
| `mask_num_cores` | `int` | `1` | Worker threads for mask pre-decoding |
| `mask_line_thickness` | `int` | `15` | Pixel width passed to `cv2.polylines` when rasterizing polylines (matches the official challenge starter notebook) |
| `include_test_split` | `bool` | `True` | Append `test/<StudyInstanceUID>.jpg` rows with `split="test"` and NaN labels |

**Required files**

| Argument | File | Required |
|---|---|---|
| `csv_path` | `train.csv` | Yes |
| `annotations_csv_path` | `train_annotations.csv` | Only when `output_mask=True` |

**Path construction**
```
StudyInstanceUID: "1.2.826.0.1.3680043.8.498.46923145579096002357117760813047428"
→ patient_id : <PatientID column>             (falls back to study_id for test rows)
→ study_id   : "1.2.826.0.1.3680043.8.498.46923…"
→ image_path : "train/<StudyInstanceUID>.jpg"  (or "test/<StudyInstanceUID>.jpg")
→ split      : "train" | "test"
→ mask_path  : <JSON polyline list>            (rewritten to a PNG path after preprocess_masks)
```

**Predefined splits.** The competition's official train/test partition is exposed via the `split` column. Call `get_datasets_predefined()` to obtain `(train_ds, test_ds)` directly. Test rows have NaN labels (no public labels) and are useful for inference / submission only.

**Labels:** 11 binary multi-label targets — `cvc_abnormal`, `cvc_borderline`, `cvc_normal`, `ett_abnormal`, `ett_borderline`, `ett_normal`, `ngt_abnormal`, `ngt_borderline`, `ngt_incompletely_imaged`, `ngt_normal`, `swan_ganz_catheter_present`. Raw CSV uses `"CVC - Abnormal"` style names — the harmonizer renames to clean snake_case in `_build_labels`.

**Note on masks.** Polyline annotations are line traces, not regions; `output_bbox` is unsupported (silently ignored with a warning). `output_report` is also unsupported. When `output_mask=True`, polylines are rasterized into binary segmentation masks via `cv2.polylines` (configurable line width). Train rows without annotations resolve to all-zero masks — slice the harmonized DataFrame to `mask_path.notna()` before training a segmentation head if that matters.

---

### VinDr-CXR (Train)

```python
from radharmony.dataset import VinDrCXRTrainDataset

ds = VinDrCXRTrainDataset(
    base_image_dir="/data/VinDr-CXR/vindr-cxr/1.0.0/train/",
    csv_path=None,          # auto-discovered: image_labels_train.csv
    bbox_csv_path=None,     # auto-discovered: annotations_train.csv
    output_cls=True,
    output_bbox=True,
)
```

**Constructor arguments** (dataset-specific)

| Argument | Type | Default | Description |
|---|---|---|---|
| `csv_path` | `str` | auto | Path to `image_labels_train.csv` |
| `bbox_csv_path` | `str` | auto | Path to `annotations_train.csv` |

**Required CSV files**

| Argument | File | Required |
|---|---|---|
| `csv_path` | `image_labels_train.csv` | Yes |
| `bbox_csv_path` | `annotations_train.csv` | No (required for `output_bbox=True`) |

**Note:** All images are PA-view DICOM chest X-rays (`.dicom`). Labels are aggregated from three independent radiologist annotations via majority vote. Bounding boxes (multiple per image) are normalized by reading DICOM header dimensions.

**Labels:** 28 findings — `aortic_enlargement`, `atelectasis`, `calcification`, `cardiomegaly`, `clavicle_fracture`, `consolidation`, `copd`, `edema`, `emphysema`, `enlarged_pa`, `ild`, `infiltration`, `lung_cavity`, `lung_cyst`, `lung_opacity`, `lung_tumor`, `mediastinal_shift`, `no_finding`, `nodule/mass`, `other_diseases`, `other_lesion`, `pleural_effusion`, `pleural_thickening`, `pneumonia`, `pneumothorax`, `pulmonary_fibrosis`, `rib_fracture`, `tuberculosis`.

---

### VinDr-CXR (Test)

```python
from radharmony.dataset import VinDrCXRTestDataset

ds = VinDrCXRTestDataset(
    base_image_dir="/data/VinDr-CXR/vindr-cxr/1.0.0/test/",
    csv_path=None,          # auto-discovered: image_labels_test.csv
    bbox_csv_path=None,     # auto-discovered: annotations_test.csv
    output_cls=True,
    output_bbox=True,
)
```

**Constructor arguments** (dataset-specific)

| Argument | Type | Default | Description |
|---|---|---|---|
| `csv_path` | `str` | auto | Path to `image_labels_test.csv` |
| `bbox_csv_path` | `str` | auto | Path to `annotations_test.csv` |

**Required CSV files**

| Argument | File | Required |
|---|---|---|
| `csv_path` | `image_labels_test.csv` | Yes |
| `bbox_csv_path` | `annotations_test.csv` | No (required for `output_bbox=True`) |

**Note:** Labels and bounding boxes are already aggregated (verified by 5 radiologists). Unlike the training set, no majority-vote step is needed. The test CSV uses `"Other disease"` (singular); the harmonizer renames it to `"Other diseases"` to match the train set convention.

**Labels:** Same 28 findings as VinDr-CXR (Train).

---

### TAIX-Ray

215,381 bedside chest radiographs from 47,724 ICU patients (RWTH Aachen, 2010–2023), with structured prospective radiologist annotations on a 5-point ordinal severity scale. Distributed via HuggingFace under CC BY 4.0: <https://huggingface.co/datasets/TLAIM/TAIX-Ray>.

Two configurations ship in parallel — pick the matching dataset class:

| Class | base_image_dir | Image size | Approx. on-disk |
|---|---|---|---|
| `TAIXRay512Dataset` | `…/TAIX-Ray/data_512/images/` | longer dim = 512 px | ~58 GB |
| `TAIXRayDataset` | `…/TAIX-Ray/data_original/images/` | native acquisition res | ~1.27 TB |

Both share the same `annotation.csv` schema, `TAIXRayHarmonizer`, label set, and predefined patient-level split. Use `notebooks/download_huggingface.ipynb` to fetch the parquet shards and unpack them to `images/<UID>.png` + `annotation.csv`.

```python
from radharmony.dataset import TAIXRay512Dataset

ds = TAIXRay512Dataset(
    base_image_dir="/path/to/TAIX-Ray/data_512/images/",
    csv_path=None,             # auto-discovered: annotation.csv
    label_mode="binary",       # "binary" (default) or "ordinal"
    output_cls=True,
)

# Use the dataset's own train/val/test split (paper's official partition)
train_ds, val_ds, test_ds = ds.get_datasets_predefined()

# Or run the standard random patient-level split
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

**Constructor arguments** (dataset-specific; same for both classes)

| Argument | Type | Default | Description |
|---|---|---|---|
| `csv_path` | `str` | auto | Path to `annotation.csv` |
| `label_mode` | `str` | `"binary"` | `"binary"` maps severity ≥ 1 → 1.0; `"ordinal"` preserves raw 0–4 grades (0–3 for `heart_size`) |

**Required files**

| Argument | File | Required |
|---|---|---|
| `csv_path` | `annotation.csv` | Yes |

**Note:** TAIX-Ray has no segmentation masks, no radiology reports, and no bounding boxes — `output_mask`, `output_report`, and `output_bbox` are silently ignored with a warning. The harmonized DataFrame carries `sex`, `age` (in days), `study_date`, `fold`, `split`, and `physician_id` as extra metadata columns alongside the labels.

**Predefined splits.** TAIX-Ray ships its own patient-level partition (137,593 / 34,860 / 42,928 train / val / test). Call `get_datasets_predefined()` to honour it; the `Fold` column (0–4) supports k-fold variants if needed.

**Path construction**
```
UID: "549a816ae020fb7da68a31d7d62d73c418a069c77294fc084dd9f7bd717becb9"
→ patient_id : <PatientID hash>
→ study_id   : <UID>
→ image_path : "<UID>.png"   (relative to base_image_dir)
```

**Labels:** 8 ordinal findings — `heart_size` (0–3: normal / borderline / enlarged / massively enlarged) and `pulmonary_congestion`, `pleural_effusion_left`, `pleural_effusion_right`, `pulmonary_opacities_left`, `pulmonary_opacities_right`, `atelectasis_left`, `atelectasis_right` (each 0–4: none / questionable / mild / moderate / severe).

---

### RSNA 2022 Cervical Spine

2,019 cervical-spine CT volumes from the [RSNA 2022 Cervical Spine Fracture Detection Kaggle competition](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection). 8 binary labels per volume: `patient_overall` (any fracture present) plus per-vertebra `c1`–`c7`. 87 of the 2,019 studies ship with whole-volume NIfTI segmentation masks; 235 ship with slice-level fracture bounding boxes.

Like all Kaggle competitions in this repo, only **train** is wired up — the test set has no labels (held by Kaggle for leaderboard scoring). See `docs/reviews/dataset_split_handling.md`.

Two complementary dataset classes:

| Class | Granularity | Use for |
|---|---|---|
| `RSNA2022CervicalSpineDataset` | 3-D per-volume | Whole-volume classification; optional segmentation masks (87 studies) |
| `RSNA2022CervicalSpineBboxDataset` | 2-D per-slice | Slice-level fracture-localization with bounding boxes (235 studies, 7,217 slices) |

```python
# 3-D per-volume
from radharmony.dataset import RSNA2022CervicalSpineDataset

ds = RSNA2022CervicalSpineDataset(
    base_image_dir="~/datasets/external/rsna-2022-cervical-spine-fracture-detection/train_images/",
    csv_path=None,           # auto-discovered: ../train.csv
    segmentation_dir=None,   # auto-discovered: ../segmentations/
    hu_window=(-200, 1800),  # bone-centric default; pass (-1000, 1000) for soft tissue
    output_cls=True,
    output_mask=False,       # True → restricts to the 87 mask-having studies
)

# 2-D per-slice with bboxes
from radharmony.dataset import RSNA2022CervicalSpineBboxDataset

ds_bbox = RSNA2022CervicalSpineBboxDataset(
    base_image_dir="~/datasets/external/rsna-2022-cervical-spine-fracture-detection/train_images/",
    csv_path=None,           # auto-discovered: ../train.csv
    bbox_csv_path=None,      # auto-discovered: ../train_bounding_boxes.csv
    output_cls=True,
    output_bbox=True,
)
```

**Constructor arguments** (3-D class)

| Argument | Type | Default | Description |
|---|---|---|---|
| `csv_path` | `str` | auto | Path to `train.csv` |
| `segmentation_dir` | `str` | auto | Directory of `<StudyUID>.nii` masks; auto-discovered as `<parent>/segmentations` |
| `hu_window` | `(min, max)` | `(-200, 1800)` | HU clipping window — bone-centric default for cervical spine |

**Constructor arguments** (2-D bbox class)

| Argument | Type | Default | Description |
|---|---|---|---|
| `csv_path` | `str` | auto | Path to `train.csv` (cls labels per study) |
| `bbox_csv_path` | `str` | auto | Path to `train_bounding_boxes.csv` |

**Path construction**

3-D — image_path is the per-study DICOM **directory** (flat layout, no `<SeriesUID>/` nesting unlike RSNA PE):
```
StudyInstanceUID: "1.2.826.0.1.3680043.6200"
→ patient_id : "1.2.826.0.1.3680043.6200"   (no separate patient column; one study per patient)
→ study_id   : "1.2.826.0.1.3680043.6200"
→ image_path : "1.2.826.0.1.3680043.6200"   (directory; ITKReader assembles 3-D volume)
→ mask_path  : ".../segmentations/1.2.826.0.1.3680043.6200.nii"  (NaN if not in 87-subset)
```

2-D — image_path is a single `.dcm` slice file:
```
image_path : "1.2.826.0.1.3680043.10051/133.dcm"   (StudyUID/<slice_number>.dcm)
bbox       : [[y_min, y_max, x_min, x_max], ...]   (normalized to [0,1] against 512×512 slice)
```

**Note:** Slice dimensions are 512×512 throughout the release (verified via DICOM Rows/Columns). The bbox harmonizer relies on this constant rather than reading per-slice headers, which would add 200k+ stat calls.

---

### CT-RATE

```python
from radharmony.dataset import CTRATEDataset

ds = CTRATEDataset(
    base_image_dir="/data/CT-RATE/dataset/train_fixed/",
    csv_path=None,                   # auto-discovered: train_predicted_labels.csv
    view_position_csv_path=None,     # auto-discovered: train_metadata.csv
    hu_window=(-1000, 1000),         # HU clipping before percentile normalisation
    output_cls=True,
)
```

**Constructor arguments** (dataset-specific)

| Argument | Type | Default | Description |
|---|---|---|---|
| `csv_path` | `str` | auto | Path to `train_predicted_labels.csv` or `validation_predicted_labels.csv` |
| `view_position_csv_path` | `str` | auto | Path to `train_metadata.csv` or `validation_metadata.csv` |
| `hu_window` | `tuple[float, float] \| None` | `(-1000, 1000)` | HU clipping range; `None` skips clipping |

**Required CSV files**

| Argument | File | Required |
|---|---|---|
| `csv_path` | `train_predicted_labels.csv` or `validation_predicted_labels.csv` | Yes |
| `view_position_csv_path` | `train_metadata.csv` or `validation_metadata.csv` | No |

**Path construction**
```
VolumeName: "P001_S01_CT_Image"
→ patient_id : "P001_S01"
→ study_id   : "P001_S01_CT"
→ image_path : "P001_S01/P001_S01_CT/P001_S01_CT_Image"   (relative to base_image_dir)
```

---

### RAD-ChestCT

```python
from radharmony.dataset import RadChestCTDataset

ds = RadChestCTDataset(
    base_image_dir="/data/RAD-ChestCT/images/",
    csv_path=None,          # auto-discovered: CT_Scan_Metadata_Complete_35747.csv
    label_csv_path=None,    # auto-discovered: imgtrain_Abnormality_and_Location_Labels.csv
    bbox_csv_path=None,     # auto-discovered: Extrema_35747.csv
    output_cls=True,
    output_bbox=True,
)
```

**Constructor arguments** (dataset-specific)

| Argument | Type | Default | Description |
|---|---|---|---|
| `csv_path` | `str` | auto | Path to `CT_Scan_Metadata_Complete_35747.csv` |
| `label_csv_path` | `str` | auto | Path to one of the `img{train,valid,test}_Abnormality_and_Location_Labels.csv` files |
| `bbox_csv_path` | `str` | auto | Path to `Extrema_35747.csv` |

**Required CSV files**

| Argument | File | Required |
|---|---|---|
| `csv_path` | `CT_Scan_Metadata_Complete_35747.csv` | Yes |
| `label_csv_path` | `imgtrain/imgvalid/imgtest_Abnormality_and_Location_Labels.csv` | Yes |
| `bbox_csv_path` | `Extrema_35747.csv` | No |

**Labels:** 84 findings, hardcoded and sorted alphabetically. Override `ds.LABEL_COLS` with a subset before calling `get_datasets()` to restrict the `cls` tensor.

---

## Adding a custom dataset

### Step 1 — Write a harmonizer

Create a subclass of `BaseHarmonizer`. You must implement three methods that populate columns on `self.df`:

```python
from radharmony.harmonizer.base import BaseHarmonizer

class MyDatasetHarmonizer(BaseHarmonizer):
    LABEL_COLS = ["finding_a", "finding_b"]

    # Required join keys for the label CSV (if separate)
    LABEL_JOIN_COLS = ["patient_id", "study_id"]

    def _build_patient_id(self):
        """Populate self.df["patient_id"]."""
        self.df["patient_id"] = self.df["PatientID"]

    def _build_study_id(self):
        """Populate self.df["study_id"]."""
        self.df["study_id"] = self.df["StudyInstanceUID"]

    def _build_image_path(self):
        """Populate self.df["image_path"] with relative paths."""
        self.df["image_path"] = (
            self.df["patient_id"] + "/" + self.df["dicom_id"] + ".dcm"
        )
```

**Optional hooks** — override any of these for richer outputs:

| Method | Populates | When to override |
|---|---|---|
| `_build_labels` | label columns from `LABEL_COLS` | when labels are in a separate CSV |
| `_preprocess_label_df` | — | to clean/transform the label CSV before merge |
| `_build_view_position` | `view_position` | when a view-position CSV exists |
| `_build_mask_path` | `mask_path` | when segmentation masks are available |
| `_build_bbox` | `bbox` | when bounding boxes are available |
| `_build_report` | `report` | when radiology reports are available |
| `_decode_mask` | — | when masks are stored as encoded strings (e.g. RLE) |

The `harmonize()` method is inherited — it reads the CSV, calls each `_build_*` method in order, drops rows with missing `image_path`, and returns the standardised DataFrame.

If your harmonizer’s `__init__` takes arguments beyond those stored on `BaseHarmonizer` (for example a custom root directory), override `_harmonizer_init_snapshot()` so that `save()` / `load_from_saved()` can reconstruct it. Built-in harmonizers do this for fields such as `dicom_base_dir` or `image_base_dir`.

### Step 2 — Write a dataset class

Subclass `BaseRadiologicalDataset` and implement `_get_harmonized_df`. Decorate with `@register_dataset` to make it discoverable by name.

Set the class attribute **`_HARMONIZER_CLS`** to your harmonizer type. This is required if callers use **`harmonizer_path=`** (a pickle from `BaseHarmonizer.save`): the base class calls `_HARMONIZER_CLS.load_from_saved(path)` to restore the harmonized table without re-running CSV work. It is harmless if you only ever harmonize inline.

You may also forward **`harmonized_df`**, **`harmonizer`**, and **`harmonizer_path`** from your `__init__` into `super().__init__(...)`. At the start of `_get_harmonized_df`, call **`_try_resolve_preset_harmonized()`** and return its result when it is not `None`; otherwise run your harmonizer as usual. Resolution order is: `harmonized_df` → `harmonizer_path` → `harmonizer` → your normal path.

```python
from radharmony.dataset.base import BaseRadiologicalDataset
from radharmony.dataset.transforms import RadiologyTransform2D
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path


@register_dataset("my_dataset")
class MyDataset(BaseRadiologicalDataset):
    LABEL_COLS = ["finding_a", "finding_b"]
    _HARMONIZER_CLS = MyDatasetHarmonizer

    def __init__(
        self,
        base_image_dir: str,
        csv_path: str = None,
        label_csv_path: str = None,
        transform=None,
        cache_dir: str = "./cache",
        output_cls: bool = False,
        output_mask: bool = False,
        output_report: bool = False,
        output_bbox: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
    ):
        # Auto-discover CSVs near base_image_dir
        csv_path = infer_path(base_image_dir, "metadata.csv", user_path=csv_path or "") or csv_path
        label_csv_path = infer_path(base_image_dir, "labels.csv", user_path=label_csv_path or "") or label_csv_path

        output_keys = {"img"}
        if output_cls:    output_keys.add("cls")
        if output_mask:   output_keys.add("mask")
        if output_report: output_keys.add("report")
        if output_bbox:   output_keys.add("bbox")

        super().__init__(
            base_image_dir=base_image_dir,
            transform=transform or RadiologyTransform2D(output_keys=output_keys).get_transform(),
            cache_dir=cache_dir,
            output_cls=output_cls,
            output_mask=output_mask,
            output_report=output_report,
            output_bbox=output_bbox,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
        )
        self._harmonizer = MyDatasetHarmonizer(
            csv_path=csv_path,
            label_csv_path=label_csv_path,
        )

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        return self._harmonizer.harmonize()
```

### Step 3 — Import and use

```python
import my_dataset_module   # triggers @register_dataset at import time

from radharmony.registry import resolve_dataset
MyDataset = resolve_dataset("my_dataset")

ds = MyDataset(base_image_dir="/data/my_dataset/images/")
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

Or import the class directly:

```python
from my_dataset_module import MyDataset

ds = MyDataset(base_image_dir="/data/my_dataset/images/", output_cls=True)
```

To also expose the dataset in the Gradio app, see [app_guide.md](app_guide.md).

---

## Saving and loading harmonized data (pickle)

Harmonization (CSV parsing, merging, path construction) can be slow for large datasets. RadHarmony lets you save the harmonized result to a pickle file and reload it later, skipping all CSV work on subsequent runs.

### Saving a harmonizer

After calling `harmonize()`, use `save()` to write the harmonized DataFrame, label columns, and init parameters to a pickle file:

```python
from radharmony.harmonizer import CheXpertHarmonizer

h = CheXpertHarmonizer(csv_path="/data/CheXpert-v1.0/train.csv")
h.harmonize()

# Save to disk
h.save("chexpert_harmonized.pkl")
```

The pickle stores:
- The harmonized DataFrame
- The effective `LABEL_COLS`
- An init snapshot (CSV paths and other constructor arguments) so the harmonizer can be reconstructed

### Loading a saved harmonizer

Use the class method `load_from_saved()` to restore a harmonizer without re-running CSV work:

```python
h = CheXpertHarmonizer.load_from_saved("chexpert_harmonized.pkl")

# The harmonized DataFrame is ready immediately
print(h.harmonized_df.head())
print(h.get_label_cols())
```

You can override init parameters at load time (e.g. if data moved to a new path):

```python
h = CheXpertHarmonizer.load_from_saved(
    "chexpert_harmonized.pkl",
    csv_path="/new/path/train.csv",
)
```

### Using preset DataFrames with dataset classes

Dataset classes accept three constructor arguments that bypass normal harmonization. The resolution order is:

1. **`harmonized_df`** — pass a DataFrame directly
2. **`harmonizer_path`** — path to a pickle file saved via `harmonizer.save()`
3. **`harmonizer`** — pass an already-harmonized harmonizer instance

If none are provided, the dataset runs its own harmonization.

#### Option 1: Pass a pickle path (`harmonizer_path`)

The most common approach — save once, reload on every subsequent run:

```python
from radharmony.dataset import CheXpertDataset

# First run: harmonize and save
ds = CheXpertDataset(base_image_dir="/data/CheXpert-v1.0/train/", output_cls=True)
df = ds.get_harmonized_df()  # triggers harmonization

# Save the harmonizer for later
from radharmony.harmonizer import CheXpertHarmonizer
h = CheXpertHarmonizer(csv_path="/data/CheXpert-v1.0/train.csv")
h.harmonize()
h.save("chexpert_harmonized.pkl")

# Subsequent runs: skip CSV work entirely
ds = CheXpertDataset(
    base_image_dir="/data/CheXpert-v1.0/train/",
    harmonizer_path="chexpert_harmonized.pkl",
    output_cls=True,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

This works because each dataset class sets `_HARMONIZER_CLS`, which tells the base class how to call `load_from_saved()`.

#### Option 2: Pass a DataFrame directly (`harmonized_df`)

Useful when you want to filter, modify, or inspect the DataFrame before handing it to the dataset:

```python
import pandas as pd

# Load a previously saved harmonizer
h = CheXpertHarmonizer.load_from_saved("chexpert_harmonized.pkl")
preset_df = h.harmonized_df

# Inspect or filter
print(preset_df.shape)
print(preset_df.columns.tolist())
preset_df = preset_df[preset_df["view_position"] == "PA"]

# Pass the filtered DataFrame to the dataset
ds = CheXpertDataset(
    base_image_dir="/data/CheXpert-v1.0/train/",
    harmonized_df=preset_df,
    output_cls=True,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

#### Option 3: Pass a harmonizer instance (`harmonizer`)

Useful when you've already harmonized in memory and want to reuse the result:

```python
h = CheXpertHarmonizer(csv_path="/data/CheXpert-v1.0/train.csv")
h.harmonize()

ds = CheXpertDataset(
    base_image_dir="/data/CheXpert-v1.0/train/",
    harmonizer=h,
    output_cls=True,
)
```

### Updating the preset after construction

You can also swap the harmonized DataFrame at any time using `set_harmonized_df()`:

```python
ds = CheXpertDataset(base_image_dir="/data/CheXpert-v1.0/train/", output_cls=True)

# Later, pin a specific DataFrame
ds.set_harmonized_df(preset_df)

# All subsequent get_datasets() calls use preset_df
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

### Typical workflow

```
┌─────────────────────────┐
│  First run              │
│  harmonizer.harmonize() │
│  harmonizer.save(path)  │
└──────────┬──────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Subsequent runs             │
│  Dataset(harmonizer_path=…)  │
│  → skips CSV parsing         │
│  → loads pickle instantly    │
└──────────────────────────────┘
```

---

## Observed dataset paths 

| Dataset | `base_image_dir` | main CSV | label CSV | other |
|---|---|---|---|---|
| CheXpert | `/path/to/CheXpert-v1.0/train/` | `../train.csv` | — | — |
| CheXpert-Plus (DICOM) | `/path/to/CheXpert_Plus/chexpertplus/DICOM/Uncompressed/` | `df_chexpert_plus_240401.csv` | `report_fixed.json` (JSONL) | report: inline in CSV |
| ChestX-ray14 | `/path/to/.../NIH_CXR/CXR14/` | `Data_Entry_2017.csv` | — | bbox exclusion: `BBox_List_2017.csv` |
| ChestX-ray14 (BBox) | `/path/to/.../NIH_CXR/CXR14/` | `Data_Entry_2017.csv` | — | bbox: `BBox_List_2017.csv` |
| MIMIC-CXR (DICOM) | `/path/to/.../mimic-cxr/2.1.0/files/` | `../cxr-record-list.csv.gz` | `mimic-cxr-2.0.0-chexpert.csv` | report: `cxr-study-list.csv.gz` |
| MIMIC-CXR-JPG | `/path/to/.../mimic-cxr-jpg/2.0.0/files/` | `../mimic-cxr-2.0.0-metadata.csv` | `mimic-cxr-2.0.0-chexpert.csv` | — |
| RSNA Pneumonia | `~/Downloads/rsna/` | `pneumonia-challenge-annotations-adjudicated-kaggle_2018.json` | — | bbox: inline in JSON |
| SIIM-ACR-PTX | `/path/to/.../SIIM_ACR_Pneumothorax/dicom-images-train/` | `../train-rle.csv` | — | — |
| VinDr-CXR (Train) | `/path/to/.../VinDr-CXR/vindr-cxr/1.0.0/train/` | `image_labels_train.csv` | — | bbox: `annotations_train.csv` |
| VinDr-CXR (Test) | `/path/to/.../VinDr-CXR/vindr-cxr/1.0.0/test/` | `image_labels_test.csv` | — | bbox: `annotations_test.csv` |
| CT-RATE | `/path/to/.../CT-RATE/dataset/train_fixed/` | `../tables/train_predicted_labels.csv` | `train_metadata.csv` | — |
| RAD-ChestCT | `/path/to/.../RAD-ChestCT/images/` | `../tables/CT_Scan_Metadata_Complete_35747.csv` | `imgtrain_Abnormality_and_Location_Labels.csv` | bbox: `Extrema_35747.csv` |
| TAIX-Ray (512 px) | `/path/to/TAIX-Ray/data_512/images/` | `../annotation.csv` | — | split: inline `split` column (`train`/`val`/`test`) |
| TAIX-Ray (original) | `/path/to/TAIX-Ray/data_original/images/` | `../annotation.csv` | — | split: inline `split` column (`train`/`val`/`test`) |
| RSNA 2022 Cervical Spine | `/path/to/rsna-2022-cervical-spine-fracture-detection/train_images/` | `../train.csv` | — | masks: `../segmentations/<StudyUID>.nii` (87 of 2,019) |
| RSNA 2022 Cervical Spine (BBox) | `/path/to/rsna-2022-cervical-spine-fracture-detection/train_images/` | `../train.csv` | — | bbox: `../train_bounding_boxes.csv` (slice-level, 235 of 2,019) |
