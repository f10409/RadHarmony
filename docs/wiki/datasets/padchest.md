# PadChest

**Modality:** CXR | **Format:** PNG | **Dim:** 2D | **Labels:** 193 (174 findings + 19 diagnoses) | **Reports:** Spanish

## Overview

PadChest (BIMCV) comprises 160,868 chest X-rays (the distributed label CSV
has 160,861 rows) from ~67,000 patients at
Hospital San Juan de Alicante, Spain (2009–2017). Labels are NLP-derived
from Spanish radiology reports plus manual review; the published label
vocabulary contains 193 distinct labels — 174 radiographic findings plus
19 differential diagnoses (much richer than the 14-class
CheXpert/ChestX-ray14 taxonomy). Each row also carries a pre-processed
(stemmed/lemmatized) Spanish radiology report.

RadHarmony provides a single dataset class for this dataset:
- `PadChestDataset` — full dataset (160,758 rows after dropping 103 rows
  with NaN Labels); classification + Spanish report.

## Download

PadChest is hosted at the [BIMCV Nextcloud share](https://b2drop.bsc.es/index.php/s/BIMCV-PadChest-FULL).
The `?accept=zip` whole-folder download is a dynamically streamed Zip64
that does **not** support resume — for a ~1 TB download this is fragile;
prefer fetching the 52 per-shard zips individually (each ~20 GB,
resumable via standard `wget -c`).

Authoritative primary source: BIMCV (Barcelona Supercomputing Center /
Universitat Politècnica de València). Do not use third-party Kaggle
mirrors — they are partial re-uploads.

Expected layout after extracting each `N.zip` into `images/N/`:

```
PadChest/
  PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz
  LICENSE.md
  README.md
  Verify_Zips_ImageCounts.csv.xlsx
  images/
    0/   ~3000 PNGs
    1/   ~3000 PNGs
    ...
    50/  1 PNG       (legitimate — see notes)
    54/  4297 PNGs   (catch-all; slots 51/52/53 don't exist)
```

The PNGs live in 52 sibling subdirectories under `images/`. This is the
same structural-exception pattern as ChestX-ray14's `images_001/.../images_012/`:
`base_image_dir` is the deepest stable directory (`images/`) and the
relative `image_path` carries the subdir name (`<ImageDir>/<ImageID>`).

## Label columns

193 labels (174 radiographic findings + 19 differential diagnoses),
NLP-derived from Spanish reports. All exposed as binary
one-hot columns. A representative subset:

| Column | Description |
|--------|-------------|
| `normal` | No finding (50,616 positives) |
| `copd_signs` | COPD signs (23,280) |
| `cardiomegaly` | Enlarged heart (15,022) |
| `unchanged` | Unchanged from prior study (14,349) |
| `aortic_elongation` | Aortic elongation (10,824) |
| `pleural_effusion` | Pleural effusion (10,026) |
| `scoliosis` | Scoliosis (8,335) |
| `pneumonia` | Pneumonia (8,037) |
| `interstitial_pattern` | Interstitial pattern (7,831) |
| `chronic_changes` | Chronic changes (7,343) |
| `infiltrates` | Infiltrates (7,156) |
| `consolidation` | Consolidation |
| `pneumothorax` | Pneumothorax |
| `atelectasis` | Atelectasis |
| `nodule` | Pulmonary nodule |
| `mass` | Pulmonary mass |
| `tuberculosis` | Tuberculosis |
| `pulmonary_fibrosis` | Pulmonary fibrosis |
| `nsg_tube` | NSG tube (device) |
| `endotracheal_tube` | Endotracheal tube (device) |
| `pacemaker` | Pacemaker (device) |
| … (172 more) |  |

The full alphabetically-sorted list of 193 snake_cased columns is in
[`configs/datasets/padchest.json`](../../../configs/datasets/padchest.json)
and in `PadChestDataset.LABEL_COLS`. Counts above are post-row-drop
totals (after the 103 NaN-Labels rows are removed).

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | — | The `images/` directory containing `0/`, …, `50/`, `54/` sibling subdirs (e.g. `/data/PadChest/images/`) |
| `csv_path` | `str` | No | auto | `PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz` (or `.csv`); auto-discovered |
| `output_cls` | `bool` | No | `False` | Include `"cls"` in data dict |
| `output_report` | `bool` | No | `False` | Include `"report"` (Spanish text) in data dict |
| `output_mask` | `bool` | No | `False` | **Not supported** — warns and is ignored |
| `output_bbox` | `bool` | No | `False` | **Not supported** — warns and is ignored |
| `drop_uncertain` | `bool` | No | `True` | Drop ~3,151 rows flagged with `exclude=1` or `suboptimal_study=1`. Set to `False` to train on the full noisy distribution. |
| `cache_dir` | `str` | No | `"./cache"` | MONAI PersistentDataset cache directory |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonizer pickle |

## Dataset constructor

```python
import torch
from radharmony.dataset import PadChestDataset

ds = PadChestDataset(
    base_image_dir="/data/PadChest/images/",
    output_cls=True,
    output_report=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

## Harmonizer: instantiate and inspect

```python
from radharmony.harmonizer import PadChestHarmonizer

h = PadChestHarmonizer(
    csv_path="/data/PadChest/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz",
    base_image_dir="/data/PadChest/images/",
)
df = h.harmonize()
print(df.columns.tolist())   # patient_id, study_id, image_path, view_position, report, + 193 labels
df.to_csv("padchest_harmonized.csv", index=False)
```

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import PadChestDataset

ds = PadChestDataset(
    base_image_dir="/data/PadChest/images/",
    harmonized_df=pd.read_csv("padchest_harmonized.csv"),
    output_cls=True,
    output_report=True,
)
```

## Harmonizer notes

- **Label parsing**: the raw `Labels` column stores a Python literal
  string list per row (e.g. `"['cardiomegaly', 'pleural effusion']"`).
  The harmonizer parses with `ast.literal_eval`, strips per-label
  whitespace (some raw labels have leading-space variants like
  `' pneumonia'` vs `'pneumonia'` — both merge into the same one-hot
  column), and one-hot encodes against the 193-label vocabulary.
- **Dropped rows**: 103 of 160,861 rows have NaN Labels and are always
  dropped. With the default `drop_uncertain=True`, an additional ~3,151
  rows flagged by the annotators with `exclude` or `suboptimal_study`
  are removed, leaving **157,607 rows**.  Pass `drop_uncertain=False`
  on either the harmonizer's `harmonize()` call or the dataset
  constructor to keep them and harmonize 160,758 rows.
- **No per-label uncertainty**: unlike CheXpert, PadChest has no `-1`
  uncertain marker for individual findings.  The Labels column is
  positive-only — a label is either in the list (encoded as 1) or
  absent (encoded as 0).  Annotators expressed uncertainty at the
  image level via the `exclude` / `suboptimal_study` meta-labels
  documented above.
- **Reports**: Spanish, pre-processed (stemmed/lemmatized) by the
  PadChest authors. Stored inline in the CSV's `Report` column — the
  harmonizer copies the text directly to `report` rather than reading
  separate report files. 17 rows have NaN reports (labels were
  available but no narrative text).
- **View position**: uses the cleaned `Projection` column
  (`PA` / `L` / `AP` / `AP_horizontal` / `COSTAL`) rather than the raw
  `ViewPosition_DICOM` (`POSTEROANTERIOR` / `LATERAL` / ...). `Projection`
  is closer to what other CXR harmonizers expose and is fully populated.
- **Image paths**: `ImageDir` is stored as an int in the CSV (0..50, 54);
  the harmonizer casts to str so the relative path is
  e.g. `"0/<hash>_<slug>.png"`. Slot 50 has exactly one PNG and slot 54
  is the catch-all that absorbs the missing 51/52/53 — both are
  legitimate quirks of the official distribution and verified against
  the per-zip `*.unzip-l.txt` listings.
- **Bboxes**: PadChest's `Localizations` column contains anatomic region
  strings (`'loc basal'`, `'loc cardiac'`, …), **not** pixel-coordinate
  bounding boxes. `PadChestDataset` therefore does not expose
  `output_bbox`.

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_cls=True` | `"cls"` | `(193,)` | Multi-label binary |
| `output_report=True` | `"report"` | `str` | Spanish pre-processed text (NaN where the source report was missing) |
