# Emory CHORUS (X-ray subset)

**Modality:** CXR | **Format:** DICOM | **Dim:** 2D | **Labels:** none (image-only) | **Access:** Internal (Emory)

## Overview

Emory CHORUS is an internal Emory dataset that pairs an OMOP-CDM clinical
warehouse (patients, conditions, measurements, procedures, visits) with a
multi-modality DICOM imaging tree. RadHarmony serves the curated **X-ray
subset** and, by default, filters it to **chest** radiographs.

This release is **image-only**: no finding labels, masks, bounding boxes, or
reports ship with it. (Labels could be derived from the OMOP condition codes in
a future step; that is out of scope here.) `patient_id` (DICOM `PatientID`)
groups a patient's studies for patient-level cross-validation.

Unlike every other RadHarmony dataset, CHORUS ships **no metadata CSV** — the
study/series/image table is built by walking the DICOM tree and reading each
file's header. That scan is expensive, so the harmonizer caches the result to a
**manifest CSV**: the first run scans and writes it; later runs read it back.

The X-ray subset spans ~1,500 patients. Filtered to chest radiographs it is
about **10.3K images / 4.9K studies / 1.4K patients** (all `DX`, `MONOCHROME2`);
the remainder are abdomen, spine, skull, and extremity films, kept only when
`chest_only=False`.

## Extra metadata columns

Carried through to the harmonized DataFrame so studies can be filtered or
stratified without re-scanning the tree:

| Column | Description |
|---|---|
| `series_id` | DICOM `SeriesInstanceUID` |
| `view_position` | DICOM `ViewPosition` (AP, PA, LL, LATERAL, …) |
| `body_part` | DICOM `BodyPartExamined` (e.g. `CHEST`, `PORT CHEST`) |
| `study_date` | DICOM `StudyDate` |
| `sex` | DICOM `PatientSex` |
| `age` | DICOM `PatientAge` |
| `manufacturer` | DICOM `Manufacturer` |

## Constructor arguments

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes* | `None` | DICOM root (`images_dicom_xray`, contains per-patient folders) |
| `manifest_csv_path` | `str` | No | `None` | Manifest scan cache; read if present, else scanned and written here |
| `chest_only` | `bool` | No | `True` | Keep only chest radiographs (`DX`/`CR` with `CHEST` in body-part/description) |
| `num_workers` | `int` | No | `12` | Parallel worker processes for the header scan |

### Shared arguments (inherited from `BaseRadiologicalDataset`)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `output_cls` | `bool` | No | `False` | Not supported (no labels); silently ignored |
| `output_mask` | `bool` | No | `False` | Not supported; silently ignored |
| `output_report` | `bool` | No | `False` | Not supported; silently ignored |
| `output_bbox` | `bool` | No | `False` | Not supported; silently ignored |
| `transform` | `Compose` | No | 2D default | MONAI transform pipeline (DICOM VOI-LUT / MONOCHROME handled) |
| `cache_dir` | `str` | No | `./cache` | PersistentDataset cache root; `None` disables caching |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Image tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonized CSV |

*Required unless `harmonizer_path`, `harmonized_df`, or `harmonizer` is provided.

## Dataset constructor

```python
from radharmony.dataset import EmoryCHORUSDataset

ds = EmoryCHORUSDataset(
    base_image_dir="/path/to/Emory_CHORUS/images_dicom_xray",
    manifest_csv_path="/path/to/Emory_CHORUS/chorus_xray_manifest.csv",
    cache_dir="./cache",
)
sample = ds[0]
# sample["img"] → image tensor (C, H, W); image-only, no other keys
```

## Harmonizer

```python
from radharmony.harmonizer import EmoryCHORUSHarmonizer

h = EmoryCHORUSHarmonizer(
    base_image_dir="/path/to/Emory_CHORUS/images_dicom_xray",
    csv_path="/path/to/Emory_CHORUS/chorus_xray_manifest.csv",  # scan cache
    chest_only=True,
)
df = h.harmonize()
# df columns: patient_id, study_id, image_path,
#             series_id, view_position,
#             body_part, study_date, sex, age, manufacturer
```

## Harmonizer notes

- **No metadata CSV.** The harmonizer scans DICOM headers to build the study /
  series / image table. Pass `manifest_csv_path` (dataset) / `csv_path`
  (harmonizer) so the scan is cached and reused; without it the tree is
  re-scanned on every construction.
- **Chest filter.** `chest_only=True` keeps `Modality` in `{DX, CR}` with
  `CHEST` in the body-part or study description. This admits combined
  chest+abdomen and rib studies whose description mentions the chest.
- **Image-only.** No labels, masks, bounding boxes, or reports in this release,
  so only the `img` key is produced.

## Example paths

| Resource | Path |
|---|---|
| Images (DICOM) | `/path/to/Emory_CHORUS/images_dicom_xray/` |
| Manifest CSV (scan cache) | user-chosen; written on first harmonize |
| OMOP-CDM tables | `/path/to/Emory_CHORUS/tables/` (not used by this release) |

Image paths follow the structure
`<person_id>/Images/<StudyInstanceUID>/<SeriesInstanceUID>/<sop>.dcm` relative
to the image root.
