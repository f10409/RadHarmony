# ReXGradient-160K

**Modality:** CXR | **Format:** PNG | **Dim:** 2D | **Labels:** none | **Reports:** English (4-section)

## Overview

ReXGradient-160K is the largest publicly available multi-site chest
X-ray dataset: 273,004 PNG images from 160,000 studies / 109,487
patients across 3 U.S. health systems (79 medical sites), distributed
via HuggingFace by Rajpurkar Lab. Images were downsampled to 25 % of
their original dimensions with cubic interpolation + anti-aliasing
under the data-use agreement; the original-resolution images are
commercially available through Gradient Health.

The dataset ships **no structured pathology labels** — only free-text
reports — and pre-segments each report into four sections:
**Indication**, **Comparison**, **Findings**, **Impression**. The
harmonizer concatenates them with section headers into a single
`report` string per image (one report shared across all images of the
same study).

RadHarmony provides three split-specific classes:
- `ReXGradientTrainDataset` — 238,968 images from 140,000 studies / 95,716 patients
- `ReXGradientValidDataset` —  17,007 images from  10,000 studies / 6,964 patients
- `ReXGradientTestDataset`  —  17,029 images from  10,000 studies / 6,807 patients (public test)

An additional private test set of 10,000 studies is reserved for the
[ReXrank](https://rexrank.ai) benchmark and is not distributed here.

## Download

Primary source: HuggingFace dataset
[`rajpurkarlab/ReXGradient-160K`](https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K).
Access is gated under the **ReXGradient-160K Non-Commercial Data
Access and Use Agreement** (Harvard Medical School / Gradient Health) —
non-commercial research only, no clinical use.

After accepting the agreement on HuggingFace:

```bash
hf download rajpurkarlab/ReXGradient-160K \
    --repo-type dataset \
    --local-dir /data/ReXGradient-160K/download
```

The PNGs ship as 10 concatenated zstd-compressed tar parts
(`deid_png.part00` … `deid_png.part09`, ~15 GB each). Concatenate +
extract:

```bash
cd /data/ReXGradient-160K
cat download/deid_png.part* | tar --zstd -xvf - -C .
# → produces ./deid_png/<PatientID>/<AccessionNumber>/studies/<StudyUID>/...
```

After extraction the part files can be removed (`tar` returns 0 only
on a clean extract). Total extracted size is ~145 GB.

## Expected layout

```
ReXGradient-160K/
  deid_png/                                       ← base_image_dir
    <PatientID>/                                  e.g. GRDNLZHK1CJMB9DS/
      <AccessionNumber>/                          e.g. GRDNLD4ATLU63FN8/
        studies/
          <StudyInstanceUid>/
            series/
              <SeriesInstanceUid>/
                instances/
                  <SopInstanceUid>.png
  download/
    metadata/
      train_metadata_view_position.json           ← csv_path (per split)
      valid_metadata_view_position.json
      test_metadata_view_position.json
      train_metadata.csv                          (per-study, not used by harmonizer)
      valid_metadata.csv
      test_metadata.csv
      train_metadata.json
      valid_metadata.json
      test_metadata.json
      interstitial_pattern_bbox.json              (~398 boxes; not currently wired)
    README.md
    LICENSE
```

`base_image_dir` is the **deepest stable directory** — `deid_png/` —
whose immediate children are `<PatientID>/` directories. The
harmonizer's `image_path` is the rest of the nested path under
`base_image_dir`.

## Source of truth: JSON, not CSV

Two parallel metadata forms ship per split:

| File | Granularity | Image paths? |
|---|---|---|
| `<split>_metadata.csv` | 1 row per **study** | no |
| `<split>_metadata_view_position.json` | 1 entry per study with an `ImagePath` **list** | yes |

The CSV alone cannot produce per-image rows — every study has one or
more images and the CSV is study-keyed. The harmonizer therefore
reads the JSON. Passing a `.csv` to `csv_path` raises `ValueError`
with a message pointing at the right JSON filename.

## Label columns

**None.** ReXGradient ships no CheXpert-style structured labels.
`LABEL_COLS` is empty and `output_cls=True` is not supported (the
base class will emit a `UserWarning`). Deriving labels from the
reports (e.g. via CheXbert) is left to the user.

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | `None` | Root of the extracted PNG tree, e.g. `/data/ReXGradient-160K/deid_png/`. Immediate children are `<PatientID>/` directories. |
| `csv_path` | `str` | No | auto | Path to `<split>_metadata_view_position.json`. **Must be the JSON, not the CSV.** Auto-discovered near sibling/ancestor `metadata/` directories. |

### Shared arguments (inherited from `BaseRadiologicalDataset`)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `output_cls` | `bool` | No | `False` | **Not supported** — dataset has no structured labels |
| `output_mask` | `bool` | No | `False` | **Not supported** |
| `output_report` | `bool` | No | `False` | Include `"report"` (4-section concatenated text) in the data dict |
| `output_bbox` | `bool` | No | `False` | **Not supported** by this dataset class (a separate interstitial-pattern bbox JSON exists for ~398 examples but is out of scope here) |
| `transform` | MONAI transform | No | `None` | MONAI Compose transform; `None` uses the default pipeline |
| `cache_dir` | `str` | No | `"./cache"` | MONAI PersistentDataset cache directory |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to a saved harmonizer pickle |

## Dataset constructor: train

```python
import torch
from radharmony.dataset import ReXGradientTrainDataset

ds = ReXGradientTrainDataset(
    base_image_dir="/data/ReXGradient-160K/deid_png/",
    output_report=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=10)
sample = train_ds[0]
# sample["img"]    → torch.Tensor, shape (1, 224, 224), float in [-1, 1]
# sample["report"] → "INDICATION:\n...\n\nCOMPARISON:\n...\n\nFINDINGS:\n...\n\nIMPRESSION:\n..."
```

## Dataset constructor: validation

```python
from radharmony.dataset import ReXGradientValidDataset

ds = ReXGradientValidDataset(
    base_image_dir="/data/ReXGradient-160K/deid_png/",
    output_report=True,
)
full_ds = ds.get_datasets()   # 17,007 samples
```

## Dataset constructor: public test

```python
from radharmony.dataset import ReXGradientTestDataset

ds = ReXGradientTestDataset(
    base_image_dir="/data/ReXGradient-160K/deid_png/",
    output_report=True,
)
test_ds = ds.get_datasets()   # 17,029 samples
```

## Harmonizer

```python
from radharmony.harmonizer import ReXGradientValidHarmonizer

h = ReXGradientValidHarmonizer(
    csv_path="/data/ReXGradient-160K/download/metadata/valid_metadata_view_position.json",
    base_image_dir="/data/ReXGradient-160K/deid_png/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'view_position', 'report',
#  'accession_number', 'patient_sex', 'patient_age', 'study_date']
df.to_csv("rexgradient_valid_harmonized.csv", index=False)
```

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import ReXGradientValidDataset

ds = ReXGradientValidDataset(
    base_image_dir="/data/ReXGradient-160K/deid_png/",
    harmonized_df=pd.read_csv("rexgradient_valid_harmonized.csv"),
    output_report=True,
)
```

## Harmonizer notes

- **JSON source, per-image rows**: each JSON entry is one *study* with
  parallel lists (`ImagePath`, `ImageViewPosition`, `ImageShape`,
  `ImageModality`, `ImageBodyPart`). The harmonizer explodes by
  `ImagePath` so that each row in the harmonized DataFrame corresponds
  to one image. The 4 report sections (Indication / Comparison /
  Findings / Impression) are duplicated across all images of the same
  study.
- **Path stripping**: JSON `ImagePath` entries are written relative to
  `metadata/` as `"../deid_png/<...>"`; the harmonizer strips the
  leading `"../deid_png/"` so the result is relative to
  `base_image_dir`.
- **`csv_path` argument**: kept for interface consistency with the
  rest of RadHarmony, but it must point at the per-image JSON. A
  `.csv` path is rejected with an explicit error.
- **View position**: read inline from `ImageViewPosition` (one entry
  per image). Values include `PA`, `AP`, `LATERAL`, `LL`,
  `POSTERO_ANTERIOR`, `LAO`, `RAO`, `UNKNOWN`, `N/A`, and ~2 % `None`.
  No DICOM header reads required.
- **Reports**: combined with section headers via
  `_format_report(indication, comparison, findings, impression)`.
  Empty / `"None"` / NaN sections are skipped; rows with all four
  sections empty get a `None` report.
- **Extra metadata columns**: `accession_number`, `patient_sex`,
  `patient_age` (raw string, e.g. `"007Y"` / `"002D"`), `study_date`
  (YYYYMMDD int) are exposed via `EXTRA_OUTPUT_COLS` for downstream
  filtering.
- **Bboxes**: `metadata/interstitial_pattern_bbox.json` contains ~398
  Label-Studio-format boxes (x / y / width / height in original pixel
  coords, label = subtype of interstitial pattern). This v1
  integration **does not** wire them in. To add them later, normalize
  coordinates by the per-image `ImageShape` and join on the synthetic
  `<study_id>_<sop_uid>.png` key the bbox file uses.

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_report=True` | `"report"` | `str` | 4-section concatenated text (`INDICATION` / `COMPARISON` / `FINDINGS` / `IMPRESSION`). `None` when all 4 source sections are empty. |
| `output_cls=True` | — | — | **Not supported**; no structured labels |
| `output_mask=True` | — | — | **Not supported** |
| `output_bbox=True` | — | — | **Not supported** (see Harmonizer notes for the interstitial-pattern bbox file) |
