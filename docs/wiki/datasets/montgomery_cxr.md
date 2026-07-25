# Montgomery County CXR (TB)

**Modality:** CXR | **Format:** PNG | **Dim:** 2D | **Labels:** TB classification (binary) + lung segmentation masks

## Overview

The Montgomery County Tuberculosis Chest X-ray dataset was collected in collaboration with the National Library of Medicine (LHNCBC) and the Montgomery County Department of Health and Human Services (Maryland, USA). It contains 138 posterior-anterior (PA) chest radiographs — 80 normal and 58 TB-positive — with binary TB labels, patient demographics, and manually drawn left/right lung segmentation masks.

Despite its small size, the dataset is widely used for TB classification and lung segmentation benchmarking.

## Download

Available from the LHNCBC Tuberculosis Chest X-ray Datasets collection:

> https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/

After download, the expected layout is:

```
MontgomerySet/
  CXR_png/                   # base_image_dir points here
    MCUCXR_0001_0.png        # normal  (suffix _0)
    MCUCXR_0001_1.png        # TB      (suffix _1)
    ...
  ManualMask/                # sibling of CXR_png/
    leftMask/
      MCUCXR_0001_0.png
      ...
    rightMask/
      MCUCXR_0001_0.png
      ...
  ClinicalReadings/          # sibling of CXR_png/
    MCUCXR_0001_0.txt        # one file per patient
    ...
```

`base_image_dir` points at the `CXR_png/` images directory itself — the
last common folder containing every image (matching the convention used by
other RadHarmony datasets). `ClinicalReadings/` and `ManualMask/` are read
as sibling directories.

### Downloading

The dataset is publicly available — no account required. Download and unzip with:

```bash
BASE_URL="https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set"

wget -q "${BASE_URL}/MontgomerySet.zip" -O MontgomerySet.zip
unzip -q MontgomerySet.zip
# → produces MontgomerySet/ in the current directory
```

If the zip URL redirects or is unavailable, visit the index page and follow the download link:
> https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/index.html

## Label columns

| Column | Type | Description |
|--------|------|-------------|
| `tb` | int (0/1) | Tuberculosis status — 0 = normal, 1 = TB-positive |

## Extra columns

| Column | Description |
|--------|-------------|
| `sex` | Patient sex (M/F) |
| `age` | Patient age in years |
| `mask_path_left` | Relative path to left lung mask PNG (if present) |
| `mask_path_right` | Relative path to right lung mask PNG (if present) |

## Constructor arguments (harmonizer)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_dir` | `str` | Yes | — | The `CXR_png/` images directory itself (sibling `ClinicalReadings/` and `ManualMask/` are read relative to it) |

`MontgomeryCXRHarmonizer.harmonize()` also accepts `mask_output_dir` and
`mask_num_cores` kwargs (see *Mask output*).

## Constructor arguments (dataset)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `base_image_dir` | `str` | `None` | The `CXR_png/` images directory |
| `transform` | MONAI Compose | standard 224 px 2-D | Default 2-D pipeline; `output_keys` extended with `mask` when `output_mask=True` |
| `cache_dir` | `str` | `"./cache"` | MONAI `PersistentDataset` cache; `None` disables |
| `mask_output_dir` | `str` | `None` | Required when `output_mask=True` — directory the fused left+right lung-mask PNGs are written to |
| `mask_num_cores` | `int` | `1` | Reserved for signature parity with SIIM-ACR PTX; currently unused (fusion loop is small) |
| `output_cls` | `bool` | `False` | Yield TB label under the `"cls"` key |
| `output_mask` | `bool` | `False` | Yield fused lung mask under the `"mask"` key; requires `mask_output_dir` |
| `output_report` | `bool` | `False` | Yield clinical reading text under the `"report"` key |
| `dtype` | `torch.dtype` | `torch.bfloat16` | Output tensor dtype |
| `harmonized_df` | `pd.DataFrame` | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | `None` | Path to saved harmonized CSV |
| `output_bbox` | `bool` | `False` | Not supported; ignored |

## Mask output

When `output_mask=True`, the dataset class invokes
`MontgomeryCXRHarmonizer.preprocess_masks(mask_output_dir, ...)` which
**fuses the left and right manual lung masks into a single binary PNG per
image** (`fused = (left > 0) | (right > 0)`) and writes one file per image
into `mask_output_dir`. The harmonized DataFrame's `mask_path` column is
then populated with the **absolute** path to each fused PNG, so MONAI can
load it directly. The fusion loop is idempotent — existing fused PNGs are
not rewritten on subsequent runs.

Passing `output_mask=True` without `mask_output_dir` raises `ValueError`.

## Usage

```python
from radharmony.harmonizer import MontgomeryCXRHarmonizer
from radharmony.dataset import MontgomeryCXRDataset

# Harmonizer-only flow
h = MontgomeryCXRHarmonizer(base_dir="/data/MontgomerySet/CXR_png/")
df = h.harmonize()
print(df.shape)                       # (138, ...)
print(df["tb"].value_counts())        # 0: 80, 1: 58
print(df.columns.tolist())

# Dataset flow with classification + lung mask output
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    output_cls=True,
    output_mask=True,
    mask_output_dir="/scratch/montgomery_fused_masks/",
)
```

## Notes

- The filename suffix (`_0` / `_1`) encodes TB status; this is the
  authoritative source of the `tb` label.
- Lung masks are optional — `mask_path_left` and `mask_path_right` are
  `NaN` if `ManualMask/` is absent. In that case `output_mask=True` rows
  will be dropped from the data stream.
- `ClinicalReadings/<stem>.txt` is one file per patient (free-text plus
  `Patient's Sex` / `Patient's Age` headers); the harmonizer parses
  `sex`, `age`, and `report` from it.
