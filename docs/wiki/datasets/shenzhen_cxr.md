# Shenzhen Hospital CXR Set

**Modality:** CXR | **Format:** PNG | **Dim:** 2D | **Labels:** Tuberculosis (binary) + 19 TB finding types

## Overview

The Shenzhen Hospital CXR Set is a tuberculosis screening dataset created by the National Library of Medicine (NLM/LHNCBC) in collaboration with Shenzhen No.3 People's Hospital, Guangdong Medical College, Shenzhen, China. Images were captured as part of daily out-patient clinical routine using a Philips DR Digital Diagnose system.

- **662 total images**: 336 TB-positive, 326 normal
- **Resolution**: approximately 3000 × 3000 px (varies per image)
- **Label encoding**: embedded in each filename (`_0` = normal, `_1` = TB)
- **TB finding annotations**: 19 binary finding-type columns + per-finding region masks (TB patients only)

This is the companion dataset to the [Montgomery County CXR Set](montgomery_cxr.md), from the same NLM TB screening collaboration.

!!! note
    There is no primary CSV — `ShenzhenCXRHarmonizer` scans the `CXR_png/` directory and builds the DataFrame from filenames.

## Download

Available at [NLM/LHNCBC Tuberculosis CXR Datasets](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/index.html). No authentication required.

The NLM server blocks directory listing on the root URL, so the dataset must be mirrored per-subdirectory:

```bash
BASE="https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set"
DEST="./Shenzhen-Hospital-CXR-Set"

wget --mirror --no-parent --no-host-directories --cut-dirs=3 \
     --reject='index.html*' -e robots=off -P "$DEST" \
     "${BASE}/CXR_png/" "${BASE}/ClinicalReadings/"

wget -P "$DEST" \
     "${BASE}/NLM-ChinaCXRSet-ReadMe.docx" \
     "${BASE}/shenzhen_consensus_roi.csv"
```

For annotation files (`Annotations/`, `Annotations-2/`), index pages must be parsed to discover file names. See the [Acquiring Datasets](../acquiring-datasets.md#shenzhen-hospital-cxr) page for a complete download script.

Please cite the following publications if you use this dataset:

> Jaeger S, et al. Automatic tuberculosis screening using chest radiographs. *IEEE Trans Med Imaging*. 2014;33(2):233–45.
>
> Candemir S, et al. Lung segmentation in chest radiographs using anatomical atlases with nonrigid registration. *IEEE Trans Med Imaging*. 2014;33(2):577–90.

## Expected layout

```
Shenzhen-Hospital-CXR-Set/
  CXR_png/                              ← base_image_dir points here
    CHNCXR_0001_0.png                   # Normal
    CHNCXR_0327_1.png                   # TB-positive
    ...                                 # 662 total
  ClinicalReadings/                     # sibling of CXR_png/
    CHNCXR_0001_0.txt                   # "male 45yrs\nnormal"
    CHNCXR_0327_1.txt                   # "female 28yrs\nSTB"
    ...
  Annotations-2/                        # sibling of CXR_png/
    Statistics_ShenzhenDataset_US.csv   # 19 binary TB finding columns (TB only)
    masks/
      CHNCXR_0330_1_Pleural_Effusion_2.png
      CHNCXR_0332_1_Linear_Density_1.png
      ...                               # ~4,000 per-finding region masks
  shenzhen_consensus_roi.csv
  NLM-ChinaCXRSet-ReadMe.docx
```

`base_image_dir` points at the `CXR_png/` images directory itself — the
last common folder containing every image (matching the convention used by
other RadHarmony datasets). `ClinicalReadings/` and `Annotations-2/` are
read as sibling directories.

## Label columns

| Column | Type | Description |
|--------|------|-------------|
| `tuberculosis` | `int` (0 or 1) | `0` = normal, `1` = active tuberculosis |

## TB finding annotation columns

Present only for TB-positive patients (0327–0662); `NaN` for normal patients.
Values are counts of annotated regions (0 = absent, ≥1 = present).

| Column | Prevalence (TB patients) |
|--------|--------------------------|
| `Clustered_Nodule_(2mm-5mm_apart)` | 605 |
| `Linear_Density` | 339 |
| `Small_Infiltrate_(non-linear)` | 219 |
| `Single_Nodule_(non-calcified)` | 193 |
| `Calcified_Nodule` | 171 |
| `Pleural_Thickening_(non-apical)` | 113 |
| `Cavity` | 101 |
| `Moderate_Infiltrate_(non-linear)` | 97 |
| `Apical_Thickening` | 92 |
| `Pleural_Effusion` | 36 |
| `Adenopathy` | 30 |
| `Miliary` | 27 |
| `Retraction` | 22 |
| `Severe_Infiltrate_(Consolidation)` | 20 |
| `Other` | 71 |
| `Unknown` | 19 |
| `Thickening_of_interlobar_fissure` | 8 |
| `Calcified_lymph_node` | 8 |
| `Calcification_(other_than_nodule&lymphnod)` | 6 |

## Extra metadata columns

| Column | Type | Description |
|--------|------|-------------|
| `patient_age` | `int` | Patient age in years (from ClinicalReadings/) |
| `patient_sex` | `str` | `"M"` or `"F"` (from ClinicalReadings/) |
| `report` | `str` | Raw clinical reading text |
| `finding_masks` | `str` (JSON) | JSON list of relative paths (e.g. `"../Annotations-2/masks/..."`) to per-finding region masks; `None` for normal patients |

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes* | `None` | The `CXR_png/` images directory itself |
| `mask_output_dir` | `str` | No | `None` | Required when `output_mask=True` — directory the unioned TB-region PNGs are written to |
| `mask_num_cores` | `int` | No | `1` | Reserved for signature parity with SIIM-ACR PTX; currently unused |

### Shared arguments (inherited from `BaseRadiologicalDataset`)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `output_cls` | `bool` | No | `False` | Yield TB label under `"cls"` |
| `output_mask` | `bool` | No | `False` | Yield fused per-finding "TB region" mask under `"mask"`; requires `mask_output_dir` |
| `output_report` | `bool` | No | `False` | Yield clinical reading text under `"report"` |
| `output_bbox` | `bool` | No | `False` | Not supported; ignored |
| `transform` | Compose | No | standard 2-D 224 px | MONAI Compose transform; `output_keys` extended with `mask` when `output_mask=True` |
| `cache_dir` | `str` | No | `"./cache"` | MONAI cache directory; `None` disables |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to a saved harmonized pickle |

\* Required unless `harmonizer_path` or `harmonized_df` is provided.

## Mask output

When `output_mask=True`, the dataset class invokes
`ShenzhenCXRHarmonizer.preprocess_masks(mask_output_dir, ...)` which builds
one binary "TB region" PNG per image:

- **TB-positive cases**: union the per-finding masks listed in
  `finding_masks` (`fused = OR(mask_i for i in findings)`).
- **TB-negative cases**: write an all-zero PNG of the same size as the
  source image (matches the SIIM-ACR PTX convention so the row stays in
  the stream).

The `mask_path` column is populated with the **absolute** path to each
fused PNG. The loop is idempotent — existing PNGs are not rewritten.

Passing `output_mask=True` without `mask_output_dir` raises `ValueError`.

## Usage

```python
from radharmony.dataset import ShenzhenCXRDataset

ds = ShenzhenCXRDataset(
    base_image_dir="/data/Shenzhen-Hospital-CXR-Set/CXR_png/",
    output_cls=True,
    output_report=True,
    cache_dir="./cache",
)

sample = ds[0]
print(sample["img"].shape)   # torch.Size([1, 224, 224])
print(sample["cls"])         # tensor([0.]) or tensor([1.])

# With segmentation
ds_seg = ShenzhenCXRDataset(
    base_image_dir="/data/Shenzhen-Hospital-CXR-Set/CXR_png/",
    output_cls=True,
    output_mask=True,
    mask_output_dir="/scratch/shenzhen_tb_region_masks/",
)
```

Access TB finding annotations and mask paths via the harmonized DataFrame:

```python
df = ds.harmonized_df
tb = df[df.tuberculosis == 1]
print(tb[["patient_id", "Cavity", "Pleural_Effusion", "finding_masks"]].head())

import json
masks = json.loads(tb.iloc[0]["finding_masks"])
# ['../Annotations-2/masks/CHNCXR_0327_1_Small_Infiltrate_(non-linear)_1.png', ...]
```

## Harmonizer

```python
from radharmony.harmonizer import ShenzhenCXRHarmonizer

h = ShenzhenCXRHarmonizer(base_dir="/data/Shenzhen-Hospital-CXR-Set/CXR_png/")
df = h.harmonize()
print(df.shape)                           # (662, 27)
print(df["tuberculosis"].value_counts())  # 1: 336, 0: 326
```

## Harmonizer notes

- **No primary CSV**: labels are derived from filename suffix (`_0` / `_1`).
- **Clinical readings**: `ClinicalReadings/<stem>.txt` format — line 1 is `"<sex> <age>yrs"`, remaining lines are diagnosis text.
- **TB findings**: joined from `Annotations-2/Statistics_ShenzhenDataset_US.csv`; NaN for normal patients.
- **Finding masks**: `~4,000` per-finding region mask PNGs in `Annotations-2/masks/`; stored as a JSON list per patient in `finding_masks` (paths are sibling-relative, i.e. start with `"../Annotations-2/..."`).
- **TB-negative sample retention**: with `output_mask=True`, normals get a black-PNG mask rather than being dropped (matches SIIM-ACR PTX convention).
- **Companion dataset**: shares filename convention with [Montgomery County CXR](montgomery_cxr.md).
