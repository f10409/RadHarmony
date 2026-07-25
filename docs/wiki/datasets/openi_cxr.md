# OpenI Indiana University CXR (IU X-Ray)

**Modality:** CXR | **Format:** PNG + DICOM | **Dim:** 2D | **Labels:** 8 binary findings (CheXpert-compatible)

## Overview

3,955 radiology reports with 7,470 paired frontal and lateral chest X-ray images from the Indiana University hospital network, released by the NLM/LHNCBC under an open-access licence.

- **7,470 images** — 2 per study (frontal PA + lateral), each stored as both PNG and DICOM
- **3,955 XML reports** — structured with FINDINGS, IMPRESSION, INDICATION sections and MeSH major terms
- **Labels** — 8 binary columns derived from MeSH major terms via prefix matching

## Download

```bash
BASE="https://openi.nlm.nih.gov/imgs/collections"
DEST="./OpenI-IU-CXR"

# PNG images (~1.4 GB)
wget -P "$DEST" "${BASE}/NLMCXR_png.tgz"
tar -xzf "$DEST/NLMCXR_png.tgz" -C "$DEST"
mv "$DEST"/*.png "$DEST/images/"

# Reports (~1.1 MB)
wget -P "$DEST" "${BASE}/NLMCXR_reports.tgz"
tar -xzf "$DEST/NLMCXR_reports.tgz" -C "$DEST"

# DICOM (~75 GB, optional)
wget -P "$DEST" "${BASE}/NLMCXR_dcm.tgz"
tar -xzf "$DEST/NLMCXR_dcm.tgz" -C "$DEST"
```

## Expected layout

```
OpenI-IU-CXR/                 ← base_image_dir points here
  ecgen-radiology/
    1.xml  ...  3955.xml
  images/
    CXR1_1_IM-0001-3001.png
    CXR1_1_IM-0001-4001.png
    ...
  1/                           ← DICOM folders (one per patient, optional)
    1_IM-0001-3001.dcm
    1_IM-0001-4001.dcm
  2/
    2_IM-0652-1001.dcm
    ...
```

## Label columns

Derived from `<MeSH><major>` tags in the XML via case-insensitive prefix matching.

| Column | Positive count | MeSH triggers |
|--------|---------------|----------------|
| `no_finding` | 2,783 | "normal" |
| `cardiomegaly` | 568 | "cardiomegaly" |
| `edema` | 118 | "pulmonary edema", "pulmonary congestion", "edema" |
| `atelectasis` | 539 | "atelectasis", "pulmonary atelectasis", "lung/hypoinflation" |
| `consolidation` | 270 | "consolidation", "pneumonia", "airspace disease" |
| `pleural_effusion` | 203 | "pleural effusion", "effusion/pleural" |
| `pneumothorax` | 35 | "pneumothorax" |
| `support_devices` | 225 | "catheters", "pacemaker", "leads", "tube", "defibrillator" |

## Extra metadata columns

| Column | Type | Description |
|--------|------|-------------|
| `dicom_path` | `str` | Relative path to DICOM file (e.g. `1/1_IM-0001-3001.dcm`) |
| `view_position` | `str` | `"frontal"`, `"lateral"`, or `"other"` (from figureId in XML) |
| `report` | `str` | Concatenated FINDINGS + IMPRESSION text |
| `findings` | `str` | FINDINGS section text |
| `impression` | `str` | IMPRESSION section text |
| `indication` | `str` | INDICATION section text |
| `mesh_major` | `str` (JSON) | Raw MeSH major terms as JSON list |

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes* | `None` | Dataset root (contains `ecgen-radiology/` and `images/`) |

### Shared arguments (inherited from `BaseRadiologicalDataset`)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `output_cls` | `bool` | No | `False` | Include `"cls"` tensor in data dict |
| `output_mask` | `bool` | No | `False` | Include `"mask"` in data dict |
| `output_report` | `bool` | No | `False` | Include `"report"` string in data dict |
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
from radharmony.dataset import OpenICXRDataset

ds = OpenICXRDataset(
    base_image_dir="/data/OpenI-IU-CXR/",
    output_cls=True,
    output_report=True,
    cache_dir="./cache",
)

sample = ds.get_datasets()[0]
print(sample["img"].shape)   # torch.Size([3, 224, 224])  — RGB PNG
print(sample["cls"])         # tensor([0., 1., 0., ...])  — 8 binary labels
print(sample["report"])      # "Heart size is normal. ..."
```

Access DICOM paths via the harmonized DataFrame:

```python
df = ds.get_harmonized_df()
print(df[["patient_id", "image_path", "dicom_path", "view_position"]].head(4))
```

## Harmonizer

```python
from radharmony.harmonizer import OpenICXRHarmonizer

h = OpenICXRHarmonizer(base_dir="/data/OpenI-IU-CXR/")
df = h.harmonize()
print(df.shape)           # (7470, 18)
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'view_position', 'report',
#  'no_finding', 'cardiomegaly', ..., 'dicom_path', 'findings', ...]
```

## Harmonizer notes

- **No primary CSV** — all data is parsed from `ecgen-radiology/*.xml`.
- **Images are RGB PNG** — the `img` tensor has shape `[3, 224, 224]`, not `[1, 224, 224]`.
- **DICOM path derivation**: XML `parentImage id` (e.g. `CXR1_1_IM-0001-3001`) maps to `1/1_IM-0001-3001.dcm` using the regex `CXR(\d+)_(?:\1_)?(.*)` → `{n}/{n}_{rest}.dcm`. Two XML naming variants exist and are both handled.
- **MeSH label coverage**: labels are only as complete as the MeSH annotations in the XML. Studies without MeSH tags get all zeros.
