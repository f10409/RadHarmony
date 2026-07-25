# RAD-ChestCT

**Modality:** CT | **Format:** NumPy (`.npz`) | **Dim:** 3D | **Labels:** 84 findings + bboxes

## Overview

RAD-ChestCT is a large chest CT dataset containing 35,747 CT scans with radiologist-annotated labels for 84 pathological findings. It also includes abnormality location labels (bounding boxes). The dataset is notable for the breadth of its label taxonomy, covering findings from nodules and consolidation to surgical hardware and transplants.

## Download

Available from [Zenodo: RAD-ChestCT](https://zenodo.org/records/6406114). Note: the
Zenodo record hosts an initial release of ~3,630 volumes (~10% of the dataset) alongside
the full `CT_Scan_Metadata_Complete_35747.csv` metadata; obtaining all 35,747 scans
requires the separate data-use agreement described on the record page.

Expected layout:

```
RAD-ChestCT/
  CT_Scan_Metadata_Complete_35747.csv
  imgtrain_Abnormality_and_Location_Labels.csv
  images/
    <NoteAcc_DEID>.npz
```

## Label columns

84 findings:
`air_trapping`, `airspace_disease`, `aneurysm`, `arthritis`, `aspiration`, `atelectasis`, `atherosclerosis`, `bandlike_or_linear`, `breast_implant`, `breast_surgery`, `bronchial_wall_thickening`, `bronchiectasis`, `bronchiolectasis`, `bronchiolitis`, `bronchitis`, `cabg`, `calcification`, `cancer`, `cardiomegaly`, `catheter_or_port`, `cavitation`, `chest_tube`, `clip`, `congestion`, `consolidation`, `coronary_artery_disease`, `cyst`, `debris`, `deformity`, `density`, `dilation_or_ectasia`, `distention`, `emphysema`, `fibrosis`, `fracture`, `gi_tube`, `granuloma`, `groundglass`, `hardware`, `heart_failure`, `heart_valve_replacement`, `hemothorax`, `hernia`, `honeycombing`, `infection`, `infiltrate`, `inflammation`, `interstitial_lung_disease`, `lesion`, `lucency`, `lung_resection`, `lymphadenopathy`, `mass`, `mucous_plugging`, `nodule`, `nodulegr1cm`, `opacity`, `other_path`, `pacemaker_or_defib`, `pericardial_effusion`, `pericardial_thickening`, `plaque`, `pleural_effusion`, `pleural_thickening`, `pneumonia`, `pneumonitis`, `pneumothorax`, `postsurgical`, `pulmonary_edema`, `reticulation`, `scarring`, `scattered_calc`, `scattered_nod`, `secretion`, `septal_thickening`, `soft_tissue`, `staple`, `stent`, `sternotomy`, `suture`, `tracheal_tube`, `transplant`, `tree_in_bud`, `tuberculosis`

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | `None` | `images/` directory — flat dir of `<NoteAcc_DEID>.npz` files (e.g. `/data/RAD-ChestCT/images/`). Note: the harmonizer constructor parameter is named `image_base_dir` (not `base_image_dir`) for legacy reasons |
| `csv_path` | `str` | No | auto | `CT_Scan_Metadata_Complete_35747.csv`; auto-discovered |
| `label_csv_path` | `str` | No | auto | Abnormality labels CSV; auto-discovered |
| `bbox_csv_path` | `str` | No | auto | Location labels CSV for bounding boxes |

### Shared arguments (inherited from `BaseRadiologicalDataset`)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `output_cls` | `bool` | No | `False` | Include `"cls"` in data dict |
| `output_mask` | `bool` | No | `False` | Include `"mask"` in data dict |
| `output_report` | `bool` | No | `False` | Include `"report"` in data dict |
| `output_bbox` | `bool` | No | `False` | Include `"bbox"` and `"bbox_labels"` in data dict |
| `transform` | MONAI transform | No | `None` | MONAI Compose transform; `None` uses the default pipeline |
| `cache_dir` | `str` | No | `"./cache"` | MONAI cache directory |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonized CSV |

## Dataset constructor

```python
import torch
from radharmony.dataset import RadChestCTDataset

ds = RadChestCTDataset(
    base_image_dir="/data/RAD-ChestCT/images/",
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

## Harmonizer: instantiate and inspect

```python
from radharmony.harmonizer import RadChestCTHarmonizer

h = RadChestCTHarmonizer(
    csv_path="/data/RAD-ChestCT/tables/CT_Scan_Metadata_Complete_35747.csv",
    image_base_dir="/data/RAD-ChestCT/images/",
    label_csv_path="/data/RAD-ChestCT/tables/imgtrain_Abnormality_and_Location_Labels.csv",
)
df = h.harmonize()
print(df.columns.tolist())
df.to_csv("radchestct_harmonized.csv", index=False)
```

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import RadChestCTDataset

ds = RadChestCTDataset(
    base_image_dir="/data/RAD-ChestCT/images/",
    harmonized_df=pd.read_csv("radchestct_harmonized.csv"),
    output_cls=True,
)
```

## Harmonizer notes

- Labels are multi-label binary from radiologist annotations
- Bounding boxes come from per-finding abnormality location labels; normalised to fractional 3D coordinates
- Very broad label taxonomy (84 classes) — not all classes have balanced representation

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_cls=True` | `"cls"` | `(84,)` | Multi-label binary findings |
| `output_mask=True` | `"mask"` | `(1, D, H, W)` | Segmentation mask |
| `output_report=True` | `"report"` | str | Free-text report |
| `output_bbox=True` | `"bbox"`, `"bbox_labels"` | list | 3D location boxes |

## Example paths

| Role | Path |
|---|---|
| `image_base_dir` | `/path/to/RAD-ChestCT/images/` |
| `csv_path` (`CT_Scan_Metadata_Complete_35747.csv`) | `/path/to/RAD-ChestCT/tables/CT_Scan_Metadata_Complete_35747.csv` |
| `label_csv_path` (train) | `/path/to/RAD-ChestCT/tables/imgtrain_Abnormality_and_Location_Labels.csv` |
| `label_csv_path` (valid) | `/path/to/RAD-ChestCT/tables/imgvalid_Abnormality_and_Location_Labels.csv` |
| `label_csv_path` (test) | `/path/to/RAD-ChestCT/tables/imgtest_Abnormality_and_Location_Labels.csv` |

The constructor parameter is `image_base_dir=` (not `base_image_dir=`) — legacy name. CSVs live in `tables/` sibling dir, auto-discoverable from `images/`.
