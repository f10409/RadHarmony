# Chest ImaGenome

**Modality:** CXR | **Format:** DICOM | **Dim:** 2D | **Labels:** per-box findings (no image-level cls)

## Overview

Chest ImaGenome is an automatically constructed scene-graph dataset built **on top of
MIMIC-CXR**. For each frontal chest X-ray it provides anatomical-region **bounding boxes**
(right lung, cardiac silhouette, trachea, ...) and, per region, the radiographic
**findings** present there. It ships **annotations only** - the pixels come from the
MIMIC-CXR **DICOM** tree, keyed by `dicom_id`.

Two releases are wired into RadHarmony as two configs:

| Config | Registry key | Images | Notes |
|---|---|---|---|
| Gold | `chest_imagenome_gold` | 1,000 | Manually verified, merged ground-truth boxes (26 regions) |
| Silver | `chest_imagenome_silver` | ~240,000 | Auto-generated scene graphs (up to ~36 regions), official train/valid/test splits |

**Per-box, not image-level.** Each box carries its own findings. Anatomy lives in
`bbox_labels` (e.g. `"right lung"`); the region's findings live in a parallel
`bbox_findings` column (e.g. `"lung opacity|pleural effusion"`, empty when the region has
none). There is **no** image-level `cls` vector - findings are deliberately not aggregated
to the image level.

## Download

Chest ImaGenome: [PhysioNet: Chest ImaGenome](https://physionet.org/content/chest-imagenome/1.0.0/).
Images: [PhysioNet: MIMIC-CXR](https://physionet.org/content/mimic-cxr/). Both require
PhysioNet credentialing, CITI training, and a signed DUA. This is MIMIC-derived data - do
not redistribute the annotations, harmonized tables, or images.

Expected layout:

```
CHEST-IMAGENOME/                       <- annotation_dir
  gold_dataset/
    gold_bbox_coordinate_annotations_1000images.csv
    gold_object_attribute_with_coordinates.txt
  silver_dataset/
    scene_graph.zip
    splits/{train,valid,test}.csv, images_to_avoid.csv
  utils/cxr-record-list_view.csv

MIMIC-CXR-V2-AWS/files/                <- base_image_dir (DICOM tree)
  p10/p10000032/s50414267/<dicom_id>.dcm
  ...
```

## Findings

Per-box findings are the positive (relation `yes`) scene-graph attributes in the
categories `anatomicalfinding`, `disease`, `tubesandlines`, and `device` (the `nlp` and
`technicalassessment` meta-categories are excluded). Examples: `lung opacity`,
`pleural effusion`, `pneumothorax`, `enlarged cardiac silhouette`, `atelectasis`,
`endotracheal tube`. The exact vocabulary is defined by the dataset
(`semantics/attribute_relations_v1.txt`).

## Constructor arguments

Both `ChestImaGenomeGoldDataset` and `ChestImaGenomeSilverDataset` share this signature
(silver adds `split`):

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | — | MIMIC-CXR **DICOM** `files/` root (children are `p10/`, `p11/`, ...), e.g. `/data/MIMIC-CXR-V2-AWS/files/` |
| `annotation_dir` | `str` | Yes | — | Chest ImaGenome release root (contains `gold_dataset/`, `silver_dataset/`, `utils/`), e.g. `/data/CHEST-IMAGENOME/` |
| `split` | `str` | No | `"all"` | **Silver only.** One of `"all"`, `"train"`, `"valid"`, `"test"` |
| `output_bbox` | `bool` | No | `True` | Include `"bbox"` and `"bbox_labels"` in the sample (boxes are the sole annotation) |
| `cache_dir` | `str` | No | `"./cache"` | MONAI cache directory |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to a saved harmonizer |

## Dataset constructor

### Gold

```python
import torch
from radharmony.dataset import ChestImaGenomeGoldDataset

ds = ChestImaGenomeGoldDataset(
    base_image_dir="/path/to/MIMIC-CXR-V2-AWS/files/",
    annotation_dir="/path/to/CHEST-IMAGENOME/",
    output_bbox=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
```

### Silver

```python
from radharmony.dataset import ChestImaGenomeSilverDataset

ds = ChestImaGenomeSilverDataset(
    base_image_dir="/path/to/MIMIC-CXR-V2-AWS/files/",
    annotation_dir="/path/to/CHEST-IMAGENOME/",
    split="valid",         # "all" | "train" | "valid" | "test"
    output_bbox=True,
    dtype=torch.float32,
)
```

## Harmonizer: instantiate and inspect

```python
from radharmony.harmonizer import ChestImaGenomeGoldHarmonizer

h = ChestImaGenomeGoldHarmonizer(
    annotation_dir="/path/to/CHEST-IMAGENOME/",
    base_image_dir="/path/to/MIMIC-CXR-V2-AWS/files/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'view_position',
#  'bbox', 'bbox_labels', 'bbox_findings', 'image_width', 'image_height']
```

## Load from a saved harmonizer

```python
from radharmony.dataset import ChestImaGenomeGoldDataset

# after h.save("chest_imagenome_gold.pkl")
ds = ChestImaGenomeGoldDataset(
    harmonizer_path="chest_imagenome_gold.pkl",
    output_bbox=True,
)
```

## Harmonizer notes

- Boxes are normalized to `[0, 1]` as `[y_min, y_max, x_min, x_max]` (dim0/dim1 order)
  against the original DICOM dimensions (`Rows` = H, `Columns` = W), read from the header
  and cached - mirroring the VinDr-CXR harmonizer.
- `bbox`, `bbox_labels` (anatomy), and `bbox_findings` (findings) are three parallel,
  index-aligned per-box lists. **All** detected regions are kept, including finding-free
  ones (their `bbox_findings` entry is an empty string).
- `bbox_findings` is carried through via the harmonizer's `EXTRA_OUTPUT_COLS`. With
  `output_bbox=True` it is also threaded into each sample (the dataset sets
  `SUPPORTS_BBOX_FINDINGS = True`) alongside `bbox` + `bbox_labels`, index-aligned through
  augmentation. In the visualizer app, each box is labelled `anatomy: findings`.
- Gold resolves `dicom_id -> subject/study/path` via `utils/cxr-record-list_view.csv`;
  silver reads it from the split CSVs. Instantiating a full silver split parses
  `scene_graph.zip` and reads one DICOM header per image (a one-time cost cached by
  `harmonizer.save()`).

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_bbox=True` | `"bbox"` | `(N, 4)` | One box per anatomical region, normalized `[y_min,y_max,x_min,x_max]` |
| `output_bbox=True` | `"bbox_labels"` | list[str] | Per-box **anatomy** region name |
| `output_bbox=True` | `"bbox_findings"` | list[str] | Per-box **findings**, `\|`-joined (empty string when none) |

## Example paths

| Role | Path |
|---|---|
| `base_image_dir` | `/path/to/MIMIC-CXR-V2-AWS/files/` |
| `annotation_dir` | `/path/to/CHEST-IMAGENOME/` |
