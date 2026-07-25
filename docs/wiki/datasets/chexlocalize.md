# CheXlocalize

**Modality:** CXR | **Format:** JPEG | **Dim:** 2D | **Labels:** 14 pathologies

## Overview

CheXlocalize is a benchmark built on the CheXpert "test" split (668 frontal/lateral chest X-rays, 500 patients) for evaluating saliency-method localization against expert-drawn ground truth. In addition to the standard 14 CheXpert pathology labels, 499 of the 668 images have per-pathology segmentation masks (up to 10 masks per image, one per finding present) hand-drawn by a board-certified radiologist.

RadHarmony's mask pipeline supports a single combined mask per image (like SIIM-ACR unions multiple annotator rows), so `CheXlocalizeDataset` unions every per-pathology mask for an image into one binary "any abnormal region" mask when `output_mask=True`. If you need per-pathology masks kept separate, read `gt_segmentations_test.json` directly instead of going through this dataset class.

## Download

Available from the [CheXlocalize GitHub repo](https://github.com/rajpurkarlab/cheXlocalize) (Stanford AIMI / Rajpurkar Lab), which links to the Stanford AIMI download for both the CheXpert test-split images/labels and the CheXlocalize annotations/segmentations.

Expected layout after extraction (CheXpert images + CheXlocalize annotations are separate downloads that sit side by side):

```
chexlocalize/
  CheXpert/
    test_labels.csv
    test/
      patient64741/
        study1/
          view1_frontal.jpg
      ...
  CheXlocalize/
    gt_segmentations_test.json     # per-pathology COCO-RLE masks, 499/668 images
    gt_annotations_test.json       # raw radiologist polygons (not used by RadHarmony)
    hb_annotations_test.json       # human-benchmark radiologist annotations (not used)
    hb_salient_pt_test.json        # human-benchmark salient points (not used)
    gradcam_segmentations_val.json # Grad-CAM outputs on the val split (not used)
```

## Label columns

14 binary multi-label targets (same taxonomy as CheXpert; this split is fully verified — no `-1` uncertainty values):

| Column | Description |
|--------|-------------|
| `atelectasis` | Partial lung collapse |
| `cardiomegaly` | Enlarged heart |
| `consolidation` | Airspace consolidation |
| `edema` | Pulmonary edema |
| `enlarged_cardiomediastinum` | Widened mediastinum |
| `fracture` | Rib/bone fracture |
| `lung_lesion` | Lung lesion |
| `lung_opacity` | Lung opacity |
| `no_finding` | No pathology detected |
| `pleural_effusion` | Pleural effusion |
| `pleural_other` | Other pleural abnormality |
| `pneumonia` | Pneumonia |
| `pneumothorax` | Pneumothorax |
| `support_devices` | Support devices present |

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes | `None` | `CheXpert/test/` directory — direct parent of `patient.../study.../view*.jpg` (e.g. `/data/chexlocalize/CheXpert/test/`) |
| `csv_path` | `str` | No | auto | Path to `test_labels.csv`; auto-discovered near `base_image_dir` if omitted |
| `mask_json_path` | `str` | No | auto | Path to `gt_segmentations_test.json`; auto-discovered (searches sibling `CheXlocalize/` dir) if omitted. Only loaded when `output_mask=True` |
| `mask_output_dir` | `str` | No | `None` | Directory for decoded, unioned mask PNGs; required for `output_mask=True` |
| `mask_num_cores` | `int` | No | `1` | Parallel worker threads for mask decoding |

### Shared arguments (inherited from `BaseRadiologicalDataset`)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `output_cls` | `bool` | No | `False` | Include `"cls"` tensor in data dict |
| `output_mask` | `bool` | No | `False` | Include unioned segmentation mask under `"mask"` |
| `transform` | MONAI transform | No | `None` | MONAI Compose transform; `None` uses the default pipeline |
| `cache_dir` | `str` | No | `"./cache"` | MONAI cache directory; `None` = no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonized CSV |

## Dataset constructor

### Classification only

```python
import torch
from radharmony.dataset import CheXlocalizeDataset

ds = CheXlocalizeDataset(
    base_image_dir="/data/chexlocalize/CheXpert/test/",
    output_cls=True,
    dtype=torch.float32,
)
dataset = ds.get_datasets()
sample = dataset[0]
# sample["img"]  → Tensor(1, 224, 224)
# sample["cls"]  → Tensor(14,)
```

### With unioned segmentation masks

```python
from radharmony.dataset import CheXlocalizeDataset

ds = CheXlocalizeDataset(
    base_image_dir="/data/chexlocalize/CheXpert/test/",
    mask_json_path="/data/chexlocalize/CheXlocalize/gt_segmentations_test.json",
    mask_output_dir="/data/chexlocalize_masks/",
    output_cls=True,
    output_mask=True,
)
dataset = ds.get_datasets()
sample = dataset[0]
# sample["mask"] → Tensor(1, 224, 224); all-zero for images with no annotated finding
```

## Harmonizer: instantiate and inspect

```python
from radharmony.harmonizer import CheXlocalizeHarmonizer

h = CheXlocalizeHarmonizer(
    csv_path="/data/chexlocalize/CheXpert/test_labels.csv",
    mask_json_path="/data/chexlocalize/CheXlocalize/gt_segmentations_test.json",
    base_image_dir="/data/chexlocalize/CheXpert/test/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'view_position', 'mask_path', 'atelectasis', ...]
df.to_csv("chexlocalize_harmonized.csv", index=False)
```

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import CheXlocalizeDataset

ds = CheXlocalizeDataset(
    base_image_dir="/data/chexlocalize/CheXpert/test/",
    harmonized_df=pd.read_csv("chexlocalize_harmonized.csv"),
    output_cls=True,
)
```

## Harmonizer notes

- Reads `test_labels.csv`; there is no `Patient ID` or study-id column, so `patient_id` (`patient64741`) and `study_id` (`patient64741_study1`) are both extracted from the `Path` column
- `image_path` strips the `test/` prefix from `Path` so it resolves relative to `base_image_dir`
- `view_position` is derived from the image filename (`frontal` / `lateral`), since there is no dedicated CSV column
- Segmentation masks come from `gt_segmentations_test.json`, keyed by `<study_id>_<image_basename>` (e.g. `patient64741_study1_view1_frontal`), mapping to a dict of `{pathology_name: {size: [H, W], counts: <COCO-RLE>}}`. `_decode_mask` unions (bitwise OR) every pathology's mask for an image into one binary mask via `pycocotools.mask.decode` — pathology names in the JSON (e.g. `Airspace Opacity`, CheXpert's older name for `Lung Opacity`) don't need to match `LABEL_COLS` since every mask is unioned regardless of name
- Images absent from the segmentation JSON (169/668) get an all-zero mask when `output_mask=True`, the same convention SIIM-ACR uses for no-finding rows
- Only the "test" split is wired up here; CheXlocalize also ships a smaller "val" split (`gt_segmentations_val.json`, Grad-CAM comparisons) that isn't integrated

## Outputs

| Flag | Key | Shape | Notes |
|------|-----|-------|-------|
| `output_cls=True` | `"cls"` | `(14,)` | Binary labels; no uncertain values in this split |
| `output_mask=True` | `"mask"` | `(1, 224, 224)` | Unioned per-pathology mask; all-zero for the 169/668 images with no segmentation entry |

## Example paths

| Role | Path |
|---|---|
| `base_image_dir` | `/path/to/chexlocalize/CheXpert/test/` |
| `csv_path` (`test_labels.csv`) | `/path/to/chexlocalize/CheXpert/test_labels.csv` |
| `mask_json_path` (`gt_segmentations_test.json`) | `/path/to/chexlocalize/CheXlocalize/gt_segmentations_test.json` |
