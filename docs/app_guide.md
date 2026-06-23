# RadHarmony — Gradio App Guide

## Launching the app

```bash
python app.py
# or
gradio app.py
```

The app lets you load any registered dataset, configure output flags (cls, mask, report, bbox), and browse random samples with live transform previews. For 3-D datasets it also exposes HU window controls and a slice axis selector.

The **Labels** textbox shows the names of every label whose `cls` value is positive (`> 0.5`). For datasets with ordinal severity grades (e.g. TAIX-Ray with `label_mode="ordinal"`), values `≥ 2` are surfaced inline as `name (N)` — e.g. `pleural_effusion_left (3)`. Plain `name` (no parens) means severity 1, which is identical to a binary label being on. Standard binary datasets (CheXpert, MIMIC, etc.) are unaffected since their cls values never exceed 1.

---

## Adding a dataset to the app

Two things are needed: a **build function** and a **`DATASET_REGISTRY` entry**. Both go in `app.py`. The UI and loader update automatically — no other changes needed.

### The build function

The build function has a fixed signature and must return `(dataset_obj, error_str | None)`:

```python
def _build_my_dataset(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    # Return (None, "message") to abort with a user-visible error
    if not base_dir:
        return None, "Base image directory is required."

    return (
        MyDataset(
            base_image_dir=base_dir,
            csv_path=csv_path or None,
            label_csv_path=extra_field or None,   # maps to the first extra textbox
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=flags.get("output_mask", False),
            output_report=flags.get("output_report", False),
            output_bbox=flags.get("output_bbox", False),
        ),
        None,  # no error
    )
```

**Parameter mapping**

| Parameter | UI element | Notes |
|---|---|---|
| `base_dir` | "Base image dir" textbox | Always present |
| `csv_path` | "CSV path" textbox | Always present |
| `extra_field` | First extra textbox | Shown only when `extra_field_label` is non-empty |
| `extra_field2` | Second extra textbox | Shown only when `extra_field2_label` is non-empty |
| `cache_dir` | "Cache dir" textbox | Always present |
| `**flags` | Output flag checkboxes | Keys: `output_cls`, `output_mask`, `output_report`, `output_bbox` |

### DatasetConfig fields

```python
@dataclass
class DatasetConfig:
    is_3d: bool           # True  → 3-D transform pipeline + HU window controls shown
                          # False → 2-D pipeline
    extra_field_label: str          # Label for the first extra textbox; "" hides it
    build: Callable                 # The _build_* function above
    extra_field2_label: str = ""    # Label for the second extra textbox; "" hides it
    base_dir_placeholder: str = ""  # Grey hint text in the base dir textbox
    csv_placeholder: str = ""       # Grey hint text in the CSV textbox
    extra_placeholder: str = ""     # Grey hint text in the first extra textbox
    extra_placeholder2: str = ""    # Grey hint text in the second extra textbox
```

---

## Registry examples

### 2-D dataset — one extra field

```python
DATASET_REGISTRY["My Dataset"] = DatasetConfig(
    is_3d=False,
    extra_field_label="Label CSV (optional)",
    build=_build_my_dataset,
    base_dir_placeholder="e.g. /data/my_dataset/images/",
    csv_placeholder="auto: metadata.csv",
    extra_placeholder="auto: labels.csv",
)
```

### 3-D dataset — two extra fields

```python
DATASET_REGISTRY["My CT Dataset"] = DatasetConfig(
    is_3d=True,
    extra_field_label="Label CSV (optional)",
    extra_field2_label="BBox CSV (optional)",
    build=_build_my_ct_dataset,
    base_dir_placeholder="e.g. /data/my_ct_dataset/volumes/",
    csv_placeholder="auto: metadata.csv",
    extra_placeholder="auto: labels.csv",
    extra_placeholder2="auto: bboxes.csv",
)
```

### Dataset with a conditionally required extra field

When a flag combination makes a field mandatory, validate inside the build function and return an error string:

```python
def _build_my_seg_dataset(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    if flags.get("output_mask") and not extra_field:
        return None, "Mask output dir is required when Mask output is enabled."
    return (
        MyDataset(
            base_image_dir=base_dir,
            mask_output_dir=extra_field or None,
            output_mask=flags.get("output_mask", False),
        ),
        None,
    )

DATASET_REGISTRY["My Seg Dataset"] = DatasetConfig(
    is_3d=False,
    extra_field_label="Mask output dir",
    build=_build_my_seg_dataset,
    base_dir_placeholder="e.g. /data/my_dataset/dicoms/",
    csv_placeholder="auto: annotations.csv",
    extra_placeholder="/data/my_dataset/masks/",
)
```

### No extra fields (CSV-only dataset)

Set `extra_field_label=""` to hide the first extra textbox entirely:

```python
DATASET_REGISTRY["My Simple Dataset"] = DatasetConfig(
    is_3d=False,
    extra_field_label="",
    build=_build_my_simple_dataset,
    base_dir_placeholder="e.g. /data/my_dataset/images/",
    csv_placeholder="auto: train.csv",
)
```

---

## Existing registry entries (reference)

| UI name | `is_3d` | extra field 1 | extra field 2 |
|---|---|---|---|
| CheXpert | `False` | — | — |
| CheXpert-Plus | `False` | Label JSON (optional) | — |
| ChestX-ray14 | `False` | BBox CSV (to exclude bbox images) | — |
| ChestX-ray14 (BBox) | `False` | BBox CSV | — |
| MIMIC-CXR | `False` | Label CSV (optional) | Report CSV (optional) |
| MIMIC-CXR-JPG | `False` | Label CSV (optional) | — |
| RSNA Pneumonia | `False` | — | — |
| SIIM-ACR-PTX | `False` | Mask output dir | — |
| VinDr-CXR (Train) | `False` | BBox CSV (optional) | — |
| VinDr-CXR (Test) | `False` | BBox CSV (optional) | — |
| CT-RATE | `True` | Metadata CSV (optional) | — |
| RAD-ChestCT | `True` | Label CSV | BBox CSV (optional) |
