# Extending the App — Adding a Dataset

To make a registered dataset selectable in the Gradio app, two things are
needed: a **build function** and a **`DATASET_REGISTRY` entry**. Both go in
`app.py`. The UI and loader update automatically — no other changes needed.

## The build function

The build function has a fixed signature and must return
`(dataset_obj, error_str | None)`:

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

## DatasetConfig fields

```python
@dataclass
class DatasetConfig:
    is_3d: bool           # True  → 3-D transform pipeline + HU window controls shown
                          # False → 2-D pipeline
    extra_field_label: str          # Label for the first extra textbox; "" hides it
    build: Callable                 # The _build_* function above
    modality: str                   # REQUIRED. One of SUPPORTED_MODALITIES — used for UI grouping
                                     # ("CXR", "Radiograph", "CT", "MRI", "VQA", "Other"). Validated
                                     # in __post_init__: an unknown value raises ValueError.
    extra_field2_label: str = ""    # Label for the second extra textbox; "" hides it
    csv_label: str = "CSV path (optional)"  # Label for the CSV textbox
    base_dir_placeholder: str = ""  # Grey hint text in the base dir textbox
    csv_placeholder: str = ""       # Grey hint text in the CSV textbox
    extra_placeholder: str = ""     # Grey hint text in the first extra textbox
    extra_placeholder2: str = ""    # Grey hint text in the second extra textbox
    extra_dropdown_label: str = ""  # Optional dataset-specific dropdown; "" hides it
    extra_dropdown_choices: tuple = ()   # Dropdown choices (include an "(all)" sentinel if needed)
    extra_dropdown_kwarg: str = ""  # kwarg name forwarded from the dropdown to the build function
    is_group_header: bool = False   # Visual separator in the dataset dropdown; selecting it is a no-op
```

`modality` is mandatory for every real dataset entry (only `is_group_header=True`
separator rows skip the check). It is independent of `is_3d`: `is_3d` drives the
transform pipeline, `modality` only drives UI grouping.

---

## Registry examples

### 2-D dataset — one extra field

```python
DATASET_REGISTRY["My Dataset"] = DatasetConfig(
    is_3d=False,
    extra_field_label="Label CSV (optional)",
    build=_build_my_dataset,
    modality="CXR",
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
    modality="CT",
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
    modality="CXR",
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
    modality="CXR",
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
