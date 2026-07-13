---
name: add-dataset
description: >
  Scaffold a new radiological dataset into the RadHarmony pipeline. Creates the harmonizer,
  dataset class, app.py integration, and updates all __init__.py exports. Use when the user
  says "add a dataset", "integrate a new dataset", "scaffold a dataset", or provides CSV
  samples and asks to wire them into RadHarmony.
---

# Add Dataset Skill

Generate all files and edits needed to integrate a new radiological dataset into RadHarmony.
This skill produces four deliverables: a harmonizer, a dataset class, app.py wiring, and
package exports.

## When This Skill Triggers

- User asks to add, integrate, or scaffold a new dataset
- User provides CSV samples or column names and asks to wire them into the pipeline
- User says "add-dataset" or references this skill

## Step 0: Dataset Config JSON

Before asking the user any questions, **create a JSON config file** at
`configs/datasets/<registry_key>.json` using the template below.
Pre-fill any fields you can infer from the user's message (dataset name, paths, etc.)
and leave the rest as `null`. Then show the user the file and ask them to review / fill in
the missing fields.

If the user leaves fields as `null` or says they don't know, **try to infer** from:
1. Inspecting the CSV files at the provided paths (read headers, sample rows)
2. Listing the image directory structure (file extensions, folder hierarchy)
3. Looking for companion CSVs near `base_image_dir`
4. Reading DICOM headers for metadata (view position, modality)

Only ask the user if you truly cannot infer a value.

### JSON Template

```json
{
  "_comment": "Free-text description of the dataset, its source, and any relevant context.",

  "dataset_name": "<human-readable name, e.g. ChestX-ray14>",
  "registry_key": "<snake_case key, e.g. chestxray14>",
  "modality": "<2D or 3D>",
  "image_format": "<PNG, JPEG, DICOM, NIfTI, or NPZ>",

  "base_image_dir": "<deepest directory that is the direct parent of the entries in image_path; e.g. 'train_images/' for split datasets, NOT the competition root>",

  "primary_csv": {
    "_comment": "The main metadata CSV that lists every image row.",
    "path": "<path to the main metadata CSV>",
    "filename_variants": ["<all known filenames for infer_path, e.g. metadata.csv, metadata.csv.gz>"]
  },

  "patient_id": {
    "_comment": "How to obtain the patient ID. Provide column if it exists, or describe derivation.",
    "column": "<column name, or null if derived>",
    "derivation": "<how to derive if not a direct column, e.g. 'extract from Path column'>"
  },

  "study_id": {
    "_comment": "How to obtain the study ID. Provide column if it exists, or describe derivation.",
    "column": "<column name, or null if derived>",
    "derivation": "<how to derive if not a direct column>"
  },

  "image_path": {
    "_comment": "How to obtain the image file path relative to base_image_dir. Must NOT contain a hardcoded subdirectory prefix that could be folded into base_image_dir instead — see the 'base_image_dir Convention' section.",
    "column": "<column name containing image paths, or null if constructed>",
    "derivation": "<how to construct if not a direct column>",
    "prefix_to_strip": "<prefix to remove so paths are relative to base_image_dir, or null>"
  },

  "labels": {
    "_comment": "Label columns and how to handle uncertain values.",
    "columns": ["<list of label column names as they appear in the CSV, sorted alphabetically>"],
    "separate_csv": {
      "_comment": "If labels are in a separate CSV, describe path and how to join.",
      "path": "<path to label CSV, or null if labels are in primary CSV>",
      "filename_variants": ["<all known filenames>"],
      "join_columns": ["<columns to join on>"]
    },
    "uncertain_handling": {
      "_comment": "How to handle missing or uncertain label values.",
      "nan_action": "<'keep' or 'to_zero' — what to do with NaN values>",
      "negative_one_action": "<'keep', 'to_nan', or 'drop' — what to do with -1 values>",
      "drop_uncertain_rows": "<true or false — drop rows that still have NaN after mapping>"
    }
  },

  "view_position": {
    "_comment": "View position info (e.g. AP/PA/Lateral). Can come from CSV or DICOM headers.",
    "column": "<CSV column name, or null>",
    "source": "<'csv', 'dicom_header', or null>",
    "separate_csv": {
      "path": null,
      "filename_variants": [],
      "join_columns": []
    }
  },

  "masks": {
    "_comment": "Segmentation mask support.",
    "supported": false,
    "column": null,
    "format": "<'rle', 'file_path', or null>"
  },

  "bboxes": {
    "_comment": "Bounding box support.",
    "supported": false,
    "column": null
  },

  "reports": {
    "_comment": "Radiology report support. Describe format and location.",
    "supported": false,
    "format": "<'text_files', 'csv_column', or null>",
    "separate_csv": {
      "_comment": "CSV that maps studies to report file paths.",
      "path": null,
      "filename_variants": [],
      "join_columns": []
    },
    "base_dir": "<directory containing report files, or null>",
    "path_column": "<column in report CSV containing report file paths, or null>"
  },

  "extra_init_params": {
    "_comment": "Any additional constructor parameters not covered above.",
    "<param_name>": "<description, e.g. dicom_base_dir: root of DICOM tree for header reads>"
  },

  "hu_window": "<[min, max] for 3D CT, or null>",
  "custom_preprocessing": "<description of any special preprocessing needed, or null>"
}
```

### Discovery Notebook

Users can also fill in the JSON interactively using `notebooks/dataset_discovery.ipynb`.
The notebook guides them through inspecting CSVs, directories, and DICOM headers,
then exports the completed JSON to `configs/datasets/<registry_key>.json`.

### Where to download from: prefer the primary source

Before suggesting any download method, **identify the dataset's primary source**
— usually the hosting institution (RSNA, NIH, Stanford AIMI, PhysioNet, etc.)
or a peer-reviewed paper's data availability statement. That's what to
download from, and that's what to point users at in the planning doc / README.

**Do not default to Kaggle mirrors**, even when the data is available there:

- Many Kaggle "datasets" are third-party re-uploads with no guarantee of
  byte-for-byte parity with the primary release (duplicated rows, dropped
  edge-case images, recompressed/reprocessed files, stripped metadata).
- Even official Kaggle *competitions* sometimes diverge from the primary
  release — e.g. include only a subset, re-encode images, or bundle
  pre-processed variants.
- Reproducibility expectations in downstream research assume the primary
  release.

Kaggle `kagglehub` is the right tool **only** when the dataset's primary
distribution channel *is* Kaggle (competitions whose data isn't released
elsewhere, e.g. SIIM-FISABIO-RSNA COVID-19 Detection).  Flag this explicitly
in the planning doc for each dataset.

### Kaggle-hosted datasets

When Kaggle *is* the primary source, suggest `kagglehub` to the user so they
can fetch it locally before harmonization. The cached path becomes
`base_image_dir`.

```python
import kagglehub

# For competitions
path = kagglehub.competition_download("<competition-slug>")
# For public datasets
# path = kagglehub.dataset_download("<owner>/<dataset-slug>")

print("Path to files:", path)
```

Authentication notes:

- Kaggle's current token format is `KGAT_<hex>` (from Settings → API → Create New API Token).
- Use it via the `KAGGLE_API_TOKEN` **environment variable**, not the legacy
  `~/.kaggle/kaggle.json` file — the new format isn't read from `kaggle.json`.
- Public datasets can download with no auth; anything requiring auth (competitions,
  `kaggle competitions list`, etc.) needs the env var set.
- Competition downloads also require accepting the rules on the competition page first.

This downloads into `~/.cache/kagglehub/` (or `$KAGGLEHUB_CACHE`). Do **not** embed
kagglehub calls inside harmonizer or dataset code — keep downloading as a user-side
step and let the constructor take the resolved path.

### Workflow After JSON Creation

1. **Create the JSON** at `configs/datasets/<registry_key>.json` with as much pre-filled
   as possible.
2. **Show the user** the JSON and ask them to review. Highlight any `null` fields that are
   required.
3. **Try to infer** missing fields:
   - Read CSV headers and sample rows
   - List image directory structure
   - Search for companion CSVs near `base_image_dir`
   - Read DICOM headers if applicable
4. **Ask the user** only for fields you cannot infer.
5. **Update the JSON** with inferred/provided values before proceeding to code generation.

## Architecture Overview

Every dataset in RadHarmony consists of:

```
radharmony/harmonizer/<name>/     -- reads raw CSVs, returns standardized DataFrame
  __init__.py                          (re-exports the harmonizer class)
  <name>.py                            (implementation)
radharmony/dataset/<name>/         -- MONAI Dataset wrapper with transforms & splitting
  __init__.py                          (re-exports the dataset class)
  <name>.py                            (implementation)
app.py                            -- Gradio app build function + registry entry
```

**Every dataset lives in a folder on both sides**, even when it starts with
a single class. This leaves room for future variants (train/test split,
release versions like `<Name>Plus`, alternate readers like `<Name>PNG`)
without a later restructure. Only shared utilities (`base.py`, `base_vqa.py`,
`transforms.py`) remain as top-level `.py` files inside `harmonizer/` and
`dataset/`.

## `base_image_dir` Convention

`base_image_dir` MUST point at the **direct parent of the entries named in
`image_path`** — i.e. the deepest directory under which the harmonizer's
`image_path` resolves the image file or DICOM series directory. Do **not**
hardcode an extra subdirectory prefix into `image_path` and then ask the user
to provide the parent.

**Right** — image_path is fully relative to the directory the user provides:

```python
# base_image_dir = .../stage_2_train_images/
self.df["image_path"] = self.df["patientId"].astype(str) + ".dcm"
```

**Wrong** — hardcoded prefix forces the user to enter a different (less
specific) path:

```python
# base_image_dir = .../rsna-pneumonia-detection-challenge/  (parent)
self.df["image_path"] = "stage_2_train_images/" + self.df["patientId"].astype(str) + ".dcm"
```

**For Train/Test split datasets**: each split takes its own image directory
as `base_image_dir` (e.g. `train_images/` and `test_images/`), the same way
RSNA PE Detection, RSNA Abdominal Trauma, RSNA Cervical Spine, RSNA Lumbar
Spine, VinDr-CXR, and SIIM COVID-19 do. Don't share a parent root across
splits — the user enters the split-specific directory.

**Stripping cruft is fine**: when the upstream CSV stores paths with a
prefix that the user shouldn't have to provide (e.g. CheXpert's `Path`
column starts with `CheXpert-v1.0/train/...`, MIMIC-CXR's `path` starts
with `files/...`), strip that prefix in `_build_image_path` so
`base_image_dir` aligns with the deepest stable directory.

**CSV auto-discovery still works**: `infer_path` walks up from
`base_image_dir` to its parent (and grandparent) when looking for CSVs, so
moving `base_image_dir` deeper does not break label / metadata file
resolution as long as the CSVs live in a sibling or ancestor directory.

**Structural exception — multiple sibling image directories inside the
base**: when the images are genuinely split across several sibling subdirs
that live *inside* `base_image_dir`, `image_path` has to carry the subdir
name. Examples: ChestX-ray14 (`base_image_dir` is the CXR14 root,
`image_path` starts with `images_001/`, …, `images_012/`); RSNA Bone Age
val (`base_image_dir` is `Bone Age Validation Set/`, `image_path` starts
with `boneage-validation-dataset-1/` or `-2/`). The exception must be
self-contained — `base_image_dir` should still be the **deepest** stable
directory; never inflate it just to share a base across splits. Document
this explicitly in the harmonizer's class docstring and on the wiki page.

When the wiki says "`base_image_dir`: `train/` directory", the user types
the path to `train/` — not the competition root.

## Style Notes

- Include **concrete example paths** in docstrings (e.g. ``.../mimic-cxr/2.1.0/files/``).
- Add **inline comments** for non-obvious logic (e.g. why a prefix is stripped, why a column
  is cast to str). Future readers should understand the "why" without reading the CSV.

## File 1: Harmonizer (`radharmony/harmonizer/<name>/<name>.py`)

Create the folder `radharmony/harmonizer/<name>/` containing:

- `<name>.py` — implementation (template below)
- `__init__.py` — re-export line: `from .<name> import <Name>Harmonizer`

Relative imports inside `<name>.py` must use `..` to reach sibling modules
at the `harmonizer/` level — e.g. `from ..base import BaseHarmonizer` and
`from ..base_vqa import BaseVQAHarmonizer` — because the file is one level
deeper than `base.py`.

### Template

```python
"""Harmonizer for the <HumanName> dataset."""

import os

import pandas as pd

from ..base import BaseHarmonizer


class <Name>Harmonizer(BaseHarmonizer):
    """Harmonize <HumanName> into the standard RadHarmony format.

    Args:
        csv_path: Path to the primary metadata CSV.
        label_csv_path: Path to the label CSV (if separate).
        <extra args as needed>
    """

    LABEL_COLS = [
        # Original column names exactly as they appear in the CSV.
        # BaseHarmonizer auto-snake_cases them.
        # MUST be sorted alphabetically.
    ]

    # --- Join / source column configuration ---
    # Set to [] or None when the feature is not supported.
    LABEL_JOIN_COLS = []
    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = []
    REPORT_PATH_COL = None

    def __init__(self, csv_path: str, label_csv_path: str = None, base_image_dir: str = None):
        super().__init__(csv_path=csv_path, label_csv_path=label_csv_path)
        self.base_image_dir = base_image_dir

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = ...  # derive from CSV columns

    def _build_study_id(self) -> None:
        self.df["study_id"] = ...  # derive from CSV columns

    def _build_image_path(self) -> None:
        self.df["image_path"] = ...  # relative path from base_image_dir

    # Override _harmonizer_init_snapshot if __init__ has extra params beyond BaseHarmonizer:
    # def _harmonizer_init_snapshot(self) -> dict:
    #     snap = super()._harmonizer_init_snapshot()
    #     snap["base_image_dir"] = self.base_image_dir
    #     return snap
```

### Key Rules

1. **LABEL_COLS**: Use original CSV column names (e.g. `"Pleural Effusion"`). The base class
   snake_cases them automatically. Sort alphabetically.
   **Snake-case convention**: The base class uses `col.lower().replace(" ", "_")` — it does
   **NOT** replace `/` or other punctuation. So `"Nodule/Mass"` → `"nodule/mass"` (not
   `"nodule_mass"`). When overriding `_build_labels()`, use the same convention or the
   column will be silently dropped by `_select_harmonized_columns()`.
2. **`_build_image_path()`**: Must produce paths relative to `base_image_dir`. The dataset
   class prepends `base_image_dir` via `get_data_dict()`.
3. **`super().__init__()`**: Pass ALL CSV path kwargs that the base class accepts:
   `csv_path`, `label_csv_path`, `view_position_csv_path`, `mask_csv_path`, `bbox_csv_path`,
   `report_csv_path`, `report_base_dir`.
4. **`_harmonizer_init_snapshot()`**: Override if your `__init__` has extra params not in
   BaseHarmonizer. This is required for `save()` / `load_from_saved()` to work.
5. **Optional hooks**: Override `_preprocess_label_df()`, `_preprocess_bbox_df()`, etc. to
   clean CSVs before merge. Override `_decode_mask()` for RLE-encoded masks.
6. **Custom `harmonize()`**: Only override if the base flow (read CSV -> build columns ->
   merge labels -> merge optional CSVs) doesn't fit. See RadChestCTHarmonizer for an example
   of a fully custom `harmonize()` that melts/pivots label columns.
7. **Multi-bbox datasets**: When a dataset has multiple bounding boxes per image (e.g.
   VinDr-CXR), override `_build_bbox()` to aggregate them. Store as a list of
   `[dim0_min, dim0_max, dim1_min, dim1_max]` lists (normalized to `[0, 1]`).
   The pipeline (`_BboxToMask`, `_overlay_bbox`) handles both single and multi-bbox
   formats.  To normalize pixel-coordinate bboxes, read DICOM headers
   (`pydicom.dcmread(path, stop_before_pixels=True)`) to get `Rows`/`Columns` and
   divide accordingly.  Only read headers for images that actually have bboxes.

8. **Bboxes on 3-D volumes**: encode every box as a 6-tuple
   `[d_min, d_max, y_min, y_max, x_min, x_max]` normalized to `[0, 1]`.
   The same shape covers both supported flavors:

   - **True volumetric bbox** spanning many slices, e.g. RadChestCT's
     lung-region extrema (`sup_axis0min` … `lef_axis2max` normalized
     by `(shape0, shape1, shape2)` — see
     `radharmony/harmonizer/radchestct/radchestct.py:_preprocess_bbox_df`).
     Typical case: one bbox per scan describing an organ-scale region.
   - **Per-slice 2-D annotation** on a single slice, e.g. RSNA Lumbar
     Spine and RSNA Cervical Spine BBox. The depth range pins the
     box to its slice (`eps_z = 0.5 / n_instances`) while the in-plane
     coords stay native; conceptually 2-D, encoded as 6-tuple so the
     pipeline machinery is uniform.

   Coords must align with the post-`OrientationD(IPL)` +
   `TransposeD([0,3,2,1])` axis order — dim 0 = I (slice/D),
   dim 1 = P (H), dim 2 = L (W). Anchor the slice axis on **ascending
   `ImagePositionPatient[2]`** (z-position), not `InstanceNumber` —
   some series store the two in opposite directions and ITKReader
   always loads ascending-z. See
   `radharmony/harmonizer/rsna_2024_lumbar_spine/rsna_2024_lumbar_spine.py`
   for the canonical pattern.

   **Sub-pixel rendering**: per-slice annotations end up 1-voxel-thick
   in the depth axis after resampling, which on coronal/sagittal panels
   maps to one or zero pixels. `_overlay_bbox` auto-inflates such
   rectangles so the green outline still renders visibly — no special
   handling needed in the harmonizer or dataset class.

   **No-finding rows**: harmonizer outputs `[]` for rows without
   annotations. The transform pipeline writes the all-zeros
   placeholder `[0.0] * (2*ndim)` for empty channels, and
   `_overlay_bbox` skips that sentinel automatically — no special
   handling required in the dataset class.

   **Dense per-anatomy annotations** (one bbox per vertebra/level/etc.,
   typical of the per-slice flavor): add the dataset's display name to
   `_BBOX_FOCUS_FIRST_DATASETS` in `app.py` (around line 1447). This
   switches the 3-D Gradio renderer to pick slice indices from the
   focused bbox's center on all three panels, render only the selected
   bbox, and reveal the "Next bbox →" button for cycling through
   annotations without re-rolling the sample. Volumetric-bbox datasets
   like RadChestCT do **not** belong in this set — their single
   organ-scale bbox already spans the volume's mid-slices, so default
   mid-slice rendering is correct.

## File 2: Dataset (`radharmony/dataset/<name>/`)

Each dataset lives in its own subdirectory package:

```
radharmony/dataset/<name>/
  __init__.py   # re-exports the public dataset class(es)
  <name>.py     # implementation
```

`__init__.py` should contain only the re-export, e.g.:

```python
from .<name> import <Name>Dataset

__all__ = ["<Name>Dataset"]
```

Note: relative imports inside `<name>.py` must use `..` to reach sibling modules
at the `dataset/` level — e.g. `from ..base import BaseRadiologicalDataset` and
`from ..transforms import RadiologyTransform2D`.

### Template

```python
"""MONAI dataset for the <HumanName> dataset."""

import torch

import pandas as pd

from radharmony.harmonizer import <Name>Harmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D  # or RadiologyTransform3D


@register_dataset("<registry_key>")
class <Name>Dataset(BaseRadiologicalDataset):
    """PyTorch dataset for <HumanName>.

    Args:
        base_image_dir: Root directory containing image files.
        csv_path: Path to the primary metadata CSV. Auto-inferred when None.
        label_csv_path: Path to the label CSV (if separate). Auto-inferred when None.
        transform: MONAI Compose transform. Defaults to standard pipeline.
        cache_dir: PersistentDataset cache directory. None disables caching.
        output_cls: Yield label vector under key ``cls``.
        output_mask: Yield segmentation mask under key ``mask``.
        output_report: Yield report text under key ``report``.
        output_bbox: Yield bounding box under key ``bbox``.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls"})  # set to the outputs this dataset provides
    LABEL_COLS = [
        # snake_cased label names matching the harmonized DataFrame.
        # MUST be sorted alphabetically.
    ]
    _HARMONIZER_CLS = <Name>Harmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        label_csv_path: str = None,
        transform=None,
        cache_dir: str = "./cache",
        output_cls: bool = False,
        output_mask: bool = False,
        output_report: bool = False,
        output_bbox: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        # Infer base_image_dir from saved harmonizer if needed
        if harmonizer_path is not None and base_image_dir is None:
            _h = <Name>Harmonizer.load_from_saved(harmonizer_path)
            base_image_dir = _h.base_image_dir

        if base_image_dir is None and harmonizer_path is None and harmonized_df is None and harmonizer is None:
            raise ValueError("base_image_dir is required when harmonizer_path is not provided.")

        # Auto-discover CSVs near base_image_dir
        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._csv_path = infer_path(
                base_image_dir, "<primary_csv_filename>", user_path=csv_path or "",
            ) or csv_path
            self._label_csv_path = infer_path(
                base_image_dir, "<label_csv_filename>", user_path=label_csv_path or "",
            ) or label_csv_path
        else:
            self._csv_path = csv_path
            self._label_csv_path = label_csv_path

        if transform is None:
            output_keys = {"img"}
            if output_cls:
                output_keys.add("cls")
            if output_mask:
                output_keys.add("mask")
            if output_bbox:
                output_keys.add("bbox")
            if output_report:
                output_keys.add("report")
            # Use RadiologyTransform3D for CT volumes
            transform = RadiologyTransform2D(
                img_size=224, output_keys=output_keys, dtype=dtype,
            ).get_transform()

        super().__init__(
            base_image_dir=base_image_dir,
            transform=transform,
            cache_dir=cache_dir,
            output_cls=output_cls,
            output_mask=output_mask,
            output_report=output_report,
            output_bbox=output_bbox,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
        )

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        harmonizer = <Name>Harmonizer(
            csv_path=self._csv_path,
            label_csv_path=self._label_csv_path,
        )
        return harmonizer.harmonize()
```

### Key Rules

1. **LABEL_COLS**: Snake_cased versions of harmonizer labels. Sorted alphabetically.
   Must match exactly what the harmonized DataFrame produces.
2. **`_HARMONIZER_CLS`**: Required for `harmonizer_path=` to work.
3. **`_get_harmonized_df()`**: Always call `_try_resolve_preset_harmonized()` first.
4. **`super().__init__()`**: Forward `harmonized_df`, `harmonizer`, `harmonizer_path`.
5. **`infer_path()`**: Use for CSV auto-discovery. Pattern: `infer_path(...) or csv_path`.
   List **all known filename variants** (from `filename_variants` in the JSON config) as
   positional args so auto-discovery works regardless of which version the user has on disk.
6. **Default transform**: `RadiologyTransform2D` for 2D, `RadiologyTransform3D` for 3D.
   For 3D, include `hu_window` and set `base_transpose` as needed.
7. **Unsupported outputs**: Declare `SUPPORTED_OUTPUTS: frozenset` on the class listing the output keys the dataset actually provides (e.g. `frozenset({"cls"})` for classification-only datasets). The base class `__init__` automatically emits a `UserWarning` for any flag the user enables that is absent from `SUPPORTED_OUTPUTS`. Do **not** add per-class warning logic — the base handles it.
8. **`dtype` parameter**: Always add `dtype=torch.bfloat16` to `__init__` and pass it to
   the default transform builder (`RadiologyTransform2D(... dtype=dtype)` or
   `RadiologyTransform3D(... dtype=dtype)`). This lets callers override the output tensor
   precision (e.g. `dtype=torch.float32` for inference) without having to construct the
   full transform themselves. Requires `import torch` at the top of the dataset file.
   When the user passes a custom `transform=`, `dtype` is ignored — it only applies to
   the default transform built inside `__init__`.
9. **RLE mask guard**: When `masks.format = "rle"` in the config, the harmonizer stores
   raw RLE strings in `mask_path`. MONAI's `LoadImageD` will receive those strings as
   file paths and crash with a cryptic `OSError: File name too long`. Add a `ValueError`
   in `__init__` that fires when `output_mask=True`, `mask_output_dir=None`, and no
   preset harmonizer/df is provided. Example:
   ```python
   if output_mask and mask_output_dir is None and harmonizer_path is None \
           and harmonized_df is None and harmonizer is None:
       raise ValueError(
           "output_mask=True requires mask_output_dir to be set. "
           "<Dataset> masks are RLE strings and must be decoded to image files "
           "before MONAI can load them. Pass mask_output_dir='/path/to/masks'."
       )
   ```
   The guard is skipped for preset paths because those DataFrames may already contain
   decoded file paths.

## File 3: App Integration (`app.py`)

### Add a build function

```python
def _build_<name>(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    csv_path = _infer(base_dir, "<primary_csv_filename>", user_path=csv_path)
    if not csv_path:
        return None, "<primary CSV> is required."
    label_csv = _infer(base_dir, "<label_csv_filename>", user_path=extra_field) if extra_field is not None else None
    return (
        <Name>Dataset(
            base_image_dir=base_dir,
            csv_path=csv_path,
            label_csv_path=label_csv,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=flags.get("output_mask", False),
            output_report=flags.get("output_report", False),
            output_bbox=flags.get("output_bbox", False),
        ),
        None,
    )
```

### Add to DATASET_REGISTRY

```python
"<HumanName>": DatasetConfig(
    is_3d=False,  # True for CT
    extra_field_label="Label CSV (optional)",  # or "" to hide
    build=_build_<name>,
),
```

### Rules

1. Use `_infer()` (app.py's wrapper around `infer_path`) for CSV discovery.
2. Return `(None, "error message")` if a required CSV is missing.
3. All `output_*` flags should be passed through from `**flags`.
4. Add the import at the top of `app.py`.

## File 4: Package Exports

### `radharmony/harmonizer/<name>/__init__.py`
Create (part of File 1): `from .<name> import <Name>Harmonizer`

### `radharmony/harmonizer/__init__.py`
Add: `from .<name> import <Name>Harmonizer`

### `radharmony/dataset/<name>/__init__.py`
Create (part of File 2): `from .<name> import <Name>Dataset`

### `radharmony/dataset/__init__.py`
Add: `from .<name> import <Name>Dataset`

## Base Class Policy

**Do not modify base classes** (`BaseHarmonizer`, `BaseRadiologicalDataset`,
`RadiologyTransform2D`, `RadiologyTransform3D`, etc.) when adding a new dataset.
All new logic should live in the new dataset's own harmonizer and dataset files.

If a base class change is truly unavoidable, **stop and notify the user** before making
the edit. Explain what needs to change and why, and get explicit approval first.

## Checklist

After generating all files, verify:

- [ ] Harmonizer LABEL_COLS sorted alphabetically (original case)
- [ ] Dataset LABEL_COLS sorted alphabetically (snake_case)
- [ ] Dataset LABEL_COLS matches harmonizer's snake_cased output exactly
- [ ] `_HARMONIZER_CLS` set on dataset class
- [ ] `_harmonizer_init_snapshot()` overridden if harmonizer has extra `__init__` params
- [ ] `_try_resolve_preset_harmonized()` called first in `_get_harmonized_df()`
- [ ] `harmonized_df`, `harmonizer`, `harmonizer_path` forwarded to `super().__init__()`
- [ ] `infer_path()` uses **all filename variants** from JSON config as candidates
- [ ] App build function uses `_infer()` consistently
- [ ] App `DATASET_REGISTRY` entry has correct `is_3d` flag
- [ ] Both `__init__.py` files updated with imports
- [ ] `SUPPORTED_OUTPUTS` frozenset set on the class listing exactly the output keys the dataset provides; base class will emit `UserWarning` automatically for any unsupported flag
- [ ] `dtype=torch.bfloat16` param added to `__init__` and threaded to the default transform builder
- [ ] JSON config file saved and up-to-date in `configs/datasets/`
- [ ] Integrity check registered in **both** `notebooks/datasets/dataset_integrity_check.ipynb` (human) and `scripts/check_datasets.py` (agent/CI), and passes — verify with `uv run python scripts/check_datasets.py --datasets <registry_key>` (exit 0)
- [ ] **When `image_path` is a directory** (DICOM series), override `verify_images`
      in **both** `BaseHarmonizer` and `BaseRadiologicalDataset` subclasses — the
      base uses `os.path.isfile` in both places and will drop every row otherwise.
- [ ] **For MRI**, instantiate `RadiologyTransform3D(..., hu_window=None)` so the
      HU-windowing step is skipped; the percentile-based intensity step in
      `_base_load_3d` handles normalisation cross-scanner.
- [ ] **For ordinal/multi-class targets** (e.g. severity grades), flatten to a
      binary one-hot column per `(target, class)` — plays natively with
      `output_cls` and avoids base-class customisation.
- [ ] **For RLE mask datasets** (`masks.format = "rle"`), add a `ValueError` guard in
      `__init__` when `output_mask=True` and `mask_output_dir is None` (and no preset
      harmonizer/df provided) — prevents a cryptic `OSError: File name too long` from
      deep inside MONAI. See Dataset Key Rule 8 for the exact pattern.
- [ ] **For bboxes on 3-D volumes**, encode each as a 6-tuple
      `[d_min, d_max, y_min, y_max, x_min, x_max]` (post-`OrientationD(IPL)`
      + `TransposeD` axis order), normalized by the original volume
      shape. Volumetric bboxes (e.g. RadChestCT lung region) use the
      full `[0,1]` extent on each axis as appropriate. Per-slice
      2-D-on-3-D annotations (e.g. lumbar/cervical) use
      `eps_z = 0.5/n_instances` and must anchor slice ranks on
      ascending `ImagePositionPatient[2]`, not `InstanceNumber`. For
      dense per-anatomy (vertebra/level) annotations, register the
      display name in `_BBOX_FOCUS_FIRST_DATASETS` in `app.py` to
      enable bbox-centered slicing on all three panels and the "Next
      bbox →" cycle button. See Harmonizer Key Rule 8.
- [ ] **`base_image_dir` points at the deepest stable image directory** — no
      hardcoded subdirectory prefix in `image_path` that could be folded into
      `base_image_dir` instead. For Train/Test splits, each variant takes its own
      split-specific directory (e.g. `train_images/`, `test_images/`). See the
      "`base_image_dir` Convention" section. Exception: datasets with multiple
      sibling image dirs at the same level (ChestX-ray14, RSNA Bone Age val).
- [ ] **`base_image_dir` confirmed with the user** (workflow step 4a) before
      generating code — the user has signed off on the deepest-stable-directory
      choice and the relative `image_path` shape, and `base_image_dir + image_path`
      has been verified to resolve to a real file/directory for one sample row.
- [ ] **Wiki page created** at `docs/wiki/datasets/<registry_key>.md` with
      the standard sections (Overview, Download, Expected layout, Label columns,
      Constructor arguments, Dataset / Harmonizer / saved-CSV samples,
      Harmonizer notes, Outputs). Train/Test split datasets show samples for
      each split using the split-specific classes.
- [ ] **Wiki inventory rows added** in `docs/wiki/datasets.md` (top table)
      and `docs/wiki/index.md`; for split datasets also a row in the
      train/test split table inside `docs/wiki/datasets.md`.

## Workflow

1. **Create JSON config** -- Write `configs/datasets/<registry_key>.json` from the template.
   Pre-fill everything you can from the user's message.
2. **Show & ask** -- Present the JSON to the user. Highlight required fields that are `null`.
3. **Infer missing fields** -- Read CSVs, list directories, check DICOM headers. Update the
   JSON with inferred values.
4. **Ask the user** -- Only for fields you cannot infer. Update the JSON.
4a. **Confirm `base_image_dir` (relative-path check)** -- Before generating
    code, state explicitly:
    - what `base_image_dir` is going to be (e.g. `train_images/`, the
      `files/` root, …),
    - what `image_path` will look like as a relative string (e.g.
      `<study_id>/<series_id>` or `<id>.png`),
    - and whether the join `base_image_dir + image_path` resolves to a
      real file/directory on disk for one sample row.

    Apply the **`base_image_dir` Convention** (see section above): the base
    must be the **deepest stable directory** whose immediate contents are
    the entries named in `image_path`. If you find yourself baking a
    hardcoded subdirectory prefix into `image_path`, deepen `base_image_dir`
    instead. Get explicit user sign-off if there's any ambiguity (e.g.
    multiple plausible roots, or a structural exception like ChestX-ray14
    with sibling `images_001/…/images_012/`).
5. **Read existing code** -- Read at least one similar harmonizer + dataset pair for reference.
   For 2D datasets, reference CheXpert or MIMIC-CXR. For 3D, reference CT-RATE or RadChestCT.
6. **Generate harmonizer** -- Create `radharmony/harmonizer/<name>/` with `<name>.py` (implementation) and `__init__.py` (re-export) using the JSON config. Use `..base` for relative imports inside `<name>.py`.
7. **Generate dataset** -- Create `radharmony/dataset/<name>/` with `<name>.py` (implementation) and `__init__.py` (re-export) using the JSON config. Use `..base` and `..transforms` for relative imports inside `<name>.py`.
8. **Update exports** -- Edit both `__init__.py` files.
9. **Update app.py** -- Add import, build function, and registry entry.
10. **Run checklist** -- Verify all items above.
11. **Integration test** -- The integrity harness (`IntegrityReport`, `check_dataset`,
    `print_report`, `subsample_dataset`) lives in `radharmony/integrity.py` and is driven
    two ways that share that one module: the **notebook** (for a human — it also renders the
    images) and **`scripts/check_datasets.py`** (for an agent / CI — deterministic JSON plus
    a non-zero exit on failure). Register the new dataset in **both** surfaces:

    - **Notebook** `notebooks/datasets/dataset_integrity_check.ipynb`:
      - Add the class to the imports cell.
      - Add the path constant(s) to the paths cell.
      - Add a `build(<Name>Dataset, base_image_dir=..., csv_path=..., output_cls=True, ...)`
        cell mirroring the existing dataset cells (enable `output_report` / `output_mask` /
        `output_bbox` as supported). `build()` sub-samples, runs `check_dataset`, prints the
        report, and previews a few samples with images.
    - **Script** `scripts/check_datasets.py`:
      - Add the matching path constant(s) near the top.
      - Add a `DatasetSpec("<registry_key>", <Name>Dataset, dict(base_image_dir=...,
        csv_path=..., output_cls=True, ...))` to `get_specs()`. For 3-D volumes pass
        `workers=2` to bound memory, matching the other 3-D specs.

    - **Self-verify (agent):** run
      ```bash
      uv run python scripts/check_datasets.py --datasets <registry_key> --max-samples 30
      ```
      Read the report and confirm: status `OK`, image dtype/shape as expected, `img range`
      within `[-1, 1]`, and — when `output_cls` — sane per-label positive counts. If the
      status is `ERROR` (or the stats look wrong), read the traceback, fix the generated
      harmonizer / dataset code, and re-run until it passes (exit 0). Add `--json -` for a
      machine-parseable report.
12. **Update the wiki** -- Required, not optional. The wiki is the
    user-facing reference for dataset usage; a new dataset without a wiki
    page is invisible to most users.
    - Create `docs/wiki/datasets/<registry_key>.md` based on a similar
      existing page (e.g. `chexpert.md` for 2D classification,
      `rsna_pe_detection.md` for 3D, `rsna_2024_lumbar_spine.md` for MRI).
      Required sections: Overview, Download (link to the primary source —
      see "Where to download from" above), Expected layout, Label columns,
      Constructor arguments table, Dataset constructor sample, Harmonizer
      sample, Load-from-saved-CSV sample, Harmonizer notes, Outputs.
    - For Train/Test split datasets, show one code sample per split using
      the split-specific class (e.g. `FooTrainDataset`, `FooTestDataset`)
      — do **not** show only the base class.
    - Make the **`base_image_dir`** row in the constructor-args table
      concrete: name the deepest stable directory and give an example
      path; reproduce the convention agreed in step 4a verbatim.
    - Add an inventory row to `docs/wiki/datasets.md` and
      `docs/wiki/index.md`.
    - For Train/Test split datasets, also add a row to the train/test
      split table in `docs/wiki/datasets.md` showing the split-specific
      `base_image_dir` for each variant.
    - Optionally also update the `README.md` inventory table.
13. **Retrospective** -- Summarize what you learned from this implementation:
    - What worked well, what was tricky, what required unexpected fixes.
    - Any patterns, edge cases, or CSV quirks that future datasets might share.
    - If any of these lessons are general enough to improve this skill (e.g. a missing
      template pattern, a new checklist item, a better default), update this SKILL.md
      and `docs/add_dataset_skill.md` accordingly. Keep the skill up-to-date as a
      living document.
