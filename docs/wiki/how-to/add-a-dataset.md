# Add a dataset

A new dataset needs a harmonizer, a dataset class, an entry in the app, its paths, an integrity check and
a wiki page. The
[add-dataset skill](https://github.com/f10409/RadHarmony/blob/main/.claude/skills/add-dataset/SKILL.md)
does all the steps with you in Claude Code. Ask it to "add a dataset", and give it the paths of the
source files. This page lists the files to make, and shows how to add the dataset to the app.

Before you start, have the source files of the new dataset on your disk, and the list of its labels.

## 1. Make the files

Do not change base code, for example `BaseHarmonizer` or `BaseRadiologicalDataset`. Put the new logic in
the files of the new dataset. A base code change needs approval first (D-20).

In this list, `<name>` is the folder name and `<key>` is the registry key. For Montgomery County CXR, both are
`montgomery_cxr`.

| File | What it holds |
|---|---|
| `configs/datasets/<name>.json` | The facts of the dataset: the source files, the ID columns, the labels, the outputs, and `expected_stats` |
| `radharmony/harmonizer/<name>/<name>.py` and `__init__.py` | The harmonizer: it reads the source files and gives the harmonized table |
| `radharmony/dataset/<name>/<name>.py` and `__init__.py` | The dataset class, with `@register_dataset("<key>")`, `LABEL_COLS`, `SUPPORTED_OUTPUTS` and `_HARMONIZER_CLS` |
| `radharmony/harmonizer/__init__.py` and `radharmony/dataset/__init__.py` | An import line and an `__all__` entry each |
| `app.py` | An import, a build function and a `DATASET_REGISTRY` entry (see step 2) |
| `scripts/check_datasets.py` and `notebooks/datasets/dataset_integrity_check.ipynb` | The integrity check: a `DatasetSpec` in `get_specs()` of the script, and a cell in the notebook |
| `docs/wiki/reference/datasets/<group>/<name>.md` | The dataset page, a row in the [dataset catalog](../reference/datasets/index.md), and a line in the `mkdocs.yml` nav |
| `docs/wiki/how-to/get-the-data.md` | The download steps of the dataset |
| `tests/test_<name>.py` and `tests/fixtures/fake_datasets.py` | The tests, and a fake-data builder so that CI runs the dataset with no real data |
| The data paths file | An entry for `<key>`, so that the checks on real data find the folders ([Install: data paths](../get-started/install.md#data-paths-for-the-checks)) |

`base_image_dir` is the deepest folder that all image files share (D-04). For a dataset with
official splits, each split is its own class with its own `base_image_dir` (D-09).

## 2. Show the dataset in the app

The app needs a build function and a `DatasetConfig` entry in `DATASET_REGISTRY`, both in `app.py`. The
dropdown, the textboxes and the load step use them, so the rest of the app needs no change.

The build function takes the values of the textboxes and the output checkboxes. It returns
`(dataset, None)`, or `(None, "<message>")` to show an error in the app.

<!-- block: run=skip reason="MyDataset is your own dataset class" -->
```python
def _build_my_dataset(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    csv = _infer(base_dir, "metadata.csv", user_path=csv_path)  # finds the file next to the images
    if not csv:
        return None, "metadata.csv is required."
    return (
        MyDataset(
            base_image_dir=base_dir,
            csv_path=csv,
            label_csv_path=extra_field or None,  # the first extra textbox
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=flags.get("output_mask", False),
            output_report=flags.get("output_report", False),
            output_bbox=flags.get("output_bbox", False),
        ),
        None,
    )
```

| Argument | Textbox or checkbox in the app |
|---|---|
| `base_dir` | "Base image directory". The app checks that the folder exists before it calls the function |
| `csv_path` | "CSV path" |
| `extra_field` | The first extra textbox. The app shows it when `extra_field_label` or `extra_placeholder` is not empty |
| `extra_field2` | The second extra textbox. The app shows it when `extra_field2_label` or `extra_placeholder2` is not empty |
| `cache_dir` | "Cache dir" |
| `**flags` | The output checkboxes: `output_cls`, `output_mask`, `output_report`, `output_bbox` and `output_reg` |

`_infer` is `radharmony.utils.infer.infer_path`, with another name in `app.py`. It looks for the file only in
`base_dir` and its parent folder (D-35). An empty textbox gives an empty string, not `None`, except an empty
`cache_dir`, which gives `None`. A required field that depends on a checkbox goes in the build function too,
for example `if flags.get("output_mask") and not extra_field: return None, "..."`.

Then add the entry. The app groups the dropdown by modality and then by dataset family. A family header
is a row that starts a family, made with `_header("<modality>")`. Put each entry after a family header
of the same modality. An entry with no header before it raises
`ValueError` when the app starts.

<!-- block: run=skip reason="lines to add inside DATASET_REGISTRY in app.py" -->
```python
    "── My Dataset ──": _header("CXR"),
    "My Dataset": DatasetConfig(
        is_3d=False,
        extra_field_label="Label CSV (optional)",
        build=_build_my_dataset,
        modality="CXR",
        base_dir_placeholder="e.g. /data/my_dataset/images/",
        csv_placeholder="auto: metadata.csv",
        extra_placeholder="auto: labels.csv",
    ),
```

An official split gets its own entry, for example "CheXpert (Train)" and "CheXpert (Valid)" (D-09).

The fields of `DatasetConfig`:

| Field | Type | Default | Description |
|---|---|---|---|
| `is_3d` | `bool` | required | `True` uses the 3-D transform and shows the HU window controls |
| `extra_field_label` | `str` | required | The label of the first extra textbox. `""` with no `extra_placeholder` hides it |
| `build` | callable | required | The build function |
| `modality` | `str` | required | One of `SUPPORTED_MODALITIES`: `"CXR"`, `"Radiograph"`, `"CT"`, `"CT (2D)"`, `"MRI"`, `"VQA"` or `"Other"`. Another value raises `ValueError`. It sets only the group in the dropdown, not the transform |
| `extra_field2_label` | `str` | `""` | The label of the second extra textbox. `""` with no `extra_placeholder2` hides it |
| `csv_label` | `str` | `"CSV path (optional)"` | The label of the CSV textbox |
| `base_dir_placeholder` | `str` | `""` | The gray hint text in the base folder textbox |
| `csv_placeholder` | `str` | `""` | The gray hint text in the CSV textbox |
| `extra_placeholder` | `str` | `""` | The gray hint text in the first extra textbox |
| `extra_placeholder2` | `str` | `""` | The gray hint text in the second extra textbox |
| `extra_dropdown_label` | `str` | `""` | The label of an extra dropdown for the dataset. `""` hides it |
| `extra_dropdown_choices` | `tuple` | `()` | The choices of the extra dropdown. `"(all)"` or an empty choice gives `None` |
| `extra_dropdown_kwarg` | `str` | `""` | The name of the build function argument that gets the dropdown value |
| `is_group_header` | `bool` | `False` | `True` for a family header. `_header()` sets it |
| `bbox_default` | `bool` | `False` | The start value of the BBox checkbox for this entry (D-11) |

To add datasets to the app with no edit of `app.py`, write a module with a function
`register_app_datasets(registry, DatasetConfig, _header, app_dtype)` that adds entries to `registry`.
Then start the app with `RADHARMONY_APP_PLUGINS=<module> python app.py`. A module that raises an error, or
adds an entry with no family header, is not loaded. The app prints a message and starts without it.

## Check

The new registry key is in the registry. This block uses `montgomery_cxr`. Use your own key.

<!-- block: run=ci -->
```python
from radharmony.registry import list_datasets, resolve_dataset

print("montgomery_cxr" in list_datasets())  # True
print(resolve_dataset("montgomery_cxr").__name__)  # MontgomeryCXRDataset
```

The integrity check of the dataset passes (exit code 0). Name its `DatasetSpec` by the registry key, or
`<registry key>_<variant>` for a second check of the same class. `--list` prints all names.

<!-- block: run=skip reason="needs the data of the new dataset" -->
```bash
python scripts/check_datasets.py --datasets <key> --max-samples 30
```

In the app (`python app.py`), the dataset is in the dropdown, and a load shows the first sample.

## See also

- [Use the app](use-the-app.md): how to load and view a dataset in the app.
- [Montgomery](../reference/datasets/cxr-boxes-masks/montgomery_cxr.md): a small dataset page to copy.
- [Design decisions](../design.md): the rules of the harmonized table (IDs, labels, boxes, splits).
