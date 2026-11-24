# The config JSON

The config JSON `configs/datasets/<name>.json` holds the facts of a dataset: the source files, the ID
columns, the labels, the outputs, and `expected_stats`. Write it before any code. The config reviewer (step
5) and the stats test (`tests/test_dataset_stats.py`) read it.

## Rules

- `tests/test_configs.py` needs 3 keys: `dataset_name`, `registry_key` and `modality` (`"2D"` or `"3D"`).
- `registry_key` must be a registry key. For a dataset with official splits, use the key of one split, for
  example `vindr_cxr_train`. `expected_stats` has a block for each split key. The test of the key fails
  until the dataset class exists (step 7). That is expected.
- The file holds no real path. `tests/test_configs.py` fails on a lab data path. Write a path as a file
  name, or as a path relative to `base_image_dir`, or with `/path/to/...`. The user passes the real paths
  as arguments, and they go in the data paths file (step 3).
- An ID from a file name uses the ID part of the name only. Remove a part that holds a label or a view.
  Example: Montgomery `MCUCXR_0113_1.png` ends in `_1` for TB, so its patient is `MCUCXR_0113`, not
  `MCUCXR_0113_1`. A label in an ID leaks the label. Write the rule and one example in `derivation`.
- Fill a field from the source files first. Read the table headers and a few rows, the folder tree, the
  files next to the images, and the DICOM headers (`pydicom.dcmread(path, stop_before_pixels=True)`). Ask the user only for a value that you cannot find. Leave a value that nobody knows as `null`.

## Template

```json
{
  "_comment": "What the dataset is, where it comes from, and the version of the release.",

  "dataset_name": "<name for people, for example ChestX-ray14>",
  "registry_key": "<registry key, for example chestxray14_train>",
  "modality": "<2D or 3D>",
  "image_format": "<PNG, JPEG, DICOM, NIfTI or NPZ>",

  "base_image_dir": "<the deepest folder that holds every image, for example train_images/ (D-04)>",

  "primary_csv": {
    "_comment": "The main table. It lists every image.",
    "path": "<file name or /path/to/... placeholder>",
    "filename_variants": ["<every known file name, for example metadata.csv and metadata.csv.gz>"]
  },

  "patient_id": {"column": "<column, or null>", "derivation": "<how to make it when no column exists>"},
  "study_id": {"column": "<column, or null>", "derivation": "<how to make it when no column exists>"},

  "image_path": {
    "_comment": "The image file, relative to base_image_dir (D-04).",
    "column": "<column, or null>",
    "derivation": "<how to make it when no column exists>",
    "prefix_to_strip": "<a fixed prefix of the column to remove, or null>"
  },

  "labels": {
    "columns": ["<the label columns, with the names of the source>"],
    "separate_csv": {
      "path": "<label table, or null when the labels are in the main table>",
      "filename_variants": [],
      "join_columns": ["<the columns to join on>"]
    },
    "uncertain_handling": {
      "nan_action": "<'keep' or 'to_zero': a blank (not mentioned) label (D-05)>",
      "negative_one_action": "<'keep' or 'drop': -1 (uncertain) stays -1 in cls, never NaN (D-05)>",
      "drop_uncertain_rows": "<true or false. CheXpert-style labels use chexpert_codes (D-05)>"
    }
  },

  "view_position": {
    "column": "<column, or null>",
    "source": "<'csv', 'dicom_header' or null>",
    "separate_csv": {"path": null, "filename_variants": [], "join_columns": []}
  },

  "masks": {"supported": false, "column": null, "format": "<'rle', 'file_path' or null>"},
  "bboxes": {"supported": false, "column": null},

  "reports": {
    "supported": false,
    "format": "<'text_files', 'csv_column' or null>",
    "separate_csv": {"path": null, "filename_variants": [], "join_columns": []},
    "base_dir": "<folder of the report files, or null>",
    "path_column": "<column with the report file paths, or null>"
  },

  "extra_init_params": {"<argument>": "<what it is, for example dicom_base_dir: the DICOM tree for the headers>"},

  "hu_window": "<[min, max] for CT, or null>",
  "custom_preprocessing": "<a special step, or null>",

  "expected_stats": {"_comment": "step 4: see references/stats.md. Keep it an object, not a string"}
}
```

The config has no field yet for the official splits, the age or the sex. Write them in `_comment`, and in
the docstring of the harmonizer.

## Where to download from

- Use the primary source: the group that made the dataset (for example RSNA, NIH, Stanford AIMI or
  PhysioNet), or the data statement of the paper.
- Do not use a Kaggle copy by default. A third-party upload can drop or repeat rows, or change the image
  files or the metadata. Even a Kaggle competition can hold a subset of the release.
- Use Kaggle only when Kaggle is the primary source, for example a competition with no other release. Say
  so in the config `_comment` and on the wiki page.
- Do not put a download call (for example `kagglehub`) in the harmonizer or the dataset class. The user
  downloads the data first. The class takes the folder.
- The download steps for a person go in `docs/wiki/how-to/get-the-data.md` (step 13).

## A new Python package

If the harmonizer needs a package that is not in `pyproject.toml`, ask the user first. Then add the package
to `pyproject.toml`. Then install it with `uv pip install -e ".[all]"` (or a smaller extra). Do not run a plain
`uv sync`. It syncs to the default group of the lock file and removes the extras that `uv pip install`
added, for example `gradio`, `transformers` and `torchvision`. `uv pip install -e ".[all]"` puts them back.
