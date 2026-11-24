# The harmonizer

The harmonizer reads the source files and gives the harmonized table. Read D-01, D-03, D-04, D-05 and D-30
in `docs/wiki/design.md` first.

## Files

- `radharmony/harmonizer/<name>/<name>.py`: the class.
- `radharmony/harmonizer/<name>/__init__.py`: the import and `__all__` (ruff F401 fails on an import with no
  `__all__`). Example for `my_dataset`:

  ```python
  from .my_dataset import MyDatasetHarmonizer

  __all__ = ["MyDatasetHarmonizer"]
  ```

Every dataset has its own folder, also with one class. Then a later variant (a split class, a release, a
PNG reader) needs no move. Inside `<name>.py`, import the base files with `..`, for example
`from ..base import BaseHarmonizer` and `from ..base_vqa import BaseVQAHarmonizer`.

## Template

A 2-D dataset with a main table (one row for each image) and a label table (one row for each study). The
template runs on fake files. Change the names and columns to the source.

```python
"""Harmonizer for My Dataset (2-D chest X-rays with a separate label table)."""

from ..base import BaseHarmonizer


class MyDatasetHarmonizer(BaseHarmonizer):
    """Harmonize My Dataset into the harmonized table.

    One row is one image (D-03). ``image_path`` is relative to ``base_image_dir`` (D-04).
    Example: ``p001/s01/view1.png`` for ``/data/my_dataset/images/p001/s01/view1.png``.

    Args:
        csv_path: The main table, ``metadata.csv``. One row for each image.
        label_csv_path: The label table, ``labels.csv``. One row for each study.
        base_image_dir: The ``images/`` folder.
    """

    LABEL_COLS = ["Cardiomegaly", "Pleural Effusion"]  # the source names. harmonize() sorts them (D-05)
    LABEL_JOIN_COLS = ["patient", "study"]  # the join columns of labels.csv
    VIEW_POSITION_SOURCE_COL = "view"  # the base step gives PA, AP or LATERAL
    AGE_SOURCE_COL = "age"
    AGE_SOURCE_UNIT = "years"  # "years", "months", "days" or "dicom_as" (D-06)
    SEX_SOURCE_COL = "sex"  # the base step gives M, F or NaN (D-26)

    def __init__(self, csv_path: str, label_csv_path: str = None, base_image_dir: str = None):
        super().__init__(csv_path=csv_path, label_csv_path=label_csv_path)
        self.base_image_dir = base_image_dir  # the base snapshot saves it for load_from_saved (D-08)

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["patient"].astype(str)

    def _build_study_id(self) -> None:
        # A study number is unique only inside a patient, so join the two.
        self.df["study_id"] = self.df["patient"].astype(str) + "_" + self.df["study"].astype(str)

    def _build_image_path(self) -> None:
        # The source path starts with "images/", and base_image_dir is that folder (D-04).
        self.df["image_path"] = self.df["path"].str.removeprefix("images/")
```

## Rules

1. **`LABEL_COLS`** holds the label columns with the names of the source, for example `"Pleural Effusion"`.
   `_build_labels()` makes each name snake_case with `col.lower().replace(" ", "_")`. It does not change
   `/` or other marks, so `"Nodule/Mass"` gives `nodule/mass`. `harmonize()` sorts `LABEL_COLS` by these
   snake_case names (D-05). An override of `_build_labels()` must use the same names, or
   `_select_harmonized_columns()` drops the column and gives no warning.
2. **Label codes** (D-05). Keep 1, 0, -1 (uncertain) and NaN (not mentioned or unknown). For CheXpert-style
   labels, call `chexpert_codes` from `radharmony/harmonizer/_labels.py`, as `CheXpertHarmonizer` does. A
   row with no label at all gets NaN in every label, not 0. A regression target goes in `REG_COLS`, not in
   `LABEL_COLS`.
3. **The order of the steps.** `harmonize()` runs `_read_source()`, then `_build_patient_id()` and
   `_build_study_id()`, then the `study_ids` and `patient_ids` filter, then the other `_build_*` steps. So a
   column that an ID needs (for example a "same patient as" note in a report file) must be in the table of
   `_read_source()`. Do not read it in a later step.
4. **`_build_image_path()`** gives a path relative to `base_image_dir` (D-04). The dataset class adds
   `base_image_dir`. See "`base_image_dir`" in `data_traps.md`.
5. **`super().__init__()`** takes the source table arguments of `BaseHarmonizer`: `csv_path`,
   `label_csv_path`, `view_position_csv_path`, `mask_csv_path`, `bbox_csv_path`, `report_csv_path` and
   `report_base_dir`. Pass the ones that the dataset has. A table in a second file needs its `*_JOIN_COLS`
   and its `*_SOURCE_COL`. The join is an inner join, and it raises `ValueError` when a key has more than
   one line (D-03).
6. **Save and load** (D-08). The base `_harmonizer_init_snapshot()` saves the base arguments,
   `base_image_dir` and `dicom_base_dir`. Override it for any other argument of `__init__`, as
   `CheXpertHarmonizer` does for `demographics_path`. Then `save()` and `load_from_saved()` give the same
   table.
7. **No copy of `harmonize()`** (D-30). Override `_read_source()` and the `_build_*` steps. Override
   `harmonize()` only to take an option or to add a step after `super().harmonize()`. See
   `RadChestCTHarmonizer._read_source()` for a source with its label columns in another shape.
8. **Clean a second table first.** Override `_preprocess_label_df()`, `_preprocess_bbox_df()` and the other
   `_preprocess_*` hooks to clean a table before the join. Override `_decode_mask()` for RLE masks.
9. **Age and sex** (D-06, D-26). Set `AGE_SOURCE_COL` with `AGE_SOURCE_UNIT`, and `SEX_SOURCE_COL`. The
   base steps give `age_years` and `sex` (`M`, `F` or NaN). Do not put `sex` in `EXTRA_OUTPUT_COLS`. When
   the values are only in the DICOM headers, use `add_patient_tags()` from
   `radharmony/harmonizer/_dicom_tags.py`, and say so in the docstring and on the wiki page.
10. **View** (D-01). For a 2-D image, set `VIEW_POSITION_SOURCE_COL`. The base step gives the shared names
   `PA`, `AP` and `LATERAL` (D-01). A 3-D class with a plane or series name sets
   `VIEW_POSITION_NAMES = False`.
11. **Other columns.** Put a covariate that is not a target (for example a device maker) in
    `EXTRA_OUTPUT_COLS`. Set `SERIES_ID_SOURCE_COL` for a dataset with DICOM series.
12. **Boxes, masks and a source with no table.** The patterns are in `data_traps.md`. For boxes, read
    "Boxes" and "Boxes on 3-D volumes". For masks, read "More than one mask for each image" and "Two mask
    files for each image". For a source with no CSV, read "A source with no table".
13. **Docstrings.** Give a real example path in the class docstring, for example
    `/data/my_dataset/images/p001/s01/view1.png`. Write a short comment for a step that is not clear,
    for example why a prefix is removed.
