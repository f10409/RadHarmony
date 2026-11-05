# Checklist

Go through each item after step 13. The test or check in brackets proves the item. Fix each item that
fails.

## Config and stats

- [ ] `configs/datasets/<name>.json` has `dataset_name`, `registry_key` and `modality`, and no real data
      path (`tests/test_configs.py`).
- [ ] `expected_stats` has a block for each official split, from the paper and the source files, with the
      reasons in `_comment` (`tests/test_dataset_stats.py`).
- [ ] The user answered each finding of the config reviewer, and the config JSON follows the answers.
- [ ] The data paths file has an entry for each new registry key, with `status` (D-21, L-007). Dev repo only:
      each entry has `verified` (`tests/test_data_paths.py`), and the paths page is up to date
      (`scripts/gen_nas_paths_page.py --check`).

## Harmonizer

- [ ] `LABEL_COLS` holds the source names of every label column.
- [ ] The labels keep 1, 0, -1 and NaN as D-05 says. A row with no label at all gets NaN.
- [ ] `image_path` is relative to `base_image_dir`, with no folder that `base_image_dir` could hold (D-04).
      The user agreed on `base_image_dir`.
- [ ] One row for each image, or `rows` in `expected_stats` for a table with one row for each study (D-03).
- [ ] `_harmonizer_init_snapshot()` covers each argument of `__init__` that the base does not save (D-08,
      `tests/test_save_load.py`).
- [ ] No copy of `harmonize()` (D-30).
- [ ] Age and sex come from `AGE_SOURCE_COL`, `AGE_SOURCE_UNIT` and `SEX_SOURCE_COL` (D-06, D-26).
- [ ] Boxes: row first, in [0, 1], checked on the images, and `clip_boxes` called (D-07, L-015).
- [ ] 3-D boxes: 6 values in the axis order of D-12. A box on one slice uses the file order of the 3-D loader.
- [ ] A DICOM series folder as `image_path`: `verify_images` is overridden in the harmonizer and the dataset
      class.

## Dataset class

- [ ] `LABEL_COLS` holds the snake_case names, sorted with `sorted()`. The `cls` order is
      `sorted(ds.LABEL_COLS)` (`tests/test_label_order.py`, D-05).
- [ ] `_HARMONIZER_CLS` is set. `_get_harmonized_df()` calls `_try_resolve_preset_harmonized()` first.
- [ ] `__init__` takes the 5 output flags and passes them, with `harmonized_df`, `harmonizer` and
      `harmonizer_path`, to `super().__init__()` (`tests/test_output_flags.py`, D-11).
- [ ] `SUPPORTED_OUTPUTS` lists exactly the outputs that the class gives. The default transform keeps only
      these keys.
- [ ] `infer_path()` gets every known file name, and `required=` follows D-35.
- [ ] `dtype=torch.bfloat16` goes to the default transform.
- [ ] Each official split has its own class and registry key (D-09). A split with no labels has no `cls`
      (D-25).
- [ ] RLE masks: a `ValueError` for `output_mask=True` with no `mask_output_dir`.
- [ ] MRI: `hu_window=None`. CT: a `hu_window`.

## Wiring and tests

- [ ] Both `__init__.py` files import the new classes and list them in `__all__`.
- [ ] `app.py` has an entry for each split class, after a family header of its modality (`tests/test_app.py`).
- [ ] `tests/fixtures/fake_datasets.py` has a builder in `FAKE` (`tests/test_harmonizer_contract.py`, L-008).
- [ ] `tests/test_<name>.py` has a test for each trap that the harmonizer handles, with values from the source
      (D-23).
- [ ] `scripts/check_datasets.py` has a `DatasetSpec` named by the registry key (D-21,
      `tests/test_scripts.py`). The integrity check gives `OK` (step 12).
- [ ] The notebook `notebooks/datasets/dataset_integrity_check.ipynb` has a cell, with no data path text in
      it.

## Docs

- [ ] The dataset page follows the template in `wiki_page.md` (`wiki_audit.py`), and every count has a claim note
      (`claims.py`).
- [ ] The catalog has a row for each split class, and the count of registry keys is new.
- [ ] `mkdocs.yml` has a nav line. `get-the-data.md` has a row and a section.
- [ ] The docstrings and comments of the new code pass `scripts/evalkit/ste_lint.py`.
