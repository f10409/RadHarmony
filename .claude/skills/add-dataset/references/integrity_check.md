# The integrity check

The integrity check loads a few samples of a dataset class and checks the image, `cls`, `mask`, `bbox` and
`report` outputs (`radharmony/integrity.py`, D-39). It has a spec in `scripts/check_datasets.py` and a cell in
the notebook `notebooks/datasets/dataset_integrity_check.ipynb`.

## Add it (SKILL.md step 11)

1. Add a `DatasetSpec` for each new registry key to `get_specs()` in `scripts/check_datasets.py`, and
   import the class. The edit needs no data. The form, with the Montgomery spec as the example:

   ```python
   S("montgomery_cxr", MontgomeryCXRDataset, dict(
       **P("montgomery_cxr"),
       mask_output_dir=os.path.join(MASK_OUT, "montgomery"),
       output_cls=True, output_mask=True, output_report=True)),
   ```

   `S` is `DatasetSpec`. `P(key)` gives the paths of the key from the data paths file. A key with no entry
   gets a hint in place of the paths, so a default run skips it.
   - Name the check by the registry key, or `<key>_<variant>` for a second check of the same class (D-21).
   - Turn on every output that the class supports. A class that writes its masks in `harmonize()` also
     gets `mask_output_dir=os.path.join(MASK_OUT, "<name>")`. `MASK_OUT` is a cache folder that
     `scripts/check_datasets.py` defines.
   - A 3-D dataset adds `workers=2`.
   - Data with the status `restricted` adds `default=False`. Data with the status `not_on_nas` also gives the
     arguments as a function, because `P()` raises for it: `lambda: dict(**P("<key>"), output_cls=True)`.
   - A dataset class of the internal overlay of the dev repo (`radharmony_internal/`) goes in
     `scripts/check_datasets_internal.py`. A public dataset never needs it.
2. In `notebooks/datasets/dataset_integrity_check.ipynb`, add the class to the imports cell and add one
   `build(...)` cell. Do not put path text in the cell (D-21). The other cells use path constants (a known
   gap of D-21). So the new cell is the first to read the paths file. Add
   `sys.path.insert(0, "../../scripts")` and `from dataset_paths import paths` to it, and pass
   `**paths("<key>")`. Use the NotebookEdit tool (the Edit tool refuses `.ipynb` files). It changes only the
   cells that you name. Do not load and dump the whole notebook with `json`, because that rewrites hundreds of
   lines. Check with `git diff --stat`: the notebook diff is about 15 lines.

## Run it (SKILL.md step 12.2)

`.venv/bin/python scripts/check_datasets.py --datasets <key> --max-samples 30` must exit with 0. In a git
worktree, start it with `PYTHONPATH=$PWD`. `--json -` prints the report as JSON. With no entry for the key,
it stops with the hint. With a folder for this run only, run the same check from Python:

```python
from radharmony.dataset import MontgomeryCXRDataset  # your new class
from radharmony.integrity import check_dataset, print_report, subsample_dataset

ds = MontgomeryCXRDataset(base_image_dir="/data/MontgomerySet/CXR_png/", output_cls=True, cache_dir=None)
subsample_dataset(ds, 30)  # a fast pass on 30 rows. It changes ds in place
report = check_dataset(ds, num_workers=0)
print_report(report)  # report.status is "OK", "WARN" or "ERROR"
```

Read the report: the status is `OK`, the image shape and dtype are right, and the image range is in [-1, 1]
(D-14). With `output_cls`, the positive count of each label is possible. With `WARN` or `ERROR`, find the
cause before step 13.
