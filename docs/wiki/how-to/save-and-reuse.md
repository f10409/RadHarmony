# Save and reuse a harmonized table

<!-- blocks: run=local data=montgomery_cxr -->

Harmonizing a big dataset can take minutes. Save the harmonized table once, and give the file to the
dataset class in the next runs. You can also filter the table first. The examples use
[Montgomery County CXR](../reference/datasets/cxr-boxes-masks/montgomery_cxr.md).

Before you start, [get the Montgomery data](get-the-data.md#montgomery-county-cxr).

## 1. Harmonize and save

`save()` writes one pickle file. It holds the harmonized table, the label columns and the arguments of
the harmonizer.

```python
from radharmony.harmonizer import MontgomeryCXRHarmonizer

h = MontgomeryCXRHarmonizer(base_dir="/data/MontgomerySet/CXR_png/")
df = h.harmonize()
h.save("montgomery_harmonized.pkl")
print(len(df))
# 138
```

`save()` writes `h.harmonized_df`. This is always the table of the last `harmonize()` call (D-24).

## 2. Load the file into a dataset

`harmonizer_path=` loads the file with the harmonizer class of the dataset. The dataset does not read the
source files.

```python
from radharmony.dataset import MontgomeryCXRDataset

ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    harmonizer_path="montgomery_harmonized.pkl",
    output_cls=True,
    cache_dir=None,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
print(ds.LABEL_COLS)
# ['tuberculosis']
```

- The label columns are the labels of the harmonizer that are in the table (D-08). Other columns, for
  example `age` and `age_years`, are not labels.
- A file from another harmonizer class raises `ValueError`. The message names the class of the file.
- A split class can load a file from an old class that mixes the splits. This works only when every row
  of the file is in the split of the class. Otherwise it raises `ValueError` (D-09).

## 3. Load the harmonizer from the file

`load_from_saved()` gives the harmonizer back, with its table. It does not read the source files.

```python
h2 = MontgomeryCXRHarmonizer.load_from_saved("montgomery_harmonized.pkl")
print(h2.harmonized_df.shape)
# (138, 10)
```

If you moved the data, give the new paths as keyword arguments. They replace the saved arguments of the
harmonizer, for example `load_from_saved("montgomery_harmonized.pkl", base_dir="/data/new/CXR_png/")`.

## 4. Filter the table and give it to the dataset

Harmonize the full dataset first. Then filter the table with pandas (D-02). Give the result with
`harmonized_df=`.

```python
table = h2.harmonized_df
tb_only = table[table["tuberculosis"] == 1]

ds_tb = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    harmonized_df=tb_only,
    output_cls=True,
    cache_dir=None,
)
print(len(ds_tb.get_datasets()))
# 58
```

- The table can also come from a CSV file: `h.harmonize().to_csv(...)`, then `pd.read_csv(...)`. The
  transform reads a box column in CSV text ([Transforms: Boxes](../reference/transforms.md#boxes)).
- With `harmonized_df=`, the dataset keeps the `LABEL_COLS` of its class. If that list is empty, it takes
  the labels of its harmonizer class that are in the table (D-08).
- The filter arguments of a dataset class also filter a given table, for example `series_filter` of
  RSNA 2024 lumbar (D-02).

On a big dataset, you can harmonize only some studies or patients. `harmonize()` then keeps only these
rows right after it makes the IDs. It still reads the full source file.

```python
few = MontgomeryCXRHarmonizer(base_dir="/data/MontgomerySet/CXR_png/").harmonize(patient_ids=["MCUCXR_0001"])
print(len(few))
# 1
```

- The result is the same as a filter after `harmonize()` (D-02).
- All image harmonizers take `study_ids=` and `patient_ids=`. Of the VQA harmonizers, only
  MIMIC-Ext-CXR-QBA takes them.
- A harmonizer of your own that does not call `super().harmonize()` filters after it makes every row, and
  warns.

## 5. Give a harmonizer or change the table

Give a harmonizer object with `harmonizer=`, after its `harmonize()` or `load_from_saved()`.
`set_harmonized_df()` gives a dataset a new table. The next `get_datasets()` and `get_folds()` calls use
it.

```python
ds2 = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/", harmonizer=h, output_cls=True, cache_dir=None
)
ds2.set_harmonized_df(tb_only)
print(len(ds2.get_datasets()))
# 58
```

If you give more than one table, see the
[order of the table arguments](../reference/api-datasets.md#order-of-the-table-arguments).

## Check

The table from the file is the same as the table of `harmonize()` (D-08):

```python
import pandas as pd

pd.testing.assert_frame_equal(ds.get_harmonized_df(), df)  # no error
```

## See also

- [Load a dataset and pick outputs](load-a-dataset.md)
- [Dataset API](../reference/api-datasets.md): the shared arguments, with `harmonized_df`, `harmonizer`
  and `harmonizer_path`.
- [D-02](../design.md#d-02-harmonize-first-then-filter) and
  [D-08](../design.md#d-08-save-and-load-give-the-same-table) in Design decisions.
