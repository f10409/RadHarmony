# Record the dataset stats

Write the `expected_stats` block of `configs/datasets/<name>.json` right after the config JSON, before any
code. These numbers are the test of the harmonizer: `tests/test_dataset_stats.py` compares each one with
`harmonize()` in the full run. The wiki page states the same numbers with claim notes (step 13).

## The format

The full format is in the docstring of `tests/test_dataset_stats.py`. Read it there. A summary:

- `_comment`: the sources (paper, website, file names, date), the label policy, and the reason for each
  number that differs from the paper.
- `patients_in_two_splits`: the number of patients in more than one official split. It is usually 0. When
  the authors' splits share patients, keep the splits as published and give the number (D-34).
- One block for each official split, keyed by its registry key: `images`, `studies`, `patients`,
  `positives`, `uncertain`, `rows_with_bbox`, `boxes`, `rows_with_mask`, `rows_with_report`, `sex` and
  `age_years`. Leave out a stat that the table cannot have.
- Write the numbers of the harmonized table that the user agreed on: after the label policy and the dropped
  rows of the config. Write the paper number in `_comment` when it differs.

What the test counts (`_stat` in `tests/test_dataset_stats.py`):

| Stat | What the test counts in the harmonized table |
|---|---|
| `images` | unique `image_path` values. The row count must be the same (D-03), unless the block gives `rows` |
| `studies`, `patients` | unique `study_id` and `patient_id` values |
| `positives` | the cells equal to 1 in each label column. List every label |
| `uncertain` | the NaN cells in each label column. List only the labels with 1 or more (the code of `_stat`, not the docstring). A `-1` cell is not counted here |
| `rows_with_bbox`, `rows_with_mask`, `rows_with_report` | the rows with a value in `bbox`, `mask_path` or `report` of `harmonize()` with no `mask_output_dir`. An empty list or `""` is no value. When `mask_path` is filled only in the mask step, leave `rows_with_mask` out, and say why in `_comment` (as `montgomery_cxr.json` does) |
| `boxes` | the boxes in all `bbox` cells |
| `sex` | the counts of `M` and `F` (NaN is not counted) |
| `age_years` | `[min, max]` of `age_years` |

Other keys: `harmonizer_args` maps a harmonizer argument to a `data_paths.json` argument with another name.
`known_gaps` marks a stat that the table cannot give yet, with the design entry and the finding ID.

## The sources

1. The paper, the website or the data card: the image, study and patient counts, the split sizes, and the
   label counts.
2. The source files, read with pandas or plain Python. Do not import `radharmony`. A number from the code
   under test makes the test pass on the bug (D-23).

Example: the CheXpert valid split. Get the path from the data paths file with
`paths("chexpert_valid")["csv_path"]`, or use the folder that the user gives (see "Real data" in SKILL.md).

```python
import pandas as pd

src = pd.read_csv("/data/CheXpert-v1.0/valid.csv")  # the main table, read with pandas only
patient = src["Path"].str.split("/").str[2]  # patient00001 in CheXpert-v1.0/valid/patient00001/...
print("images", src["Path"].nunique(), "patients", patient.nunique())
print("positives", {c: int((src[c] == 1).sum()) for c in ["Cardiomegaly", "Edema"]})
print("sex", src["Sex"].value_counts().to_dict(), "age", [src["Age"].min(), src["Age"].max()])
# images 234 patients 200
# positives {'Cardiomegaly': 68, 'Edema': 45}
# sex {'Male': 128, 'Female': 106} age [18, 90]
```

Apply the policy of the config in the same pandas code. Examples: drop the rows that the config drops, and
map the sex values to `M` and `F` (D-26).

## Differences

Each difference between the paper and the source files gets a written reason in `_comment`. Examples: the
release has fewer images than the paper. A row of the table has no image file. One patient has 2 IDs.
Ask the user about a difference that has no clear reason.

## Check the block

Check the shape of the block before the code exists, in any clone. When a data paths file exists, the key
needs its entry first (step 3):

`.venv/bin/python -m pytest tests/test_dataset_stats.py -k "<key> and entry_is_valid" -v`

After step 7, the full test compares the numbers (step 12).
