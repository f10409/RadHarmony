# The wiki page and the catalog

A new dataset changes 4 places in the docs. The model page is
`docs/wiki/reference/datasets/cxr-labels/chexpert.md`. A small page to copy is
`docs/wiki/reference/datasets/cxr-boxes-masks/montgomery_cxr.md`.

## 1. The dataset page

Write `docs/wiki/reference/datasets/<group>/<name>.md`. `<group>` is the folder of the main output:
`cxr-labels`, `cxr-boxes-masks`, `ct-mri-other` or `vqa-text`.

The page follows the dataset page template below. `wiki_audit.check_template` (in `scripts/evalkit/wiki_audit.py`)
checks the At a glance rows and the `##` sections, in this order:

```markdown
# <Dataset name>

<!-- claim: ... one note for each number in the table below -->
| | |
|---|---|
| Registry keys | `<key>`, ... |
| Task and outputs | Classification, <n> labels (`cls`). Boxes (`bbox`) ... |
| Images in the table | train <n> (<n> patients), test <n> (<n> patients) |
| Modality | Chest X-ray, 2-D, PNG |
| License | <license, and the citation rule> |
| Download | [Get the data: <Dataset name>](../../../how-to/get-the-data.md#<anchor>) |

<What the dataset is, who made it, and how the labels were made. The counts have claim notes.>

## Quick use
## How it is harmonized
## Labels
## Splits
## Arguments
### Shared arguments (inherited from BaseRadiologicalDataset)
## Harmonizer
## Known data problems
```

- **Quick use.** The `base_image_dir` folder (the deepest folder, as agreed in step 2), a short folder
  tree, and one code block for each split class. Use `/data/...` paths. Show the split classes, not only a
  parent class.
- **How it is harmonized.** Write one bullet for each topic. The topics are the source files, the IDs
  (with one example path), the labels and their policy for -1 and blank (D-05), and the view. Then the age
  and sex, and the dropped rows.
- **Labels.** The label columns in the order of `cls` (`sorted(ds.LABEL_COLS)`), with a short meaning. A
  VQA page uses "Questions and answers". A caption page uses "Captions". Other sections can come after
  this one.
- **Splits.** A table: class, registry key, images, source file. Say how `get_datasets(n_splits=...)`
  makes a val split (D-10), and how many patients the splits share (D-34).
- **Arguments.** Put `<!-- args: radharmony.dataset.<Name>Dataset -->` right above the table. Columns:
  Argument, Type, Required, Default, Description. Then the `### Shared arguments` table, as on the CheXpert
  page. Each output flag that the class does not support says "Not supported. The class warns and adds no
  key (D-11)".
- **Harmonizer.** A code block that makes the harmonizer and calls `harmonize()`, then the argument table
  with `<!-- args: radharmony.harmonizer.<Name>Harmonizer -->` above it.
- **Known data problems.** Each trap of the source, with numbers and the design entry. Write "None known"
  when there is none.

### Claim notes

Every count on the page has a claim note right before its paragraph or table. The `expect` value must be
in that paragraph, with the same commas.

- A count of the harmonized table comes from `expected_stats` (tier `ci`):

  `<!-- claim: id=<name>.<split>.images tier=ci cmd="python scripts/evalkit/dataset_stat.py <name> <key>.images" expect="1,234" -->`

- A count of a source file (tier `nas`) needs the entry in the data paths file. `claims.py --full` runs it
  only where the data is:

  `<!-- claim: id=<name>.<split>.csv_rows tier=nas cmd="python scripts/evalkit/dataset_stat.py --source <key> csv_path rows" expect="2,345" -->`

- A number from the paper has `source` in place of `cmd`:

  `<!-- claim: id=<name>.paper.patients source="<authors>, <journal> <year>" expect="1,000" -->`

`python scripts/evalkit/dataset_stat.py --help` gives the forms of the second argument.

### Code blocks

The check compiles a block with no note. It does not run it. A block that cannot run in CI gets
`<!-- block: run=skip reason="..." -->`. The format is in the docstring of `scripts/evalkit/blocks.py`.

## 2. The catalog

In `docs/wiki/reference/datasets/index.md`:

- Add a row to the "Inventory" table for each split class: dataset, modality, dim, labels, outputs,
  registry keys and the page link.
- For a dataset with official splits, add a row to the table in "`base_image_dir` convention".
- Change the number of registry keys in the first paragraph and in its claim note
  (`registry.n_dataset_keys`).

## 3. The nav

Add a line for the page in `mkdocs.yml`, under the heading of its group in the "Reference" section (for
example "Datasets - chest X-ray, labels").

## 4. Get the data

In `docs/wiki/how-to/get-the-data.md`:

- Add a row to the "Quick reference" table: dataset, source, access, size.
- Add a `###` section under the section of its download site. Give the download commands and the folder
  for `base_image_dir`. The page note gives every block `run=skip`.
- Link to the primary source (see "Where to download from" in `config.md`).

## Check the page

```bash
.venv/bin/python scripts/evalkit/wiki_audit.py --page reference/datasets/<group>/<name>.md
.venv/bin/python scripts/evalkit/run_blocks.py --level static --page docs/wiki/reference/datasets/<group>/<name>.md
.venv/bin/python scripts/evalkit/claims.py
.venv/bin/python scripts/evalkit/ste_lint.py docs/wiki/reference/datasets/<group>/<name>.md
```

The first 3 commands must give no problem for the page. The STE linter must print nothing.
