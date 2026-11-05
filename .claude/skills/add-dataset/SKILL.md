---
name: add-dataset
description: >
  Add a new radiology dataset to RadHarmony.
  It writes the config JSON with the expected stats, gets a config review from a new agent, then
  writes the harmonizer, the dataset class, the exports, the app entry, a fake-data builder,
  the integrity check and the dataset page. Use when the user says "add a dataset", "integrate a
  new dataset", or gives source files or CSV samples to wire into RadHarmony. To add an encoder,
  use add-backbone. To load or use a dataset that exists, use use-radharmony. To fix a finding of
  a report, use fix. To review the branch, use evaluate.
---

# Add a dataset

This skill has the full steps to add a dataset. The page for a person is
`docs/wiki/how-to/add-a-dataset.md`. It lists the main files.

Names in this skill (example: Montgomery County CXR):

| Name | Meaning | Example |
|---|---|---|
| `<name>` | The folder of the harmonizer and the dataset class, and the config file name | `montgomery_cxr` |
| `<key>` | A registry key. A dataset with official splits has one key for each split (D-09) | `montgomery_cxr`, `chexpert_train` |
| `<Name>` | The prefix of the class names | `MontgomeryCXR` |

## Real data

Some steps read the source files on disk. The data folders come from the data paths file:
`configs/data_paths.json`, or the file that `RADHARMONY_DATA_PATHS` names (`docs/wiki/get-started/install.md`,
"Data paths for the checks"). Read them with `paths(key)` of `scripts/dataset_paths.py`, from the repo root
after `sys.path.insert(0, "scripts")`. If the file or the key is missing, `paths()` raises
`MissingPathsError` with a hint. Then ask the user for the folder, and offer one of 3 choices:

1. Add the entry to their data paths file (copy `configs/data_paths.example.json` first if the file is
   missing).
2. Give the folder for this run only.
3. Skip the real-data check, and say so in the report.

A sub-agent cannot ask the user. It skips, and it reports what it skipped. The dev repo is the clone where
`git ls-files configs/data_paths.json` prints the file (the file is in git).

**No user to ask.** Sometimes no user can answer a question (a sub-agent, a headless run). Then write the
question and the chosen answer in the report, and go on.

**A git worktree.** `.venv/bin/python scripts/check_datasets.py` and `python -c` can import the main clone
in place of the worktree. From the worktree root, start them with `PYTHONPATH=$PWD`. `pytest` and the
scripts in `scripts/evalkit/` check the code of the clone or worktree where they run.

## Rules

- **Base code.** Follow `CLAUDE.md`, section "Base code needs explicit approval". Put the new logic in the
  files of the new dataset.
- **Design decisions.** Follow `CLAUDE.md`, section "Design decisions and rules". Read these entries of
  `docs/wiki/design.md` first: D-01, D-03, D-04, D-05, D-06, D-07, D-08, D-09, D-10, D-11, D-21, D-23, D-25, D-26,
  D-30, D-33, D-34 and D-35.
- **Data paths.** Code, tests, docs and the config JSON get no real data path. In examples, use
  `/data/...`. The real paths go only in the data paths file (see "Real data", D-21).
- **No code before the review.** Write no harmonizer or dataset code before step 5 is complete.
- **The expected stats come from the source files** (D-23). Never change an expected stat to match the code.
- **A code bug is not a doc fix** (D-22). Fix only the code of the new dataset. If code that existed before
  this task looks wrong, report it with `file:line` and a command.

## Step 1. Read

1. Ask the user for the folder of the source files, the paths of the tables, and the paper or the website.
   With CSV samples only: do steps 1 to 3. In step 4, write only the paper numbers, and write in
   `_comment` that the source counts wait for the full data. Do steps 5 to 11 and 13 on the samples. Step 12
   and the source counts of step 4 wait for the full data. Tell the user.
2. Read `docs/wiki/how-to/add-a-dataset.md`, and the design entries above.
3. Read a similar dataset: its harmonizer, dataset class, config JSON and wiki page.
   - 2-D labels: `chexpert` (`docs/wiki/reference/datasets/cxr-labels/chexpert.md`).
   - 2-D masks and reports, small: `montgomery_cxr`.
   - 2-D boxes: `vindr_cxr`.
   - 3-D CT: `radchestct` or `rsna_pe_detection`. MRI: `rsna_2024_lumbar_spine`.
4. Read the source files: the file list, the table headers and a few rows. Then the paper, and the website or
   data card.
   `notebooks/datasets/dataset_discovery.ipynb` helps a person do the same.

## Step 2. Write the config JSON

1. Write `configs/datasets/<name>.json` from the template in `references/config.md`.
2. Fill each field from the source files. Ask the user only for a value that you cannot find.
3. Choose the classes. Each official split is its own class and registry key (D-09). Images with no labels
   go in a separate test class (D-25).
4. Choose `base_image_dir` with the user. Read "`base_image_dir`" in `references/data_traps.md`. Show the
   user the folder, the form of `image_path`, and one sample row where `base_image_dir` + `image_path` is a
   real file. Wait for the user to agree.

## Step 3. Add the data paths

1. Add one entry for each new registry key to the data paths file (see "Real data"). The dev repo must
   have it. A user who has the data adds it to their own file. The format is in the docstring of
   `scripts/dataset_paths.py`. The keys of `paths` are the argument names of the dataset class.

   ```json
   "<key>": {
     "status": "ok",
     "paths": {"base_image_dir": "/data/my_dataset/images/", "csv_path": "/data/my_dataset/metadata.csv"},
     "verified": "<YYYY-MM-DD>",
     "note": "<a fact that the paths do not tell>"
   }
   ```

   Use the status `restricted` when some users cannot read the folder, and `not_on_nas` when the lab does
   not have the data.
2. Dev repo only: `tests/test_data_paths.py` needs `verified` for an `ok` or `restricted` entry. Rebuild the
   paths page with `.venv/bin/python scripts/gen_nas_paths_page.py`. In a user's own file, do not run it:
   it writes the folders of that file into a docs page.
3. In Python, read the paths with `paths("<key>")`. Do not copy a path into code, tests or docs.

## Step 4. Record the dataset stats

Do this right after the config JSON, before any code. Follow `references/stats.md` to write the
`expected_stats` block of `configs/datasets/<name>.json`. The numbers come from the paper and from the source
files read with pandas, with no RadHarmony code. `tests/test_dataset_stats.py` compares them with `harmonize()` later.

## Step 5. Config review by a new agent

1. The reviewer cannot ask the user. So get the data folders first (see "Real data"), and pass them in its
   task. Then start a new agent with the Agent tool (`general-purpose`, not a fork), so that it has no
   earlier context.
   Tell it to read the brief `.claude/skills/add-dataset/references/config_review.md`. Give it the config
   path, the registry keys, the data folders and the full folder
   `reports/add-dataset/<name>/config_review/`. Do not tell it your own doubts.
2. The reviewer returns its findings as text (a sub-agent cannot write report files). Save the text in
   `reports/add-dataset/<name>/config_review/findings.md`. Each finding has a command and its output. Run
   the command again and check the fact.
3. Ask the user about each finding, one at a time, with
   `.claude/skills/evaluate/references/decisions.md`. Update the config JSON and `expected_stats` after each
   answer, before the next finding. An answer for this dataset only (for example "keep 2 patient IDs") goes
   in the config `_comment` and on the wiki page, not in `docs/wiki/design.md`. A D- entry is only for a rule
   of more than one dataset.
4. Start the code only when the user answered every finding.

## Step 6. Write the harmonizer

1. Write `radharmony/harmonizer/<name>/<name>.py` and `radharmony/harmonizer/<name>/__init__.py` from
   `references/harmonizer.md`. For each trap of the source, read `references/data_traps.md`. Each official
   split gets its own harmonizer class (D-09).
2. Export it now, because the dataset class imports it: add `from .<name> import <Name>Harmonizer` and an
   `__all__` entry to `radharmony/harmonizer/__init__.py`.

## Step 7. Write the dataset class

1. Write `radharmony/dataset/<name>/<name>.py` and `radharmony/dataset/<name>/__init__.py` from
   `references/dataset_class.md`. Each official split gets its own class and `@register_dataset("<key>")`.
2. Export it: add `from .<name> import <Name>Dataset` and an `__all__` entry to
   `radharmony/dataset/__init__.py`.

## Step 8. Check the exports

Check that the registry finds each new class:

```bash
.venv/bin/python -c "import radharmony.dataset; from radharmony.registry import resolve_dataset as r; print(r('<key>'))"
```

## Step 9. Add the dataset to the app

Follow `docs/wiki/how-to/add-a-dataset.md`, step 2 ("Show the dataset in the app"). Do not copy it here.

- Each split class gets its own entry, after a family header of the same modality.
- `tests/test_app.py::test_app_lists_the_split_keys_and_every_dataset` checks that every new key has an
  entry.
- An output that needs a folder (for example `mask_output_dir`) needs a textbox. Copy the build function of a
  similar dataset in `app.py`, for example Shenzhen CXR.
- Dense boxes on 3-D volumes need one more line. See "Boxes on 3-D volumes" in `references/data_traps.md`.

## Step 10. Add a fake-data builder for CI

CI has no data. The fake data lets the CI tests check the harmonizer contract (rule L-008 in `eval/RULES.md`).

1. In `tests/test_<name>.py`, write `write_fake(tmp_path)`. It writes a few rows with the real file names
   and the real columns of the source. Copy the shape of `write_fake` in `tests/test_chexphoto.py`.
2. In `tests/fixtures/fake_datasets.py`, add a builder that returns a harmonizer on these files. Add it to
   `FAKE`, with the name `<name>` (or `<key>` for a split class).
3. In `tests/test_<name>.py`, add a test for each trap that the harmonizer handles. Get the expected value
   from the source (D-23). A test on real data must skip when the data is not there. Read the path with
   `paths("<key>", missing_ok=True).get("<argument>", "")`, then use `pytest.mark.skipif`, as
   `tests/test_node21.py` does. A plain `paths("<key>")` at import time raises before the skip.
4. Run `.venv/bin/python -m pytest tests/test_harmonizer_contract.py tests/test_<name>.py -v`. It must pass.

## Step 11. Add the integrity check

Follow "Add it" in `references/integrity_check.md`:

1. Add a `DatasetSpec` for each new registry key to `get_specs()` in `scripts/check_datasets.py`, named by
   the registry key (D-21). Turn on every output that the class supports. The edit needs no data.
2. Add a cell to `notebooks/datasets/dataset_integrity_check.ipynb` with the NotebookEdit tool. The cell
   reads the paths file, with no path text (D-21).

## Step 12. Run the checks on the data

These checks read the data folders (see "Real data").

1. The stats test: `.venv/bin/python -m pytest tests/test_dataset_stats.py -k <key> -v`. With a folder for
   this run only, run `harmonize()` on it instead. Count each stat as the table "What the test counts" in
   `references/stats.md` says.

   Find the cause of each difference. Fix the new code, or write the reason in `expected_stats._comment`.
   Do not change a number to match the code (D-23).
2. The integrity check: `.venv/bin/python scripts/check_datasets.py --datasets <key> --max-samples 30`. It
   must exit with 0 and give the status `OK`. Read "Run it" in `references/integrity_check.md` for the
   report and for a folder given for this run only.
3. The fast tests. These tests read every registry key or every config JSON, and they take about 1 minute.
   They must pass. With no data paths file, the tests that need it skip.

   ```bash
   .venv/bin/python -m pytest tests/test_registry.py tests/test_configs.py tests/test_label_order.py \
       tests/test_output_flags.py tests/test_save_load.py tests/test_app.py tests/test_data_paths.py \
       tests/test_scripts.py tests/test_harmonizer_contract.py tests/test_<name>.py -q
   ```

   Do not run `pytest tests` in the foreground. The full run can take more than 1 hour. Step 15 runs it.

   With a data paths file that has only some entries, the tests that check every key skip. Only the
   complete file of the dev repo (`"complete": true`) runs them.

## Step 13. Write the wiki page

Follow `references/wiki_page.md`. It makes or changes these files:

- `docs/wiki/reference/datasets/<group>/<name>.md`: the dataset page, from the template in `references/wiki_page.md`.
- `docs/wiki/reference/datasets/index.md`: a catalog row, and the new count of registry keys.
- `mkdocs.yml`: a nav line in the group.
- `docs/wiki/how-to/get-the-data.md`: the download steps.

## Step 14. Check the list

Go through `references/checklist.md`. Fix each item that fails.

## Step 15. Prove it

1. Run `.venv/bin/python scripts/evalkit/run_all.py --group lint,docs` (fast). It must pass with no new
   problem.
2. Run the full `.venv/bin/python scripts/evalkit/run_all.py` once, in the background (`run_in_background`),
   and wait for the notice. It can take more than 1 hour. Add `--full` only where the data of the
   `tier=nas` claims is: in the dev repo,
   `.venv/bin/python -m pytest tests/test_data_paths.py -k paths_exist_on_nas -q` reports passed tests, not
   skipped ones. It must pass with no new problem.
3. Run `/evaluate` on the work. Work not committed: use `--base <current branch>`. Work committed: use
   `--base <the commit before the first change>`. Do not run the full checks a second time: `/evaluate`
   step 3 runs them.
4. If `/evaluate` reports an open high finding, fix it, or tell the user why not. The task is complete only
   when the automatic checks pass and `/evaluate` has no open high finding.

## Step 16. Lessons

A mistake from this task becomes a check, not text in this skill. Follow `eval/RULES.md`, section "How to
add a rule (lessons)". If a computer cannot check it, the same section makes it a judgement rule.
