# How we test RadHarmony

A small change in shared code can change the numbers of every dataset, with no error. Example: the code
swaps x and y in a bounding box. Each box then points to the wrong part of the image, but the code still
runs. The tests and checks on this page find such problems before a change merges. This page gives an
overview, and then the commands to run the tests yourself.

## What we test

The tests are in `tests/`, and pytest runs them.

| Kind | What it checks | Data it needs |
|---|---|---|
| Unit tests | One function or class on small made-up inputs. Examples: a transform moves each box with the image, and a metric gives the same value as scikit-learn. | None |
| Fake-data tests | Some harmonizers on a few rows of fake source files, with the real file names and columns. One shared test (the contract test) checks the output table. Examples: the table is the same after save and load, the label columns are sorted, and `image_path` is relative to the image folder. | None |
| Real-data tests | `harmonize()` on the real source files of each dataset. The numbers of images, patients, positives for each label, and patients of each sex must match `expected_stats` in `configs/datasets/<key>.json`. The contract test also runs on each real table. | The dataset |
| Lesson tests | A real mistake from the past, turned into a test so that it cannot come back. Each test has a row in `eval/RULES.md`. | Some need a dataset |
| Repo tests | `pyproject.toml` lists each package that the code imports. Each registry key gives a class. The config files are valid. The script and app functions work on fake files. | None |

A test skips when the things it needs are not there. Examples:

- A data test skips when `configs/data_paths.json` has no entry for its dataset, or when the folder cannot
  be read ([Data paths for the checks](../get-started/install.md#data-paths-for-the-checks)).
- A test skips when it needs an optional package or a GPU that is not installed.

A skip is not a pass. CI has no data, so it skips all the real-data tests.

## Other automatic checks

Pytest is one part of the automatic checks. `scripts/evalkit/run_all.py` runs all of them, in 3 groups:

| Group | Checks |
|---|---|
| `lint` | ruff, with only the rules that find bugs. A word check (the STE linter) on docs, docstrings and comments, for example no em dashes. |
| `docs` | Each argument table in the docs matches the code. Each import and keyword argument in a doc code block exists, and the blocks marked `run=ci` run. Each number with a claim note matches its source. Some lesson checks that are scripts, not pytest tests. |
| `test` | Pytest. It also lists the changed files that are base code. |

Base code is the code that all datasets or evaluators share, for example `radharmony/harmonizer/base.py`. A
change there can change every result, so it needs the approval of a maintainer.

## Old problems: the known-problems lists

When a new check starts, the code can already have problems of that kind. We do not fix them all on day one.
The lists in `eval/known_problems/` hold these old problems.

- A `lint` or `docs` check fails only on a problem that is not on its list. A failing test always fails.
- Only a new rule may add its old problems to a list. The pull request names the rule on its own line, for
  example `New rule: L-013`.
- No other change may make a list longer. CI checks this on each pull request.
- When you fix an old problem, remove it from the list. So the lists get shorter.

Example: a new docstring with an em dash fails the `lint` group. An old docstring with an em dash on the
list does not fail.

## When the checks run

| When | What runs | Data |
|---|---|---|
| Each commit (commit hook) | `run_all.py --fast`: the `lint` and `docs` checks on the staged files only. No pytest, and no doc block runs. A commit that changes base code needs a `BASE CODE CHANGE:` line in its message. | None |
| Before you push | The quick run, `run_all.py`: the 3 groups on all files. With many datasets in your data paths file, it can take more than 1 hour. | Only the datasets in your data paths file |
| Each pull request (GitHub CI) | `tests.yml`: pytest. `checks.yml`: the `lint` and `docs` groups, the check that no list got longer, and the check that a base-code change has approval text. | None |
| Before a release (maintainers) | The full run, `run_all.py --full`: the quick run, plus every doc code block and the number claims that need the data. A block that cannot run because data, a package or a GPU is missing is "blocked", not failed. | All datasets, and a GPU for some blocks |

CI installs the newest versions of the packages that `pyproject.toml` allows, the same way a new user
installs them. So CI also finds a problem that comes from a new package release, not only from our code.

## Run the tests yourself

Run these commands in a clone of the repo, with its venv
([Install](../get-started/install.md#install-with-uv)).

1. Install the test tools once:

    <!-- block: run=skip reason="installs packages and changes the git config" -->
    ```bash
    uv pip install -e . --group dev        # radharmony, pytest, ruff and vulture
    git config core.hooksPath .githooks    # turns on the commit hooks
    ```

2. Run the checks:

    <!-- block: run=skip reason="runs the check suite" -->
    ```bash
    .venv/bin/python scripts/evalkit/run_all.py --group lint,docs   # lint and docs only, about 20 seconds
    .venv/bin/python scripts/evalkit/run_all.py                     # quick run: lint, docs and pytest
    ```

    `run_all.py` prints a table with one row for each check. A new problem starts with `NEW`, for example
    `NEW ste: docs/wiki/how-to/cache.md:12: [STE-01] 27 words: ...`. The last line is
    `automatic checks: PASS` or `automatic checks: FAIL`.

3. To run only some tests, call pytest:

    <!-- block: run=skip reason="runs tests" -->
    ```bash
    .venv/bin/python -m pytest tests/test_transforms.py -v   # one file
    .venv/bin/python -m pytest tests/ -k montgomery -rs     # tests with "montgomery" in the name, with skip reasons
    ```

4. To run the real-data tests of a dataset, give its folders in a data paths file
   ([Data paths for the checks](../get-started/install.md#data-paths-for-the-checks)):

    <!-- block: run=skip reason="copies a file in a clone" -->
    ```bash
    cp configs/data_paths.example.json configs/data_paths.json
    # Keep the entry of your dataset, for example montgomery_cxr, and give its real folders. Then:
    .venv/bin/python -m pytest tests/test_dataset_stats.py -k montgomery -v
    ```

    If a test still skips, `-rs` prints the reason. Example: the folder cannot be read.

## A new test for each fix

- A bug fix starts with a test that fails. Then the fix makes it pass.
- The expected value comes from the source data or the paper, not from the code. A value copied from the
  code passes even when the code is wrong. Example: the expected image count comes from the dataset paper,
  not from a run of `harmonize()`.
- A mistake that happens 2 times, or one time with a large effect, becomes a rule in
  [eval/RULES.md](https://github.com/f10409/RadHarmony/blob/main/eval/RULES.md). If a computer can check
  the rule, the rule gets a test or a check. A script proves that the test fails on the old mistake and
  passes on the fix.

## Review by agents

The automatic checks find only the problems that we already know how to check. The `/evaluate` skill of
Claude Code adds a review by agents. Bug hunters must prove each bug with a command and its output. A second
agent, the checker, runs each proof again and drops the proofs that do not repeat. A lesson from the review
becomes a new rule, as in the section above. The steps are in
[eval/PIPELINE.md](https://github.com/f10409/RadHarmony/blob/main/eval/PIPELINE.md).

## See also

- [Design decisions](../design.md): the expected behaviors that the tests check.
- [eval/RULES.md](https://github.com/f10409/RadHarmony/blob/main/eval/RULES.md): the rules and the check for
  each rule.
- [CONTRIBUTING.md](https://github.com/f10409/RadHarmony/blob/main/CONTRIBUTING.md): the commands, and how to
  send a pull request.
