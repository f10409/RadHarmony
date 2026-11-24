# Contributing to RadHarmony

This page tells you how to set up a clone, which rules to follow, and how to send a change.

## How the public repo gets its files

The maintainers develop RadHarmony in a private repo. Sometimes they copy its files to this public repo in one
commit (a snapshot). The private repo has only a few more files: the real data paths of the team, internal
docs and a folder for private datasets.

- Send your change as a pull request (PR) to `f10409/RadHarmony`.
- A maintainer merges the PR. The next sync brings it into the private repo, so the next snapshot keeps it.

## 1. Set up a clone (once)

1. Fork `f10409/RadHarmony` on GitHub. Clone your fork.
2. Make a venv ([Install](docs/wiki/get-started/install.md#install-with-uv)).
3. Run these lines in the clone:

<!-- block: run=skip reason="installs packages and changes the git config" -->
```bash
uv pip install -e . --group dev        # radharmony, pytest, ruff and vulture
uv pip install nbstripout
nbstripout --install                   # removes the notebook outputs on commit
git config core.hooksPath .githooks    # turns on the commit hooks
```

The commit hooks run the automatic checks on the staged files. They also stop a commit that changes base code
when the commit message has no `BASE CODE CHANGE:` line.

The dataset checks need the folders of your data. To check some datasets, make `configs/data_paths.json` for
them ([Data paths for the checks](docs/wiki/get-started/install.md#data-paths-for-the-checks)).

## 2. Make the change

Make a branch first: `git checkout -b fix/<topic>`. Then read these files:

- [CLAUDE.md](CLAUDE.md): a change to base code needs approval first.
- [docs/wiki/design.md](docs/wiki/design.md): the design decisions.
- [eval/RULES.md](eval/RULES.md): the rules that the automatic checks and `/evaluate` enforce.

| Change | Read | Claude Code skill |
|---|---|---|
| Add a dataset | [Add a dataset](docs/wiki/how-to/add-a-dataset.md) | `add-dataset` |
| Add a backbone | [Add a backbone](docs/wiki/how-to/add-a-backbone.md) | `add-backbone` |
| Fix a bug | Write a failing test first. Then fix the code. | |
| Review a branch | [eval/PIPELINE.md](eval/PIPELINE.md) | `evaluate`, then `fix` for its findings |

The skills are in `.claude/skills/`. Claude Code loads them by itself when you open the clone
([Claude Code skills](docs/wiki/get-started/install.md#claude-code-skills)).

To change base code, open an issue first and ask a maintainer. The PR description then starts with the
base-code section of [CLAUDE.md](CLAUDE.md). Its `Approved by:` line has the name of the maintainer and the date.

## 3. Check, then send a PR

Run the automatic checks before you push:

<!-- block: run=skip reason="runs the check suite" -->
```bash
.venv/bin/python scripts/evalkit/run_all.py --group lint,docs   # lint and docs, about 20 seconds
.venv/bin/python scripts/evalkit/run_all.py                     # all checks with pytest
```

With no data paths file, the full run takes some minutes. With many datasets, it can take more than 1 hour.
Then push the branch to your fork (`git push origin fix/<topic>`), and open a PR on `f10409/RadHarmony`.

GitHub runs the same checks on each PR. A check fails only on a new problem. The old problems are in
`eval/known_problems/`, and a PR must not add to these lists.

## Do not do these

- Do not write real data paths (the folders on your machine or server) in code, tests or docs.
- Do not commit `configs/data_paths.json`. Git ignores it.
- Do not commit data with a data use agreement (for example MIMIC report text or images), or an API key.
- Do not commit notebook outputs. Exception: `notebooks/tutorials/radharmony_hands_on.ipynb` keeps its outputs
  (`keep_output` in its metadata).
