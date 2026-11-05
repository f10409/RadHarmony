# Intent: app, scripts, registry and integrity checker

**Design decisions that apply** (`docs/wiki/design.md`). Test each one, and read its known gaps:
- D-11: output flags, for the integrity check
- D-21: one file for data paths (a user's own file: `docs/wiki/get-started/install.md`, "Data paths for the checks")
- D-39: `check_dataset` gives ERROR or WARN
- D-45: a known-problems list only gets shorter (`scripts/evalkit/`)
- Any other entry in the design.md section "Repo rules" that applies to this part.

**Other test steps:**

1. `radharmony/registry.py` and `radharmony/evaluator/_registry.py`:
   - Every registered key resolves to the right class.
   - A duplicate key raises an error instead of overwriting the old one.
   - `list_datasets()` and `list_evaluators()` are complete after a plain `import radharmony`, or the docs say which import is needed.
2. `app.py`, and `radharmony_internal/app.py` if the repo has it:
   - Every `DATASET_REGISTRY` entry builds with its default arguments.
   - The defaults match the dataset docs.
   - The Gradio functions return what the UI shows. Call them directly. Do not start a server.
3. `radharmony/integrity.py`:
   - `check_dataset` reports a real problem as WARN or ERROR, and gives no false OK or false ERROR.
   - Check an earlier finding again: `check_dataset` does not parse JSON-string boxes (NODE21 style), so it
     counts valid boxes as invalid.
   - `subsample_dataset` keeps the dataset usable.
4. `scripts/*.py`:
   - The CLI flags work as their help text says.
   - `--json -` prints only JSON to stdout.
   - The exit codes are right.
   - `scripts/check_datasets.py` runs. A default run skips the datasets with no path. `--datasets <key>` with no
     path stops with the hint. If the repo has `scripts/check_datasets_internal.py`, it runs too.
   - Other scripts (benchmark runners, notebooks helpers) run with `--help` and do what their docstring says.
5. `pyproject.toml` extras: each extra installs what its backbone or feature imports. Read the imports to check. Do not install anything.
