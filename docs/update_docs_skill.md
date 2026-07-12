---
name: update-docs
description: >
  Audit and update project documentation to match the current codebase state.
  Use this skill when the user asks to "update the docs", "sync the README",
  "update the wiki", "update tutorials", "keep docs up to date", or after a
  batch of code changes and the user wants to reconcile documentation. Also
  trigger when the user asks "does the README need updating?" or "are the docs
  stale?". Covers the full documentation surface — README, wiki pages
  (docs/wiki/), long-form guides (docs/*.md), mkdocs nav, and notebook
  tutorials.
---

# Update Docs Skill

Audit project documentation against the current codebase and update anything
stale, broken, or missing. The goal is documentation that a new contributor
can trust without cross-referencing the source code.

## Core Philosophy

Documentation drifts in predictable ways:
- **File paths** change when things are reorganized
- **Tables** (extras, datasets, notebooks, evaluators) grow stale as new
  entries are added
- **Quick-start examples** reference old APIs
- **Feature lists** lag behind new capabilities
- **Counts** ("N datasets", "M evaluators") fall behind reality
- **Nav indexes** (`mkdocs.yml`) miss orphaned pages

This skill finds those drifts and fixes them precisely — no rewrites, no
expansions beyond what the code actually provides.

## When This Skill Triggers

- User asks to "update docs", "sync README", "update wiki", or "update tutorials"
- User asks "does the README need updating?" or "are the docs stale?"
- After a batch of code changes (dataset additions/restructure, new evaluators,
  new backbone recipes, harmonizer changes, transform changes, notebook
  reorganization, new optional dependencies)
- User asks to reconcile documentation with recent git changes

## Workflow

### 1. Audit what changed

```bash
git diff HEAD --stat          # files changed since last commit
git diff HEAD --name-only     # just the paths
git log --oneline -20         # recent commit messages give the change theme
```

Map each changed code path to the doc surface(s) it likely impacts:

| Code change | Likely doc surface(s) |
|---|---|
| `radharmony/dataset/<name>/` added or renamed | `docs/wiki/datasets/<name>.md`, `docs/wiki/datasets.md`, README dataset list, `mkdocs.yml` nav |
| `LABEL_COLS` modified in a dataset | `docs/wiki/datasets/<name>.md` label count, `docs/wiki/datasets.md` table row |
| `radharmony/harmonizer/<name>.py` added | `docs/wiki/datasets/<name>.md` (if harmonizer is dataset-specific), `notebooks/tutorials/examples.ipynb` |
| `radharmony/evaluator/classification/<name>.py` added | `docs/wiki/evaluator/<name>.md`, `docs/wiki/evaluator/index.md` table, README evaluator list, `mkdocs.yml` |
| `radharmony/evaluator/backbones/<name>.py` added | `docs/wiki/backbones/<name>.md` (new page), `docs/wiki/backbones/index.md` table, `docs/wiki/backbones/custom_backbones.md` (only when authoring patterns change), README extras table, `mkdocs.yml` nav |
| `pyproject.toml` extras changed | README extras table |
| `radharmony/dataset/transforms.py` changed | `docs/wiki/transforms.md` |
| `app/` changed | `docs/wiki/app.md` (including its Remote-access section) |
| `notebooks/` reorganized | README notebook table, any wiki page linking to notebook paths |

### 2. Audit each doc surface

Inventory: every doc surface in this repo and what to check on each. Read the
current file and compare against reality.

**[README.md](README.md)**
- Optional extras table: matches `[project.optional-dependencies]` in `pyproject.toml`?
- Quick-start example: imports + class names still exist in `radharmony/dataset/__init__.py`?
- Evaluator section: code block imports valid? Backbone factory names match `radharmony/evaluator/backbones/__init__.py`?
- Supported datasets count + prose list: matches `radharmony/dataset/__init__.py` + the wiki overview's "N underlying datasets" line?
- Notebook table: every linked path exists on disk?
- Links to docs/guides: files exist at the linked paths?
- Wiki page URLs (`https://f10409.github.io/RadHarmony/...`) — confirm path matches mkdocs nav, accounting for `use_directory_urls: false`.

**[docs/wiki/index.md](docs/wiki/index.md)** (mkdocs home)
- Top-line description / value prop still accurate?
- Any links to other wiki pages — do they resolve?

**[docs/wiki/quickstart.md](docs/wiki/quickstart.md)**
- Import statements, class names, constructor kwargs valid against source?
- Dataset paths: still mention NAS or local paths that exist?

**[docs/wiki/architecture.md](docs/wiki/architecture.md)**
- Subpackage layout matches `radharmony/`?
- Base class names (`BaseRadiologicalDataset`, `RadiologicPreprocessor`, etc.) match source?
- Any diagrams (`docs/figs/*`) reference current names?

**[docs/wiki/api.md](docs/wiki/api.md)** (Dataset API reference)
- `BaseRadiologicalDataset` constructor args + method signatures match `radharmony/dataset/base.py`?
- `get_datasets()` signature + return type match source?
- `output_*` flags (`output_cls`, `output_mask`, `output_bbox`, `output_report`) — only mention flags that exist.

**[docs/wiki/transforms.md](docs/wiki/transforms.md)**
- `RadiologyTransform2D` / `RadiologyTransform3D` constructor args + fluent-builder methods (`with_flip`, `with_rotate`, etc.) match source?
- `output_keys` defaults and accepted values match source?

**[docs/wiki/datasets.md](docs/wiki/datasets.md)** (dataset overview / inventory table)
- The "N dataset configurations across M underlying datasets" header — counts match reality? (Count table rows for configs; count distinct wiki dataset pages for underlying datasets.)
- Every row points to a real `datasets/<name>.md` page.
- Every dataset wiki page that exists is represented as ≥1 row.

**[docs/wiki/datasets/*.md](docs/wiki/datasets/)** (per-dataset pages)
- `base_image_dir` default path on NAS — verify against the dataset class default and the [NAS dataset paths memory](../memory/reference_nas_paths.md).
- Label count — matches `len(LABEL_COLS)` in the dataset class?
- Access notes (Kaggle, PhysioNet, HuggingFace, manual download) still accurate?
- Constructor kwargs shown in any example — exist in the class signature?
- Output keys mentioned (`img`, `cls`, `mask`, `bbox`, `report`) — match what the dataset actually emits?

**[docs/wiki/acquiring-datasets.md](docs/wiki/acquiring-datasets.md)**
- Download URLs / commands still valid?
- Dataset-specific notes (path conventions, login requirements) align with the dataset's wiki page?

**[docs/wiki/nas_paths.md](docs/wiki/nas_paths.md)**
- Each NAS path matches the dataset's `base_image_dir` default or the [NAS dataset paths memory](../memory/reference_nas_paths.md).
- Cross-check against `git log` for recent dataset-path-change commits.

**[docs/wiki/evaluator/index.md](docs/wiki/evaluator/index.md)**
- Evaluator table lists every `*Evaluator` class exported from `radharmony/evaluator/__init__.py` with the matching registry key from the `@register_evaluator(...)` decorator.
- Backbone recipes table — every `make_*` exported from `radharmony/evaluator/backbones/__init__.py` is listed with embed dim + input size + return tuple length matching the source.
- Quick-start code blocks compile against current signatures.

**[docs/wiki/evaluator/{linear_probe,knn_probe,svm_probe,prototype_probe,zero_shot,finetune}.md](docs/wiki/evaluator/)** (per-evaluator pages)
- Registry key matches `@register_evaluator("...")` in the source.
- Constructor args table — every kwarg + default + type matches `__init__` in the source class.
- Default-kwargs dict (e.g. `DEFAULT_PROBE_KWARGS`) is reproduced verbatim from the source.

**[docs/wiki/backbones/index.md](docs/wiki/backbones/index.md)** (recipes table)
- Every `make_*` exported from `radharmony/evaluator/backbones/__init__.py` is in the table.
- Each row links to a real `docs/wiki/backbones/<name>.md` page.
- Embed dim / input size / returns-tuple length match the source.

**[docs/wiki/backbones/<name>.md](docs/wiki/backbones/)** (per-backbone pages)
- Embed dim + input size + segmentation-mode patch grid match the source.
- HuggingFace hub identifier (where applicable) matches the `_<NAME>_HUB` constant.
- For manual-setup backbones (`ark_plus`, `medical_mae`, `eva_x`, `chexfound`): install steps, override kwargs, and default-path table match the factory signature.

**[docs/wiki/backbones/custom_backbones.md](docs/wiki/backbones/custom_backbones.md)**
- `ImageEncoderWrapper.from_huggingface(..., pool=...)` options list matches `radharmony/evaluator/wrappers.py`.
- `EncoderPreprocessTransform.from_huggingface(..., reader=...)` options list matches `radharmony/evaluator/transforms.py`.
- The skeleton recipe matches the structure in `radharmony/evaluator/backbones/_template.py`.

**[docs/wiki/app.md](docs/wiki/app.md)**
- Launch command (`python -m radharmony.app` or similar) still works?
- Feature list reflects what the app currently does?
- Remote-access section: port numbers / SSH commands still valid?

**[mkdocs.yml](mkdocs.yml)**
- Every `nav:` entry references a file that exists.
- Every `docs/wiki/**/*.md` file is referenced by ≥1 `nav:` entry (orphaned pages don't ship).
- `use_directory_urls` value: site URLs in README and other docs must match the chosen style (with `false`, URLs use `path/file.html`; with `true`, they use `path/file/`).

**[notebooks/tutorials/*.ipynb](notebooks/tutorials/)**
- Every `from radharmony.X import Y` import — `Y` is still exported from `X/__init__.py`.
- Dataset constructor kwargs — match current class signatures.
- Hardcoded paths (`/data/...`, `../../...`) — still resolve in the canonical environment.
- Cross-notebook links — target files still exist.

### 3. Verify all code blocks

**Every code block in every doc surface must be verified before the audit
is considered complete.** This is non-negotiable — stale examples are
worse than no examples.

For each code block:

- **Import statements**: confirm the symbol is exported from the module
  listed. Use `grep` on `__init__.py` or the source file.
  ```bash
  grep -n "SVMProbeEvaluator\|VinDrCXRTrainDataset" \
       radharmony/evaluator/__init__.py radharmony/dataset/__init__.py
  ```
- **Class / function signatures**: every kwarg used in the example actually
  exists in the constructor / function.
  ```bash
  grep -n "def __init__" radharmony/dataset/vindr_cxr/vindr_cxr_train.py
  ```
- **Default values**: if the doc states a default, verify it matches source.
  Off-by-one or renamed defaults are common drift.
- **Return types / shapes**: if the doc says `→ Tensor[B, D]` or
  `→ pd.DataFrame`, spot-check against what the code actually returns.
- **Registry keys**: doc-quoted registry keys (`"linear_probe"`, `"chexpert"`,
  …) match the `@register_evaluator` / `@register_dataset` decorators.
- **Path strings**: file paths in code blocks (`/data/...`, NAS paths) align
  with dataset defaults and the [NAS paths memory](reference_nas_paths.md).
- **CLI commands**: shown shell commands (`uv pip install ...`,
  `python -m radharmony.app ...`) — confirm against `pyproject.toml`
  `[project.scripts]` and any entry-point modules.

Fix any discrepancy — update the doc to match the code, never the reverse.
Flag anything that requires a live import to verify (e.g., a third-party
model hub call) with a `<!-- verify: <reason> -->` comment so a human can
check it.

### 4. Cross-cutting consistency checks

After per-surface audits, run these checks across surfaces:

- **Dataset count**: README prose count == `docs/wiki/datasets.md` header
  count == distinct dataset wiki pages.
- **Extras table**: README extras table rows == `pyproject.toml`
  `[project.optional-dependencies]` keys.
- **Evaluator list**: README evaluator prose == `docs/wiki/evaluator/index.md`
  table == `radharmony/evaluator/__init__.py` exports.
- **Backbone list**: README extras table backbone entries ==
  `docs/wiki/backbones/index.md` recipes table ==
  `radharmony/evaluator/backbones/__init__.py` exports.
- **mkdocs nav vs files on disk**: every `docs/wiki/**/*.md` is in nav;
  every nav target exists.

### 5. Fix only what's wrong

Make targeted edits:
- Update broken paths to their new locations
- Add new rows to tables for new entries (extras, backbones, evaluators, notebooks, datasets)
- Remove rows for deleted files
- Fix import names that changed
- Update counts (dataset count, configuration count, parameter counts, etc.)
- Add missing `mkdocs.yml` nav entries for orphan wiki pages

Do NOT:
- Rewrite prose that is still accurate
- Add sections about features not yet implemented
- Expand examples beyond what's needed to be correct
- Change formatting or style for its own sake

### 6. Report what you changed

After editing, provide a concise summary of what was stale and what was fixed.
Group by document. Flag anything you found that needs human judgment (e.g.,
a NAS path you cannot verify, a deprecation you are unsure about, content
behind an auth wall).

### 7. Sync this skill file

After all doc edits are done, sync the canonical skill file into the project:

```bash
cp ~/.claude/skills/update-docs/SKILL.md docs/update_docs_skill.md
```

This keeps `docs/update_docs_skill.md` in version control so the skill
definition is committed alongside the project it serves. If the skill was
edited in `docs/update_docs_skill.md` directly, sync in the other direction:

```bash
cp docs/update_docs_skill.md ~/.claude/skills/update-docs/SKILL.md
```

Always sync whichever copy is newer (check `git diff docs/update_docs_skill.md`
and compare against `~/.claude/skills/update-docs/SKILL.md` with `md5sum`).

---

## RadHarmony quick reference

Subsystems and the documentation surfaces they back:

| Subsystem | Source root | Primary docs |
|---|---|---|
| Datasets | `radharmony/dataset/` | `docs/wiki/datasets.md`, `docs/wiki/datasets/*.md`, `docs/wiki/api.md`, README quick-start |
| Harmonizers | `radharmony/harmonizer/` | `docs/wiki/datasets/*.md` (per-dataset notes), `notebooks/tutorials/examples.ipynb` |
| Preprocessors | `radharmony/preprocessor/` | `docs/wiki/architecture.md`, `notebooks/tutorials/examples.ipynb` |
| Transforms | `radharmony/dataset/transforms.py` | `docs/wiki/transforms.md` |
| Registry | `radharmony/registry.py` | `docs/wiki/api.md`, `notebooks/tutorials/custom_dataset_tutorial.ipynb` |
| Evaluator core | `radharmony/evaluator/`, `radharmony/evaluator/classification/` | `docs/wiki/evaluator/*.md`, README evaluator section |
| Evaluator backbones | `radharmony/evaluator/backbones/` | `docs/wiki/backbones/index.md`, `docs/wiki/backbones/<name>.md` |
| Encoder wrappers | `radharmony/evaluator/wrappers.py`, `radharmony/evaluator/transforms.py` | `docs/wiki/backbones/custom_backbones.md` |
| App | `app/` | `docs/wiki/app.md` |

## Style Rules

- **Paths**: use repo-root-relative paths in documentation. Link with
  markdown link syntax (`[file](path)`), not bare backticks, so the IDE
  preview makes them clickable.
- **Tables**: keep column order consistent with existing tables; add new
  rows at the bottom unless there's a natural sort order (alphabetical,
  logical grouping, modality).
- **Code blocks**: use the exact import syntax that works, tested against
  the current codebase. Prefer the public top-level import
  (`from radharmony.evaluator import X`) over deep imports
  (`from radharmony.evaluator.classification.linear_probe import X`)
  unless the deep path is the only one exported.
- **Install commands**: use `uv pip install -e ".[extra]"` (not `pip
  install`) to match the project convention.
- **Counts**: prefer enumerable counts (table rows, file counts) over
  hand-typed totals — drift is silent and frequent.
- **Do not create new doc files** unless the user explicitly asks; update
  existing ones.
- **Do not invent content**. If the source doesn't document something
  (e.g., a kwarg's purpose), the doc shouldn't either — refer to the
  source instead of guessing.

## Output Format

For each document updated:
1. Edit the file directly (use Edit tool, not Write).
2. After all edits: one-sentence summary per file of what changed.
3. Flag anything that needs human verification (NAS paths, external URLs,
   deprecated APIs you are not certain about, content behind an auth wall).
