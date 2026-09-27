---
name: audit-wiki-docs
description: >
  Audit the RadHarmony wiki docs (docs/wiki/) against the real source code. Extracts every
  fenced code block and every constructor-arguments table, resolves the classes / factories
  they reference against the live radharmony package, and reports where docs and code disagree:
  invalid imports/kwargs, undocumented or phantom arguments, undocumented or wrong defaults,
  stale registry keys, and bogus pip extras. Use when the user says "audit the wiki docs",
  "verify the docs against the code", "check the wiki code blocks", "are the wiki examples
  correct", or "make sure every argument is documented". Reports findings first; edits only
  on approval. Complements (does not replace) the update-docs drift-sync skill.
---

# Audit Wiki Docs Skill

Systematically verify that every code block and every argument table in `docs/wiki/`
is correct and complete against the shipped `radharmony` source. This is a
**correctness + completeness audit**, not a drift/staleness sync — for the latter
(counts, nav, tables of contents, broken paths) use the `update-docs` skill instead.
The two are complementary: run `audit-wiki-docs` when the question is "does the code
in these docs actually work and is every argument documented?"

The audit is driven by a shipped Python helper, `audit_wiki_docs.py`, that does the
extraction and introspection so results are deterministic and repeatable. It never
executes doc code (it parses with `ast`); it only imports the package.

## When This Skill Triggers

- User asks to "audit the wiki docs", "verify the docs against the code", "check the
  wiki code blocks", "are the wiki examples correct".
- User asks "is every argument documented?", "do the arg tables match the signatures?",
  "are the documented defaults right?".
- After a batch of API changes (new/renamed constructor args, changed defaults, new
  registry keys, new pip extras) when you want to catch every doc that fell behind.

## What this audits (seven checks)

| User's check | Mechanized as | Finding type(s) |
|---|---|---|
| 1. Code blocks are correct | imports resolve; every kwarg in a constructor/factory call exists in the real signature | `unknown-import`, `unknown-kwarg` |
| 2 + 4. Every argument is documented (table completeness) | the arg table lists every constructor parameter; no phantom rows | `missing-arg-row`, `phantom-arg-row` |
| 3. Descriptions are accurate | registry key is registered; documented default matches code; verbatim default-kwargs dicts match | `registry-key-mismatch`, `default-mismatch`, `default-dict-mismatch` |
| 5. Defaults are documented | every parameter that carries a default has a non-empty `Default` cell | `missing-default` |
| 6. Layout is consistent (house style) | inherited/base-class args live under a `### Shared arguments` subsection, not mixed into the class-specific table nor hidden behind a cross-link to a central table | `shared-arg-not-grouped`, `missing-shared-subsection` |
| 7. Section titles are consistent (house style) | dataset pages share one `##` section vocabulary; no page reintroduces a deprecated synonym | `nonstandard-section-title` |
| (bash) install/CLI correctness | `pip install -e ".[<extra>]"` extras exist in pyproject | `unknown-extra`, `unknown-command` |

Structural facts the helper handles — keep them in mind when reading results:

- **House style: self-contained pages with a `### Shared arguments` subsection.** Every
  evaluator and dataset page's `## Constructor arguments` section is split into a
  class-specific table followed by a `### Shared arguments (inherited from <BaseClass>)`
  subsection (`BaseClsEvaluator` / `BaseSegEvaluator` / `GenerativeEvaluator` for
  evaluators; `BaseRadiologicalDataset` for image datasets; `BaseVQADataset` for the VQA
  datasets, a separate hierarchy whose shared block adds `output_struct` /
  `output_question_type`). No page cross-links to a central
  common-args table. Check 6 enforces this: a base-class arg sitting in the specific table
  is `shared-arg-not-grouped`; an evaluator that inherits shared args but has no subsection
  (the old cross-link pattern) is `missing-shared-subsection`. `docs/wiki/evaluator/index.md`
  keeps its **Common constructor arguments** table as an at-a-glance overview only.
- **Section grouping.** The helper unions all tables in one constructor section — the
  specific table plus its `### Shared arguments` / `### Training arguments` subtables — by
  heading depth *and* subsection heading (rsna_2022 uses `###` for both), so symbol-matching
  sees the section's combined arg set and the split halves don't map to different sibling
  symbols (e.g. a dataset vs its harmonizer). Only actual dataset (`BaseRadiologicalDataset`
  or `BaseVQADataset`) and evaluator symbols are layout-checked; transforms and backbone
  factories are exempt.
- **Closed vs open signatures.** A dataset leaf that declares all its args explicitly
  (no `**kwargs`) contributes only its own signature — base-only params are *not* demanded
  in its table. `unknown-kwarg` is only emitted for a *closed* signature (the forwarding
  chain terminates without `**kwargs`); a genuinely open passthrough is left alone.
- **Section-title vocabulary (check 7, dataset pages).** All dataset pages share one
  `##` section vocabulary (standardized 2026-07-25): `Overview` → `Download` →
  `Expected layout` → label/column sections → `Constructor arguments` → `Dataset constructor`
  → `Harmonizer` → `Load from saved harmonized CSV` → `Harmonizer notes` → `Outputs`
  (or `VQA data dict`) → `Example paths`. Check 7 is a **denylist, not an allowlist**: it
  flags only known deprecated synonyms (`Usage`/`Dataset (MONAI)` → `Dataset constructor`;
  `Harmonizer: instantiate and inspect` → `Harmonizer`; `Extra columns` → `Extra metadata
  columns`; `Data paths` → `Example paths`; bare `Notes` → `Harmonizer notes`). Legitimate
  dataset-specific sections (`Mask output`, `Bounding box classes`, rsna_2022's per-variant
  H2s, montgomery's split `Constructor arguments (harmonizer)/(dataset)`) are **not** flagged.
  Scoped to `docs/wiki/datasets/*.md` only — a bare `## Notes` is canonical on evaluator and
  backbone pages, so those categories are never checked. It does **not** enforce section
  *order*, only titles.

Severity: `error` = mechanical, high-confidence (fix these). `warning` = soft, needs a
human glance (e.g. `default-mismatch`, which tolerates quote style and parenthetical notes
but can still surface an intentional doc simplification like `auto`).

## Workflow

### 1. Run the helper

Always run with the Python env where `radharmony` is installed (Python 3.12) — `import radharmony.evaluator` pulls torch/MONAI
(~6s one-time warmup):

```bash
python \
  docs/audit_wiki_docs.py \
  --json <scratchpad>/audit.json
```

Scope flags:
- Whole wiki (default): omit `--page`.
- One page / glob: `--page docs/wiki/evaluator/linear_probe.md` (accepts an absolute or
  cwd-relative path, a bare filename, or a glob resolved under `--wiki`).
- `--wiki <dir>` overrides the wiki root (defaults to `<repo>/docs/wiki`).
- `--format json` prints JSON to stdout when `--json` is not given.

Exit code is `1` when any `error`-severity finding exists, else `0` — usable in CI.

### 2. Triage the findings

Read the JSON (or the grouped table). Sort by severity; confirm a representative finding of
each type by hand before acting. Known nuances to apply judgment on:

- **Generic API pages** (`api.md`, `architecture.md`) document the *base* dataset API but
  use a concrete class (e.g. `CheXpertTrainDataset`) as the example. The helper resolves
  each table to its best-matching concrete symbol, so base-only args the concrete class
  narrows out (`output_mask`, `output_reg`, …) can appear as `phantom-arg-row`, and the
  example passing them appears as `unknown-kwarg`. The `unknown-kwarg` is usually a **real
  bug** (the example would raise `TypeError`); the `phantom-arg-row` on the generic table
  is often intended. Decide per case.
- **`default-mismatch` (warning)** is deliberately soft. `auto`, "standard pipeline", and
  `None (some note)` are common intentional simplifications — verify before changing.
- **Unmapped tables** (a harmonizer table on a page that doesn't import the harmonizer) are
  silently skipped, not falsely mapped to the dataset. If you expect a table to be checked
  and it isn't, make sure the symbol is imported in a code block on that page.
- **Layout findings (check 6)** are `warning`-severity house-style nudges. `shared-arg-not-grouped`
  means a base-class arg sits in the class-specific table — move it under the section's
  `### Shared arguments` subsection. `missing-shared-subsection` means an evaluator inherits
  shared args but the page inlines none (usually a leftover cross-link) — add the subsection.
  Both are gated to dataset/evaluator symbols; transforms and backbone factories never trip them.
- **Section-title findings (check 7)** are `warning`-severity house-style nudges. A
  `nonstandard-section-title` names the deprecated heading and its canonical replacement —
  just rename the `##` heading (dataset pages only). If a genuinely new dataset-specific
  section is being flagged, it isn't: check 7 only knows the fixed synonym denylist, so a
  finding here always means one of the known synonyms crept back in.

### 3. Prose spot-check (the non-mechanized part of check 3)

The helper cannot grade free-text accuracy. For a sample of arg-table `Description` cells
and page narrative, cross-read the class/function docstring (they are rich and per-arg —
e.g. `MIMICCXRDataset`, `make_raddino`) and flag descriptions that are wrong or misleading.
Inspect a signature directly when unsure:

```bash
python -c \
  "import inspect; from radharmony.dataset import MIMICCXRDataset as C; print(inspect.signature(C.__init__)); print(inspect.getdoc(C))"
```

### 4. Present the report — do NOT edit yet

Summarize findings grouped by page (counts by type + the notable individual items) and
present them to the user. Wait for approval before touching any doc.

### 5. Apply fixes on approval, then re-run

Fix the **doc to match the code**, never the reverse (same rule as `update-docs`). Typical
fixes: add a missing arg row (with the real default), remove/rename a phantom row, fill an
empty `Default` cell, correct a stale registry key, remove an invalid kwarg from an example,
move inherited args out of the class-specific table into a `### Shared arguments (inherited
from <BaseClass>)` subsection (check 6); rename a deprecated dataset-page `##` heading to
its canonical form (check 7).
After editing, re-run the helper (whole wiki, or `--page` for the touched files) and confirm
the targeted findings are gone and no new ones appeared.

## What NOT to do

- **Do not execute doc code** to test it — the helper uses `ast`; keep it that way.
- **Do not fix the code to match the docs.** Docs follow the source.
- **Do not rewrite accurate prose.** Only change what a finding identifies.
- **Do not centralize shared args to silence check 6.** The house style inlines them on
  each page under `### Shared arguments`; do not "fix" a layout finding by moving args to a
  central table and cross-linking. (The helper still unions `evaluator/index.md`'s common-args
  table into the *documented* set so it stays an allowed overview; keep that union if you edit
  the behavior.)
- **Do not "fix" a generic-API-page phantom** without confirming the class really lacks the
  arg — it usually documents the base API on purpose.
- **Do not delete or rename source symbols** to silence a finding.

## Output Format

1. Run the helper; keep the JSON in the scratchpad.
2. Report grouped by page: per-type counts, then the notable findings with `file:line`.
3. Flag anything needing human judgment (soft `default-mismatch`, generic-page phantoms,
   prose descriptions you could not verify against a docstring).
4. After fixes: one-line-per-file summary of what changed, and the re-run result (zero
   remaining errors on the touched pages).

## After editing — sync the skill file

This skill is repo-specific, so mirror both the skill and its helper into the repo (public
`RadHarmony`), keeping them version-controlled alongside the docs they check:

```bash
cp ~/.claude/skills/audit-wiki-docs/SKILL.md            docs/audit_wiki_docs_skill.md
cp ~/.claude/skills/audit-wiki-docs/audit_wiki_docs.py  docs/audit_wiki_docs.py
md5sum ~/.claude/skills/audit-wiki-docs/SKILL.md           docs/audit_wiki_docs_skill.md
md5sum ~/.claude/skills/audit-wiki-docs/audit_wiki_docs.py docs/audit_wiki_docs.py
```

If the repo copy was edited instead, sync in the other direction. Always confirm both
copies match with `md5sum`.
