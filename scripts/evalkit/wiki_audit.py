#!/usr/bin/env python3
"""Audit the argument tables of the RadHarmony wiki against the real source code.

It reads every argument table (a table whose first column is ``Argument``). It finds the
class or factory that each table documents. It reports where the docs and the code disagree:

  arg-table completeness -> missing-arg-row, phantom-arg-row
  defaults               -> missing-default, default-mismatch
  registry key line      -> registry-key-mismatch
  layout                 -> shared-arg-not-grouped, missing-shared-subsection
  section titles         -> nonstandard-section-title
  bash blocks            -> unknown-extra
  args line              -> args-line-unresolved, args-line-orphan

An ``<!-- args: radharmony.X.Class -->`` line right above a table names the class of the
table. The line of the first table of a section, or of its ``### Shared arguments`` table,
names the class of the whole section. Without a line, the class is a guess from the
argument names. Put the shared table right after the table of its class.

The imports and the call arguments of the doc code blocks are checked by
``run_blocks.py`` (static level, L-005). This script never runs doc code. It only
imports the package::

    .venv/bin/python scripts/evalkit/wiki_audit.py --json out.json
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


# --------------------------------------------------------------------------
# Repo layout
# --------------------------------------------------------------------------
def find_repo_root(start: Path) -> Path:
    """Walk up from *start* until a directory containing pyproject.toml."""
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").is_file():
            return p
    return start


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(SCRIPT_DIR)
sys.path.insert(0, str(REPO_ROOT))  # repo root first: test this clone, not an installed copy
sys.path.insert(0, str(SCRIPT_DIR))
from blocks import extract_blocks  # noqa: E402
from signatures import SigModel, page_imports, resolve_symbol, signature_model  # noqa: E402


# --------------------------------------------------------------------------
# Finding model
# --------------------------------------------------------------------------
# severity: "error" = mechanical, high-confidence; "warning" = soft, human-review.
@dataclass
class Finding:
    page: str
    line: int
    type: str
    severity: str
    symbol: str
    detail: str
    suggested_fix: str = ""


# --------------------------------------------------------------------------
# Truth model: registries, importable symbols, pyproject extras
# --------------------------------------------------------------------------
@dataclass
class TruthModel:
    dataset_keys: set[str] = field(default_factory=set)
    evaluator_keys: set[str] = field(default_factory=set)
    extras: set[str] = field(default_factory=set)
    resolve_dataset: object = None
    resolve_evaluator: object = None
    # base-class argument NAMES, for the layout check (inherited args belong
    # under a "### Shared arguments" subsection, not the class-specific table)
    base_eval_args: set[str] = field(default_factory=set)
    base_dataset_args: set[str] = field(default_factory=set)
    base_vqa_args: set[str] = field(default_factory=set)  # BaseVQADataset hierarchy
    base_classes: set = field(default_factory=set)  # base objects to exempt


def build_truth(wiki_root: Path) -> TruthModel:
    tm = TruthModel()
    try:
        from radharmony.registry import list_datasets, resolve_dataset
        tm.dataset_keys = set(list_datasets())
        tm.resolve_dataset = resolve_dataset
    except Exception as e:  # pragma: no cover
        print(f"[warn] could not load dataset registry: {e}", file=sys.stderr)
    try:
        from radharmony.evaluator import list_evaluators, resolve_evaluator
        tm.evaluator_keys = set(list_evaluators())
        tm.resolve_evaluator = resolve_evaluator
    except Exception as e:  # pragma: no cover
        print(f"[warn] could not load evaluator registry: {e}", file=sys.stderr)

    # pyproject extras
    pyproject = REPO_ROOT / "pyproject.toml"
    if tomllib and pyproject.is_file():
        with pyproject.open("rb") as fh:
            data = tomllib.load(fh)
        tm.extras = set(data.get("project", {}).get("optional-dependencies", {}).keys())

    # base-class arg NAMES for the layout check. Collect each base __init__'s
    # own parameters (not the MRO walk) so we know which names are "inherited
    # plumbing" that the house style groups under a "### Shared arguments"
    # subsection. Base classes themselves are exempted from the check.
    def _own_params(cls) -> set[str]:
        try:
            return {
                n for n, p in inspect.signature(cls.__init__).parameters.items()
                if n != "self" and p.kind not in
                (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            }
        except (TypeError, ValueError):
            return set()

    try:
        from radharmony.dataset.base import BaseRadiologicalDataset
        # base_image_dir stays in the class-specific table by convention;
        # dtype is subclass-added but conventionally part of the shared block.
        tm.base_dataset_args = (_own_params(BaseRadiologicalDataset) - {"base_image_dir"}) | {"dtype"}
        tm.base_classes.add(BaseRadiologicalDataset)
    except Exception as e:  # pragma: no cover
        print(f"[warn] could not load dataset base class: {e}", file=sys.stderr)
    try:
        from radharmony.dataset.base_vqa import BaseVQADataset
        # VQA datasets are a separate hierarchy; base_image_dir stays specific.
        tm.base_vqa_args = _own_params(BaseVQADataset) - {"base_image_dir"}
        tm.base_classes.add(BaseVQADataset)
    except Exception as e:  # pragma: no cover
        print(f"[warn] could not load VQA dataset base class: {e}", file=sys.stderr)
    for mod, cname in (
        ("radharmony.evaluator.base", "BaseEvaluator"),
        ("radharmony.evaluator.classification.base", "BaseClsEvaluator"),
        ("radharmony.evaluator.segmentation.base", "BaseSegEvaluator"),
        ("radharmony.evaluator.language.base_generative", "GenerativeEvaluator"),
    ):
        try:
            base = getattr(importlib.import_module(mod), cname)
            tm.base_eval_args |= _own_params(base)
            tm.base_classes.add(base)
        except Exception:
            pass
    return tm


# --------------------------------------------------------------------------
# Markdown parsing: arg tables
# --------------------------------------------------------------------------
@dataclass
class ArgRow:
    arg: str
    type: str
    required: str
    default: str
    line: int


@dataclass
class ArgTable:
    heading: str
    rows: list[ArgRow]
    columns: list[str]
    level: int = 0  # heading depth (# count) the table sits under
    symbol: str | None = None  # the dotted name of an ``<!-- args: X -->`` line above the table
    symbol_line: int = 0  # the line of that args line


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    # split on unescaped pipes
    return [c.strip() for c in re.split(r"(?<!\\)\|", s)]


def _strip_code(cell: str) -> str:
    return cell.strip().strip("`").strip()


_ARGS_LINE = re.compile(r"^\s*<!--\s*args:\s*([\w.]+)\s*-->\s*$")
# Any line that looks like an args line, also with a typo, for the orphan check.
_ANY_ARGS_LINE = re.compile(r"^\s*<!--\s*args?\s*:")
_FENCE = re.compile(r"^\s*(```|~~~)")


def _args_line(lines: list[str], i: int) -> tuple[str | None, int]:
    """Return (X, line number) of an ``<!-- args: X -->`` line just above line ``i``.

    Blank lines between are fine. Without such a line, return ``(None, 0)``.
    """
    k = i - 1
    while k >= 0 and not lines[k].strip():
        k -= 1
    m = _ARGS_LINE.match(lines[k]) if k >= 0 else None
    return (m.group(1), k + 1) if m else (None, 0)


def _outside_fences(text: str) -> list[str]:
    """Return the lines of ``text``, with each line inside a code fence made empty."""
    out, in_fence = [], False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            out.append("")
        else:
            out.append("" if in_fence else line)
    return out


def orphan_args_lines(rel: str, text: str, tables: list[ArgTable]) -> list[Finding]:
    """Flag each args line that no argument table uses, for example with a typo or text below it."""
    used = {t.symbol_line for t in tables if t.symbol_line}
    return [Finding(rel, n, "args-line-orphan", "error", line.strip(),
                    "This args line names no argument table",
                    "Put `<!-- args: radharmony.X.Class -->` right above the table (blank lines are fine).")
            for n, line in enumerate(_outside_fences(text), start=1)
            if _ANY_ARGS_LINE.match(line) and n not in used]


def find_arg_tables(text: str, heading: str | None = None) -> list[ArgTable]:
    """Find markdown tables whose header first column is 'Argument'.

    If *heading* is given, only tables under a matching ``##`` heading are
    returned (used for the central common-args table). A table keeps the
    class name of its ``<!-- args: X -->`` line. Lines inside code fences are skipped.
    """
    lines = _outside_fences(text)
    tables: list[ArgTable] = []
    cur_heading = ""
    cur_level = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        hm = re.match(r"^(#{1,6})\s+(.*)$", line)
        if hm:
            cur_level = len(hm.group(1))
            cur_heading = hm.group(2).strip()
            i += 1
            continue
        # table header?
        if line.lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^[ \t]*\|?[ \t:*-]+\|", lines[i + 1]):
            header = _split_row(line)
            # A real constructor-arguments table has an "Argument" first column
            # AND a "Type" column. Requiring "Type" skips look-alike explanatory
            # tables (e.g. "| Argument | Loads | Use when |").
            has_type = any(h.strip().lower() == "type" for h in header)
            if header and header[0].strip().lower().startswith("argument") and has_type:
                if heading is None or heading.lower() in cur_heading.lower():
                    cols = [h.strip() for h in header]
                    ci = {c.lower(): k for k, c in enumerate(cols)}
                    rows: list[ArgRow] = []
                    j = i + 2
                    while j < len(lines) and lines[j].lstrip().startswith("|"):
                        cells = _split_row(lines[j])
                        if len(cells) < len(cols):
                            cells += [""] * (len(cols) - len(cells))

                        def get(key):
                            k = ci.get(key)
                            return cells[k] if k is not None and k < len(cells) else ""
                        rows.append(ArgRow(
                            arg=_strip_code(cells[0]),
                            type=get("type"),
                            required=get("required"),
                            default=get("default"),
                            line=j + 1,
                        ))
                        j += 1
                    symbol, symbol_line = _args_line(lines, i)
                    tables.append(ArgTable(heading=cur_heading, rows=rows, columns=cols, level=cur_level,
                                           symbol=symbol, symbol_line=symbol_line))
                    i = j
                    continue
        i += 1
    return tables


# --------------------------------------------------------------------------
# Default-value normalization for comparison
# --------------------------------------------------------------------------
_EMPTY_DEFAULT = {"", "-", "—", "–", "n/a", "na", "none required"}
# doc-friendly placeholders that stand in for a real default (not a mismatch)
_SOFT_DEFAULT_PLACEHOLDERS = {"auto", "auto-discovered", "auto-inferred", "auto-detected"}


def _canon(s: str) -> str:
    """Lowercase, drop backticks, and strip one layer of surrounding quotes."""
    s = s.strip().strip("`").strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        s = s[1:-1]
    return s.lower().strip()


def norm_default(val) -> str:
    """Canonical form of a real (code) default value."""
    if val is inspect.Parameter.empty:
        return "<required>"
    if val is None:
        return "none"
    if isinstance(val, str):
        return val.lower()
    if inspect.isclass(val) or inspect.isfunction(val):  # a class or a function default: its name
        return val.__name__.lower()
    return str(val).lower()


def doc_default_is_empty(cell: str) -> bool:
    return _canon(cell) in _EMPTY_DEFAULT


def doc_default_matches(cell: str, val) -> bool:
    """True if the documented default cell is consistent with the code default.

    Tolerant of quote style, backticks, and trailing parenthetical notes
    (e.g. ``None`` (demo's template)). Placeholders like ``auto`` count as a
    match against ``None``.
    """
    target = norm_default(val)
    # first backtick-quoted span, if any, else the text up to the first '('
    m = re.search(r"`([^`]+)`", cell)
    lead = m.group(1) if m else re.split(r"[(]", cell, 1)[0]
    lead = _canon(lead)
    if lead == target:
        return True
    if lead in _SOFT_DEFAULT_PLACEHOLDERS and target == "none":
        return True
    # whole-cell canon fallback (handles values with no backticks)
    return _canon(cell) == target


# --------------------------------------------------------------------------
# Section-title consistency (check 7)
# --------------------------------------------------------------------------
# Dataset pages share one section vocabulary (standardized 2026-07-25). This
# denylist maps each deprecated synonym to its canonical heading so a new page
# cannot reintroduce the drift. Scoped to dataset pages ONLY -- a bare "Notes"
# heading is canonical on evaluator/backbone pages but on a dataset page the
# trailing notes section is "Harmonizer notes". It is a denylist, not an
# The 6c page templates (docs/proposals/wiki_outline_6c.md). Each kind gives the fixed At a glance rows and
# the rule for the ``##`` sections. Extra sections may stand only where the rule says.
_LABEL_SECTIONS = ("Labels", "Questions and answers", "Captions")
_TEMPLATES = {
    "dataset": {
        "glance": ["Registry keys", "Task and outputs", "Images in the table", "Modality", "License", "Download"],
        "head": ["Quick use", "How it is harmonized"],
        "tail": ["Splits", "Arguments", "Harmonizer", "Known data problems"],
    },
    "evaluator": {
        "glance": ["Registry key", "Task", "Dataset outputs it needs", "Split modes", "Encoder it needs", "Output"],
        "head": ["Quick use"],
        "tail": ["Arguments", "Output", "Notes"],
    },
    "backbone": {
        "glance": ["Recipe", "Model card", "Input size", "Normalization", "Feature", "Install extra",
                   "Segmentation features"],
        "head": ["Quick use", "How the input is prepared", "Segmentation mode"],
        "tail": ["Arguments", "Notes"],
    },
    "howto": {"glance": None, "head": [], "tail": ["Check", "See also"]},
}


def _page_kind(rel: str) -> str | None:
    """Return the template of a wiki page from its path, or None for a page with no template."""
    parts = Path(rel).parts
    if Path(rel).name == "index.md":
        return None
    if "how-to" in parts:
        return "howto"
    for folder, kind in (("datasets", "dataset"), ("evaluators", "evaluator"), ("backbones", "backbone")):
        if folder in parts and parts.index(folder) >= 1 and parts[parts.index(folder) - 1] == "reference":
            # A dataset page is one level below its group folder.
            return kind if kind != "dataset" or len(parts) - parts.index(folder) == 3 else None
    return None


def check_template(rel: str, text: str) -> list[Finding]:
    """Check that a page has the sections and the At a glance rows of its 6c template."""
    kind = _page_kind(rel)
    if kind is None:
        return []
    tpl = _TEMPLATES[kind]
    lines = _outside_fences(text)
    h2 = [(i, m.group(1).strip()) for i, line in enumerate(lines, start=1)
          if (m := re.match(r"^##\s+(.*?)\s*$", line))]
    titles = [t for _, t in h2]
    findings: list[Finding] = []

    def bad(kind_: str, line: int, detail: str, fix: str) -> None:
        findings.append(Finding(rel, line, kind_, "error", kind, detail, fix))

    head, tail = tpl["head"], tpl["tail"]
    middle = titles[len(head):len(titles) - len(tail)] if len(titles) >= len(head) + len(tail) else None
    if titles[:len(head)] != head or titles[len(titles) - len(tail):] != tail or middle is None:
        bad("template-sections", h2[0][0] if h2 else 1, f"The ## sections are {titles}",
            f"A {kind} page starts with {head} and ends with {tail}.")
    elif kind == "dataset" and (not middle or middle[0] not in _LABEL_SECTIONS):
        bad("template-sections", h2[len(head)][0], f"The section after {head[-1]!r} is {middle[:1]}",
            f"Use one of {list(_LABEL_SECTIONS)} there.")
    if kind == "howto":
        steps = [(i, t) for i, t in h2 if re.match(r"^\d+\.", t)]
        for n, (i, t) in enumerate(steps, start=1):
            if not t.startswith(f"{n}. "):
                bad("template-sections", i, f"Step `{t}` is not step {n}", "Number the steps 1, 2, 3 with no gap.")
                break
    if tpl["glance"] is not None:
        top = lines[:h2[0][0] - 1] if h2 else lines
        rows = [c[0] for line in top if line.startswith("|") and (c := _split_row(line)) and c[0] not in ("", "---")]
        if rows != tpl["glance"]:
            bad("template-glance", 1, f"The At a glance rows are {rows}", f"Use the rows {tpl['glance']}.")
    return findings


# --------------------------------------------------------------------------
# Per-page audit
# --------------------------------------------------------------------------
def audit_page(path: Path, wiki_root: Path, tm: TruthModel) -> list[Finding]:
    rel = str(path.relative_to(wiki_root.parent))
    text = path.read_text(encoding="utf-8")
    findings: list[Finding] = []

    blocks = extract_blocks(text)

    # The radharmony names that the page imports. run_blocks.py reports a bad import (L-005).
    page_symbols, _ = page_imports(blocks)

    sig_cache: dict[int, SigModel | None] = {}

    def get_sig(obj):
        key = id(obj)
        if key not in sig_cache:
            sig_cache[key] = signature_model(obj)
        return sig_cache[key]

    # --- registry-key line ---
    keym = re.search(r"Registry key:\s*`\"([^\"]+)\"`", text)
    reg_key = None
    if keym:
        reg_key = keym.group(1)
        key_line = text[: keym.start()].count("\n") + 1
        under_eval = "/evaluators/" in rel
        valid = (reg_key in tm.evaluator_keys) if under_eval else \
                (reg_key in tm.evaluator_keys or reg_key in tm.dataset_keys)
        if not valid:
            findings.append(Finding(
                rel, key_line, "registry-key-mismatch", "error", reg_key,
                f'Registry key `"{reg_key}"` is not registered',
                "Update to a real key from list_datasets()/list_evaluators()."))

    # --- arg tables: rows and defaults ---
    # A page can have several argument tables, for example one for the dataset and one
    # for the harmonizer. Each table gets the class that fits it best.
    # A "Common arguments" table documents a base class. Its rows count as the
    # inherited arguments, so it is not matched to a class here.
    all_tables = find_arg_tables(text)
    findings.extend(orphan_args_lines(rel, text, all_tables))
    tables = [t for t in all_tables if "common" not in t.heading.lower()]
    candidates: dict[str, object] = dict(page_symbols)
    if reg_key is not None:
        resolver = tm.resolve_evaluator if "/evaluators/" in rel else \
            (tm.resolve_dataset if reg_key in tm.dataset_keys else tm.resolve_evaluator)
        try:
            if resolver:
                obj = resolver(reg_key)
                candidates.setdefault(obj.__name__, obj)
        except Exception:
            pass

    def match_symbol(args: set):
        """Best-overlap symbol for an arg set, or None if nothing maps well."""
        best_obj, best_sig, best_ov = None, None, 0
        for obj in candidates.values():
            sig = get_sig(obj)
            if sig is None:
                continue
            ov = len(args & set(sig.params))
            if ov > best_ov:
                best_obj, best_sig, best_ov = obj, sig, ov
        # require the symbol to cover a majority of the args, so a harmonizer
        # table doesn't get mapped to the dataset on one stray name.
        if best_obj is None or best_ov < max(1, (len(args) + 1) // 2):
            return None, None
        return best_obj, best_sig

    # A constructor section is one heading and all deeper (###) subtables under
    # it -- e.g. a "## Constructor arguments" table plus its "### Shared
    # arguments" / "### Training arguments" subtables. Group by heading depth so
    # symbol-matching sees the section's *combined* arg set; otherwise the split
    # halves can map to different sibling symbols on a multi-class page (e.g. a
    # dataset's specific rows overlap its harmonizer equally) and the inherited
    # rows stop being unioned into the leaf symbol.
    # _shared_sub: the narrow "### Shared arguments" subsection (used by the
    #   layout check to tell inherited-arg tables from class-specific ones).
    # _sub_heading: any subsection that continues a constructor section (also
    #   "### Training arguments (shared with ...)"), used only for grouping.
    _shared_sub = re.compile(r"shared arguments|inherited from", re.I)
    _sub_heading = re.compile(r"shared arguments|inherited from|training arguments|shared with", re.I)
    groups: list[list[ArgTable]] = []
    open_level = None
    for t in tables:
        # A table continues the current section if it is an explicit sub-heading
        # (shared/training) OR sits at a deeper heading level than the section's
        # opening heading. Sibling sub-headings (e.g. rsna_2022 uses ### for both
        # "Constructor arguments" and "Shared arguments") merge on the former.
        # A table with its own ``args:`` line for another class starts a new section.
        # A shared table may give the line for a section that has none.
        is_sub = bool(_sub_heading.search(t.heading))
        group_symbol = groups[-1][0].symbol if groups else None
        if groups and is_sub and t.symbol and group_symbol is None:
            groups[-1][0].symbol, groups[-1][0].symbol_line = t.symbol, t.symbol_line
            group_symbol = t.symbol
        new_symbol = t.symbol is not None and t.symbol != group_symbol
        if groups and not new_symbol and (is_sub or (open_level is not None and t.level > open_level)):
            groups[-1].append(t)
        else:
            groups.append([t])
            open_level = t.level

    # group tables by resolved symbol so completeness unions its tables
    group_matches: list[tuple[object, SigModel, list[ArgTable]]] = []
    by_symbol: dict[int, tuple[object, SigModel, list[ArgTable]]] = {}
    for grp in groups:
        combined = {r.arg for t in grp for r in t.rows}
        if grp[0].symbol:
            # The args: line names the class. It wins over the guess from the argument names.
            module, _, name = grp[0].symbol.rpartition(".")
            obj = resolve_symbol(module, name) if module else None
            sig = get_sig(obj) if obj is not None else None
            if sig is None:
                findings.append(Finding(
                    rel, grp[0].symbol_line, "args-line-unresolved", "error", grp[0].symbol,
                    f"`<!-- args: {grp[0].symbol} -->` does not name a class or function of radharmony",
                    "Write the full dotted name, for example radharmony.dataset.CheXpertTrainDataset."))
                continue
        else:
            obj, sig = match_symbol(combined)
        if obj is None:
            continue  # unmappable section (symbol not imported on the page) -> skip
        group_matches.append((obj, sig, grp))
        by_symbol.setdefault(id(obj), (obj, sig, []))[2].extend(grp)

    for obj, sig, tbls in by_symbol.values():
        name_of = getattr(obj, "__name__", str(obj))
        # Each page lists every argument of its class, also the shared ones (6c template).
        documented = {r.arg for t in tbls for r in t.rows}
        sig_names = set(sig.params)

        # missing-arg-row: a real param documented nowhere for this symbol
        for pname in sorted(sig_names - documented):
            p = sig.params[pname]
            findings.append(Finding(
                rel, tbls[0].rows[0].line - 1 if tbls[0].rows else 1,
                "missing-arg-row", "error", name_of,
                f"Parameter `{pname}` (of `{name_of}`) is not in the constructor-arguments table",
                f"Add a row for `{pname}` (default `{_pretty_default(p.default)}`)."))

        for t in tbls:
            has_default_col = any(c.lower() == "default" for c in t.columns)
            has_required_col = any(c.lower() == "required" for c in t.columns)
            for r in t.rows:
                if r.arg not in sig_names:
                    findings.append(Finding(
                        rel, r.line, "phantom-arg-row", "error", name_of,
                        f"Documented argument `{r.arg}` is not a parameter of `{name_of}`",
                        "Remove the row or fix the name."))
                    continue
                p = sig.params[r.arg]
                has_default = p.default is not inspect.Parameter.empty
                marked_required = r.required.strip().lower() in ("yes", "y", "required")
                if not (has_default and has_default_col):
                    continue
                if doc_default_is_empty(r.default):
                    # check 5: a defaulted param must document its default.
                    if has_required_col and marked_required:
                        # documenting a defaulted arg as Required=Yes is a common
                        # convention (e.g. base_image_dir) -> soft nudge only.
                        findings.append(Finding(
                            rel, r.line, "missing-default", "warning", name_of,
                            f"`{r.arg}` has code default `{_pretty_default(p.default)}` but is marked Required=Yes with no Default",
                            f"Confirm it's effectively required; else set Required=No and Default `{_pretty_default(p.default)}`."))
                    else:
                        findings.append(Finding(
                            rel, r.line, "missing-default", "error", name_of,
                            f"`{r.arg}` has default `{_pretty_default(p.default)}` but the Default cell is empty",
                            f"Set Default to `{_pretty_default(p.default)}`."))
                elif not doc_default_matches(r.default, p.default):
                    # check 3: documented default should match the code (soft).
                    findings.append(Finding(
                        rel, r.line, "default-mismatch", "warning", name_of,
                        f"`{r.arg}` documented default `{r.default}` vs code `{_pretty_default(p.default)}`",
                        f"Verify; set to `{_pretty_default(p.default)}` if stale."))

    # --- layout consistency: inherited/shared args must live under a
    #     "### Shared arguments (inherited from <Base>)" subsection, not mixed
    #     into the class-specific table, and must be inlined (not cross-linked).
    #     Enforces the house style across evaluator + dataset pages. ---
    for obj, sig, grp in group_matches:
        if obj in tm.base_classes:
            continue  # a page documenting a base class itself (for example reference/api-datasets.md)
        is_eval = _is_evaluator(obj)
        mro_names = [b.__name__ for b in getattr(obj, "__mro__", [])] if inspect.isclass(obj) else []
        is_dataset = "BaseRadiologicalDataset" in mro_names
        is_vqa = "BaseVQADataset" in mro_names
        if is_eval:
            shared_set = tm.base_eval_args
        elif is_vqa:
            shared_set = tm.base_vqa_args
        elif is_dataset:
            shared_set = tm.base_dataset_args
        else:
            continue  # transforms, backbone factories, etc. have no shared-arg convention
        shared_in_sig = shared_set & set(sig.params)
        if not shared_in_sig:
            continue
        name_of = getattr(obj, "__name__", str(obj))
        base_hint = _base_name(obj, is_eval)
        spec_tables = [t for t in grp if not _shared_sub.search(t.heading)]
        sub_tables = [t for t in grp if _shared_sub.search(t.heading)]
        misplaced = [r for t in spec_tables for r in t.rows if r.arg in shared_in_sig]
        if misplaced:
            names = ", ".join(f"`{r.arg}`" for r in misplaced)
            findings.append(Finding(
                rel, misplaced[0].line, "shared-arg-not-grouped", "warning", name_of,
                f"Inherited arg(s) {names} appear in the class-specific table",
                f"Move them under a `### Shared arguments (inherited from {base_hint})` subsection."))
        elif is_eval and not sub_tables:
            # shared args exist in the signature but are inlined nowhere on the
            # page -- the old cross-link-to-central-table pattern. (For datasets
            # a missing shared arg is already a hard `missing-arg-row`.)
            findings.append(Finding(
                rel, grp[0].rows[0].line if grp[0].rows else 1,
                "missing-shared-subsection", "warning", name_of,
                f"`{name_of}` inherits shared args but the page has no `### Shared arguments` subsection",
                f"Inline them under `### Shared arguments (inherited from {base_hint})` rather than linking to a central table."))

    # --- section titles (dataset pages) ---
    findings.extend(check_template(rel, text))

    # --- bash blocks ---
    for blk in blocks:
        if blk.lang not in ("bash", "sh", "shell", "console"):
            continue
        for off, ln in enumerate(blk.code.splitlines()):
            for extra in re.findall(r'pip install[^\n]*?-e\s+["\']?\.\[([^\]]+)\]', ln):
                for name in re.split(r"\s*,\s*", extra.strip()):
                    name = name.strip().strip("\"'")
                    if name and tm.extras and name not in tm.extras:
                        findings.append(Finding(
                            rel, blk.start_line + off, "unknown-extra", "error", name,
                            f'pip extra `[{name}]` is not in pyproject optional-dependencies',
                            "Fix the extra name or add it to pyproject."))
    return findings


def _is_evaluator(obj) -> bool:
    try:
        return inspect.isclass(obj) and any("Evaluator" in b.__name__ for b in obj.__mro__)
    except Exception:
        return False


def _base_name(obj, is_eval: bool) -> str:
    """Name of the base class whose args belong in the shared subsection."""
    names = [b.__name__ for b in getattr(obj, "__mro__", [])]
    if not is_eval:
        return "BaseVQADataset" if "BaseVQADataset" in names else "BaseRadiologicalDataset"
    for cand in ("GenerativeEvaluator", "BaseSegEvaluator", "BaseClsEvaluator"):
        if cand in names:
            return cand
    return "BaseEvaluator"


def _pretty_default(val) -> str:
    if val is inspect.Parameter.empty:
        return "required"
    if isinstance(val, str):
        return repr(val)
    if inspect.isclass(val) or inspect.isfunction(val):
        return val.__name__
    return str(val)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wiki", default=str(REPO_ROOT / "docs" / "wiki"),
                    help="wiki root dir (default: <repo>/docs/wiki)")
    ap.add_argument("--page", default=None,
                    help="restrict to a page or glob (relative to --wiki or absolute)")
    ap.add_argument("--json", default=None, help="write findings JSON here")
    ap.add_argument("--format", choices=["table", "json"], default="table")
    args = ap.parse_args()

    wiki_root = Path(args.wiki).resolve()
    if not wiki_root.is_dir():
        print(f"[error] wiki root not found: {wiki_root}", file=sys.stderr)
        return 2

    if args.page:
        p = Path(args.page)
        if p.exists():  # absolute or cwd-relative path to a real file
            pages = [p.resolve()]
        else:
            pages = sorted(wiki_root.glob(args.page)) or sorted(wiki_root.glob(f"**/{args.page}"))
            if not pages and (wiki_root / args.page).exists():
                pages = [wiki_root / args.page]
    else:
        pages = sorted(p for p in wiki_root.rglob("*.md") if ".ipynb_checkpoints" not in p.parts)

    tm = build_truth(wiki_root)

    all_findings: list[Finding] = []
    for page in pages:
        try:
            all_findings.extend(audit_page(page, wiki_root, tm))
        except Exception as e:  # keep going across pages
            print(f"[warn] {page}: audit error: {e}", file=sys.stderr)

    if args.json:
        Path(args.json).write_text(
            json.dumps([asdict(f) for f in all_findings], indent=2), encoding="utf-8")

    if args.format == "json" and not args.json:
        print(json.dumps([asdict(f) for f in all_findings], indent=2))
    else:
        _print_table(all_findings, len(pages))

    # exit 1 if any hard errors, else 0
    return 1 if any(f.severity == "error" for f in all_findings) else 0


def _print_table(findings: list[Finding], n_pages: int) -> None:
    by_page: dict[str, list[Finding]] = {}
    for f in findings:
        by_page.setdefault(f.page, []).append(f)
    errors = sum(1 for f in findings if f.severity == "error")
    warnings = sum(1 for f in findings if f.severity == "warning")
    for page in sorted(by_page):
        print(f"\n=== {page} ===")
        for f in sorted(by_page[page], key=lambda x: x.line):
            tag = "ERR " if f.severity == "error" else "warn"
            print(f"  L{f.line:<4} [{tag}] {f.type}: {f.detail}")
            if f.suggested_fix:
                print(f"           -> {f.suggested_fix}")
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.type] = counts.get(f.type, 0) + 1
    print(f"\n--- audited {n_pages} page(s): {errors} error(s), {warnings} warning(s) ---")
    for t in sorted(counts):
        print(f"    {t}: {counts[t]}")


if __name__ == "__main__":
    raise SystemExit(main())
