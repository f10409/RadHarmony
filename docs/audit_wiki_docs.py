#!/usr/bin/env python3
"""Audit the RadHarmony wiki docs against the real source code.

Extracts every fenced code block and every ``## Constructor arguments`` table
from the wiki markdown, resolves the classes / factories they reference against
the live ``radharmony`` package, and reports where the docs and the code
disagree. The five checks it mechanizes:

  1. code-block correctness  -> unknown-import, unknown-kwarg
  2/4. arg-table completeness -> missing-arg-row, phantom-arg-row
  3. description accuracy      -> registry-key-mismatch, default-mismatch,
                                  type-mismatch, default-dict-mismatch
  5. defaults documented       -> missing-default
  bash blocks                  -> unknown-extra, unknown-command

It never executes doc code (uses ``ast``); it only imports the shipped package.
Run under the full venv so ``import radharmony.evaluator`` succeeds::

    /home/fli40/RadHarmony/.venv/bin/python audit_wiki_docs.py --json out.json

Prose-description accuracy that can't be mechanized is left to the calling
agent (cross-read against the class/function docstrings).
"""
from __future__ import annotations

import argparse
import ast
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
REPO_ROOT = find_repo_root(SCRIPT_DIR if (SCRIPT_DIR / "pyproject.toml").is_file() else Path.cwd())


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
    common_eval_args: set[str] = field(default_factory=set)  # from evaluator/index.md


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

    # evaluator common-args table (shared args documented centrally)
    idx = wiki_root / "evaluator" / "index.md"
    if idx.is_file():
        for tbl in find_arg_tables(idx.read_text(encoding="utf-8"), heading="Common constructor arguments"):
            tm.common_eval_args |= {r.arg for r in tbl.rows}
    return tm


# --------------------------------------------------------------------------
# Symbol resolution
# --------------------------------------------------------------------------
_MOD_CACHE: dict[str, object] = {}


def _import(mod: str):
    if mod not in _MOD_CACHE:
        _MOD_CACHE[mod] = importlib.import_module(mod)
    return _MOD_CACHE[mod]


def resolve_symbol(module: str, name: str):
    """Resolve ``from <module> import <name>`` to a live object, or None.

    Only radharmony.* is resolved; third-party imports return None (skipped).
    """
    if not module.startswith("radharmony"):
        return None
    try:
        obj = getattr(_import(module), name)
        return obj
    except Exception:
        pass
    # deep-path fallback: from radharmony.evaluator.backbones import make_x
    try:
        return _import(f"{module}.{name}")
    except Exception:
        return None


# --------------------------------------------------------------------------
# Signature model (walks the MRO to absorb **base_kwargs forwarding)
# --------------------------------------------------------------------------
@dataclass
class SigModel:
    params: dict[str, inspect.Parameter]  # name -> Parameter (leaf wins)
    closed: bool  # False if the base-most __init__ still has **kwargs


def signature_model(obj) -> SigModel | None:
    if inspect.isclass(obj):
        return _class_sig(obj)
    if inspect.isfunction(obj) or inspect.isbuiltin(obj) or callable(obj):
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            return None
        params, var_kw = _collect(sig)
        return SigModel(params=params, closed=not var_kw)
    return None


def _collect(sig: inspect.Signature):
    params: dict[str, inspect.Parameter] = {}
    var_kw = False
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL,):
            continue
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            var_kw = True
            continue
        params[name] = p
    return params, var_kw


def _class_sig(cls) -> SigModel | None:
    """Accepted-parameter set for a class constructor.

    A base class's parameters are only absorbed when the subclass forwards via
    ``**kwargs`` (e.g. evaluators' ``**base_kwargs``). A leaf that declares all
    its args explicitly (no ``**kwargs``, e.g. every dataset) contributes its
    own signature and nothing from its bases -- otherwise base-only params would
    be spuriously flagged as undocumented.
    """
    inits = [k for k in cls.__mro__ if k is not object and "__init__" in vars(k)]
    if not inits:
        try:
            sig = inspect.signature(cls)
        except (TypeError, ValueError):
            return None
        params, var_kw = _collect(sig)
        return SigModel(params=params, closed=not var_kw)
    merged: dict[str, inspect.Parameter] = {}
    forwards = True  # keep absorbing bases while the previous init had **kwargs
    closed = True
    for klass in inits:  # MRO order: leaf -> base
        if not forwards:
            break
        try:
            sig = inspect.signature(klass.__init__)
        except (TypeError, ValueError):
            continue
        params, var_kw = _collect(sig)
        for name, p in params.items():
            merged.setdefault(name, p)  # leaf wins
        forwards = var_kw           # only descend further if this init has **kwargs
        closed = not var_kw         # closed iff the chain terminated without **kwargs
    return SigModel(params=merged, closed=closed)


# --------------------------------------------------------------------------
# Markdown parsing: fenced code blocks + arg tables
# --------------------------------------------------------------------------
@dataclass
class CodeBlock:
    lang: str
    start_line: int  # 1-based line of the code (first line after the fence)
    code: str


_FENCE = re.compile(r"^([ \t]*)(`{3,}|~{3,})\s*([A-Za-z0-9_+-]*)\s*$")


def extract_blocks(text: str) -> list[CodeBlock]:
    blocks: list[CodeBlock] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        indent, fence, lang = m.group(1), m.group(2), (m.group(3) or "").lower()
        body: list[str] = []
        start = i + 2  # 1-based line number of first body line
        j = i + 1
        closed = False
        while j < len(lines):
            cm = _FENCE.match(lines[j])
            if cm and cm.group(2)[0] == fence[0] and len(cm.group(2)) >= len(fence) and not cm.group(3):
                closed = True
                break
            body.append(lines[j])
            j += 1
        blocks.append(CodeBlock(lang=lang, start_line=start, code="\n".join(body)))
        i = (j + 1) if closed else j
    return blocks


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


def find_arg_tables(text: str, heading: str | None = None) -> list[ArgTable]:
    """Find markdown tables whose header first column is 'Argument'.

    If *heading* is given, only tables under a matching ``##`` heading are
    returned (used for the central common-args table).
    """
    lines = text.splitlines()
    tables: list[ArgTable] = []
    cur_heading = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        hm = re.match(r"^#{1,6}\s+(.*)$", line)
        if hm:
            cur_heading = hm.group(1).strip()
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
                    tables.append(ArgTable(heading=cur_heading, rows=rows, columns=cols))
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
    if inspect.isclass(val):
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
# Per-page audit
# --------------------------------------------------------------------------
def audit_page(path: Path, wiki_root: Path, tm: TruthModel) -> list[Finding]:
    rel = str(path.relative_to(wiki_root.parent))
    text = path.read_text(encoding="utf-8")
    findings: list[Finding] = []

    blocks = extract_blocks(text)
    page_symbols: dict[str, object] = {}  # imported name -> live object

    # --- pass 1: collect imports across all python blocks ---
    for blk in blocks:
        if blk.lang not in ("python", "py", "python3"):
            continue
        try:
            tree = ast.parse(blk.code)
        except SyntaxError:
            continue  # partial snippet; skip AST checks for this block
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    obj = resolve_symbol(node.module, alias.name)
                    local = alias.asname or alias.name
                    if node.module.startswith("radharmony"):
                        if obj is None:
                            findings.append(Finding(
                                rel, blk.start_line + (node.lineno - 1), "unknown-import",
                                "error", f"{node.module}.{alias.name}",
                                f"`from {node.module} import {alias.name}` does not resolve",
                                "Fix the import path/name or remove the reference."))
                        else:
                            page_symbols[local] = obj

    # --- pass 2: kwarg validity on constructor / factory calls ---
    sig_cache: dict[int, SigModel | None] = {}

    def get_sig(obj):
        key = id(obj)
        if key not in sig_cache:
            sig_cache[key] = signature_model(obj)
        return sig_cache[key]

    for blk in blocks:
        if blk.lang not in ("python", "py", "python3"):
            continue
        try:
            tree = ast.parse(blk.code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name):
                continue
            obj = page_symbols.get(node.func.id)
            if obj is None:
                continue
            sig = get_sig(obj)
            if sig is None or not sig.closed:
                continue  # open signature (genuine **kwargs passthrough) -> can't be sure
            for kw in node.keywords:
                if kw.arg is None:  # **kwargs splat in the call
                    continue
                if kw.arg not in sig.params:
                    findings.append(Finding(
                        rel, blk.start_line + (node.lineno - 1), "unknown-kwarg",
                        "error", node.func.id,
                        f"`{node.func.id}(...)` passes `{kw.arg}=` which is not a parameter",
                        f"Remove `{kw.arg}` or fix the name."))

    # --- pass 3: registry-key line ---
    keym = re.search(r"Registry key:\s*`\"([^\"]+)\"`", text)
    reg_key = None
    if keym:
        reg_key = keym.group(1)
        key_line = text[: keym.start()].count("\n") + 1
        under_eval = "/evaluator/" in rel
        valid = (reg_key in tm.evaluator_keys) if under_eval else \
                (reg_key in tm.evaluator_keys or reg_key in tm.dataset_keys)
        if not valid:
            findings.append(Finding(
                rel, key_line, "registry-key-mismatch", "error", reg_key,
                f'Registry key `"{reg_key}"` is not registered',
                "Update to a real key from list_datasets()/list_evaluators()."))

    # --- pass 4/5: arg-table completeness + defaults ---
    # A page can carry several arg tables for different symbols (e.g. a dataset
    # table AND a "Constructor arguments (harmonizer)" table). Resolve EACH
    # table to its own best-matching symbol rather than one page-wide primary.
    # Skip shared *reference* tables ("Common [constructor] arguments") -- they
    # document a base class and are already consumed as the inherited set;
    # mapping them to a concrete leaf would wrongly demand leaf-specific rows.
    tables = [t for t in find_arg_tables(text) if "common" not in t.heading.lower()]
    candidates: dict[str, object] = dict(page_symbols)
    if reg_key is not None:
        resolver = tm.resolve_evaluator if "/evaluator/" in rel else \
            (tm.resolve_dataset if reg_key in tm.dataset_keys else tm.resolve_evaluator)
        try:
            if resolver:
                obj = resolver(reg_key)
                candidates.setdefault(obj.__name__, obj)
        except Exception:
            pass

    def match_symbol(table: ArgTable):
        """Best-overlap symbol for a table, or None if nothing maps well."""
        args = {r.arg for r in table.rows}
        best_obj, best_sig, best_ov = None, None, 0
        for obj in candidates.values():
            sig = get_sig(obj)
            if sig is None:
                continue
            ov = len(args & set(sig.params))
            if ov > best_ov:
                best_obj, best_sig, best_ov = obj, sig, ov
        # require the symbol to cover a majority of the table's rows, so a
        # harmonizer table doesn't get mapped to the dataset on one stray name.
        if best_obj is None or best_ov < max(1, (len(args) + 1) // 2):
            return None, None
        return best_obj, best_sig

    # group tables by resolved symbol so completeness unions its tables
    by_symbol: dict[int, tuple[object, SigModel, list[ArgTable]]] = {}
    for t in tables:
        obj, sig = match_symbol(t)
        if obj is None:
            continue  # unmappable table (symbol not imported on the page) -> skip
        by_symbol.setdefault(id(obj), (obj, sig, []))[2].append(t)

    for obj, sig, tbls in by_symbol.values():
        name_of = getattr(obj, "__name__", str(obj))
        inherited = tm.common_eval_args if _is_evaluator(obj) else set()
        doc_args = {r.arg for t in tbls for r in t.rows}
        documented = doc_args | inherited
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
                    if r.arg not in inherited:
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

    # --- pass 6: bash blocks ---
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


def _pretty_default(val) -> str:
    if val is inspect.Parameter.empty:
        return "required"
    if isinstance(val, str):
        return repr(val)
    if inspect.isclass(val):
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
        pages = sorted(wiki_root.rglob("*.md"))

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
