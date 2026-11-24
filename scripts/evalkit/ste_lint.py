"""STE linter for docs prose, docstrings and comments.

It checks only the rules that a computer can check. The "follows our rules?"
review part checks the others (passive voice, one instruction per sentence).

Rules:
    STE-01  sentence with more than 25 words (a code span counts as 1 word)
    STE-02  em dash, or " -- " used as a dash
    STE-03  semicolon in prose
    STE-04  paragraph with more than 6 sentences (Markdown only)
    STE-06  word from the avoid list in ste_words.toml
    STE-10  British spelling
    STE-11  future tense (use the present tense)

Usage::

    python scripts/evalkit/ste_lint.py [FILE ...] [--json OUT]
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root first: test this clone, not an installed copy
sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocks import REPO, doc_pages, read_text  # noqa: E402
from problems import problem  # noqa: E402

CHECK = "ste"
PY_GLOBS = ["radharmony/**/*.py", "radharmony_internal/**/*.py", "scripts/**/*.py", "tests/**/*.py", "app.py",
            ".claude/skills/**/*.py"]
MD_EXTRA = ["CLAUDE.md", "eval/*.md", ".claude/skills/*/SKILL.md", ".claude/skills/*/references/**/*.md"]
# Generated or copied files that the linter does not read.
SKIP_FILES = {"docs/internal/nas_paths.md"}
SKIP_DIRS = {".ipynb_checkpoints", "legacy", "__pycache__"}

MAX_WORDS = 25
MAX_SENTENCES = 6

_WORDS = tomllib.loads((Path(__file__).with_name("ste_words.toml")).read_text())


def _word_re(words: dict) -> re.Pattern:
    alts = sorted((re.escape(w) for w in words), key=len, reverse=True)
    return re.compile(r"(?<![\w.-])(" + "|".join(alts) + r")(?![\w-])", re.IGNORECASE)


AVOID = _WORDS["avoid"]
BRITISH = _WORDS["british"]
_AVOID_RE = _word_re(AVOID)
_BRITISH_RE = _word_re(BRITISH)
_WILL_RE = re.compile(r"\b(will|won't)\b", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(\[*_])")
_WORD = re.compile(r"[A-Za-z0-9][\w'/.+-]*")


# --------------------------------------------------------------------------
# Text units: one paragraph, list item, table cell, docstring entry or comment
# --------------------------------------------------------------------------
def _clean_md(line: str) -> str:
    """Remove inline code, links, URLs and HTML from one Markdown line."""
    line = re.sub(r"`[^`]*`", " CODE ", line)
    line = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", line)
    line = re.sub(r"https?://\S+", " ", line)
    line = re.sub(r"<[^>]+>", " ", line)
    return line


def md_units(text: str) -> list[tuple[int, str, bool]]:
    """Split Markdown prose into units: (line, text, is_paragraph).

    Fenced blocks, HTML comments and front matter are skipped. A table cell is one
    unit. A heading is one unit. Only plain paragraphs count for STE-04.
    """
    text = re.sub(r"<!--.*?-->", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.DOTALL)
    lines = text.splitlines()
    units: list[tuple[int, str, bool]] = []
    cur: list[str] = []
    cur_start, cur_para = 0, False
    in_fence, fence_char = False, ""

    def flush():
        nonlocal cur
        if cur:
            units.append((cur_start, " ".join(cur), cur_para))
        cur = []

    start = 0
    if lines and lines[0].strip() == "---":  # YAML front matter
        for k in range(1, len(lines)):
            if lines[k].strip() == "---":
                start = k + 1
                break
    for idx in range(start, len(lines)):
        raw = lines[idx]
        n = idx + 1
        fm = re.match(r"^\s*(`{3,}|~{3,})", raw)
        if fm:
            if not in_fence:
                flush()
                in_fence, fence_char = True, fm.group(1)[0]
            elif fm.group(1)[0] == fence_char:
                in_fence = False
            continue
        if in_fence:
            continue
        s = raw.strip()
        if not s:
            flush()
            continue
        if s.startswith("|"):
            flush()
            if re.fullmatch(r"[|:\-\s]+", s):
                continue
            for cell in s.strip("|").split("|"):
                cell = _clean_md(cell).strip()
                if cell:
                    units.append((n, cell, False))
            continue
        heading = re.match(r"^#{1,6}\s+(.*)", s)
        item = re.match(r"^([-*+]|\d+[.)])\s+(.*)", s)
        quote = s.startswith(">")
        if heading:
            flush()
            units.append((n, _clean_md(heading.group(1)), False))
            continue
        if item or quote:
            flush()
            cur_start, cur_para = n, False
            cur.append(_clean_md(item.group(2) if item else s.lstrip("> ")))
            continue
        if not cur:
            cur_start, cur_para = n, True
        cur.append(_clean_md(s))
    flush()
    return units


_DOC_HEADER = re.compile(
    r"^\s*(Args|Arguments|Returns?|Yields?|Raises|Parameters|Attributes|Examples?|Notes?|"
    r"See Also|References|Other Parameters|Warnings?|Methods|Keyword Arguments)\s*:?\s*$")
_DOC_ENTRY = re.compile(r"^\s*\**\w+\**\s*(\([^)]*\))?\s*:(\s|$)|^\s*\w+\s*:\s*\S")
# An indented definition line: a term, 2 or more spaces, then the text.
_DOC_TERM = re.compile(r"^\s+[\w().,-]+(\s[\w().-]+)?\s{2,}\S")


def docstring_units(doc: str, first_line: int) -> list[tuple[int, str, bool]]:
    """Split a docstring into units. Headers, doctest lines and literal blocks are skipped."""
    units = []
    cur: list[str] = []
    cur_start = first_line
    literal_indent = None

    def flush():
        nonlocal cur
        if cur:
            units.append((cur_start, " ".join(cur), False))
        cur = []

    for off, raw in enumerate(doc.splitlines()):
        n = first_line + off
        s = raw.strip()
        indent = len(raw) - len(raw.lstrip())
        if literal_indent is not None:
            if s and indent <= literal_indent:
                literal_indent = None
            else:
                continue
        if not s or _DOC_HEADER.match(raw) or re.fullmatch(r"-{3,}|={3,}", s):
            flush()
            continue
        if s.startswith((">>>", "...")):
            flush()
            continue
        if _DOC_ENTRY.match(raw) or _DOC_TERM.match(raw) or re.match(r"^([-*]|\d+[.)])\s", s):
            flush()
            cur_start = n
        if not cur:
            cur_start = n
        cur.append(re.sub(r"``[^`]*``|`[^`]*`", " CODE ", s))
        if s.endswith("::"):
            flush()
            literal_indent = indent
    flush()
    return units


def _looks_like_code(text: str) -> bool:
    """Say if a comment is commented-out code (the linter skips it)."""
    if not re.search(r"[=()\[\]]", text):
        return False
    try:
        ast.parse(text.strip())
    except SyntaxError:
        return False
    return True


def py_units(source: str) -> list[tuple[int, str, bool]]:
    """Return the docstrings and comments of a Python file as units."""
    units: list[tuple[int, str, bool]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return units
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                units += docstring_units(body[0].value.value, body[0].lineno)
    group: list[str] = []
    g_start, g_line, g_col = 0, -10, -1
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError):
        return units
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        line, col = tok.start
        text = tok.string.lstrip("#").strip()
        if line == 1 and tok.string.startswith("#!") or not text \
                or re.match(r"(noqa|type:|pragma|fmt:|-\*-|ruff:|pylint:)", text) \
                or _looks_like_code(text):
            continue
        if line == g_line + 1 and col == g_col:
            group.append(text)
        else:
            if group:
                units.append((g_start, " ".join(group), False))
            group, g_start = [text], line
        g_line, g_col = line, col
    if group:
        units.append((g_start, " ".join(group), False))
    return units


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------
def _n_words(sentence: str) -> int:
    return len(_WORD.findall(sentence))


def lint_units(units, file: str, markdown: bool) -> list[dict]:
    """Apply the rules to the units of one file.

    Each problem keeps only its own sentence, so an edit to another sentence of the
    paragraph does not change the problem.
    """
    out = []
    for line, text, is_para in units:
        text = " ".join(text.split())
        if not text:
            continue
        sentences = [s for s in _SENT_SPLIT.split(text) if s.strip()]
        for s in sentences:
            n = _n_words(s)
            if n > MAX_WORDS:
                out.append(problem(CHECK, "STE-01", file, line, f"{n} words: {s[:160]}"))
            if "\u2014" in s:
                out.append(problem(CHECK, "STE-02", file, line, f"em dash: {s[:160]}"))
            elif markdown and re.search(r"\s--\s", s):
                out.append(problem(CHECK, "STE-02", file, line, f"'--' as a dash: {s[:160]}"))
            if (markdown and ";" in s) or (not markdown and re.search(r"[A-Za-z)];\s+[a-z]", s)):
                out.append(problem(CHECK, "STE-03", file, line, f"semicolon: {s[:160]}"))
            for m in _AVOID_RE.finditer(s):
                w = m.group(1)
                out.append(problem(CHECK, "STE-06", file, line, f"'{w}' (use: {AVOID[w.lower()]}): {s[:120]}"))
            for m in _BRITISH_RE.finditer(s):
                w = m.group(1)
                out.append(problem(CHECK, "STE-10", file, line, f"'{w}' (use: {BRITISH[w.lower()]}): {s[:120]}"))
            if _WILL_RE.search(s):
                out.append(problem(CHECK, "STE-11", file, line, f"'will': {s[:160]}"))
        if markdown and is_para and len(sentences) > MAX_SENTENCES:
            out.append(problem(CHECK, "STE-04", file, line,
                               f"paragraph with more than {MAX_SENTENCES} sentences: {sentences[0][:120]}"))
    return out


def lint_file(path: Path) -> list[dict]:
    """Lint one Markdown or Python file."""
    rel = path.relative_to(REPO).as_posix()
    text = read_text(rel)
    if path.suffix == ".md":
        return lint_units(md_units(text), rel, markdown=True)
    return lint_units(py_units(text), rel, markdown=False)


def target_files(files: set[str] | None = None) -> list[Path]:
    """List the files the linter reads. ``files`` limits the list."""
    found = set(doc_pages())
    for pattern in PY_GLOBS + MD_EXTRA:
        found.update(REPO.glob(pattern))
    out = []
    for p in sorted(found):
        rel = p.relative_to(REPO).as_posix()
        if rel in SKIP_FILES or SKIP_DIRS & set(p.parts) or not p.is_file():
            continue
        if files is not None and rel not in files:
            continue
        out.append(p)
    return out


def run(files: set[str] | None = None) -> list[dict]:
    """Lint every target file and return the problems."""
    out = []
    for p in target_files(files):
        out += lint_file(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*", help="files to lint (default: all target files)")
    ap.add_argument("--json", help="write the problems as JSON to this file")
    args = ap.parse_args()
    if args.files:
        problems = []
        for f in args.files:
            problems += lint_file(Path(f).resolve())
    else:
        problems = run()
    if args.json:
        Path(args.json).write_text(json.dumps(problems, indent=1))
    for p in problems:
        print(f"{p['file']}:{p['line']}: [{p['rule']}] {p['text']}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
