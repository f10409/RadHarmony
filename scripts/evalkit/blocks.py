"""Read fenced code blocks from Markdown, with their run notes.

A run note is an HTML comment on the line before the fence::

    <!-- block: run=ci|local|gpu|skip data=<key> timeout=300 reason="..." -->

A page note sets the default for every block on the page::

    <!-- blocks: run=local data=chexpert_train -->

A block with no note gets ``run=static``: the checks compile it but do not run it.
"""
from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# True in a commit run (run_all.py --fast): the checks read the staged copy of each file.
STAGED = False

# Markdown files that the docs checks read. Generated copies and drafts are left out.
DOC_GLOBS = ["docs/wiki/**/*.md", "README.md", "CONTRIBUTING.md"]
SKIP_PARTS = {".ipynb_checkpoints"}

RUN_LEVELS = {"static", "ci", "local", "gpu", "skip"}
PY_LANGS = {"python", "py", "python3", "pycon"}


@dataclass
class CodeBlock:
    lang: str
    start_line: int  # 1-based line of the code (first line after the fence)
    code: str
    opts: dict = field(default_factory=dict)


_FENCE = re.compile(r"^([ \t]*)(`{3,}|~{3,})\s*([A-Za-z0-9_+-]*)\s*$")
_BLOCK_NOTE = re.compile(r"^\s*<!--\s*block:\s*(.*?)\s*-->\s*$")
_PAGE_NOTE = re.compile(r"^\s*<!--\s*blocks:\s*(.*?)\s*-->\s*$", re.MULTILINE)


def parse_note(body: str) -> dict:
    """Parse ``k=v k2="a b"`` into a dict. A bad note gives {"_error": message}."""
    try:
        parts = shlex.split(body)
    except ValueError as e:
        return {"_error": str(e)}
    opts = {}
    for part in parts:
        if "=" not in part:
            return {"_error": f"no '=' in {part!r}"}
        k, v = part.split("=", 1)
        opts[k] = v
    return opts


def extract_blocks(text: str) -> list[CodeBlock]:
    """Return every fenced block in ``text``, with the page note and its own note merged."""
    m = _PAGE_NOTE.search(text)
    page_opts = parse_note(m.group(1)) if m else {}
    blocks: list[CodeBlock] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        indent, fence, lang = m.group(1), m.group(2), (m.group(3) or "").lower()
        opts = dict(page_opts)
        k = i - 1
        while k >= 0 and not lines[k].strip():
            k -= 1
        note = _BLOCK_NOTE.match(lines[k]) if k >= 0 else None
        if note:
            opts.update(parse_note(note.group(1)))
        body: list[str] = []
        start = i + 2
        j = i + 1
        closed = False
        while j < len(lines):
            cm = _FENCE.match(lines[j])
            if cm and cm.group(2)[0] == fence[0] and len(cm.group(2)) >= len(fence) and not cm.group(3):
                closed = True
                break
            # A fence inside a list item is indented. Remove that indent from the code.
            body.append(lines[j][len(indent):] if lines[j].startswith(indent) else lines[j].lstrip())
            j += 1
        blocks.append(CodeBlock(lang=lang, start_line=start, code="\n".join(body), opts=opts))
        i = (j + 1) if closed else j
    return blocks


def python_source(block: CodeBlock) -> tuple[str, list[int]]:
    """Return the Python code of a block, and the Markdown line of each code line.

    In a pycon block (or a block that starts with ``>>>``), only the prompt lines stay,
    without the prompt. So the code line ``n`` (1-based) is on the Markdown line ``lines[n - 1]``.
    """
    rows = list(enumerate(block.code.splitlines()))
    if block.lang == "pycon" or block.code.lstrip().startswith(">>>"):
        rows = [(i, ln[4:]) for i, ln in rows if ln.startswith((">>> ", "... "))]
    return "\n".join(text for _, text in rows), [block.start_line + i for i, _ in rows]


def read_text(rel: str) -> str:
    """Read a repo file (path relative to the repo). With STAGED, read the staged copy."""
    if STAGED:
        proc = subprocess.run(["git", "show", f":{rel}"], cwd=REPO, capture_output=True)
        if proc.returncode == 0:
            return proc.stdout.decode("utf-8", errors="replace")
    return (REPO / rel).read_text(encoding="utf-8", errors="replace")


def doc_pages(files: set[str] | None = None) -> list[Path]:
    """List the Markdown pages that the docs checks read.

    ``files`` (repo-relative paths) limits the list, for a run on staged files.
    """
    pages = []
    for pattern in DOC_GLOBS:
        for p in sorted(REPO.glob(pattern)):
            rel = p.relative_to(REPO).as_posix()
            if SKIP_PARTS & set(p.parts) or (files is not None and rel not in files):
                continue
            pages.append(p)
    return pages
