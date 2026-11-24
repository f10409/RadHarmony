"""Check the numbers in the docs that have a claim note.

A claim note is an HTML comment before the paragraph that states the number::

    <!-- claim: id=padchest_gr.n_images tier=nas cmd="python scripts/x.py --count" expect="4,555" -->

The check runs ``cmd`` with no shell, from the repo root. It passes when the
output (stripped) equals ``expect`` and the next paragraph contains ``expect``.
``tier=ci`` claims run in every run. ``tier=nas`` claims run only in a full run.

A number from a paper has no command. Its note names the paper in ``source`` in place of
``cmd``. The check then only confirms that ``expect`` is in the next paragraph::

    <!-- claim: id=brax.paper.patients source="Reis et al., Scientific Data 2022" expect="19,351" -->

Usage::

    python scripts/evalkit/claims.py [--full] [--unclaimed]
"""
from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root first: test this clone, not an installed copy
sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocks import REPO, doc_pages, parse_note, read_text  # noqa: E402
from problems import problem  # noqa: E402

CHECK = "claims"
CLAIM_TIMEOUT = 600
_CLAIM = re.compile(r"<!--\s*claim:\s*(.*?)\s*-->")
_BIG_NUMBER = re.compile(r"(?<![\w.,])\d{1,3}(,\d{3})+(?![\w,])")


def find_claims(text: str) -> list[tuple[int, dict, str]]:
    """Return (line, note, next paragraph) for each claim note in ``text``."""
    lines = text.splitlines()
    out = []
    for i, ln in enumerate(lines):
        m = _CLAIM.search(ln)
        if not m:
            continue
        para = []
        for nxt in lines[i + 1:]:
            if not nxt.strip():
                if para:
                    break
                continue
            para.append(nxt)
        out.append((i + 1, parse_note(m.group(1)), " ".join(para)))
    return out


def check_claim(rel: str, line: int, note: dict, para: str) -> list[dict]:
    """Run one claim and return its problems."""
    cid = note.get("id", "?")
    missing = [k for k in ("id", "expect") if k not in note]
    if ("cmd" in note) == ("source" in note):
        missing.append("cmd or source (one of them)")
    if "_error" in note or missing:
        return [problem(CHECK, "CLAIM-NOTE", rel, line, f"claim {cid}: bad note {note.get('_error', missing)}")]
    out = []
    # A whole-number match: 14,555 must not match the expected 4,555.
    if not re.search(rf"(?<![\w,.]){re.escape(note['expect'])}(?![\w]|[,.]\d)", para):
        out.append(problem(CHECK, "CLAIM-TEXT", rel, line,
                           f"claim {cid}: '{note['expect']}' is not in the next paragraph"))
    if "source" in note:  # a number from a paper: no command to run
        return out
    try:
        cmd = shlex.split(note["cmd"])
    except ValueError as e:
        cmd, out = [], out + [problem(CHECK, "CLAIM-NOTE", rel, line, f"claim {cid}: bad cmd: {e}")]
    if not cmd:
        return out or [problem(CHECK, "CLAIM-NOTE", rel, line, f"claim {cid}: empty cmd")]
    if cmd[0] == "python":
        cmd[0] = sys.executable
    try:
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=CLAIM_TIMEOUT)
        got = proc.stdout.strip()
        if proc.returncode == 3 and "dataset_stat.py" in note["cmd"]:
            pass  # no path in the data paths file of this clone: the claim cannot run here
        elif proc.returncode != 0:
            out.append(problem(CHECK, "CLAIM-ERROR", rel, line,
                               f"claim {cid}: exit code {proc.returncode}: {proc.stderr.strip()[-200:]}"))
        elif got != note["expect"]:
            out.append(problem(CHECK, "CLAIM-WRONG", rel, line,
                               f"claim {cid}: the docs say {note['expect']}, the command gives {got[:80]}"))
    except subprocess.TimeoutExpired:
        out.append(problem(CHECK, "CLAIM-ERROR", rel, line, f"claim {cid}: more than {CLAIM_TIMEOUT} s"))
    except OSError as e:
        out.append(problem(CHECK, "CLAIM-ERROR", rel, line, f"claim {cid}: cannot run the command: {e}"))
    return out


def unclaimed_numbers(page: Path) -> list[tuple[int, str]]:
    """List big numbers in prose (for example 4,555) with no claim note above them."""
    text = read_text(page.relative_to(REPO).as_posix())
    claimed = {e for _, n, _ in find_claims(text) for e in [n.get("expect")] if e}
    out, in_fence = [], False
    for i, ln in enumerate(text.splitlines(), 1):
        if re.match(r"^\s*(`{3,}|~{3,})", ln):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in _BIG_NUMBER.finditer(re.sub(r"`[^`]*`", "", ln)):
            if m.group(0) not in claimed:
                out.append((i, m.group(0)))
    return out


def run(full: bool = False, files: set[str] | None = None) -> list[dict]:
    """Check every claim that the tier allows."""
    out = []
    for page in doc_pages(files):
        rel = page.relative_to(REPO).as_posix()
        for line, note, para in find_claims(read_text(rel)):
            if note.get("tier", "ci") == "nas" and not full:
                continue
            out += check_claim(rel, line, note, para)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="also run tier=nas claims")
    ap.add_argument("--unclaimed", action="store_true", help="list big numbers with no claim")
    args = ap.parse_args()
    if args.unclaimed:
        for page in doc_pages():
            for line, num in unclaimed_numbers(page):
                print(f"{page.relative_to(REPO)}:{line}: {num}")
        return 0
    problems = run(args.full)
    for p in problems:
        print(f"{p['file']}:{p['line']}: [{p['rule']}] {p['text']}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
