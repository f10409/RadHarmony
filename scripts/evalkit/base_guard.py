"""Find base-code changes and check that they carry the approval text.

Base code is listed in CLAUDE.md ("Base code needs explicit approval").
A file also counts as base code when it defines a ``class Base...``.

Usage::

    python scripts/evalkit/base_guard.py --base origin/main [--pr-body FILE]
    python scripts/evalkit/base_guard.py --staged --commit-msg FILE

Exit code 1: a base file changed, and the PR body or commit message has no approval text.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root first: test this clone, not an installed copy
REPO = Path(__file__).resolve().parents[2]

# Keep this list the same as the list in CLAUDE.md. A test checks it.
BASE_FILES = [
    "radharmony/harmonizer/base.py",
    "radharmony/harmonizer/base_vqa.py",
    "radharmony/dataset/base.py",
    "radharmony/dataset/base_vqa.py",
    "radharmony/dataset/transforms.py",
    "radharmony/preprocessor/base.py",
    "radharmony/evaluator/base.py",
    "radharmony/evaluator/classification/base.py",
    "radharmony/evaluator/segmentation/base.py",
]
_BASE_CLASS = re.compile(r"^class Base\w*", re.MULTILINE)
PR_HEADING = "## ⚠️ Base code change"
COMMIT_TAG = "BASE CODE CHANGE:"


def changed_files(base: str | None = None, staged: bool = False) -> list[str]:
    """List the files changed since ``base``, or the staged files."""
    # --no-renames: a moved base file shows its old path too.
    cmd = ["git", "diff", "--name-only", "--no-renames", "--diff-filter=ACMD"]
    cmd += ["--cached"] if staged else [f"{base}...HEAD"]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=True)
    return out.stdout.split()


def base_files(files: list[str]) -> list[str]:
    """Return the files in ``files`` that are base code."""
    out = []
    for f in files:
        path = REPO / f
        if f in BASE_FILES:
            out.append(f)
        elif f.endswith(".py") and path.is_file() and _BASE_CLASS.search(path.read_text(errors="replace")):
            out.append(f)
    return out


# "Approved by: <name> on <YYYY-MM-DD>", as in CLAUDE.md. Bold markup may wrap the label.
_APPROVED = re.compile(r"Approved by\**:\**[ \t]*([^\W\d_][^<>\n]*?)[ \t]+on[ \t]+\d{4}-\d{2}-\d{2}")


def pr_body_ok(body: str) -> bool:
    """A PR body is OK when it starts with the base-code section and has an approval line.

    The line is ``Approved by: <name> on <YYYY-MM-DD>``. An empty line, the template
    text ``<name ...>`` or a line with no date is no approval.
    """
    return body.lstrip().startswith(PR_HEADING) and _APPROVED.search(body) is not None


def commit_msg_ok(msg: str, files: list[str]) -> bool:
    """A commit message is OK when it has the tag and names each base file.

    Lines that start with '#' do not count: git removes them from the message.
    """
    text = "\n".join(ln for ln in msg.splitlines() if not ln.startswith("#"))
    return COMMIT_TAG in text and all(f in text for f in files)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="origin/main", help="compare HEAD with this commit")
    ap.add_argument("--staged", action="store_true", help="check the staged files instead")
    ap.add_argument("--pr-body", help="file with the PR description")
    ap.add_argument("--commit-msg", help="file with the commit message")
    args = ap.parse_args()
    hits = base_files(changed_files(args.base, args.staged))
    if not hits:
        print("base code: no change")
        return 0
    print("base code changed: " + ", ".join(hits))
    if args.pr_body and not pr_body_ok(Path(args.pr_body).read_text()):
        print(f"FAIL: the PR description must start with '{PR_HEADING}' and have "
              "'Approved by: <name> on <YYYY-MM-DD>'.")
        return 1
    if args.commit_msg and not commit_msg_ok(Path(args.commit_msg).read_text(), hits):
        print(f"FAIL: the commit message must have '{COMMIT_TAG}' and the base file names (CLAUDE.md).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
