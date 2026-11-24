"""Lesson checks: rules that we added after a real mistake.

Each rule has a row in ``eval/RULES.md`` with the commit that made the mistake.
A rule that needs the package or data is a test in ``tests/test_lessons.py`` instead.

    L-002  no NAS paths (/mnt/NAS) in the public docs and the skills (.claude/skills/)
    L-006  no NAS paths in code, scripts or tests (use configs/data_paths.json)
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root first: test this clone, not an installed copy
sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocks import REPO, doc_pages, read_text  # noqa: E402
from problems import problem  # noqa: E402

CHECK = "lessons"
_NAS = re.compile(r"/mnt/NAS\d*")
# The config and its reader hold the paths. The generated page shows them on purpose.
# The other files hold a pattern that finds NAS paths, not a path.
NAS_ALLOWED = {"configs/data_paths.json", "scripts/dataset_paths.py", "docs/internal/nas_paths.md",
               "scripts/evalkit/lessons.py", "tests/test_lessons.py", "tests/test_configs.py",
               "scripts/legacy/public_manifest.toml", "scripts/legacy/PUBLISHING.md"}
CODE_DIRS = ["radharmony", "radharmony_internal", "scripts", "tests", "app.py"]


def _git_files(paths: list[str]) -> list[str]:
    out = subprocess.run(["git", "ls-files", "--", *paths], cwd=REPO, capture_output=True, text=True)
    return [f for f in out.stdout.splitlines() if f.endswith((".py", ".md", ".toml", ".sh"))]


def l002_files() -> list[str]:
    """The files that L-002 reads: the doc pages and the skills (D-21). Agents read and copy the skills."""
    docs = [p.relative_to(REPO).as_posix() for p in doc_pages()]
    return docs + [f for f in _git_files([".claude/skills"]) if f not in docs]


def nas_paths(files: set[str] | None = None) -> list[dict]:
    """L-002 and L-006: find /mnt/NAS paths outside the data-paths config."""
    docs = l002_files()
    code = _git_files(CODE_DIRS)
    out = []
    for rule, group in (("L-002", docs), ("L-006", code)):
        for rel in group:
            if rel in NAS_ALLOWED or (files is not None and rel not in files):
                continue
            if not (REPO / rel).is_file():
                continue
            for i, ln in enumerate(read_text(rel).splitlines(), 1):
                m = _NAS.search(ln)
                if m:
                    out.append(problem(CHECK, rule, rel, i, f"NAS path: {ln.strip()[:120]}"))
    return out


def run(files: set[str] | None = None) -> list[dict]:
    """Run every lesson check."""
    return nas_paths(files)


if __name__ == "__main__":
    for p in run():
        print(f"{p['file']}:{p['line']}: [{p['rule']}] {p['text']}")
