"""Lessons that the ruff config in ``pyproject.toml`` checks (``run_all.py --group lint``).

L-020: in ``radharmony/harmonizer/``, an ``except`` does not end in a silent ``pass``
(S110) or ``continue`` (S112). A file that cannot be read must warn with a count
(R-code-19, R-code-20).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUFF = shutil.which("ruff", path=str(Path(sys.executable).parent)) or shutil.which("ruff")
SILENT = "try:\n    open('x')\nexcept Exception:\n    pass\n"


def _codes(*args, stdin=None) -> list[str]:
    proc = subprocess.run([RUFF, "check", "--output-format", "json", "--exit-zero", *args], cwd=ROOT,
                          input=stdin, capture_output=True, text=True, check=True)
    return [h["code"] for h in json.loads(proc.stdout)]


@pytest.mark.skipif(RUFF is None, reason="ruff is not installed")
def test_l020_no_silent_except_in_harmonizer_code():
    """L-020: ruff S110 and S112 find no silent except in radharmony/harmonizer/."""
    # The config turns the rules on for the harmonizer folder only.
    assert "S110" in _codes("--stdin-filename", "radharmony/harmonizer/_probe.py", "-", stdin=SILENT)
    assert "S110" not in _codes("--stdin-filename", "radharmony/utils/_probe.py", "-", stdin=SILENT)
    # The harmonizer code has no silent except.
    hits = [c for c in _codes("radharmony/harmonizer") if c in ("S110", "S112")]
    assert not hits, f"{len(hits)} silent except in radharmony/harmonizer/ (L-020)"
