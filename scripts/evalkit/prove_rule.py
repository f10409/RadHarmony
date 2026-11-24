"""Prove that a lesson check catches its mistake.

It makes two temporary git worktrees: one at the commit with the mistake (--bad)
and one at the commit with the fix (--good). It copies the check files into
both (the check is usually newer than the mistake), then runs pytest. The check
must FAIL at --bad and PASS at --good.

Usage::

    python scripts/evalkit/prove_rule.py --bad 95bb587^ --good 95bb587 \\
        --with tests/test_lessons.py --with scripts/dataset_paths.py --with configs/data_paths.json \\
        -- tests/test_lessons.py::test_l001_chexpert_age_years_is_filled
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root first: test this clone, not an installed copy
REPO = Path(__file__).resolve().parents[2]


def run_at(ref: str, copies: list[str], pytest_args: list[str], tmp: Path) -> int:
    """Run pytest in a worktree at ``ref``. Returns the pytest exit code."""
    wt = tmp / ref.replace("/", "_").replace("^", "_parent")
    subprocess.run(["git", "worktree", "add", "--detach", "-q", str(wt), ref], cwd=REPO, check=True)
    try:
        for rel in copies:
            dst = wt / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / rel, dst)
        env = dict(os.environ, PYTHONPATH=str(wt))
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *pytest_args],
                              cwd=wt, env=env, capture_output=True, text=True)
        tail = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1:]
        print(f"{ref}: exit {proc.returncode}: {tail[0] if tail else ''}")
        return proc.returncode
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=REPO, check=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bad", required=True, help="commit with the mistake")
    ap.add_argument("--good", required=True, help="commit with the fix")
    ap.add_argument("--with", dest="copies", action="append", default=[],
                    help="repo file to copy into both worktrees (repeat)")
    ap.add_argument("pytest_args", nargs="+", help="pytest arguments, after --")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="prove_rule_") as tmp:
        bad = run_at(args.bad, args.copies, args.pytest_args, Path(tmp))
        good = run_at(args.good, args.copies, args.pytest_args, Path(tmp))
    ok = bad == 1 and good == 0
    print("PROVED: the check fails on the mistake and passes on the fix" if ok
          else "NOT PROVED: expected exit 1 at --bad and exit 0 at --good")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
