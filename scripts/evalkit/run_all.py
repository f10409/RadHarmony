"""Run the automatic checks: lint, docs and test.

    lint   ruff (bug rules) and the STE linter
    docs   wiki audit, doc code blocks, number claims, lesson checks
    test   pytest, and the base-code check

Run types:
    --fast   staged files only, no pytest (the commit hook)
    (none)   quick run: all files, no data and no GPU needed (CI)
    --full   full run: also runs the doc code blocks and the NAS claims

It prints one line when each check ends, then a table. The base check compares HEAD with its
merge-base with ``--base`` (default ``origin/main``).

A run fails only on a problem that is not on the known-problems list
(``eval/known_problems/``), on a failed test, or on a check that crashed.

Usage::

    python scripts/evalkit/run_all.py [--fast | --full] [--group lint,docs,test] [--out DIR] [--base REF]
    python scripts/evalkit/run_all.py --update-known      # write the lists (only shrink them)
    python scripts/evalkit/run_all.py --check-shrink HEAD [--new-rule L-013] [--pr-body FILE]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root first: test this clone, not an installed copy
sys.path.insert(0, str(Path(__file__).resolve().parent))
import base_guard  # noqa: E402
import claims  # noqa: E402
import lessons  # noqa: E402
import run_blocks  # noqa: E402
import blocks  # noqa: E402
import ste_lint  # noqa: E402
from blocks import read_text  # noqa: E402
from problems import REPO, compare, lists_at, new_rules_in, problem, save_known, shrink_errors  # noqa: E402

PY_TARGETS = ["radharmony", "radharmony_internal", "scripts", "tests", "app.py", ".claude/skills"]
# These lists change only in a full run, so only a full run may write them.
FULL_ONLY_LISTS = {"blocks_run", "claims"}


def staged_files() -> set[str]:
    out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
                         cwd=REPO, capture_output=True, text=True, check=True)
    return set(out.stdout.split())


# --------------------------------------------------------------------------
# One function per check. Each returns (problems, info). Info is a short text.
# --------------------------------------------------------------------------
def _ruff(args: list[str], stdin: str | None = None) -> list[dict]:
    ruff = shutil.which("ruff", path=str(Path(sys.executable).parent)) or shutil.which("ruff")
    if not ruff:
        raise RuntimeError("ruff is not installed (uv pip install ruff)")
    # --force-exclude: the exclude list in pyproject.toml also applies to named files.
    proc = subprocess.run([ruff, "check", "--output-format", "json", "--exit-zero", "--force-exclude", *args],
                          cwd=REPO, capture_output=True, text=True, input=stdin)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[-300:])
    return [problem("ruff", h["code"] or "syntax", h["filename"], h["location"]["row"], h["message"])
            for h in json.loads(proc.stdout or "[]")]


def check_ruff(files):
    if files is None:
        # The public repo has no radharmony_internal/, so skip a target that is not in this clone.
        return _ruff([t for t in PY_TARGETS if (REPO / t).exists()]), ""
    # Commit run: lint the staged copy of each staged Python file in the targets.
    targets = sorted(f for f in files if f.endswith(".py")
                     and any(f == t or f.startswith(t + "/") for t in PY_TARGETS))
    out = []
    for rel in targets:
        out += _ruff(["--stdin-filename", rel, "-"], stdin=read_text(rel))
    return out, "" if targets else "no Python files"


def check_ste(files):
    return ste_lint.run(files), ""


def check_wiki(files):
    with tempfile.TemporaryDirectory() as tmp:
        wiki = REPO / "docs" / "wiki"
        if files is not None:
            # Commit run: audit a copy of the wiki that has the staged pages.
            staged = [f for f in files if f.startswith("docs/wiki/") and f.endswith(".md")]
            if not staged:
                return [], "no wiki pages"
            wiki = Path(tmp) / "wiki"
            shutil.copytree(REPO / "docs" / "wiki", wiki)
            for rel in staged:
                (wiki / rel.removeprefix("docs/wiki/")).write_text(read_text(rel), encoding="utf-8")
        out_json = Path(tmp) / "wiki.json"
        proc = subprocess.run([sys.executable, str(REPO / "scripts/evalkit/wiki_audit.py"),
                               "--wiki", str(wiki), "--json", str(out_json)], cwd=REPO,
                              capture_output=True, text=True)
        if not out_json.is_file():
            raise RuntimeError(proc.stderr.strip()[-300:])
        findings = json.loads(out_json.read_text())
    out = []
    for f in findings:
        rel = f"docs/{f['page']}"  # the audit gives the page relative to docs/
        if files is None or rel in files:
            out.append(problem("wiki", f["type"], rel, f["line"], f"{f['symbol']}: {f['detail']}"))
    return out, ""


def check_blocks(files):
    probs, _ = run_blocks.run("static", files)
    return probs, ""


def make_check_blocks_run(level):
    def check_blocks_run(files):
        _, records = run_blocks.run(level, files)  # static problems are in the "blocks" check
        probs = [problem("blocks_run", "BLOCK-RUN", r["page"], r["line"], f"block: {r['detail']}")
                 for r in records if r["result"] in ("failed", "timeout")]
        counts = {}
        for r in records:
            counts[r["result"]] = counts.get(r["result"], 0) + 1
        check_blocks_run.records = records
        return probs, ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "no block to run"
    return check_blocks_run


def make_check_claims(full):
    def check_claims(files):
        return claims.run(full, files), ""
    return check_claims


def check_lessons(files):
    return lessons.run(files), ""


def check_pytest(files):
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/"],
                          cwd=REPO, capture_output=True, text=True)
    tail = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1:] or ["no output"]
    check_pytest.output = proc.stdout[-20000:]
    if proc.returncode != 0:
        failed = [ln for ln in proc.stdout.splitlines() if ln.startswith("FAILED")]
        return [problem("pytest", "TEST", ln.split()[1].split("::")[0] if len(ln.split()) > 1 else "tests",
                        0, ln) for ln in failed] or [problem("pytest", "TEST", "tests", 0, tail[0])], tail[0]
    return [], tail[0]


def make_check_base(staged, base="origin/main"):
    """Return the base-code check. It compares HEAD with its merge-base with ``base``, not with the tip."""
    def check_base(files):
        try:
            changed = base_guard.changed_files(base, staged)
        except subprocess.CalledProcessError:
            return [], f"no {base} to compare with"
        if staged:
            since = "staged"
        else:  # ``base...HEAD`` in changed_files already starts at the merge-base. Name it in the info.
            mb = subprocess.run(["git", "merge-base", base, "HEAD"], cwd=REPO, capture_output=True, text=True)
            since = f"since the merge-base of HEAD and {base} ({mb.stdout.strip()[:8] or '?'})"
        hits = base_guard.base_files(changed)
        return [], (f"base code changed {since}: " + ", ".join(hits)) if hits else f"no base-code change {since}"
    return check_base


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def build_checks(args):
    """Return [(group, check name, function, uses the known-problems list)]."""
    run_level = "full" if args.full else "ci"
    checks = [
        ("lint", "ruff", check_ruff, True),
        ("lint", "ste", check_ste, True),
        ("docs", "wiki", check_wiki, True),
        ("docs", "blocks", check_blocks, True),
        ("docs", "claims", make_check_claims(args.full), True),
        ("docs", "lessons", check_lessons, True),
        ("test", "base", make_check_base(args.fast, args.base), False),
    ]
    if not args.fast:
        checks.insert(4, ("docs", "blocks_run", make_check_blocks_run(run_level), True))
        checks.append(("test", "pytest", check_pytest, False))
    groups = set(args.group.split(","))
    return [c for c in checks if c[0] in groups]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true", help="staged files only, no pytest")
    mode.add_argument("--full", action="store_true", help="also run doc blocks and NAS claims")
    ap.add_argument("--group", default="lint,docs,test", help="comma list of groups")
    ap.add_argument("--out", help="write results.json and summary.md in this folder")
    ap.add_argument("--base", default="origin/main", metavar="REF",
                    help="the base check lists the base files changed since the merge-base of HEAD and REF")
    ap.add_argument("--update-known", action="store_true",
                    help="write the current problems as the known-problems lists")
    ap.add_argument("--check-shrink", metavar="REF",
                    help="fail if a known-problems list grew since this git commit")
    ap.add_argument("--new-rule", action="append", default=[], metavar="RULE",
                    help="with --check-shrink: a new rule whose old problems may join a list (repeat)")
    ap.add_argument("--pr-body", metavar="FILE",
                    help="with --check-shrink: also take the 'New rule: <ID>' lines of this PR description")
    args = ap.parse_args(argv)
    unknown = set(args.group.split(",")) - {"lint", "docs", "test"}
    if unknown:
        ap.error(f"unknown group: {', '.join(sorted(unknown))}. Use lint, docs or test.")

    if args.check_shrink:
        new_rules = args.new_rule + (new_rules_in(Path(args.pr_body).read_text()) if args.pr_body else [])
        errors = shrink_errors(args.check_shrink, new_rules)
        for e in errors:
            print("grew:", e)
        print(f"known-problems lists: {'grew' if errors else 'did not grow'} since {args.check_shrink}")
        if not errors and lists_at(args.check_shrink) == 0:
            print(f"WARNING: {args.check_shrink} has no known-problems lists, so nothing was compared. "
                  "Before a commit, compare with HEAD.")
        return 1 if errors else 0

    files = staged_files() if args.fast else None
    blocks.STAGED = args.fast  # a commit run checks the staged copy of each file
    if args.fast and not files:
        print("no staged files")
        return 0

    rows, results, failed = [], {}, False
    for group, name, fn, use_known in build_checks(args):
        t0 = time.time()
        try:
            probs, info = fn(files)
        except Exception as e:  # report the crash and go on with the other checks
            rows.append((group, name, "-", "-", "-", "-", "CRASH", str(e)[:120]))
            results[name] = {"group": group, "crash": str(e)}
            failed = True
            print(f"{group}/{name}: CRASH ({time.time() - t0:.0f} s) {str(e)[:120]}", flush=True)
            continue
        new, fixed = compare(name, probs, files) if use_known else (probs, 0)
        if name in FULL_ONLY_LISTS and not args.full:
            fixed = "-"  # a quick run checks less than the list holds
        bad = bool(new)
        failed |= bad
        known = len(probs) - len(new) if use_known else "-"
        rows.append((group, name, len(probs), known, len(new), fixed, "FAIL" if bad else "pass",
                     f"{info} ({time.time() - t0:.0f} s)".strip()))
        # One line when each check ends, so a long run shows its progress.
        print(f"{group}/{name}: {'FAIL' if bad else 'pass'} ({len(new)} new, {time.time() - t0:.0f} s)", flush=True)
        results[name] = {"group": group, "problems": probs, "new": new, "fixed": fixed, "info": info}
        if name == "blocks_run":
            results[name]["records"] = getattr(fn, "records", [])
        if name == "pytest":
            results[name]["output"] = getattr(fn, "output", "")
        if args.update_known and use_known and not args.fast and (args.full or name not in FULL_ONLY_LISTS):
            save_known(name, probs)

    head = ("group", "check", "problems", "known", "new", "fixed", "result", "info")
    table = [head] + [tuple(str(x) for x in r) for r in rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(head) - 1)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r[:-1], widths, strict=True)) + "  " + r[-1] for r in table]
    print()
    print("\n".join(lines))
    for name, res in results.items():
        for p in res.get("new", [])[:30]:
            print(f"NEW {name}: {p['file']}:{p['line']}: [{p['rule']}] {p['text'][:160]}")
    verdict = "FAIL" if failed else "PASS"
    print(f"\nautomatic checks: {verdict}")
    if args.update_known:
        print("known-problems lists written. Run --check-shrink HEAD before you commit them.")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(results, indent=1))
        md = ["# Automatic checks", "", f"Result: **{verdict}**", "", "```", *lines, "```", ""]
        (out / "summary.md").write_text("\n".join(md))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
