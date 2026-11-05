"""Check a code snippet before a user gets it.

In a RadHarmony clone, the script uses the functions of the block runner of the docs
(``scripts/evalkit/run_blocks.py``). It writes no file in the repo. Outside a clone (RadHarmony installed as a
package), it does a short check of its own with the same static rules.

- Default (static): the snippet must compile, and each radharmony import must exist. Each call to an imported
  radharmony name may use only its keyword arguments (L-005).
- ``--run``: the script also runs the snippet in a temporary folder, and prints the last 20 lines of its output.
  A missing package, data or GPU gives "blocked". In a clone, the script runs the snippet again with
  ``$RADHARMONY_MODEL_PYTHON`` when a package is missing.

Input: a ``.py`` file (one block), or, in a clone, a ``.md`` file (its fenced blocks, with their run notes).
Exit code: 0 pass, 1 fail, 2 blocked (the static checks passed, but the snippet did not run to the end).

Usage::

    python <skill folder>/scripts/check_snippet.py /tmp/snippet.py
    python <skill folder>/scripts/check_snippet.py /tmp/snippet.py --run
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path


def _is_clone(path: Path) -> bool:
    return (path / "scripts" / "evalkit" / "run_blocks.py").is_file() and (path / "docs" / "wiki").is_dir()


def _find_clone() -> Path | None:
    """Return the RadHarmony clone: the repo of this skill folder, else the git repo of the current folder.

    A user can copy the skill to ~/.claude/skills/, so the folder of the script is not always in the clone.
    """
    here = Path(__file__).resolve().parents[4]
    if _is_clone(here):
        return here
    proc = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    top = Path(proc.stdout.strip()) if proc.returncode == 0 else None
    return top if top is not None and _is_clone(top) else None


REPO = _find_clone()
IN_CLONE = REPO is not None


def _with_block_runner(src: Path, run: bool, timeout: int) -> int:
    """Check the snippet with the functions of the block runner. No file is written in the repo."""
    sys.path.insert(0, str(REPO / "scripts" / "evalkit"))
    import blocks as evalkit_blocks
    import run_blocks

    text = src.read_text(encoding="utf-8")
    if src.suffix != ".md":
        text = f"```python\n{text.rstrip()}\n```\n"
    # A .py file has the fence line above it: line n of the page is line n - 1 of the file.
    shift = 0 if src.suffix == ".md" else 1
    page_blocks = [b for b in evalkit_blocks.extract_blocks(text)
                   if b.lang in evalkit_blocks.PY_LANGS and b.opts.get("run") != "skip"]

    problems = []
    for b in page_blocks:
        code, lines = evalkit_blocks.python_source(b)
        try:
            compile(code, str(src), "exec")
        except SyntaxError as e:
            line = lines[min(e.lineno or 1, len(lines)) - 1] if lines else b.start_line
            problems.append({"line": line, "rule": "BLOCK-SYNTAX", "text": e.msg})
    if not problems:
        problems = run_blocks.call_problems(str(src), page_blocks)  # imports and keyword arguments (L-005)
    for p in problems:
        print(f"line {p['line'] - shift}: [{p['rule']}] {p['text']}")
    if problems:
        return 1
    if not run:
        return 0

    todo = [{"line": b.start_line, "name": f"{src}:{b.start_line - shift}", "code": evalkit_blocks.python_source(b)[0],
             "timeout": timeout} for b in page_blocks]
    records = _run_harness(run_blocks, todo, sys.executable)
    model_python = os.environ.get("RADHARMONY_MODEL_PYTHON", "")
    if model_python and any(run_blocks._classify(r, "full")[1].startswith("needs package") for r in records):
        print(f"A package is missing: running again with RADHARMONY_MODEL_PYTHON ({model_python}).")
        records = _run_harness(run_blocks, todo, model_python)
    results = [run_blocks._classify(r, "full") for r in records]
    for r, (result, detail) in zip(records, results, strict=True):
        print(f"line {r['line'] - shift}: {result} {detail}".rstrip())
    if len(records) < len(todo) or any(res in ("failed", "timeout") for res, _ in results):
        return 1
    return 2 if any(res != "ok" for res, _ in results) else 0


def _run_harness(run_blocks, todo: list[dict], python: str) -> list[dict]:
    """Run the blocks in a new process with the harness of the block runner, in a temporary folder.

    Print the last 20 lines of the block output, and return one record for each block that ran.
    """
    import json
    import tempfile

    env = dict(os.environ, PYTHONPATH=str(REPO), MPLBACKEND="Agg", GRADIO_ANALYTICS_ENABLED="False")
    with tempfile.TemporaryDirectory(prefix="check_snippet_") as tmp:
        rec_file = Path(tmp) / "_records.jsonl"
        try:
            proc = subprocess.run([python, "-c", run_blocks._HARNESS, str(rec_file)], input=json.dumps(todo),
                                  cwd=tmp, env=env, capture_output=True, text=True,
                                  timeout=sum(t["timeout"] for t in todo) + 60)
            out = (proc.stdout + proc.stderr).rstrip()
        except subprocess.TimeoutExpired:
            out = "the process stopped at the time limit"
        if out:
            print("--- output (last 20 lines) ---")
            print("\n".join(out.splitlines()[-20:]))
            print("---")
        lines = rec_file.read_text().splitlines() if rec_file.is_file() else []
    return [json.loads(ln) for ln in lines if ln.strip()]


def _static_only(code: str) -> list[str]:
    """The static rules of the block runner, for a machine with the package and no clone.

    This is a short copy of the checks of ``run_blocks.py``, because that script is only in a clone.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"line {e.lineno}: [BLOCK-SYNTAX] {e.msg}"]
    out, names = [], {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "radharmony":
            for a in node.names:
                try:
                    names[a.asname or a.name] = getattr(importlib.import_module(node.module), a.name)
                except (ImportError, AttributeError):
                    out.append(f"line {node.lineno}: [BLOCK-IMPORT] `from {node.module} import {a.name}` "
                               "does not resolve")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names:
            params = _keyword_names(names[node.func.id])
            if params is None:
                continue  # **kwargs that no base class closes: any name can be right
            out += [f"line {node.lineno}: [BLOCK-KWARG] `{node.func.id}(...)` passes `{kw.arg}=`, "
                    "which is not a parameter" for kw in node.keywords if kw.arg is not None and kw.arg not in params]
    return out


def _keyword_names(obj) -> set[str] | None:
    """Return the keyword names that ``obj`` takes, or None when it takes any name.

    A class that passes ``**kwargs`` to its base class takes the names of the base ``__init__`` too.
    """
    funcs = [c.__init__ for c in obj.__mro__ if "__init__" in vars(c)] if inspect.isclass(obj) else [obj]
    names: set[str] = set()
    for f in funcs:
        try:
            params = inspect.signature(f).parameters
        except (TypeError, ValueError):
            return None
        names |= {n for n, p in params.items()
                  if n != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
        if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
            return names
    return None


def _without_block_runner(src: Path, run: bool, timeout: int) -> int:
    if src.suffix == ".md":
        raise SystemExit("A .md file needs a RadHarmony clone. Give a .py file.")
    problems = _static_only(src.read_text(encoding="utf-8"))
    for p in problems:
        print(p)
    if problems or not run:
        return 1 if problems else 0
    try:
        proc = subprocess.run([sys.executable, str(src)], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"timeout: more than {timeout} s")
        return 1
    out = (proc.stdout + proc.stderr).rstrip()
    if out:
        print("--- output (last 20 lines) ---")
        print("\n".join(out.splitlines()[-20:]))
        print("---")
    if proc.returncode == 0:
        return 0
    # As run_blocks.py: a missing package, data file or GPU is "blocked", not a wrong snippet.
    last = out.splitlines()[-1] if out else ""
    missing_pkg = re.match(r"(ModuleNotFoundError|ImportError): No module named '(\w+)", last)
    if (missing_pkg and missing_pkg.group(2) != "radharmony") or last.startswith("FileNotFoundError") \
            or "CUDA" in last:
        return 2
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("snippet", help="a .py file, or a .md file with fenced blocks (clone only)")
    ap.add_argument("--run", action="store_true", help="also run the snippet")
    ap.add_argument("--timeout", type=int, default=300, help="seconds for each block, with --run")
    args = ap.parse_args()

    check = _with_block_runner if IN_CLONE else _without_block_runner
    code = check(Path(args.snippet), args.run, args.timeout)
    print({0: "PASS", 1: "FAIL", 2: "BLOCKED: the static checks passed, but the snippet did not run to the end"}[code])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
