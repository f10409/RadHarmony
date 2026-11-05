"""Check and run the Python code blocks in the docs.

Levels:
    static  every Python block must compile. A run note must be valid. Each radharmony
            import must exist, and each call to an imported name may use only its
            keyword arguments (L-005).
    ci      also run the blocks with ``run=ci`` (no data, no GPU). Every error fails.
    full    also run every other Python block that is not ``skip``, page by page,
            in a new process, with a time limit. A block that stops because data,
            a package or a GPU is missing is BLOCKED, not failed.

The blocks of one page run in order and share their names, as a reader runs them.
After the first block that does not pass, the next blocks on the page are "not run".
Blocks run in a temporary folder. The note ``cwd=repo`` runs the page from the repo root.

Usage::

    python scripts/evalkit/run_blocks.py --level static|ci|full [--page GLOB] [--json OUT]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root first: test this clone, not an installed copy
sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocks import PY_LANGS, REPO, RUN_LEVELS, doc_pages, extract_blocks, python_source, read_text  # noqa: E402
from problems import problem  # noqa: E402
from signatures import page_imports, signature_model  # noqa: E402

CHECK = "blocks"
BLOCK_TIMEOUT = 120
PAGE_TIMEOUT = 900
# The model venv has the packages that the dev venv does not have (transformers, timm, ...).
# Set RADHARMONY_MODEL_PYTHON to the Python of a venv with the backbone packages. With no value, no rerun.
MODEL_PYTHON = os.environ.get("RADHARMONY_MODEL_PYTHON", "")
# Modules from this repo. A missing module that is not one of these is a missing package.
REPO_MODULES = {"radharmony", "radharmony_internal", "dataset_paths", "app"}


def _source(block) -> str:
    """Return the code of a block. Prompts (``>>>``) are removed from pycon blocks."""
    return python_source(block)[0]


def _timeout(block) -> int | None:
    try:
        return int(block.opts.get("timeout", BLOCK_TIMEOUT))
    except ValueError:
        return None


def call_problems(rel: str, blocks: list) -> list[dict]:
    """Check the radharmony imports of a page, and the keyword arguments of each call to them (L-005).

    An import anywhere on the page counts for every block of the page. A block that does not
    compile is left out (BLOCK-SYNTAX reports it). A class or function with ``**kwargs`` can take
    any name, so its calls are not checked.
    """
    names, bad = page_imports(blocks)
    out = [problem(CHECK, "BLOCK-IMPORT", rel, line, f"`from {module} import {name}` does not resolve")
           for line, module, name in bad]
    for b in blocks:
        if b.lang not in PY_LANGS:
            continue
        code, lines = python_source(b)
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names):
                continue
            sig = signature_model(names[node.func.id])
            if sig is None or not sig.closed:
                continue
            for kw in node.keywords:
                if kw.arg is not None and kw.arg not in sig.params:
                    out.append(problem(CHECK, "BLOCK-KWARG", rel, lines[node.lineno - 1],
                                       f"`{node.func.id}(...)` passes `{kw.arg}=`, which is not a parameter"))
    return out


def static_problems(page: Path) -> list[dict]:
    """Compile each Python block, check the run notes, the imports and the call arguments."""
    rel = page.relative_to(REPO).as_posix()
    out = []
    page_blocks = extract_blocks(read_text(rel))
    for b in page_blocks:
        note_line = b.start_line - 1
        if "_error" in b.opts:
            out.append(problem(CHECK, "BLOCK-NOTE", rel, note_line, f"bad run note: {b.opts['_error']}"))
        run = b.opts.get("run", "static")
        if run not in RUN_LEVELS:
            out.append(problem(CHECK, "BLOCK-NOTE", rel, note_line, f"unknown run level {run!r}"))
        if run == "skip" and not b.opts.get("reason"):
            out.append(problem(CHECK, "BLOCK-NOTE", rel, note_line, "run=skip with no reason"))
        if _timeout(b) is None:
            out.append(problem(CHECK, "BLOCK-NOTE", rel, note_line,
                               f"timeout must be a number of seconds, not {b.opts['timeout']!r}"))
        if b.lang not in PY_LANGS or run == "skip":
            continue
        code, lines = python_source(b)
        try:
            compile(code, f"{rel}:{b.start_line}", "exec")
        except SyntaxError as e:
            # The error can be 1 line after the last code line (an open bracket).
            line = lines[min((e.lineno or 1), len(lines)) - 1] if lines else b.start_line
            out.append(problem(CHECK, "BLOCK-SYNTAX", rel, line,
                               f"block at line {b.start_line}: {e.msg}: {(e.text or '').strip()[:80]}"))
    return out + call_problems(rel, page_blocks)


# The child process runs this harness. It reads the blocks as JSON on stdin. It writes
# one JSON line per block to the file in argv[1], apart from the output of the blocks.
_HARNESS = r'''
import json, signal, sys
blocks = json.load(sys.stdin)
out = open(sys.argv[1], "w", buffering=1)
ns = {"__name__": "__main__"}
class _Timeout(BaseException):
    pass
def _alarm(*_):
    raise _Timeout()
signal.signal(signal.SIGALRM, _alarm)
failed_at = None
for b in blocks:
    if failed_at is not None:
        rec = {"status": "not_run", "detail": f"depends on line {failed_at}"}
    elif b.get("blocked"):
        rec = {"status": "blocked", "detail": b["blocked"]}
    else:
        signal.alarm(b["timeout"])
        try:
            exec(compile(b["code"], b["name"], "exec"), ns)
            rec = {"status": "ok"}
        except _Timeout:
            rec = {"status": "timeout", "detail": f"more than {b['timeout']} s"}
        except BaseException as e:
            msg = (str(e).splitlines() or [""])[0][:300]
            # Repo code can raise its own ImportError from the one of a missing package: take that name too.
            mod = getattr(e, "name", None) or getattr(e.__cause__, "name", None) or ""
            rec = {"status": "error", "exc": type(e).__name__, "msg": msg, "module": mod}
        finally:
            signal.alarm(0)
    rec["line"] = b["line"]
    out.write(json.dumps(rec) + "\n")
    if rec["status"] != "ok" and failed_at is None:
        failed_at = b["line"]
'''


def _classify(rec: dict, level: str) -> tuple[str, str]:
    """Turn a harness record into (result, message).

    Result is one of ok, failed, blocked, timeout, not_run.
    At the ci level every error fails: a CI block must not need data, a package or a GPU.
    """
    if rec["status"] != "error":
        return rec["status"], rec.get("detail", "")
    exc, msg, mod = rec["exc"], rec["msg"], rec["module"].split(".")[0]
    if level == "ci":
        return "failed", f"{exc}: {msg}"
    if exc in ("ModuleNotFoundError", "ImportError") and mod and mod not in REPO_MODULES:
        return "blocked", f"needs package {mod}"
    if exc == "ImportError" and "requires the following packages" in msg:
        return "blocked", f"needs a package: {msg[:160]}"
    if exc == "OSError" and "huggingface.co" in msg:
        return "blocked", "needs model files that are not in the local cache"
    # A missing file in a repo folder is a typo in the doc, not missing data.
    if exc == "FileNotFoundError" and re.search(r"['\"](?:configs|docs|eval|scripts|tests|radharmony|notebooks)/", msg):
        return "failed", f"{exc}: {msg}"
    # A class with no data raises FileNotFoundError with the file name (D-35).
    if exc == "FileNotFoundError" or (
            re.search(r"No such file|does not exist|not found|must be an existing", msg)
            and re.search(r"(/data/|/path/to|/mnt/|\.csv|\.json|\.parquet|\.png|\.dcm|\.nii)", msg)):
        return "blocked", f"needs data: {msg[:160]}"
    if "CUDA" in msg and exc in ("RuntimeError", "AssertionError", "AcceleratorError"):
        return "blocked", f"needs a GPU: {msg[:160]}"
    return "failed", f"{exc}: {msg}"


def gpu_available() -> bool:
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return False
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False
    proc = subprocess.run([smi, "-L"], capture_output=True, text=True)
    return proc.returncode == 0 and "GPU" in proc.stdout


def _todo(blocks: list, name: str, level: str, has_gpu: bool, every_block: bool = False) -> tuple[list[dict], bool]:
    """Return the blocks to run as harness items, and whether a block wants the repo as its folder.

    With ``every_block``, each Python block runs, whatever its ``run=`` note (except ``run=skip``).
    """
    todo, in_repo = [], False
    for b in blocks:
        run = b.opts.get("run", "static")
        if b.lang not in PY_LANGS or run == "skip" or _timeout(b) is None:
            continue
        if level == "ci" and run != "ci" and not every_block:
            continue
        in_repo |= b.opts.get("cwd") == "repo"
        item = {"line": b.start_line, "name": f"{name}:{b.start_line}", "code": _source(b),
                "timeout": _timeout(b)}
        if run == "gpu" and not has_gpu:
            item["blocked"] = "needs a GPU (no GPU on this machine)"
        todo.append(item)
    return todo, in_repo


def _run_todo(todo: list[dict], level: str, python: str, in_repo: bool, name: str) -> tuple[list[dict], str]:
    """Run the harness items in one child process, in a temporary folder (or the repo).

    Returns one record per item (page, line, result, detail, python) and the output of the process.
    """
    env = dict(os.environ, PYTHONPATH=str(REPO), MPLBACKEND="Agg", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", GRADIO_ANALYTICS_ENABLED="False")
    if level != "full":
        env["CUDA_VISIBLE_DEVICES"] = ""  # CI blocks must not need a GPU.
    output = ""
    with tempfile.TemporaryDirectory(prefix="radharmony_blocks_") as tmp:
        rec_file = Path(tmp) / "_records.jsonl"
        work = REPO if in_repo else Path(tmp)
        try:
            proc = subprocess.run([python, "-c", _HARNESS, str(rec_file)], input=json.dumps(todo), cwd=work,
                                  env=env, capture_output=True, text=True, timeout=PAGE_TIMEOUT)
            output = proc.stdout + proc.stderr
        except subprocess.TimeoutExpired as e:
            output = "".join(x.decode(errors="replace") if isinstance(x, bytes) else (x or "")
                             for x in (e.stdout, e.stderr))
        recs = {}
        if rec_file.is_file():
            for ln in rec_file.read_text().splitlines():
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                recs[r["line"]] = r
    out, stopped_at = [], None
    for t in todo:
        r = recs.get(t["line"])
        if r is None:
            # The process stopped here (page time limit or a crash). Later blocks did not run.
            if stopped_at is None:
                stopped_at = t["line"]
                r = {"status": "timeout", "detail": f"the page stopped here (limit {PAGE_TIMEOUT} s, or a crash)"}
            else:
                r = {"status": "not_run", "detail": f"depends on line {stopped_at}"}
        result, msg = _classify(r, level)
        out.append({"page": name, "line": t["line"], "result": result, "detail": msg, "python": python})
    return out, output


def run_page(page: Path, level: str, python: str = sys.executable, has_gpu: bool = False) -> list[dict]:
    """Run the blocks of one page that the level allows. Returns one record per block."""
    rel = page.relative_to(REPO).as_posix()
    todo, in_repo = _todo(extract_blocks(read_text(rel)), rel, level, has_gpu)
    if not todo:
        return []
    return _run_todo(todo, level, python, in_repo, rel)[0]


def run_text(text: str, level: str = "full", python: str = sys.executable, has_gpu: bool | None = None,
             name: str = "snippet") -> tuple[list[dict], str]:
    """Run every Python block of a Markdown text in a temporary folder, as the doc block check does.

    The text is not written to the repo. The blocks share one process, so a later block sees the names
    of an earlier block. A block with ``run=skip`` does not run.

    Args:
        text: Markdown with Python blocks (``python`` or ``pycon`` fences).
        level: "full" sorts a missing package, data file or GPU as "blocked". "ci" counts each error as "failed".
        python: the Python that runs the blocks, for example ``$RADHARMONY_MODEL_PYTHON``.
        has_gpu: whether ``run=gpu`` blocks may run. None asks ``nvidia-smi`` (only at the full level).
        name: the page name in the records.

    Returns:
        The records (page, line, result, detail, python), one for each block, and the output of the blocks.
    """
    if has_gpu is None:
        has_gpu = level == "full" and gpu_available()
    todo, in_repo = _todo(extract_blocks(text), name, level, has_gpu, every_block=True)
    if not todo:
        return [], ""
    return _run_todo(todo, level, python, in_repo, name)


def run_level(level: str, pages: list[Path], workers: int = 8) -> list[dict]:
    """Run all pages. In a full run, a page whose blocks need a missing package runs again
    with the model venv."""
    has_gpu = level == "full" and gpu_available()
    with ThreadPoolExecutor(workers) as ex:
        results = [r for rs in ex.map(lambda p: run_page(p, level, has_gpu=has_gpu), pages) for r in rs]
    if level == "full" and MODEL_PYTHON and Path(MODEL_PYTHON).is_file():
        again = sorted({r["page"] for r in results if r["detail"].startswith("needs package")})
        if again:
            keep = [r for r in results if r["page"] not in again]
            with ThreadPoolExecutor(workers) as ex:
                redo = [r for rs in ex.map(lambda rel: run_page(REPO / rel, level, MODEL_PYTHON, has_gpu), again)
                        for r in rs]
            results = keep + redo
    return sorted(results, key=lambda r: (r["page"], r["line"]))


def run(level: str = "static", files: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    """Return (static problems, run records)."""
    pages = doc_pages(files)
    problems = [p for page in pages for p in static_problems(page)]
    records = run_level(level, pages) if level in ("ci", "full") else []
    return problems, records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--level", choices=["static", "ci", "full"], default="static")
    ap.add_argument("--page", help="limit to pages that match this glob (relative to the repo)")
    ap.add_argument("--json", help="write problems and run records as JSON to this file")
    args = ap.parse_args()
    files = {p.relative_to(REPO).as_posix() for p in REPO.glob(args.page)} if args.page else None
    problems, records = run(args.level, files)
    if args.json:
        Path(args.json).write_text(json.dumps({"problems": problems, "records": records}, indent=1))
    for r in records:
        print(f"{r['page']}:{r['line']}: {r['result']} {r['detail']}")
    for p in problems:
        print(f"{p['file']}:{p['line']}: [{p['rule']}] {p['text']}")
    bad = problems or [r for r in records if r["result"] in ("failed", "timeout")]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
