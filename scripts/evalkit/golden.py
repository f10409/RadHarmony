"""Golden outputs for the code cleanup (plan 6b): save them before a batch, compare them after it.

For each dataset spec of ``scripts/check_datasets.py`` and ``scripts/check_datasets_internal.py``,
the tool saves:

- the shape of the harmonized table, and a hash of each column (its dtype and values, in row order);
- a hash of 5 samples ``ds[i]`` (each tensor's dtype, shape and values, other values as text).

It also saves the result tables of the 5 probes that train and of the segmentation probe on toy data.
A cleanup must not change any of these. Each dataset runs in its own process, with a time limit.

Run with::

    .venv/bin/python scripts/evalkit/golden.py save reports/golden/before.json
    .venv/bin/python scripts/evalkit/golden.py save reports/golden/after.json --datasets chexpert_train,brax
    .venv/bin/python scripts/evalkit/golden.py compare reports/golden/before.json reports/golden/after.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root first: test this clone, not an installed copy
ROOT = Path(__file__).resolve().parents[2]
N_SAMPLES = 5

#: Sample keys that hold MONAI's record of the applied steps, not data. They change from run to run.
TRACE_SUFFIXES = ("_transforms", "_meta_dict")


def _is_trace(key: str) -> bool:
    return str(key).endswith(TRACE_SUFFIXES)


def _no_address(text):
    import re

    return re.sub(r"0x[0-9a-fA-F]+", "0x", text) if isinstance(text, str) else text


def _sha(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


def _text(value) -> str:
    """Return a stable text of a table cell or a sample value (lists and dicts as sorted JSON)."""
    import numpy as np

    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, default=str)
    return repr(value)


def table_record(df) -> dict:
    """Return the shape and one hash per column of a harmonized table."""
    cols = {}
    for c in df.columns:
        values = "\n".join(_text(v) for v in df[c].tolist())
        cols[str(c)] = f"{df[c].dtype}:{_sha(values.encode())}"
    return {"shape": list(df.shape), "columns": cols}


def sample_record(sample: dict) -> dict:
    """Return one hash per key of a sample ``ds[i]``."""
    import torch

    out = {}
    for k in sorted(sample):
        if _is_trace(k):
            continue
        v = sample[k]
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu()
            data = t.float().numpy().tobytes() if t.dtype != torch.bool else t.numpy().tobytes()
            out[k] = f"{t.dtype}{tuple(t.shape)}:{_sha(data)}"
        else:
            out[k] = f"{type(v).__name__}:{_sha(_text(v).encode())}"
    return out


def one_dataset(key: str) -> dict:
    """Build the dataset of one spec and return its golden record."""
    import warnings

    import torch

    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    import check_datasets
    specs = {s.key: s for s in check_datasets.get_specs()}
    try:
        import check_datasets_internal
        specs.update({s.key: s for s in check_datasets_internal.INTERNAL_SPECS})
    except (ImportError, AttributeError):
        pass
    spec = specs[key]
    warnings.simplefilter("ignore")
    kwargs = spec.kwargs() if callable(spec.kwargs) else spec.kwargs
    ds = spec.cls(cache_dir=None, **kwargs)
    rec = {"table": table_record(ds.get_harmonized_df())}
    data = ds.get_datasets() if "num_cores" not in ds.get_datasets.__code__.co_varnames else ds.get_datasets(num_cores=1)
    n = len(data)
    idx = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1}) if n else []
    samples = {}
    for i in idx[:N_SAMPLES]:
        torch.manual_seed(0)
        samples[str(i)] = sample_record(data[i])
    rec["samples"] = samples
    return rec


def evaluator_records() -> dict:
    """Run the 5 probes that train and the segmentation probe on toy data, and hash their result tables."""
    import warnings

    import numpy as np
    import torch

    sys.path[:0] = [str(ROOT / "tests")]
    warnings.simplefilter("ignore")
    from _toy_eval import ToyClsDS, ToySegDS, toy_seg_data

    from radharmony.evaluator import (FinetuneEvaluator, KNNProbeEvaluator, LinearProbeEvaluator,
                                      PrototypeProbeEvaluator, SVMProbeEvaluator)
    from radharmony.evaluator.segmentation.linear_probe import LinearProbeSegEvaluator

    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 6)).astype(np.float32)
    y = (rng.random((200, 2)) < 0.4).astype(np.float32)
    x[:, :2] += 1.5 * y
    x[:, 2] += 3.0
    train, test = ToyClsDS(x[:120], y[:120], ["a", "b"]), ToyClsDS(x[120:], y[120:], ["a", "b"])
    common = dict(device="cpu", autocast_dtype=None, num_workers=0, threshold_strategy="fixed:0.5",
                  val_fraction=0.0, n_bootstrap=20)
    probes = {
        "linear": (LinearProbeEvaluator, torch.nn.Identity, common),
        "svm": (SVMProbeEvaluator, torch.nn.Identity, common),
        "knn": (KNNProbeEvaluator, torch.nn.Identity, dict(common, k=7)),
        "prototype": (PrototypeProbeEvaluator, torch.nn.Identity, common),
        "finetune": (FinetuneEvaluator, lambda: torch.nn.Linear(6, 6),
                     dict(common, freeze_backbone=True, epochs=3, lr=1e-2, batch_size=256)),
    }
    out = {}
    for name, (cls, make, kw) in probes.items():
        torch.manual_seed(0)
        df = cls(make(), train_dataset=train, test_dataset=test, **kw).evaluate()
        out[name] = _sha(df.round(6).to_csv(index=False).encode())
    imgs, masks = toy_seg_data(n=24)
    torch.manual_seed(0)
    seg = LinearProbeSegEvaluator(torch.nn.Identity(), train_dataset=ToySegDS(imgs[:16], masks[:16]),
                                  test_dataset=ToySegDS(imgs[16:], masks[16:]), device="cpu",
                                  num_workers=0, epochs=2, n_bootstrap=5)
    out["seg_linear"] = _sha(seg.evaluate().round(6).to_csv(index=False).encode())
    return out


def _run_one(key: str, timeout: int) -> tuple[str, dict]:
    t0 = time.time()
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    try:
        p = subprocess.run([sys.executable, __file__, "one", key], capture_output=True, text=True,
                           timeout=timeout, cwd=ROOT, env=env)
    except subprocess.TimeoutExpired:
        return key, {"error": f"timeout after {timeout} s"}
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("GOLDEN ")]
    if p.returncode or not lines:
        return key, {"error": (p.stderr.strip().splitlines() or ["no output"])[-1][:300]}
    rec = json.loads(lines[-1][len("GOLDEN "):])
    rec["seconds"] = round(time.time() - t0)
    return key, rec


def save(out: Path, keys: list[str] | None, jobs: int, timeout: int, with_slow: bool) -> int:
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    import check_datasets
    specs = check_datasets.get_specs()
    try:
        import check_datasets_internal
        specs += check_datasets_internal.INTERNAL_SPECS
    except (ImportError, AttributeError):
        pass
    if keys is None:
        keys = [s.key for s in specs if s.default or with_slow]
    result = {"commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                       text=True, cwd=ROOT).stdout.strip(),
              "datasets": {}, "evaluators": evaluator_records()}
    with ThreadPoolExecutor(jobs) as ex:
        for key, rec in ex.map(lambda k: _run_one(k, timeout), keys):
            result["datasets"][key] = rec
            print(f"{key:40s} {'ERROR ' + rec['error'] if 'error' in rec else 'ok'}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(f"wrote {out}: {len(keys)} datasets, {sum('error' in r for r in result['datasets'].values())} errors")
    return 0


def compare(a: Path, b: Path) -> int:
    """Print every difference between 2 golden files. Exit 1 if any output differs."""
    A, B = json.loads(a.read_text()), json.loads(b.read_text())
    diffs = []
    for name in sorted(set(A["evaluators"]) | set(B["evaluators"])):
        if A["evaluators"].get(name) != B["evaluators"].get(name):
            diffs.append(f"evaluator {name}: result table differs")
    for key in sorted(set(A["datasets"]) & set(B["datasets"])):
        ra, rb = A["datasets"][key], B["datasets"][key]
        if "error" in ra or "error" in rb:
            # An error text can hold a memory address (0x7f...), which changes from run to run.
            if _no_address(ra.get("error")) != _no_address(rb.get("error")):
                diffs.append(f"{key}: error before {ra.get('error')!r}, after {rb.get('error')!r}")
            continue
        ta, tb = ra["table"], rb["table"]
        if ta["shape"] != tb["shape"]:
            diffs.append(f"{key}: table shape {ta['shape']} -> {tb['shape']}")
        for col in sorted(set(ta["columns"]) | set(tb["columns"])):
            if ta["columns"].get(col) != tb["columns"].get(col):
                diffs.append(f"{key}: column {col!r} differs")
        for i in sorted(set(ra["samples"]) | set(rb["samples"]), key=int):
            sa, sb = ra["samples"].get(i, {}), rb["samples"].get(i, {})
            for k in sorted((set(sa) | set(sb)) - {k for k in set(sa) | set(sb) if _is_trace(k)}):
                if sa.get(k) != sb.get(k):
                    diffs.append(f"{key}: sample {i} key {k!r} differs")
    only = sorted(set(A["datasets"]) ^ set(B["datasets"]))
    for line in diffs:
        print(line)
    print(f"{len(diffs)} differences. Datasets in one file only: {only or 'none'}")
    return 1 if diffs else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save", help="save the golden outputs")
    s.add_argument("out", type=Path)
    s.add_argument("--datasets", help="comma-separated spec keys (default: the default specs)")
    s.add_argument("--with-slow", action="store_true", help="also the specs with default=False")
    s.add_argument("--jobs", type=int, default=6)
    s.add_argument("--timeout", type=int, default=1800, help="seconds for each dataset")
    c = sub.add_parser("compare", help="compare 2 golden files")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    o = sub.add_parser("one", help=argparse.SUPPRESS)
    o.add_argument("key")
    args = p.parse_args(argv)
    if args.cmd == "save":
        keys = [k for k in args.datasets.split(",") if k] if args.datasets else None
        return save(args.out, keys, args.jobs, args.timeout, args.with_slow)
    if args.cmd == "compare":
        return compare(args.a, args.b)
    print("GOLDEN " + json.dumps(one_dataset(args.key), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
