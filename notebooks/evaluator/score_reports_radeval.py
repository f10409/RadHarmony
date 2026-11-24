"""Stage B of the two-stage report-gen eval: score a pairs parquet with RadEval.

Deliberately standalone: it imports ONLY radeval + pandas (NOT radharmony,
torch, or transformers) so it runs in the RadEval env, decoupled from the
CheXagent-2 generation env (transformers==4.40.0). It loads the one file
``radharmony/evaluator/language/hypothesis.py`` by its path, so it cleans the
text as ``ReportGenerationEvaluator.evaluate()`` does (D-50). So run it from
its place in the repo checkout, not from a copy. Input is the parquet that
Stage A (`gen` mode / `ReportGenerationEvaluator.generate_only`) wrote:
columns ``sample_id, reference, hypothesis``, and ``patient_id`` when the
dataset has it. With ``patient_id``, each bootstrap sample draws patients and
keeps all their pairs (D-32). Without it, the bootstrap draws pairs.

Env:  uv venv --python 3.12 <venv>
      uv pip install --python <venv>/bin/python \
        "radeval[api] @ git+https://github.com/jbdel/RadEval.git" pandas pyarrow

Run:  python score_reports_radeval.py \
        --pairs-parquet /path/to/reportgen_smoke/pairs.parquet \
        --output-dir    /path/to/reportgen_smoke \
        --light                       # or omit for the full 16-metric suite
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib

import numpy as np
import pandas as pd

def _load_normalize_hypothesis():
    """Load ``normalize_hypothesis`` from ``radharmony/evaluator/language/hypothesis.py`` by its path.

    The file imports only ``re``, so this script still does not import radharmony.
    Both scoring paths then clean the text with one function (D-50).
    """
    path = pathlib.Path(__file__).resolve().parents[2] / "radharmony" / "evaluator" / "language" / "hypothesis.py"
    spec = importlib.util.spec_from_file_location("radharmony_hypothesis", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.normalize_hypothesis


normalize_hypothesis = _load_normalize_hypothesis()

FULL_METRICS = (
    "bleu", "rouge", "bertscore", "radeval_bertscore", "f1chexbert",
    "f1radbert_ct", "radgraph", "ratescore", "radgraph_radcliq", "radcliq",
    "srrbert", "temporal", "green", "mammo_green", "crimson", "radfact_ct",
)
LIGHT_METRICS = ("bleu", "rouge", "bertscore", "radgraph", "f1chexbert")


def _flatten(d: dict, prefix: str = "") -> dict:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}_"))
        elif isinstance(v, bool):
            continue
        elif isinstance(v, (int, float, np.floating)):
            out[key] = float(v)
    return out


def bootstrap_indices(n_rows, n_bootstrap, seed, groups=None):
    """Yield the row indices of each bootstrap sample (D-32).

    A copy of ``radharmony.evaluator.sampling.bootstrap_indices``, because this
    script does not import radharmony. Each sample draws patients with
    replacement and keeps all their rows. ``groups=None`` draws rows.
    """
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    if groups is None:
        for _ in range(n_bootstrap):
            yield rng.integers(0, n_rows, size=n_rows)
        return
    _, first, inverse = np.unique(np.asarray(groups), return_index=True, return_inverse=True)
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first)] = np.arange(len(first))
    patient = rank[inverse.ravel()]
    rows_by_patient = np.argsort(patient, kind="stable")
    counts = np.bincount(patient)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    for _ in range(n_bootstrap):
        drawn = rng.integers(0, len(counts), size=len(counts))
        c = counts[drawn]
        yield rows_by_patient[np.repeat(starts[drawn] - np.cumsum(c) + c, c) + np.arange(c.sum())]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs-parquet", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--light", action="store_true",
                   help="LIGHT_METRICS (no LLM/API) instead of the full 16.")
    p.add_argument("--n-bootstrap", type=int, default=0)
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--no-normalize-hyp", action="store_true",
                   help="Score the raw text. By default, the script removes the "
                        "CheXagent grounding tags and bold marks (D-50).")
    args = p.parse_args()

    df = pd.read_parquet(args.pairs_parquet)
    refs = df["reference"].astype(str).tolist()
    hyps_raw = df["hypothesis"].astype(str).tolist()
    if args.no_normalize_hyp:
        hyps = hyps_raw
    else:
        hyps = [normalize_hypothesis(h) for h in hyps_raw]
        n_changed = sum(1 for a, b in zip(hyps_raw, hyps) if a != b)
        print(f"hyp normalize: stripped grounded tags/bold from {n_changed}/{len(hyps)} samples")
    metrics = list(LIGHT_METRICS if args.light else FULL_METRICS)
    print(f"scoring {len(refs)} pairs with {len(metrics)} metrics: {metrics}")

    from radeval import RadEval  # lazy: the only heavy import

    evaluator = RadEval(metrics=metrics)

    def _score(r, h):
        return _flatten(evaluator(refs=r, hyps=h))

    rows = [{"label": "report", "bootstrap": -1, **_score(refs, hyps)}]
    groups = df["patient_id"].astype(str).tolist() if "patient_id" in df.columns else None
    for b, idx in enumerate(bootstrap_indices(len(refs), args.n_bootstrap, args.bootstrap_seed, groups)):
        rows.append({
            "label": "report", "bootstrap": b,
            **_score([refs[i] for i in idx], [hyps[i] for i in idx]),
        })

    os.makedirs(args.output_dir, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)
    point = {k: v for k, v in rows[0].items() if k not in ("label", "bootstrap")}
    with open(os.path.join(args.output_dir, "results_summary.json"), "w") as f:
        json.dump({"n_samples": len(refs), "metrics": point}, f, indent=2)
    print(f"\nn={len(refs)}  results → {args.output_dir}/")
    print(json.dumps(point, indent=2))


if __name__ == "__main__":
    main()
