"""Stage B of the two-stage report-gen eval: score a pairs parquet with RadEval.

Deliberately standalone — imports ONLY radeval + pandas (NOT radharmony,
torch, or transformers) so it runs in the RadEval env, decoupled from the
CheXagent-2 generation env (transformers==4.40.0). Input is the parquet that
Stage A (`gen` mode / `ReportGenerationEvaluator.generate_only`) wrote:
columns ``sample_id, reference, hypothesis``.

Env:  uv venv --python 3.12 <venv>
      uv pip install --python <venv>/bin/python \
        "radeval[api] @ git+https://github.com/jbdel/RadEval.git" pandas pyarrow

Run:  python score_reports_radeval.py \
        --pairs-parquet /mnt/NAS4/.../reportgen_smoke/pairs.parquet \
        --output-dir    /mnt/NAS4/.../reportgen_smoke \
        --light                       # or omit for the full 16-metric suite
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

# CheXagent-2 emits two distinct native grounded formats observed in the wild:
#   (a) "[Region: Subregion] **finding**"            (MIMIC DICOM input)
#   (b) "<|ref|> finding <|/ref|> <|box|> (x,y),(x,y) <|/box|>"  (Emory PNG input)
# Both tank BLEU/ROUGE/F1-CheXbert against free-text references. Strip the
# structural markup before scoring so we compare *content*, not format.
_GROUNDED_TAG = re.compile(r"\[[A-Za-z][A-Za-z _/&\-]+:\s*[A-Za-z][A-Za-z _/&\-]+\]\s*")
_REF_OPEN = re.compile(r"<\|ref\|>\s*")
_REF_CLOSE = re.compile(r"\s*<\|/ref\|>")
_BOX = re.compile(r"<\|box\|>.*?<\|/box\|>", re.DOTALL)
_BOLD_MARK = re.compile(r"\*+")
_WS = re.compile(r"\s+")


def normalize_hypothesis(s: str) -> str:
    s = _GROUNDED_TAG.sub("", s)
    s = _BOX.sub("", s)
    s = _REF_OPEN.sub("", s)
    s = _REF_CLOSE.sub("", s)
    s = _BOLD_MARK.sub("", s)
    s = _WS.sub(" ", s).strip()
    return s

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs-parquet", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--light", action="store_true",
                   help="LIGHT_METRICS (no LLM/API) instead of the full 16.")
    p.add_argument("--n-bootstrap", type=int, default=0)
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--no-normalize-hyp", action="store_true",
                   help="Skip the CheXagent grounded-tag stripping pass. Off by "
                        "default — keep on unless comparing raw model output.")
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
    if args.n_bootstrap > 0:
        rng = np.random.default_rng(args.bootstrap_seed & 0xFFFFFFFF)
        n = len(refs)
        for b in range(args.n_bootstrap):
            idx = rng.integers(0, n, size=n)
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
