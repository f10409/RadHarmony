"""RadEval metric adapter for the report-generation evaluator.

Wraps the RadEval framework (jbdel/RadEval, EMNLP 2025) behind the same
lazy-import contract the classification panel uses: RadEval and its heavy
per-metric dependencies are only imported when :func:`compute_report_metrics`
is actually called, so the base install stays light and importing
``radharmony.evaluator`` never pulls RadEval in.

RadEval returns a nested ``dict`` (some metrics emit sub-scores, e.g.
``radgraph`` → ``{f1, precision, recall}``). :func:`_flatten` collapses it
to a flat ``{column: float}`` panel so :meth:`BaseEvaluator._summarize`
works unchanged.
"""

from __future__ import annotations

import numpy as np

# The full 16-metric RadEval suite. Metrics that need extra resources are
# included here deliberately: GREEN / mammo_green run a local ~7B LLM;
# crimson / radfact_ct need an API key. RadEval's own lazy loader raises a
# clear per-metric error if a dependency or key is missing — we surface that
# rather than silently dropping the metric.
FULL_METRICS: tuple[str, ...] = (
    "bleu",
    "rouge",
    "bertscore",
    "radeval_bertscore",
    "f1chexbert",
    "f1radbert_ct",
    "radgraph",
    "ratescore",
    "radgraph_radcliq",
    "radcliq",
    "srrbert",
    "temporal",
    "green",
    "mammo_green",
    "crimson",
    "radfact_ct",
)

# Lexical + clinical subset: no LLM, no API keys, single-GPU friendly.
LIGHT_METRICS: tuple[str, ...] = (
    "bleu",
    "rouge",
    "bertscore",
    "radgraph",
    "f1chexbert",
)


def _flatten(d: dict, prefix: str = "") -> dict[str, float]:
    """Flatten RadEval's (possibly nested) result dict into ``{col: float}``.

    Non-numeric leaves (per-sample lists, label strings) are dropped — they
    are not part of the corpus-level metric panel.
    """
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}_"))
        elif isinstance(v, (bool,)):
            continue
        elif isinstance(v, (int, float, np.floating)):
            out[key] = float(v)
        # lists / strings / None: not a scalar metric — skip
    return out


def compute_report_metrics(
    refs: list[str],
    hyps: list[str],
    metrics: tuple[str, ...] | list[str],
) -> dict[str, float]:
    """Score hypothesis reports against references with RadEval.

    Parameters
    ----------
    refs, hyps :
        Reference and hypothesis reports, aligned by index. Same length.
    metrics :
        RadEval metric names (see :data:`FULL_METRICS`). Unknown or
        unavailable metrics raise from RadEval itself — the error is
        surfaced, not swallowed, so a misconfigured suite fails loudly.

    Returns
    -------
    dict[str, float]
        Flat, corpus-level metric panel.
    """
    if len(refs) != len(hyps):
        raise ValueError(
            f"refs / hyps length mismatch: {len(refs)} vs {len(hyps)}"
        )
    if not refs:
        raise ValueError("No (ref, hyp) pairs to score — refs is empty.")

    try:
        from radeval import RadEval  # lazy: heavy, optional [radeval] extra
    except ImportError as e:
        raise ImportError(
            "ReportGenerationEvaluator needs the RadEval framework. Install "
            'it with `uv pip install -e ".[radeval]"` (RadEval, jbdel/RadEval).'
        ) from e

    evaluator = RadEval(metrics=list(metrics))
    raw = evaluator(refs=list(refs), hyps=list(hyps))
    return _flatten(raw)
