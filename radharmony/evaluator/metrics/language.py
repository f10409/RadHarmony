"""RadEval metric adapter for the report-generation evaluator.

Wraps the RadEval framework (jbdel/RadEval, EMNLP 2025). It uses the same
lazy-import rule as the classification panel. RadEval and its heavy
per-metric dependencies load only when :func:`compute_report_metrics`
runs. So the base install stays light, and importing
``radharmony.evaluator`` never loads RadEval.

RadEval returns a nested ``dict``. Some metrics give sub-scores, for example
``radgraph`` → ``{f1, precision, recall}``. :func:`_flatten` turns it
into a flat ``{column: float}`` panel, so :meth:`BaseEvaluator._summarize`
works unchanged.
"""

from __future__ import annotations

import numpy as np

# The full 16-metric RadEval suite. It includes metrics that need extra
# resources on purpose. GREEN and mammo_green run a local ~7B LLM.
# crimson and radfact_ct need an API key. RadEval's own lazy loader raises a
# clear per-metric error if a dependency or key is missing. We show that
# error and do not drop the metric.
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

    Non-numeric leaves (per-sample lists, label strings) are dropped. They
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
        # Lists, strings and None are not scalar metrics: skip.
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
        unavailable metrics raise from RadEval itself. The error shows
        and is not hidden, so a wrong suite fails loudly.

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
            'it with `uv pip install -e ".[radeval]"` (RadEval, jbdel/RadEval). '
            "RadEval needs Python 3.11 or newer. On Python 3.10 the extra installs nothing."
        ) from e

    evaluator = RadEval(metrics=list(metrics))
    raw = evaluator(refs=list(refs), hyps=list(hyps))
    return _flatten(raw)
