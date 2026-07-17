"""Report-generation scoring over pre-computed reports (datathon26).

A thin :class:`ReportGenerationEvaluator` subclass that scores the reportbench
service's generated reports instead of calling a VLM. The paired
:class:`~radharmony.datathon26.dataset.report_dataset.ReportResultsDataset`
serves the **predicted** report under ``"img"`` and the **reference** report
under ``"report"``; :func:`identity_generator` passes the predicted text through
as the hypothesis. Everything downstream — the RadEval metric suite,
``n_bootstrap`` resampling, and ``generations.csv`` — is inherited unchanged.

Both reference and hypothesis are parsed to the same ``ref_section`` so the
comparison is symmetric (findings-vs-findings by default).

Usage::

    ds = ReportResultsDataset(results_dir, csv_path="dataset_A.csv")
    ev = DatathonReportGenerationEvaluator(dataset=ds, metrics=LIGHT_METRICS)
    results = ev.evaluate()
"""

from __future__ import annotations

from radharmony.evaluator._registry import register_evaluator
from radharmony.evaluator.language.report_generation import ReportGenerationEvaluator
from radharmony.evaluator.language._report_parser import parse_report_section
from radharmony.evaluator.metrics.language import FULL_METRICS

from ._identity import identity_generator


@register_evaluator("datathon26_report_generation")
class DatathonReportGenerationEvaluator(ReportGenerationEvaluator):
    """Score pre-computed generated reports with RadEval (no model call)."""

    _progress_desc = "scoring reports"

    def __init__(
        self,
        *,
        dataset,
        metrics=FULL_METRICS,
        ref_section: str = "findings",
        **kwargs,
    ):
        super().__init__(
            identity_generator,
            dataset=dataset,
            metrics=metrics,
            ref_section=ref_section,
            **kwargs,
        )

    def _prepare_batch(self, batch):
        if "report" not in batch or "img" not in batch:
            raise KeyError(
                "DatathonReportGenerationEvaluator expects a ReportResultsDataset "
                "sample with 'report' (reference) and 'img' (predicted) keys."
            )
        refs = [parse_report_section(str(r), self.ref_section) for r in batch["report"]]
        # Predicted text arrives under 'img'; parse to the same section so the
        # hypothesis and reference are compared on like content. These become
        # the generator input; identity_generator returns them as hypotheses.
        preds = [parse_report_section(str(p), self.ref_section) for p in batch["img"]]
        ids = [str(s) for s in batch.get("study_id", range(len(refs)))]
        return preds, [None] * len(refs), refs, ids
