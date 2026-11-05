"""Report-generation scoring over pre-computed reports (datathon26).

A thin :class:`ReportGenerationEvaluator` subclass that scores the reportbench
service's generated reports instead of calling a VLM. The paired
:class:`~radharmony.datathon26.dataset.report_dataset.ReportResultsDataset`
serves the **predicted** report under ``"img"`` and the **reference** report
under ``"report"``. The generator of this class passes the predicted text
through as the hypothesis. The rest comes from the parent class unchanged: the
RadEval metrics, the cleaning of the hypothesis (D-50), the bootstrap by
patient (``n_bootstrap``, D-32), and ``generations.csv``.

Both reference and hypothesis are parsed to the same ``ref_section`` so the
comparison is symmetric (findings-vs-findings by default).

Usage::

    ds = ReportResultsDataset(results_dir, csv_path="dataset_DS1.csv")
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

    def __init__(
        self,
        *,
        dataset,
        metrics=FULL_METRICS,
        ref_section: str = "findings",
        **kwargs,
    ):
        def prediction_section(preds):
            # The predicted text comes under 'img'. Parse it to the same section as the
            # reference, so both sides hold the same kind of content.
            return [parse_report_section(str(p), ref_section) for p in identity_generator(preds)]

        super().__init__(
            prediction_section,
            dataset=dataset,
            metrics=metrics,
            ref_section=ref_section,
            **kwargs,
        )
