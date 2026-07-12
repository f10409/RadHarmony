"""Language / report-generation evaluators.

Ships :class:`ReportGenerationEvaluator` — runs any vision-language model
over a RadHarmony dataset and scores the generated reports with the RadEval
metric suite.
"""

from .report_generation import ReportGenerationEvaluator

__all__ = ["ReportGenerationEvaluator"]
