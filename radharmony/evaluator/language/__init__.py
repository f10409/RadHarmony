"""Language evaluators.

Ships :class:`ReportGenerationEvaluator` — runs any vision-language model
over a RadHarmony dataset and scores the generated reports with the RadEval
metric suite — and :class:`VQAEvaluator` — runs a question-conditioned VLM
over a VQA dataset and scores answers with tokenized-F1 / yes-no accuracy
(MedGemma protocol).
"""

from .base_generative import GenerativeEvaluator
from .report_generation import ReportGenerationEvaluator
from .vqa import VQAEvaluator

__all__ = ["GenerativeEvaluator", "ReportGenerationEvaluator", "VQAEvaluator"]
