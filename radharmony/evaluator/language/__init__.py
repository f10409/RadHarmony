"""Language evaluators.

- :class:`ReportGenerationEvaluator` runs a vision-language model over a
  RadHarmony dataset. It scores the generated reports with the RadEval metrics.
- :func:`normalize_hypothesis` cleans a generated report before scoring (D-50).
- :class:`VQAEvaluator` runs a VLM that answers the question of each VQA sample.
  It scores the answers with token F1 and yes/no accuracy (MedGemma protocol).
- :class:`GenerativeEvaluator` is the image-to-text loop of ``VQAEvaluator``.
"""

from .hypothesis import normalize_hypothesis
from .base_generative import GenerativeEvaluator
from .report_generation import ReportGenerationEvaluator
from .vqa import VQAEvaluator

__all__ = ["GenerativeEvaluator", "ReportGenerationEvaluator", "VQAEvaluator", "normalize_hypothesis"]
