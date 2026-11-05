"""Foundation-model downstream-task evaluators for RadHarmony.

This package has classification evaluators (linear, kNN, SVM and prototype
probe, zero-shot, minimal fine-tune), frozen-feature segmentation probes, and
a language evaluator. The language evaluator is
:class:`ReportGenerationEvaluator`: VLM report generation scored with the
RadEval metric suite.
"""

from .base import BaseEvaluator
from .classification import (
    BaseClsEvaluator,
    LinearProbeEvaluator,
    KNNProbeEvaluator,
    PrototypeProbeEvaluator,
    SVMProbeEvaluator,
    ZeroShotEvaluator,
    FinetuneEvaluator,
)
from .segmentation import (
    BaseSegEvaluator,
    ConvProbeSegEvaluator,
    LinearProbeSegEvaluator,
    UPerNetSegEvaluator,
)
from .language import GenerativeEvaluator, ReportGenerationEvaluator, VQAEvaluator
from ._registry import register_evaluator, resolve_evaluator, list_evaluators
from .transforms import EncoderPreprocessTransform, RadiologyEncoderTransform
from .wrappers import ImageEncoderWrapper
from . import backbones

__all__ = [
    "BaseEvaluator",
    "BaseClsEvaluator",
    "BaseSegEvaluator",
    "LinearProbeEvaluator",
    "KNNProbeEvaluator",
    "PrototypeProbeEvaluator",
    "SVMProbeEvaluator",
    "ZeroShotEvaluator",
    "FinetuneEvaluator",
    "ConvProbeSegEvaluator",
    "LinearProbeSegEvaluator",
    "UPerNetSegEvaluator",
    "GenerativeEvaluator",
    "ReportGenerationEvaluator",
    "VQAEvaluator",
    "EncoderPreprocessTransform",
    "RadiologyEncoderTransform",
    "ImageEncoderWrapper",
    "register_evaluator",
    "resolve_evaluator",
    "list_evaluators",
    "backbones",
]
