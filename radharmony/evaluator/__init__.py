"""Foundation-model downstream-task evaluators for RadHarmony.

Six classification evaluators (linear/kNN/SVM/prototype probe, zero-shot,
minimal fine-tune) and one frozen-feature segmentation linear probe.
``language/`` remains a reserved namespace.
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
    "EncoderPreprocessTransform",
    "RadiologyEncoderTransform",
    "ImageEncoderWrapper",
    "register_evaluator",
    "resolve_evaluator",
    "list_evaluators",
    "backbones",
]
