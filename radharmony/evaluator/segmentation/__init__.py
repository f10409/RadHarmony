"""Segmentation evaluators — frozen-encoder dense-feature probes.

Public surface:

- :class:`BaseSegEvaluator` — abstract base; shared encoder + loader plumbing.
- :class:`LinearProbeSegEvaluator` — 1×1 Conv2d head on frozen patch features
  with bilinear upsample, multi-class CE, k-fold / fixed-split + bootstrap.
- :class:`UPerNetSegEvaluator` — ViTDet-style SFP + HF UPerNet head over the
  same frozen patch features; same protocol / metrics as the linear probe.
- :class:`ConvProbeSegEvaluator` — small conv-block head
  (``1×1 → LN2d → 3×3 → LN2d`` + ``1×1`` classifier); same protocol /
  metrics as the linear probe.
"""

from .base import BaseSegEvaluator
from .conv_probe import ConvProbeSegEvaluator
from .linear_probe import LinearProbeSegEvaluator
from .upernet_probe import UPerNetSegEvaluator

__all__ = [
    "BaseSegEvaluator",
    "ConvProbeSegEvaluator",
    "LinearProbeSegEvaluator",
    "UPerNetSegEvaluator",
]
