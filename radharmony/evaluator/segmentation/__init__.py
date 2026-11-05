"""Segmentation evaluators: frozen-encoder dense-feature probes.

Public surface:

- :class:`BaseSegEvaluator`: abstract base with the shared encoder and loader code.
- :class:`LinearProbeSegEvaluator`: 1×1 Conv2d head on frozen patch features
  with bilinear upsample and multi-class CE. It runs k-fold or fixed-split
  with bootstrap.
- :class:`UPerNetSegEvaluator`: ViTDet-style SFP and HF UPerNet head over the
  same frozen patch features. Same protocol and metrics as the linear probe.
- :class:`ConvProbeSegEvaluator`: small conv-block head
  (``1×1 → LN2d → 3×3 → LN2d`` and a ``1×1`` classifier). Same protocol and
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
