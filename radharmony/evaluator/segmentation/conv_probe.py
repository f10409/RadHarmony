"""Conv-block segmentation probe over frozen dense backbone features.

A natural step up from :class:`LinearProbeSegEvaluator`: instead of a single
``1×1`` conv, the head is a small convolutional block that mixes channels
and spatial context before the per-pixel classifier:

    Sequential(
        Conv2d(D, hidden, kernel_size=1, bias=False),
        LayerNorm2d(hidden),
        Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
        LayerNorm2d(hidden),
    )

then a ``1×1`` classifier and a bilinear upsample to the mask size.
``LayerNorm2d`` is channel-wise layer norm at each spatial position
(canonical SAM / ViTDet impl) — works at any batch size, unlike BN.

Sits between :class:`LinearProbeSegEvaluator` (single ``1×1`` conv) and
:class:`UPerNetSegEvaluator` (PSP + FPN top-down fuse) on the head-capacity
spectrum.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._registry import register_evaluator
from .linear_probe import LinearProbeSegEvaluator


class _LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channel dim at each spatial position.

    Permutes ``[B, C, H, W]`` to ``[B, H, W, C]`` so ``nn.LayerNorm`` (which
    normalizes over the last dim) sees channels last, then permutes back.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _ConvProbeSegHead(nn.Module):
    """Conv-block neck + 1×1 classifier + bilinear upsample."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        out_size: int,
        *,
        hidden_size: int = 256,
    ):
        super().__init__()
        self.neck = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=1, bias=False),
            _LayerNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, bias=False),
            _LayerNorm2d(hidden_size),
        )
        self.classifier = nn.Conv2d(hidden_size, num_classes, kernel_size=1)
        self.out_size = out_size

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.neck(feats)
        logits = self.classifier(x)
        return F.interpolate(
            logits,
            size=(self.out_size, self.out_size),
            mode="bilinear",
            align_corners=False,
        )


@register_evaluator("conv_probe_seg")
class ConvProbeSegEvaluator(LinearProbeSegEvaluator):
    """Frozen-feature segmentation probe with a small conv-block head.

    Same training loop, split modes, metrics, and prediction-dump behaviour
    as :class:`LinearProbeSegEvaluator`; only the head differs. The head is
    a ``1×1 → LN2d → 3×3 → LN2d`` block followed by a ``1×1`` classifier
    and a bilinear upsample.

    Extra constructor arg on top of :class:`LinearProbeSegEvaluator`:

    - ``conv_hidden_size`` — neck channel width (default 256).
    """

    def __init__(
        self,
        image_encoder,
        *,
        conv_hidden_size: int = 256,
        **kwargs,
    ):
        super().__init__(image_encoder, **kwargs)
        self.conv_hidden_size = conv_hidden_size

    def _make_head(self, feat_dim: int, out_size: int) -> nn.Module:
        return _ConvProbeSegHead(
            in_channels=feat_dim,
            num_classes=self.num_classes,
            out_size=out_size,
            hidden_size=self.conv_hidden_size,
        ).to(self.device)

    def _head_config(self) -> dict:
        return {"conv_hidden_size": self.conv_hidden_size}
