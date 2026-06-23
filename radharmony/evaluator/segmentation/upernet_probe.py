"""UPerNet head over frozen dense backbone features.

Wraps HuggingFace's :class:`UperNetHead` (PSP + FPN top-down fuse) on top
of a small **Simple Feature Pyramid** adapter that synthesises strides
{4, 8, 16, 32} from a single-scale ViT feature map (stride 16). Only the
SFP + UPerNet head are trainable; the backbone stays frozen.

Reuses the train / predict / split / metric / PNG-dump machinery from
:class:`LinearProbeSegEvaluator` by overriding only ``_make_head``.

Caveats
-------
- ``UperNetHead`` is imported from ``transformers.models.upernet.modeling_upernet``
  (a private module path). HF has kept this stable across the 4.x and 5.x
  lines but it is not part of the public ``transformers`` re-exports.
- The PSP module contains BatchNorm with a pool-to-(1,1) branch, which
  raises ``ValueError`` in training mode at batch size 1. Use
  ``batch_size >= 2`` (default 8 in :class:`BaseSegEvaluator`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._registry import register_evaluator
from .linear_probe import LinearProbeSegEvaluator


class _SimpleFeaturePyramid(nn.Module):
    """ViTDet-style multi-scale synthesis from a single stride-16 feature map.

    Produces four feature maps at strides {4, 8, 16, 32} via 2× / 1× /
    0.5× spatial rescaling, each with the same channel count as the
    input (so downstream UPerNet sees uniform ``in_channels``).
    """

    def __init__(self, in_channels: int):
        super().__init__()
        D = in_channels
        # stride 4: two 2× transposed convs
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(D, D, kernel_size=2, stride=2),
            nn.GroupNorm(32, D),
            nn.GELU(),
            nn.ConvTranspose2d(D, D, kernel_size=2, stride=2),
        )
        # stride 8: one 2× transposed conv
        self.up2 = nn.ConvTranspose2d(D, D, kernel_size=2, stride=2)
        # stride 16: identity
        # stride 32: max-pool
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [self.up4(x), self.up2(x), x, self.down2(x)]


class _UPerNetSegHead(nn.Module):
    """SFP + HF :class:`UperNetHead` + bilinear upsample to mask size."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        out_size: int,
        *,
        hidden_size: int = 256,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
    ):
        super().__init__()
        # Lazy import — only required when this head is constructed.
        from transformers import UperNetConfig
        from transformers.models.upernet.modeling_upernet import UperNetHead

        self.sfp = _SimpleFeaturePyramid(in_channels)
        cfg = UperNetConfig(
            hidden_size=hidden_size,
            pool_scales=tuple(pool_scales),
            num_labels=num_classes,
            use_auxiliary_head=False,
        )
        self.head = UperNetHead(cfg, in_channels=[in_channels] * 4)
        self.out_size = out_size

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        pyramid = self.sfp(feats)
        logits = self.head(pyramid)
        return F.interpolate(
            logits,
            size=(self.out_size, self.out_size),
            mode="bilinear",
            align_corners=False,
        )


@register_evaluator("upernet_seg")
class UPerNetSegEvaluator(LinearProbeSegEvaluator):
    """Frozen-feature segmentation probe with a UPerNet head.

    Same training loop, split modes, metrics, and prediction-dump
    behaviour as :class:`LinearProbeSegEvaluator`; the only change is
    that the head is a small ViTDet-style SFP feeding HF's
    ``UperNetHead`` (PSP + FPN fuse).

    Extra constructor args on top of :class:`LinearProbeSegEvaluator`:

    - ``upernet_hidden_size`` — UPerNet fuse channels (default 256).
    - ``upernet_pool_scales`` — PSP pool grid scales (default ``(1,2,3,6)``).
    """

    def __init__(
        self,
        image_encoder,
        *,
        upernet_hidden_size: int = 256,
        upernet_pool_scales: tuple[int, ...] = (1, 2, 3, 6),
        **kwargs,
    ):
        super().__init__(image_encoder, **kwargs)
        self.upernet_hidden_size = upernet_hidden_size
        self.upernet_pool_scales = tuple(upernet_pool_scales)

    def _make_head(self, feat_dim: int, out_size: int) -> nn.Module:
        return _UPerNetSegHead(
            in_channels=feat_dim,
            num_classes=self.num_classes,
            out_size=out_size,
            hidden_size=self.upernet_hidden_size,
            pool_scales=self.upernet_pool_scales,
        ).to(self.device)

    def _head_config(self) -> dict:
        return {
            "upernet_hidden_size": self.upernet_hidden_size,
            "upernet_pool_scales": list(self.upernet_pool_scales),
        }
