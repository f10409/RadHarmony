"""timm NaFlex ViT backbone adapted for single-channel grayscale input."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class GrayscaleViTBackbone(nn.Module):
    """NaFlex ViT-Base initialised from ImageNet weights, adapted for
    single-channel medical images.

    NaFlex ViT natively supports variable-resolution inputs, eliminating the
    need for manual positional-embedding interpolation.  The ``_gap`` variant
    uses global average pooling over patch tokens.

    NaFlex uses a linear patch embedding (not Conv2d), so timm's built-in
    ``in_chans`` adaptation does not work.  Instead we load with 3 channels
    and average the RGB weights into a single-channel linear projection.

    Args:
        model_name: timm model identifier.
        img_size: Spatial resolution of every input crop.
        pretrained: Whether to load pretrained ImageNet weights.
    """

    def __init__(
        self,
        model_name: str = "naflexvit_base_patch16_gap.e300_s576_in1k",
        img_size: int = 256,
        pretrained: bool = True,
    ):
        super().__init__()
        self.img_size = img_size

        # Load with original 3 channels — timm's in_chans adaptation
        # fails for NaFlex's linear patch embed (2D weight, not 4D Conv2d).
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,  # remove classifier head, output pooled features
        )

        # Adapt patch embedding from 3-channel to 1-channel.
        # NaFlex patch_embed.proj is nn.Linear(3*P*P, D).
        # Average across the 3 RGB input groups to get nn.Linear(1*P*P, D).
        proj = self.model.embeds.proj
        old_weight = proj.weight.data  # (D, 3*P*P)
        D, in_features_3 = old_weight.shape
        pp = in_features_3 // 3  # P*P

        new_weight = old_weight.reshape(D, 3, pp).mean(dim=1)  # (D, P*P)
        new_proj = nn.Linear(pp, D, bias=proj.bias is not None)
        new_proj.weight.data = new_weight
        if proj.bias is not None:
            new_proj.bias.data = proj.bias.data.clone()

        self.model.embeds.proj = new_proj

    @property
    def embed_dim(self) -> int:
        return self.model.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return global-average-pooled patch token features.

        Args:
            x: ``(B, 1, img_size, img_size)``

        Returns:
            ``(B, embed_dim)``
        """
        return self.model(x)
