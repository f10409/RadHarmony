"""
Custom LeJEPA encoder matching the checkpoint architecture from
mimic-cxp-from-imagenet-dp01-bs160-e10_last.ckpt.

Architecture (reverse-engineered from checkpoint state_dict):
  - NaFlex-style single-channel patch embedding  (patch_size=16, img_size=256)
  - Learnable cls_token + 4 register_tokens
  - 12 transformer blocks with:
      * Pre-norm (bias-free LayerNorm)
      * Scaled cosine attention  (logit_scale per head)
      * SwiGLU / GluMLP  (w12 → split → SiLU-gate → norm → w3)
  - Final bias-free LayerNorm
  - Global-average-pool over patch tokens → (B, 768)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Sub-modules ───────────────────────────────────────────────────────────────

class _NaFlexPatchEmbed(nn.Module):
    """Single-resolution NaFlex patch embedding.

    Checkpoint keys (prefix stripped):
        patch_embed.projs.{P*P}.weight   (D, P*P)
        patch_embed.projs.{P*P}.bias     (D,)
        patch_embed.norm.weight          (D,)
        patch_embed.patch_type_embeds.{P*P}  (D,)
    """
    def __init__(self, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        pp = in_chans * patch_size * patch_size
        self.proj = nn.Linear(pp, embed_dim)
        self.norm = nn.LayerNorm(embed_dim, bias=False)
        self.type_embed = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, N, D)"""
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * p * p)
        x = self.proj(x) + self.type_embed
        x = self.norm(x)
        return x


class _CosineSelfAttn(nn.Module):
    """Scaled cosine multi-head self-attention.

    QKV projected jointly; Q and K are L2-normalised before dot-product;
    attention logits are scaled by a learnable per-head temperature
    (logit_scale).
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.qkv  = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones(num_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                      # (B, H, N, Dh)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()  # (H,)
        attn = (q @ k.transpose(-2, -1)) * scale.view(1, H, 1, 1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class _GluMlp(nn.Module):
    """SwiGLU MLP.

    Checkpoint keys:
        mlp.w12.weight  (2*hidden, D)   — combined gate+value projection
        mlp.norm.weight (hidden,)       — post-gate LayerNorm
        mlp.w3.weight   (D, hidden)     — output projection
    """
    def __init__(self, embed_dim: int, hidden: int):
        super().__init__()
        self.w12  = nn.Linear(embed_dim, hidden * 2, bias=False)
        self.norm = nn.LayerNorm(hidden, bias=False)
        self.w3   = nn.Linear(hidden, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.norm(x)
        return self.w3(x)


class _Block(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_hidden: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, bias=False)
        self.attn  = _CosineSelfAttn(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim, bias=False)
        self.mlp   = _GluMlp(embed_dim, mlp_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ── Main encoder ──────────────────────────────────────────────────────────────

class LeJEPAEncoder(nn.Module):
    """Encoder extracted from the Custom_LeJEPA checkpoint.

    Args:
        patch_size: Patch size used during pretraining (16).
        img_size:   Input spatial resolution (256).
        embed_dim:  Token embedding dimension (768).
        num_heads:  Attention heads (12).
        depth:      Number of transformer blocks (12).
        mlp_hidden: Hidden dimension of SwiGLU MLP (2048).
        num_regs:   Number of register tokens (4).
    """

    def __init__(
        self,
        patch_size: int = 16,
        img_size: int   = 256,
        embed_dim: int  = 768,
        num_heads: int  = 12,
        depth: int      = 12,
        mlp_hidden: int = 2048,
        num_regs: int   = 4,
        pool_type: str  = "gap",
    ):
        super().__init__()
        assert pool_type in ("gap", "cls"), f"pool_type must be 'gap' or 'cls', got {pool_type!r}"
        self.pool_type      = pool_type
        self.patch_embed    = _NaFlexPatchEmbed(patch_size, in_chans=1, embed_dim=embed_dim)
        self.cls_token      = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens= nn.Parameter(torch.zeros(1, num_regs, embed_dim))
        self.blocks         = nn.ModuleList([
            _Block(embed_dim, num_heads, mlp_hidden) for _ in range(depth)
        ])
        self.norm           = nn.LayerNorm(embed_dim, bias=False)
        self._num_prefix    = 1 + num_regs  # cls + regs come before patch tokens

    @property
    def embed_dim(self) -> int:
        return self.norm.weight.shape[0]

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        device: str = "cpu",
        pool_type: str = "gap",
    ) -> "LeJEPAEncoder":
        """Instantiate and load weights from a Custom_LeJEPA .ckpt file."""
        import torch
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd  = raw["state_dict"]

        # Extract encoder.* and strip the prefix
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items()
                  if k.startswith("encoder.")}

        # Remap patch_embed keys to match this module's attribute names
        remap = {}
        for k, v in enc_sd.items():
            new_k = k
            # patch_embed.projs.{N}.weight/bias  →  patch_embed.proj.weight/bias
            if k.startswith("patch_embed.projs."):
                parts = k.split(".")  # ['patch_embed', 'projs', '256', 'weight']
                new_k = f"patch_embed.proj.{parts[-1]}"
            # patch_embed.patch_type_embeds.{N}  →  patch_embed.type_embed
            elif k.startswith("patch_embed.patch_type_embeds."):
                new_k = "patch_embed.type_embed"
            # patch_embed.norm.*  →  patch_embed.norm.*  (unchanged)
            remap[new_k] = v

        model = cls(pool_type=pool_type)
        missing, unexpected = model.load_state_dict(remap, strict=False)
        if missing:
            print(f"  [LeJEPAEncoder] missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"  [LeJEPAEncoder] unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        return model.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W)  →  (B, embed_dim)."""
        B = x.shape[0]
        tokens = self.patch_embed(x)                                       # (B, N, D)
        cls  = self.cls_token.expand(B, -1, -1)
        regs = self.register_tokens.expand(B, -1, -1)
        tokens = torch.cat([cls, regs, tokens], dim=1)                     # (B, 1+R+N, D)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        if self.pool_type == "cls":
            return tokens[:, 0]                                            # (B, D)
        patch_tokens = tokens[:, self._num_prefix:]                        # (B, N, D)
        return patch_tokens.mean(dim=1)                                    # (B, D)
