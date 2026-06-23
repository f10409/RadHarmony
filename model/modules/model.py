"""LeJEPA 2D Lightning module — self-supervised pretraining with online probe."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import lejepa
from einops import rearrange, reduce, repeat
from torchmetrics.classification import BinaryAUROC

from .backbone import GrayscaleViTBackbone


class LeJEPA2D(L.LightningModule):
    """Self-supervised LeJEPA pretraining on 2-D chest X-rays.

    Architecture mirrors ``Custom_LeJEPA/modules/sparse_model.py`` (sLeJEPA)
    but replaces the sparse ViT with a timm NaFlex ViT backbone and
    removes all 3-D / sparse-attention code paths.

    Loss:  ``(1 - λ) · prediction_MSE  +  λ · SIGReg  +  probe_BCE``

    Args:
        model_name: timm model identifier.
        img_size: Spatial size of every input crop.
        proj_dim: Output dimension of the projector MLP.
        lam: Final SIGReg weight (after warmup).
        lam_warmup_steps: Linear warmup steps for λ.
        num_global_views: Number of global crops per image.
        num_local_views: Number of local crops per image.
        lr: Peak learning rate (before world-size scaling).
        weight_decay: AdamW weight decay.
        batch_size: Per-GPU batch size (used for LR scaling).
        gradient_checkpointing: Enable gradient checkpointing on the backbone.
    """

    def __init__(
        self,
        model_name: str = "naflexvit_base_patch16_gap.e300_s576_in1k",
        img_size: int = 256,
        proj_dim: int = 128,
        lam: float = 0.025,
        lam_warmup_steps: int = 1000,
        num_global_views: int = 2,
        num_local_views: int = 8,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        batch_size: int = 128,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.warmup_steps = lam_warmup_steps

        # Backbone
        self.backbone = GrayscaleViTBackbone(
            model_name=model_name,
            img_size=img_size,
        )
        if gradient_checkpointing:
            self.backbone.model.set_grad_checkpointing(True)

        embed_dim = self.backbone.embed_dim

        # Projector  (768 → 2048 → 2048 → proj_dim)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Linear(2048, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # SIGReg regulariser
        univariate = lejepa.univariate.EppsPulley(n_points=17)
        self.sigreg_loss = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=univariate,
            num_slices=1024,
        )

        # Online pneumothorax probe (binary)
        self.online_classifier = nn.Linear(embed_dim, 1)
        self.val_auroc = BinaryAUROC()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_current_lam(self) -> float:
        if self.global_step < self.warmup_steps:
            return self.hparams.lam * (self.global_step / self.warmup_steps)
        return self.hparams.lam

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, views: torch.Tensor) -> torch.Tensor:
        """views: (B*V, 1, S, S) → features (B*V, embed_dim)."""
        return self.backbone(views)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        views = batch["views"]  # (B*V, 1, S, S)
        labels = batch["labels"]  # (B, 1)
        batch_size = batch["batch_size"]
        n_global = self.hparams.num_global_views
        n_local = self.hparams.num_local_views
        total_views = n_global + n_local

        # 1. Backbone → features
        all_feats = self(views)  # (B*V, D)

        # 2. Projector → embeddings
        all_embs = self.projector(all_feats)  # (B*V, proj_dim)

        # 3. Reshape to (B, V, proj_dim)
        all_embs_r = rearrange(
            all_embs,
            "(b v) d -> b v d",
            b=batch_size,
            v=total_views,
        )
        global_embs = rearrange(all_embs_r[:, :n_global], "b v d -> (b v) d")
        local_embs = rearrange(all_embs_r[:, n_global:], "b v d -> (b v) d")

        # 4. Online pneumothorax probe (detached features, first global view)
        probe_feats = rearrange(
            all_feats,
            "(b v) d -> b v d",
            b=batch_size,
            v=total_views,
        )[
            :, 0, :
        ].detach()  # important
        probe_logits = self.online_classifier(probe_feats)  # (B, 1)

        has_label = ~torch.isnan(labels[:, 0])
        if has_label.any():
            probe_loss = F.binary_cross_entropy_with_logits(
                probe_logits[has_label],
                labels[has_label],
            )
        else:
            probe_loss = torch.tensor(0.0, device=self.device)

        # 5. SIGReg loss
        if self.trainer.world_size > 1:
            gathered = self.all_gather(all_embs, sync_grads=True)
            pool_embs = rearrange(gathered, "rank b d -> (rank b) d")
        else:
            pool_embs = all_embs
        reg_loss = self.sigreg_loss(pool_embs)

        # 6. Prediction loss (JEPA target = mean of global views)
        target = reduce(all_embs_r[:, :n_global], "b v d -> b d", "mean")
        target_g = repeat(target, "b d -> (b v) d", v=n_global)
        target_l = repeat(target, "b d -> (b v) d", v=n_local)
        loss_g = F.mse_loss(global_embs, target_g) * n_global
        loss_l = F.mse_loss(local_embs, target_l)
        pred_loss = (loss_g + n_local * loss_l) / (n_global + n_local)

        # 7. Combine
        curr_lam = self._get_current_lam()
        ssl_loss = (1 - curr_lam) * pred_loss + curr_lam * reg_loss
        total_loss = ssl_loss + probe_loss

        # Logging
        self.log("train/ssl_loss", ssl_loss, batch_size=batch_size, sync_dist=True)
        self.log("train/pred_loss", pred_loss, batch_size=batch_size, sync_dist=True)
        self.log("train/reg_loss", reg_loss, batch_size=batch_size, sync_dist=True)
        self.log("train/probe_loss", probe_loss, batch_size=batch_size, sync_dist=True)
        self.log("train/total_loss", total_loss, batch_size=batch_size, sync_dist=True)

        return total_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validation_step(self, batch: dict, batch_idx: int) -> None:
        views = batch["views"]
        labels = batch["labels"]
        batch_size = batch["batch_size"]

        # Eval uses 1 global + 9 local = 10 views
        n_global = 1
        n_local = batch["num_views"] - 1

        all_feats = self(views)
        all_embs = self.projector(all_feats)

        total_views = n_global + n_local
        all_embs_r = rearrange(
            all_embs,
            "(b v) d -> b v d",
            b=batch_size,
            v=total_views,
        )
        global_embs = rearrange(all_embs_r[:, :n_global], "b v d -> (b v) d")
        local_embs = rearrange(all_embs_r[:, n_global:], "b v d -> (b v) d")

        # SSL losses
        if self.trainer.world_size > 1:
            gathered = self.all_gather(all_embs, sync_grads=False)
            pool_embs = rearrange(gathered, "rank b d -> (rank b) d")
        else:
            pool_embs = all_embs
        reg_loss = self.sigreg_loss(pool_embs)

        target = reduce(all_embs_r[:, :n_global], "b v d -> b d", "mean")
        target_g = repeat(target, "b d -> (b v) d", v=n_global)
        target_l = repeat(target, "b d -> (b v) d", v=n_local)
        loss_g = F.mse_loss(global_embs, target_g) * n_global
        loss_l = F.mse_loss(local_embs, target_l)
        pred_loss = (loss_g + n_local * loss_l) / (n_global + n_local)

        curr_lam = self._get_current_lam()
        ssl_loss = (1 - curr_lam) * pred_loss + curr_lam * reg_loss

        self.log("val/ssl_loss", ssl_loss, batch_size=batch_size, sync_dist=True)
        self.log("val/pred_loss", pred_loss, batch_size=batch_size, sync_dist=True)
        self.log("val/reg_loss", reg_loss, batch_size=batch_size, sync_dist=True)

        # Probe AUROC
        probe_feats = rearrange(
            all_feats,
            "(b v) d -> b v d",
            b=batch_size,
            v=total_views,
        )[:, 0, :].detach()
        probe_logits = self.online_classifier(probe_feats)

        has_label = ~torch.isnan(labels[:, 0])
        if has_label.any():
            probe_loss = F.binary_cross_entropy_with_logits(
                probe_logits[has_label],
                labels[has_label],
            )
            self.log(
                "val/probe_loss", probe_loss, batch_size=batch_size, sync_dist=True
            )
            self.val_auroc.update(
                probe_logits[has_label].sigmoid(),
                labels[has_label].int(),
            )

    def on_validation_epoch_end(self) -> None:
        try:
            auroc = self.val_auroc.compute()
            self.log("val/ptx_auroc", auroc, sync_dist=True)
        except (ValueError, RuntimeError):
            pass  # not enough samples yet
        self.val_auroc.reset()

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=total_steps,
            pct_start=0.025,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=1.0,
        )

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]
