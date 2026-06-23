"""Base class for vision segmentation evaluators.

Adds frozen-encoder dense-feature plumbing on top of :class:`BaseEvaluator`.
Expects:

- An ``image_encoder`` that returns dense patch features ``Tensor[B, D, H', W']``
  — every RadHarmony backbone supports this when built via
  ``make_<backbone>(output_keys={"img", "mask"})``.
- A dataset configured with ``output_mask=True`` so each batch includes
  a ``"mask"`` tensor aligned to the encoder's input image size.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from ..base import BaseEvaluator


class BaseSegEvaluator(BaseEvaluator):
    """Base for segmentation evaluators.

    Subclasses run a (frozen-encoder) training loop and emit a per-row
    metric DataFrame using the segmentation metric panel.
    """

    def __init__(
        self,
        image_encoder,
        *,
        dataset=None,
        train_dataset=None,
        test_dataset=None,
        num_classes: int = 2,
        device: str = "cuda",
        batch_size: int = 8,
        num_workers: int = 4,
        autocast_dtype: torch.dtype | None = torch.bfloat16,
        output_dir: str | None = None,
        n_seeds: int = 1,
        base_seed: int = 0,
        n_bootstrap: int = 0,
        bootstrap_seed: int = 0,
        threshold_strategy: str = "youden",
    ):
        super().__init__(
            dataset=dataset,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            output_dir=output_dir,
            n_seeds=n_seeds,
            base_seed=base_seed,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            threshold_strategy=threshold_strategy,
        )
        self.image_encoder = image_encoder
        self.num_classes = num_classes
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.autocast_dtype = autocast_dtype

    def _freeze_encoder(self):
        if isinstance(self.image_encoder, torch.nn.Module):
            self.image_encoder.eval()
            self.image_encoder.to(self.device)
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)

    @property
    def _autocast_device(self) -> str:
        return "cuda" if self.device.startswith("cuda") else "cpu"

    def _autocast_cm(self):
        """Autocast context manager honoring ``self.autocast_dtype``.

        Bridges the common dtype mismatch where a dataset emits bfloat16
        tensors (``BaseRadiologicalDataset.dtype`` default) but encoder
        weights are fp32 — autocast handles per-op casts transparently.
        """
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(
            device_type=self._autocast_device, dtype=self.autocast_dtype
        )

    def _make_loader(
        self, torch_ds, shuffle: bool, drop_last: bool = False
    ) -> DataLoader:
        return DataLoader(
            torch_ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=drop_last,
        )

    def _probe_shapes(self, sample_loader) -> tuple[int, int]:
        """Run one batch to discover ``(feat_dim, mask_out_size)``.

        Asserts the encoder produces a 4-D dense map. Returns the feature
        channel count and the spatial size of the ground-truth mask
        (assumed square — matches every recipe under ``backbones/``).
        """
        self._freeze_encoder()
        batch = next(iter(sample_loader))
        imgs = batch["img"].to(self.device)
        with torch.no_grad(), self._autocast_cm():
            feats = self.image_encoder(imgs)
        if feats.ndim != 4:
            raise ValueError(
                f"Segmentation encoder must return Tensor[B, D, H, W]; "
                f"got shape {tuple(feats.shape)}. Build the encoder via "
                f"make_<backbone>(output_keys={{'img', 'mask'}})."
            )
        if "mask" not in batch:
            raise KeyError(
                "Dataset did not emit a 'mask' key — pass output_mask=True "
                "when constructing the dataset."
            )
        mask_size = int(batch["mask"].shape[-1])
        return int(feats.shape[1]), mask_size
