"""Multi-crop transforms for LeJEPA 2D training."""

import torch
from monai import transforms as mt


class MultiCropTransform:
    """Generate multiple random crops (global + local) for LeJEPA training.

    Each crop is resized to ``img_size x img_size`` so that a standard ViT
    with fixed positional embeddings can process every view uniformly.

    Args:
        num_global_views: Number of large-scale crops.
        num_local_views: Number of small-scale crops.
        global_scale: (min, max) fraction of the image area for global crops.
        local_scale: (min, max) fraction of the image area for local crops.
        img_size: Target spatial size for every crop.
    """

    def __init__(
        self,
        num_global_views: int = 2,
        num_local_views: int = 6,
        global_scale: tuple[float, float] = (0.55, 1.0),
        local_scale: tuple[float, float] = (0.22, 0.55),
        img_size: int = 384,
    ):
        self.num_global_views = num_global_views
        self.num_local_views = num_local_views
        self.img_size = img_size

        self.global_pipeline = self._build_pipeline(global_scale[0], global_scale[1])
        self.local_pipeline = self._build_pipeline(local_scale[0], local_scale[1])

    def _build_pipeline(self, min_scale: float, max_scale: float) -> mt.Compose:
        return mt.Compose([
            mt.RandScaleCrop(
                roi_scale=min_scale,
                max_roi_scale=max_scale,
                random_center=True,
                random_size=True,
            ),
            mt.Resize(spatial_size=(self.img_size, self.img_size), mode="bilinear"),
            mt.RandFlip(prob=0.5, spatial_axis=[1]),
            mt.RandAdjustContrast(gamma=(0.85, 1.15), prob=0.3),
            mt.RandScaleIntensity(factors=0.1, prob=0.3),
            mt.RandShiftIntensity(offsets=0.05, prob=0.3),
            mt.RandGaussianSmooth(sigma_x=(0.1, 2.0), prob=0.5),
        ])

    def __call__(self, img: torch.Tensor) -> list[torch.Tensor]:
        """Apply multi-crop to a single image tensor (C, H, W)."""
        crops: list[torch.Tensor] = []
        for _ in range(self.num_global_views):
            crops.append(self.global_pipeline(img))
        for _ in range(self.num_local_views):
            crops.append(self.local_pipeline(img))
        return crops


class EvalMultiCropTransform:
    """Deterministic multi-crop for validation: 1 global + 5 local crops.

    Local crops are placed at 4 corners and center,
    each at ~44% of the full image size. All crops resized to ``img_size``.
    """

    def __init__(self, img_size: int = 384):
        self.img_size = img_size
        self.resize = mt.Resize(
            spatial_size=(img_size, img_size), mode="bilinear",
        )

    def __call__(self, img: torch.Tensor) -> list[torch.Tensor]:
        """Return 6 views: 1 global (full image) + 5 deterministic local."""
        # Global view — resize full image
        global_view = self.resize(img)
        views = [global_view]

        _, h_g, w_g = img.shape
        h_l = h_g // 2
        w_l = w_g // 2

        offsets = [
            (0, 0),                                    # top-left
            (0, w_g - w_l),                            # top-right
            (h_g - h_l, 0),                            # bottom-left
            (h_g - h_l, w_g - w_l),                    # bottom-right
            ((h_g - h_l) // 2, (w_g - w_l) // 2),     # center
        ]

        for h_off, w_off in offsets:
            crop = img[:, h_off : h_off + h_l, w_off : w_off + w_l]
            views.append(self.resize(crop))

        return views
