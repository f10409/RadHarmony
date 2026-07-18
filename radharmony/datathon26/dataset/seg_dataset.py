"""Dataset that serves pre-extracted PATCH features as dense maps for segmentation.

For the *embed* task the reportbench service writes one ``.npz`` per image holding
``global`` ``[dim]`` and ``patches`` ``[n_patches, dim]`` (+ a ``grid`` ``[H, W]``).
Segmentation needs the **patch** level: this dataset folds ``patches`` back into a
dense feature map ``[dim, H, W]`` (row-major, using the stored ``grid``; falling
back to a square ``H=W=√n`` when ``grid`` is absent) and serves it under the
sample's ``"img"`` key, plus the ground-truth ``"mask"`` from ``mask_path``.

Paired with :class:`IdentityEncoder` the standard RadHarmony segmentation probes
(``LinearProbeSegEvaluator`` and friends) consume the map with no image encode —
the encoder forward becomes a pass-through — so a 1×1-conv (or conv/UPerNet) head
trains on the frozen patch features and is scored with Dice/IoU.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import monai.transforms as mt
from PIL import Image

from radharmony.dataset.base import BaseRadiologicalDataset
from radharmony.registry import register_dataset


def _load_patch_map(path, dtype=torch.float32, grid_key="grid", feat_size=None):
    """Load a reportbench ``.npz`` patch array as a dense map ``[D, H, W]``.

    Module-level (picklable for DataLoader workers). Uses the stored ``grid``
    ``(H, W)`` when present, else a square grid ``H=W=√n_patches`` (raising if
    ``n_patches`` is not a perfect square and no grid is given). When
    ``feat_size`` is set the map is bilinearly resized to ``[D, feat_size,
    feat_size]`` — needed to batch models whose grid varies per image
    (e.g. Qwen2.5-VL dynamic resolution).
    """
    z = np.load(path)
    if hasattr(z, "files"):
        patches = z["patches"] if "patches" in z.files else z[z.files[0]]
        grid = z[grid_key] if grid_key in z.files else None
    else:  # bare .npy of a patch array
        patches, grid = z, None
    patches = np.asarray(patches, dtype=np.float32)
    if patches.ndim != 2:
        raise ValueError(f"{path}: expected [n_patches, dim] patches, got {patches.shape}.")
    n, d = patches.shape
    if grid is not None:
        H, W = int(np.asarray(grid).reshape(-1)[0]), int(np.asarray(grid).reshape(-1)[1])
    else:
        side = int(round(n ** 0.5))
        if side * side != n:
            raise ValueError(
                f"{path}: {n} patches is not a perfect square and the .npz has no "
                f"'{grid_key}' key — cannot fold to a 2-D map. Re-run embed on a "
                "server that writes 'grid', or pass grid_key/feat_size."
            )
        H = W = side
    if H * W != n:
        raise ValueError(f"{path}: grid {H}x{W} != {n} patches.")
    fmap = patches.reshape(H, W, d).transpose(2, 0, 1)  # [D, H, W]
    t = torch.as_tensor(fmap, dtype=torch.float32)
    if feat_size is not None:
        t = F.interpolate(t[None], size=(feat_size, feat_size), mode="bilinear",
                          align_corners=False)[0]
    return t.to(dtype)


def _load_mask(path, size=None, dtype=torch.float32):
    """Load a segmentation mask (``.png``/``.npy``) as ``[1, S, S]`` (picklable)."""
    p = str(path)
    m = np.load(p) if p.lower().endswith(".npy") else np.asarray(Image.open(p).convert("L"))
    m = np.asarray(m)
    if m.ndim == 3:
        m = m[..., 0]
    t = torch.as_tensor(m, dtype=dtype)
    if size is not None:
        t = F.interpolate(t[None, None].float(), size=(size, size), mode="nearest")[0, 0].to(dtype)
    return t[None]  # [1, S, S]


@register_dataset("datathon26_segmentation")
class PatchSegResultsDataset(BaseRadiologicalDataset):
    """Serve one dense patch-feature map ``[D, H, W]`` + a mask per image.

    Args:
        embeddings_dir: Root holding the reportbench embed results
            (``<study>/<model>_<stem>.npz`` under ``_reportbench_out/<model>``).
        harmonized_df / csv_path: Batch frame — **one row per image** — with
            ``patient_id`` / ``study_id`` / ``image_path`` and a segmentation-mask
            column (``mask_col``). Either this or ``csv_path`` is required.
        result_path: Callable ``row -> relative .npz path`` under
            ``embeddings_dir`` (use ``viewwise_embedding_path(model)``).
        mask_col: Column holding each image's mask path (default ``"mask_path"``).
        grid_key: ``.npz`` key with the patch grid ``(H, W)`` (default ``"grid"``).
        mask_size: If set, masks are resized (nearest) to ``mask_size²`` — makes
            them batchable and sets the probe's output resolution.
        feat_size: If set, feature maps are resized to ``feat_size²`` — needed for
            models whose patch grid varies per image (else the native grid is
            used, which requires a fixed grid across the batch).
        cache_dir, dtype: as for the other datathon26 datasets.
    """

    SUPPORTED_OUTPUTS = frozenset({"mask"})
    LABEL_COLS: list = []

    def __init__(
        self,
        embeddings_dir: str,
        *,
        harmonized_df: pd.DataFrame | None = None,
        csv_path: str | None = None,
        result_path=None,
        mask_col: str = "mask_path",
        grid_key: str = "grid",
        mask_size: int | None = None,
        feat_size: int | None = None,
        cache_dir: str | None = None,
        dtype: torch.dtype = torch.float32,
        transform=None,
    ):
        if harmonized_df is None and csv_path is None:
            raise ValueError("Pass either harmonized_df or csv_path.")
        df = harmonized_df.copy() if harmonized_df is not None else pd.read_csv(csv_path)

        self.dtype = dtype
        self._result_path = result_path
        self._mask_col = mask_col

        if transform is None:
            transform = mt.Compose(
                [
                    mt.Lambdad(keys="img", func=functools.partial(
                        _load_patch_map, dtype=dtype, grid_key=grid_key, feat_size=feat_size)),
                    mt.Lambdad(keys="mask", func=functools.partial(
                        _load_mask, size=mask_size, dtype=dtype)),
                    mt.SelectItemsd(keys=["img", "mask"]),
                ]
            )

        super().__init__(
            base_image_dir=embeddings_dir,
            transform=transform,
            cache_dir=cache_dir,
            output_mask=True,
            harmonized_df=df,
        )

    def _get_harmonized_df(self) -> pd.DataFrame:
        df = self._try_resolve_preset_harmonized()
        if df is None:
            raise ValueError("PatchSegResultsDataset requires a preset harmonized_df/csv_path.")
        df = df.copy()
        # Point image_path at the patch .npz so get_data_dict joins it under
        # <embeddings_dir>/<image_path>; the transform folds it to [D, H, W].
        if self._result_path is not None:
            df["image_path"] = df.apply(self._result_path, axis=1).astype(str)
        else:
            df["image_path"] = df["study_id"].astype(str) + "/patches.npz"
        # base emits 'mask' from the 'mask_path' column; honour a custom mask_col.
        if self._mask_col != "mask_path":
            df["mask_path"] = df[self._mask_col]
        return df
