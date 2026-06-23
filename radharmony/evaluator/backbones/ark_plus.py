"""Ark+ backbone recipe (Swin Transformer Large, cyclic multi-task pretraining).

Ark+ is a vision-only chest-X-ray foundation model that uses a bare
``timm.models.swin_transformer.SwinTransformer`` (Swin-L/192 @ 768 px, window 12)
pretrained on six CXR datasets via cyclic multi-task learning. Weights ship
under ``checkpoint["teacher"]`` in ``Ark6_swinLarge768_ep50.pth.tar``.

The checkpoint was saved with ``timm==0.5.4``, whose Swin layout differs
from modern timm (``downsample`` sits at the end of each ``BasicLayer`` in
0.5.4, but at the start in newer timm — every ``downsample`` key is one
layer index off). On first call this recipe side-installs timm 0.5.4 into
``<repo_root>/third_party_models/Ark/timm-054`` via
``uv pip install --target --no-deps`` and uses it only for the duration of
model construction; the rest of the Python session keeps the venv's timm,
so EVA-X and other timm-based backbones still work.

Weights are not auto-downloadable:

1. Clone the repo::

       git clone https://github.com/jlianglab/Ark <ark_dir>

2. Request weights at:
   https://forms.gle/qkoDGXNiKRPTDdCe8 (Google Form)
   or https://www.wjx.cn/vm/OvwfYFx.aspx (WeChat).

3. Place ``Ark6_swinLarge768_ep50.pth.tar`` at:
   ``<ark_dir>/Ark_Plus/Ark6_swinLarge768_ep50.pth.tar``.

Requires the ``ark_plus`` extra::

    uv pip install -e ".[ark_plus]"
"""

# ──────────────────────────────────────────────────────────────────────────────
# Implementation notes
# ──────────────────────────────────────────────────────────────────────────────
# The recipe follows the official loading code from the Ark_Plus README:
#   - bare timm SwinTransformer (no custom OmniSwinTransformer subclass)
#   - strip "module." prefix from checkpoint["teacher"]
#   - drop attn_mask buffers (input-size-dependent; rebuilt at module init)
#     and head.{weight,bias} (per-task; not used for feature extraction)
#   - load_state_dict(strict=False) to ignore the projector + omni_heads
#     keys present in the original multi-task checkpoint
#
# Side-by-side timm 0.5.4 install:
#   - uv pip install --target <legacy_dir> --no-deps timm==0.5.4
#     `--no-deps` is critical: without it, timm 0.5.4's old torchvision is
#     bundled into <legacy_dir>, sys.path picks it up ahead of the venv's
#     torchvision, and you get an ABI clash ("operator torchvision::nms
#     does not exist").
#   - Around the SwinTransformer import we (a) pop any modern-timm modules
#     from sys.modules, (b) prepend <legacy_dir> to sys.path, then in a
#     `finally:` restore the original sys.modules + sys.path so subsequent
#     code in the same session sees the venv's timm again.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import torch
import monai.transforms as mt

from radharmony.evaluator.transforms import EncoderPreprocessTransform, RadiologyEncoderTransform
from radharmony.evaluator.wrappers import ImageEncoderWrapper
from ._utils import patches_to_spatial


_INPUT_SIZE = 768
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Default: <repo_root>/third_party_models/Ark
_DEFAULT_ARK_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party_models", "Ark")
)
_DEFAULT_LEGACY_TIMM_DIR = os.path.join(_DEFAULT_ARK_DIR, "timm-054")
_DEFAULT_CHECKPOINT_PATH = os.path.join(_DEFAULT_ARK_DIR, "Ark_Plus", "Ark6_swinLarge768_ep50.pth.tar")


def _ensure_legacy_timm(legacy_dir: str) -> None:
    """Install timm==0.5.4 into legacy_dir if absent or if a prior install
    bundled torchvision (the broken-with-deps state from --no-deps omission)."""
    legacy_dir = os.path.abspath(legacy_dir)
    needs_install = (
        not os.path.exists(legacy_dir)
        or os.path.exists(os.path.join(legacy_dir, "torchvision"))
    )
    if needs_install:
        if os.path.exists(legacy_dir):
            shutil.rmtree(legacy_dir)
        subprocess.run(
            ["uv", "pip", "install",
             "--target", legacy_dir,
             "--python", sys.executable,
             "--no-deps",
             "timm==0.5.4"],
            check=True,
        )


def make_ark_plus(
    checkpoint_path: str | None = None,
    legacy_timm_dir: str | None = None,
    device: str = "cuda",
    output_keys: set[str] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[mt.Compose, ImageEncoderWrapper]:
    """Return ``(transform, encoder)`` for Ark+.

    A bare ``timm.models.swin_transformer.SwinTransformer`` (Swin-L/192 @
    768 px, window 12) is built and the teacher state dict from
    ``Ark6_swinLarge768_ep50.pth.tar`` is loaded. timm 0.5.4 is installed
    into a side-by-side directory on first call and used only for the
    duration of model construction — the rest of the Python session keeps
    using your venv's timm.

    Parameters
    ----------
    checkpoint_path :
        Path to ``Ark6_swinLarge768_ep50.pth.tar``. Defaults to
        ``<repo_root>/third_party_models/Ark/Ark_Plus/Ark6_swinLarge768_ep50.pth.tar``.
    legacy_timm_dir :
        Directory where ``timm==0.5.4`` is installed via
        ``uv pip install --target --no-deps``. Defaults to
        ``<repo_root>/third_party_models/Ark/timm-054``.
    device :
        Target device for the encoder (e.g. ``"cuda"``, ``"cuda:7"``,
        ``"cpu"``). The transform's preprocess step always runs on CPU
        in DataLoader workers regardless of this setting.
    output_keys :
        Keys to keep in each sample dict. Defaults to ``{"img", "cls"}``.
        Pass ``{"img", "mask"}`` or ``{"img", "cls", "mask"}`` for
        segmentation — the mask is resized to 768×768 with NEAREST
        interpolation and no normalization.

    dtype :
        Output tensor dtype set by the final ``ToTensorD``. Defaults to
        ``torch.bfloat16`` to match the autocast bridge in
        ``BaseClsEvaluator``/``BaseSegEvaluator``. Pass
        ``torch.float32`` to skip autocast entirely for an encoder that
        cannot autocast.

    Returns
    -------
    transform : monai.transforms.Compose
        MONAI transform suitable for passing to a RadHarmony dataset via
        the ``transform=`` kwarg.
    encoder : ImageEncoderWrapper
        Classification mode (default): ``forward(x) -> Tensor[B, 1536]``
        — pooled Swin-L features from ``forward_features``.

        Segmentation mode (``"mask"`` in *output_keys*):
        ``forward(x) -> Tensor[B, 1536, 24, 24]`` — spatial feature map
        captured from the final ``model.norm`` LayerNorm
        (stride 32 → 768 / 32 = 24).

    Example
    -------
    ::

        from radharmony.evaluator.backbones import make_ark_plus
        from radharmony.dataset import VinDrCXRTrainDataset

        transform, encoder = make_ark_plus(device="cuda:0")

        ds = VinDrCXRTrainDataset(
            base_image_dir="/data/vindr/train/",
            transform=transform,
            cache_dir="/tmp/cache/ark_plus/",
            output_cls=True,
        )
    """
    # ── Lazy imports (optional deps) ──────────────────────────────────────────
    from torchvision import transforms as tv_transforms  # type: ignore[import-untyped]
    from torchvision.transforms.functional import InterpolationMode  # type: ignore[import-untyped]
    from ._utils import squeeze_frame_dim, to_3channel, normalize_to_uint8

    legacy_dir = os.path.abspath(legacy_timm_dir or _DEFAULT_LEGACY_TIMM_DIR)
    ckpt_path = os.path.abspath(checkpoint_path or _DEFAULT_CHECKPOINT_PATH)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Ark+ checkpoint not found: {ckpt_path}\n"
            "Request weights at https://forms.gle/qkoDGXNiKRPTDdCe8 "
            "(or WeChat: https://www.wjx.cn/vm/OvwfYFx.aspx)"
        )

    _ensure_legacy_timm(legacy_dir)

    # ── Swap legacy timm onto sys.path for the duration of model construction ─
    _saved_modules = {
        k: sys.modules[k] for k in list(sys.modules)
        if k == "timm" or k.startswith("timm.")
    }
    for k in _saved_modules:
        del sys.modules[k]
    _path_added = legacy_dir not in sys.path
    if _path_added:
        sys.path.insert(0, legacy_dir)
    try:
        from timm.models.swin_transformer import SwinTransformer  # type: ignore[import-untyped]

        model = SwinTransformer(
            num_classes=1,
            img_size=_INPUT_SIZE,
            patch_size=4,
            window_size=12,
            embed_dim=192,
            depths=(2, 2, 18, 2),
            num_heads=(6, 12, 24, 48),
        )
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = {k.replace("module.", ""): v for k, v in raw["teacher"].items()}
        for k in [k for k in state if "attn_mask" in k] + ["head.weight", "head.bias"]:
            state.pop(k, None)
        model.load_state_dict(state, strict=False)
        model.eval()
    finally:
        for k in list(sys.modules):
            if k == "timm" or k.startswith("timm."):
                del sys.modules[k]
        if _path_added and legacy_dir in sys.path:
            sys.path.remove(legacy_dir)
        sys.modules.update(_saved_modules)

    # ── MONAI loader (handles PNG, JPEG, DICOM, NIfTI, …) ─────────────────────
    _loader = mt.Compose([
        mt.LoadImage(image_only=True, ensure_channel_first=True),
        mt.Lambda(squeeze_frame_dim),
        mt.Transpose(indices=[0, 2, 1]),
        mt.Lambda(to_3channel),
        mt.Lambda(normalize_to_uint8),
    ])
    _loader_mask = mt.Compose([
        mt.LoadImage(image_only=True, ensure_channel_first=True),
        mt.Lambda(squeeze_frame_dim),
        mt.Transpose(indices=[0, 2, 1]),
    ])

    # ── Image transform (ImageNet norm, BICUBIC resize to 768×768) ────────────
    _tv_transform = tv_transforms.Compose([
        tv_transforms.Resize((_INPUT_SIZE, _INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
        tv_transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    def _preprocess(path) -> torch.Tensor:
        img = _loader(str(path)).float() / 255.0  # [3, H, W] float in [0, 1]
        return _tv_transform(img)                 # [3, 768, 768]

    # ── Mask transform (NEAREST resize, no intensity normalization) ───────────
    _tv_transform_mask = tv_transforms.Compose([
        tv_transforms.Resize((_INPUT_SIZE, _INPUT_SIZE), interpolation=InterpolationMode.NEAREST),
    ])

    def _preprocess_mask(path) -> torch.Tensor:
        mask = _loader_mask(str(path))[:1].float()  # [1, H, W]
        return _tv_transform_mask(mask)             # [1, 768, 768]

    # ── Model call ────────────────────────────────────────────────────────────
    _seg_mode = output_keys is not None and "mask" in output_keys

    if _seg_mode:
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            # Swin is hierarchical; m.norm's output is [B, H*W, C] before
            # the final global avg-pool. Hook it to grab the spatial map.
            captured: dict[str, torch.Tensor] = {}

            def _hook(_module, _inp, out):
                captured["feat"] = out

            handle = m.norm.register_forward_hook(_hook)
            try:
                _ = m.forward_features(x)
            finally:
                handle.remove()
            return patches_to_spatial(captured["feat"])  # [B, 1536, 24, 24]
    else:
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            return m.forward_features(x)  # [B, 1536]

    # ── Assemble transform + encoder ──────────────────────────────────────────
    pp = EncoderPreprocessTransform.from_custom(keys=["img"], preprocess=_preprocess)
    transform = (
        RadiologyEncoderTransform(
            preprocess=pp,
            output_keys=output_keys,
            mask_preprocess=_preprocess_mask,
            dtype=dtype,
        )
        .get_transform()
    )
    encoder = ImageEncoderWrapper.from_custom(model, model_call=_model_call).to(device)

    return transform, encoder
