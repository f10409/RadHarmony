"""Ark+ backbone recipe (Swin Transformer Large, cyclic multi-task pretraining).

Ark+ is a vision-only chest-X-ray foundation model. It uses a bare
``timm.models.swin_transformer.SwinTransformer`` (Swin-L/192 @ 768 px, window 12).
It is pretrained on six CXR datasets via cyclic multi-task learning. Weights ship
under ``checkpoint["teacher"]`` in ``Ark6_swinLarge768_ep50.pth.tar``.

The checkpoint was saved with ``timm==0.5.4``, whose Swin layout differs
from modern timm. In 0.5.4, ``downsample`` sits at the end of each
``BasicLayer``. In newer timm it sits at the start, so every ``downsample``
key is one layer index off. On first call this recipe side-installs timm 0.5.4 into
``<repo_root>/third_party_models/Ark/timm-054`` via
``uv pip install --target --no-deps`` and uses it only for the duration of
model construction. The rest of the Python session keeps the venv's timm,
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
# The recipe follows the official loading code from the Ark_Plus README.
#   1. Build a bare timm SwinTransformer (no custom OmniSwinTransformer subclass).
#   2. Strip the "module." prefix from checkpoint["teacher"].
#   3. Drop the attn_mask buffers (rebuilt at module init for the input size)
#      and head.{weight,bias} (per task, not used for feature extraction).
#   4. Load with strict=False, which ignores the omni_heads keys of the
#      multi-task checkpoint.
#
# The projector (one Linear, 1536 -> 1376) is loaded. The authors' linear
# probe uses its output (D-16, Q-2).
#
# Side-by-side timm 0.5.4 install.
#   - uv pip install --target <legacy_dir> --no-deps timm==0.5.4
#     `--no-deps` is critical. Without it, the old torchvision of timm 0.5.4
#     goes into <legacy_dir>. Then sys.path picks it up before the venv's
#     torchvision. This gives an ABI clash ("operator torchvision::nms
#     does not exist").
#
#   Then, around the SwinTransformer import we (a) pop any modern-timm modules
#     from sys.modules and (b) prepend <legacy_dir> to sys.path.
#     Then a `finally:` restores the original sys.modules and sys.path.
#     So later code in the same session sees the venv's timm again.
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
# The authors' fine-tuning eval resizes to 896 x 896, then center-crops 768
# (Ark_Plus README: --img_size 896 --input_size 768, and
# Finetuning/dataloader.py build_transform_classification, "valid" mode).
_RESIZE = 896
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
# --projector_features 1376 in the Ark_Plus README, and emb_dim=1376 in
# Linearprobing/main_linearprobing.py.
_PROJECTOR_FEATURES = 1376

# Default: <repo_root>/third_party_models/Ark
_DEFAULT_ARK_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party_models", "Ark")
)
_DEFAULT_LEGACY_TIMM_DIR = os.path.join(_DEFAULT_ARK_DIR, "timm-054")
_DEFAULT_CHECKPOINT_PATH = os.path.join(_DEFAULT_ARK_DIR, "Ark_Plus", "Ark6_swinLarge768_ep50.pth.tar")


def _image_transform():
    """Return the authors' eval transform for a ``[3, H, W]`` float image in [0, 1].

    Resize to 896 x 896 (bilinear, the torchvision default in the authors'
    fine-tuning code), take one center crop of 768, then ImageNet normalize.
    The authors' pretraining code resizes with PIL (bicubic) instead.
    """
    from torchvision import transforms as tv_transforms  # type: ignore[import-untyped]

    return tv_transforms.Compose([
        tv_transforms.Resize((_RESIZE, _RESIZE), antialias=True),
        tv_transforms.CenterCrop(_INPUT_SIZE),
        tv_transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _mask_transform():
    """Return the mask transform: the image geometry, NEAREST, no normalization."""
    from torchvision import transforms as tv_transforms  # type: ignore[import-untyped]
    from torchvision.transforms.functional import InterpolationMode  # type: ignore[import-untyped]

    return tv_transforms.Compose([
        tv_transforms.Resize((_RESIZE, _RESIZE), interpolation=InterpolationMode.NEAREST),
        tv_transforms.CenterCrop(_INPUT_SIZE),
    ])


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

    This builds a bare ``timm.models.swin_transformer.SwinTransformer``
    (Swin-L/192 @ 768 px, window 12). It loads the teacher state dict from
    ``Ark6_swinLarge768_ep50.pth.tar``. timm 0.5.4 is installed
    into a side-by-side directory on first call. It is used only for the
    duration of model construction. The rest of the Python session keeps
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
        Target device for the encoder (for example ``"cuda"``, ``"cuda:7"``,
        ``"cpu"``). The transform's preprocess step always runs on CPU
        in DataLoader workers regardless of this setting.
    output_keys :
        Keys to keep in each sample dict. Defaults to ``{"img", "cls"}``.
        Pass ``{"img", "mask"}`` or ``{"img", "cls", "mask"}`` for
        segmentation. The mask gets the image geometry (resize 896,
        center-crop 768) with NEAREST interpolation and no normalization.

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
        Classification mode (default): ``forward(x) -> Tensor[B, 1376]``.
        This is the projector output of the pooled Swin-L feature, as in the
        authors' linear probe: ``generate_embeddings(x, after_proj=True)``
        (D-16). Before 2026-11-03, it was ``forward_features`` (1536 values).

        Segmentation mode (``"mask"`` in *output_keys*):
        ``forward(x) -> Tensor[B, 1536, 24, 24]``, spatial feature map
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
    from ._utils import display_255, load_image, squeeze_frame_dim, to_3channel, to_uint8

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
        # The authors' ArkSwinTransformer adds this projector after forward_features.
        model.projector = torch.nn.Linear(model.num_features, _PROJECTOR_FEATURES)
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = {k.replace("module.", ""): v for k, v in raw["teacher"].items()}
        for k in [k for k in state if "attn_mask" in k] + ["head.weight", "head.bias"]:
            state.pop(k, None)
        missing = model.load_state_dict(state, strict=False).missing_keys
        if any(k.startswith("projector.") for k in missing):
            raise KeyError(f"The Ark+ checkpoint has no projector weights: {ckpt_path}")
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
        load_image(),
        mt.Lambda(squeeze_frame_dim),
        mt.Transpose(indices=[0, 2, 1]),
        mt.Lambda(display_255),
        mt.Lambda(to_3channel),
        mt.Lambda(to_uint8),
    ])
    _loader_mask = mt.Compose([
        mt.LoadImage(image_only=True, ensure_channel_first=True),
        mt.Lambda(squeeze_frame_dim),
        mt.Transpose(indices=[0, 2, 1]),
    ])

    # ── Image transform (resize 896, center-crop 768, ImageNet norm) ──────────
    _tv_transform = _image_transform()

    def _preprocess(path) -> torch.Tensor:
        img = _loader(str(path)).float() / 255.0  # [3, H, W] float in [0, 1]
        return _tv_transform(img)                 # [3, 768, 768]

    # ── Mask transform (same geometry, NEAREST, no intensity normalization) ───
    _tv_transform_mask = _mask_transform()

    def _preprocess_mask(path) -> torch.Tensor:
        mask = _loader_mask(str(path))[:1].float()  # [1, H, W]
        return _tv_transform_mask(mask)             # [1, 768, 768]

    # ── Model call ────────────────────────────────────────────────────────────
    _seg_mode = output_keys is not None and "mask" in output_keys

    if _seg_mode:
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            # Swin is hierarchical. M.norm's output is [B, H*W, C] before
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
            # The authors' generate_embeddings(x, after_proj=True) (D-16, Q-2).
            return m.projector(m.forward_features(x))  # [B, 1376]

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
