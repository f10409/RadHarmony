"""Medical MAE backbone recipe (CXR-pretrained ViT via MAE).

Loads a Vision Transformer pre-trained with Masked Autoencoder (MAE) on a
0.3M/0.5M chest-X-ray corpus from `lambert-x/medical_mae
<https://github.com/lambert-x/medical_mae>`_. Two variants are supported
(``variant=`` kwarg):

==========  ==========  ==========
variant     embed dim   pretrain
==========  ==========  ==========
``small``   384         0.3M CXR
``base``    768         0.5M CXR
==========  ==========  ==========

Input: 224×224 RGB, **chest-X-ray normalization** (``mean=[0.5056]*3``,
``std=[0.252]*3``) — *not* ImageNet — via the canonical
``Augmentation(normalize="chestx-ray").get_augmentation("full_224", "val")``
pipeline (resize 256, center-crop 224).

Two manual steps before first use (auto-handled by the recipe if absent):

1. Clone the upstream repo for ``models_vit`` and ``util.pos_embed``::

       git clone https://github.com/lambert-x/medical_mae third_party_models/medical_mae

2. Download a pretrained checkpoint from the Google Drive link in the repo's
   README and place it under ``third_party_models/medical_mae/``:

   - ViT-Base:  ``vit-b_CXR_0.5M_mae.pth``
   - ViT-Small: ``vit_small_patch16_CXR_0.3M_mae_pretrain.pth``

Requires the ``medical_mae`` extra::

    uv pip install -e ".[medical_mae]"
"""

# ──────────────────────────────────────────────────────────────────────────────
# Implementation notes
# ──────────────────────────────────────────────────────────────────────────────
# The medical_mae ``VisionTransformer`` subclass overrides
# ``forward_features(self, x)``, matching the **timm 0.4.x** parent signature
# (per requirements.txt's ``timm==0.4.12`` pin — *not* the 0.3.2 the entry-point
# script asserts). Modern timm calls ``forward_features(x, attn_mask=...,
# is_causal=...)`` and breaks the override.
#
# On first call we side-install ``timm==0.4.12`` into
# ``<repo_root>/third_party_models/medical_mae/timm-0412/`` via
# ``uv pip install --target --no-deps`` and use it only during model
# construction; the rest of the Python session keeps the venv's timm so EVA-X,
# Ark+, and other timm-based backbones still work.
#
# Loading mirrors ``main_linprobe_chestxray.py``:
#   - checkpoint["model"] has FLAT keys (no ``encoder.`` prefix)
#   - drop ``head.{weight,bias}`` only if shape-mismatched
#   - call ``interpolate_pos_embed(model, state)`` for any non-224 input
#   - load_state_dict(strict=False) — decoder_*, mask_token are expected unexpected
#
# Default mode = CLS token (``global_pool=False``) using the pretrained
# ``self.norm``. ``global_pool=True`` would warm-start ``fc_norm`` from default
# LayerNorm init — intended for fine-tuning, not zero-shot features.
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


_INPUT_SIZE = 224
_RESIZE_SIZE = 256
_CHESTXRAY_MEAN = [0.5056, 0.5056, 0.5056]
_CHESTXRAY_STD = [0.252, 0.252, 0.252]

# Default: <repo_root>/third_party_models/medical_mae
_DEFAULT_MAE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party_models", "medical_mae")
)
_DEFAULT_LEGACY_TIMM_DIR = os.path.join(_DEFAULT_MAE_DIR, "timm-0412")

_DEFAULT_CHECKPOINT_NAMES = {
    "base":  "vit-b_CXR_0.5M_mae.pth",
    "small": "vit_small_patch16_CXR_0.3M_mae_pretrain.pth",
}
_FACTORY_NAMES = {"base": "vit_base_patch16", "small": "vit_small_patch16"}
_EMBED_DIM = {"base": 768, "small": 384}


def _ensure_legacy_timm(legacy_dir: str) -> None:
    """Install timm==0.4.12 into legacy_dir if absent or contaminated.

    ``--no-deps`` is critical: without it, an old torchvision would be bundled
    into legacy_dir and ABI-clash with the venv's torch.
    """
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
             "timm==0.4.12"],
            check=True,
        )


def make_medical_mae(
    variant: str = "base",
    checkpoint_path: str | None = None,
    mae_dir: str | None = None,
    legacy_timm_dir: str | None = None,
    device: str = "cuda",
    output_keys: set[str] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[mt.Compose, ImageEncoderWrapper]:
    """Return ``(transform, encoder)`` for Medical MAE.

    A single ViT instance is created (under side-loaded timm 0.4.12) and
    shared between the transform's preprocess closure and the returned
    ``ImageEncoderWrapper``. No weights are loaded twice.

    Parameters
    ----------
    variant :
        ``"base"`` (768-d, 86M params, pretrained on 0.5M CXRs) or
        ``"small"`` (384-d, 22M params, pretrained on 0.3M CXRs).
    checkpoint_path :
        Path to the pretrained ``.pth``. Defaults to
        ``<mae_dir>/vit-b_CXR_0.5M_mae.pth`` (base) or
        ``<mae_dir>/vit_small_patch16_CXR_0.3M_mae_pretrain.pth`` (small).
    mae_dir :
        Path to the cloned ``lambert-x/medical_mae`` repo. Defaults to
        ``<repo_root>/third_party_models/medical_mae``.
    legacy_timm_dir :
        Directory where ``timm==0.4.12`` is side-installed via
        ``uv pip install --target --no-deps``. Defaults to
        ``<mae_dir>/timm-0412``.
    device :
        Target device for the encoder (e.g. ``"cuda"``, ``"cuda:7"``,
        ``"cpu"``). The transform's preprocess step runs on CPU in DataLoader
        workers regardless.
    output_keys :
        Keys to keep in each sample dict. Defaults to ``{"img", "cls"}``.
        Pass ``{"img", "mask"}`` or ``{"img", "cls", "mask"}`` for
        segmentation — the mask is resized to 224×224 (NEAREST + center-crop)
        with no normalization.

    dtype :
        Output tensor dtype set by the final ``ToTensorD``. Defaults to
        ``torch.bfloat16`` to match the autocast bridge in
        ``BaseClsEvaluator``/``BaseSegEvaluator``. Pass
        ``torch.float32`` to skip autocast entirely for an encoder that
        cannot autocast.

    Returns
    -------
    transform : monai.transforms.Compose
        MONAI transform suitable for passing to a RadHarmony dataset via the
        ``transform=`` kwarg.
    encoder : ImageEncoderWrapper
        Classification mode (default): ``forward(x) -> Tensor[B, D]`` —
        the CLS-token embedding from the pretrained ``self.norm``.
        ``D = 768`` (base) or ``384`` (small).

        Segmentation mode (``"mask"`` in *output_keys*):
        ``forward(x) -> Tensor[B, D, 14, 14]`` — spatial patch feature map
        (224 px / 16 px patches = 14 × 14).

    Example
    -------
    ::

        from radharmony.evaluator.backbones import make_medical_mae
        from radharmony.dataset import VinDrCXRTrainDataset

        transform, encoder = make_medical_mae(variant="base", device="cuda:0")

        ds = VinDrCXRTrainDataset(
            base_image_dir="/data/vindr/train/",
            transform=transform,
            cache_dir="/tmp/cache/medical_mae/",
            output_cls=True,
        )
    """
    if variant not in _EMBED_DIM:
        raise ValueError(
            f"variant must be one of {sorted(_EMBED_DIM)}, got {variant!r}"
        )

    # ── Lazy imports (optional dependency) ────────────────────────────────────
    from torchvision import transforms as tv_transforms  # type: ignore[import-untyped]
    from torchvision.transforms.functional import InterpolationMode  # type: ignore[import-untyped]
    from ._utils import squeeze_frame_dim, to_3channel, normalize_to_uint8

    mae_dir = os.path.abspath(mae_dir or _DEFAULT_MAE_DIR)
    legacy_dir = os.path.abspath(legacy_timm_dir or os.path.join(mae_dir, "timm-0412"))

    if not os.path.isdir(mae_dir):
        raise FileNotFoundError(
            f"medical_mae directory not found: {mae_dir}\n"
            f"Clone with: git clone https://github.com/lambert-x/medical_mae {mae_dir}"
        )

    if checkpoint_path is None:
        checkpoint_path = os.path.join(mae_dir, _DEFAULT_CHECKPOINT_NAMES[variant])
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Medical MAE checkpoint not found: {checkpoint_path}\n"
            "Download from the Google Drive link in the lambert-x/medical_mae README."
        )

    _ensure_legacy_timm(legacy_dir)

    # ── Side-load legacy timm + medical_mae repo for the duration of construction ─
    # The constructed model holds references to timm 0.4.12 classes via its MRO,
    # so the forward pass continues to work after we restore the venv timm.
    _saved_modules = {
        k: sys.modules[k] for k in list(sys.modules)
        if (k == "timm" or k.startswith("timm."))
        or k in ("models_vit",)
        or k.startswith("util.")
        or k == "util"
    }
    for k in _saved_modules:
        del sys.modules[k]

    _legacy_added = legacy_dir not in sys.path
    _mae_added = mae_dir not in sys.path
    if _legacy_added:
        sys.path.insert(0, legacy_dir)
    if _mae_added:
        sys.path.insert(0, mae_dir)

    try:
        from models_vit import vit_base_patch16, vit_small_patch16  # type: ignore[import-untyped]
        from util.pos_embed import interpolate_pos_embed  # type: ignore[import-untyped]

        _factory = {"base": vit_base_patch16, "small": vit_small_patch16}[variant]
        model = _factory(num_classes=0, global_pool=False)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["model"]
        ref = model.state_dict()
        for k in ("head.weight", "head.bias"):
            if k in state and k in ref and state[k].shape != ref[k].shape:
                del state[k]
        interpolate_pos_embed(model, state)
        model.load_state_dict(state, strict=False)
        model.eval()
    finally:
        for k in list(sys.modules):
            if (k == "timm" or k.startswith("timm.")) \
               or k == "models_vit" \
               or k == "util" or k.startswith("util."):
                del sys.modules[k]
        if _legacy_added and legacy_dir in sys.path:
            sys.path.remove(legacy_dir)
        if _mae_added and mae_dir in sys.path:
            sys.path.remove(mae_dir)
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

    # ── Image transform (chestx-ray norm, default BILINEAR resize) ────────────
    # Mirrors Augmentation(normalize="chestx-ray").get_augmentation("full_224", "val"):
    # Resize(256) → CenterCrop(224) → Normalize([0.5056]*3, [0.252]*3).
    # Default torchvision Resize is BILINEAR, matching the official _full method.
    _tv_transform = tv_transforms.Compose([
        tv_transforms.Resize(_RESIZE_SIZE),
        tv_transforms.CenterCrop(_INPUT_SIZE),
        tv_transforms.Normalize(mean=_CHESTXRAY_MEAN, std=_CHESTXRAY_STD),
    ])

    def _preprocess(path) -> torch.Tensor:
        img = _loader(str(path)).float() / 255.0  # [3, H, W] float in [0, 1]
        return _tv_transform(img)                 # [3, 224, 224]

    # ── Mask transform (NEAREST resize, no intensity normalization) ───────────
    _tv_transform_mask = tv_transforms.Compose([
        tv_transforms.Resize(_RESIZE_SIZE, interpolation=InterpolationMode.NEAREST),
        tv_transforms.CenterCrop(_INPUT_SIZE),
    ])

    def _preprocess_mask(path) -> torch.Tensor:
        mask = _loader_mask(str(path))[:1].float()  # [1, H, W]
        return _tv_transform_mask(mask)             # [1, 224, 224]

    # ── Model call ────────────────────────────────────────────────────────────
    _seg_mode = output_keys is not None and "mask" in output_keys

    if _seg_mode:
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            # With global_pool=False, m.norm still exists and applies to the
            # full [CLS, patch_1, ..., patch_N] sequence; hook it to grab the
            # post-norm token sequence, skip CLS, reshape to a spatial grid.
            captured: dict[str, torch.Tensor] = {}

            def _hook(_module, _inp, out):
                captured["tokens"] = out  # [B, 1 + N, D]

            handle = m.norm.register_forward_hook(_hook)
            try:
                _ = m(x)
            finally:
                handle.remove()
            return patches_to_spatial(captured["tokens"][:, 1:])  # [B, D, 14, 14]
    else:
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            return m(x)  # [B, D] — CLS token via the pretrained self.norm

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
