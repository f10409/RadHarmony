"""MedImageInsights backbone recipe.

Uses ``lion-ai/MedImageInsights`` (Microsoft MedImageInsight, open-sourced via
Lion-AI) from HuggingFace. Architecture is DaViT (Dual Attention Vision
Transformer), 360 M params image encoder, 252 M params language encoder.
Accepts any format supported by MONAI's ``LoadImage`` (PNG, JPEG, DICOM,
NIfTI, …). Images are converted to PIL RGB and passed through the model's
torchvision preprocess chain (bicubic Resize to 480 × 480 + ImageNet
normalization). Embeddings are 1024-dim, L2-normalised.

Requires the ``medimageinsights`` extra::

    uv pip install -e ".[medimageinsights]"

Note: the model snapshot is downloaded automatically via ``huggingface_hub``
to ``~/.cache/huggingface/hub/`` on first call.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Implementation notes
# ──────────────────────────────────────────────────────────────────────────────
# 1. ALL backbone imports are LAZY (inside the factory body) so this module can
#    be imported even when the optional dependency is not installed.
#
# 2. A SINGLE model instance is created and shared between the transform
#    preprocessor and the encoder — no weights are loaded twice.
#
# 3. Image preprocessing (PIL conversion, resize, ImageNet normalization) lives
#    entirely in the MONAI transform closure. ImageEncoderWrapper never does it.
#
# 4. The model exposes a separate text encoder → return
#    (transform, image_encoder, text_encoder, None).  The fourth element is None
#    because MedImageInsights encodes text from raw strings (no external
#    tokenizer step is required by callers).
#
# 5. Segmentation mode hooks image_encoder.blocks[3][0] (the last DaViT block)
#    to capture [B, H*W, C] before the global mean-pool.  The hook is
#    registered and removed per forward call to avoid side-effects.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from typing import Any, Callable

import torch
import monai.transforms as mt

from radharmony.evaluator.transforms import EncoderPreprocessTransform, RadiologyEncoderTransform
from radharmony.evaluator.wrappers import ImageEncoderWrapper
from ._utils import squeeze_frame_dim, to_3channel, to_pil_rgb

_MEDIMAGEINSIGHTS_HUB = "lion-ai/MedImageInsights"
_WEIGHTS_SUBDIR = "2024.09.27"


def make_medimageinsights(
    device: str = "cuda",
    hub: str = _MEDIMAGEINSIGHTS_HUB,
    output_keys: set[str] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[mt.Compose, ImageEncoderWrapper, Callable, None]:
    """Return ``(transform, image_encoder, text_encoder, None)`` for MedImageInsights.

    A single ``MedImageInsight`` model instance is created and shared: its
    torchvision preprocess chain is captured in the transform closure and its
    DaViT backbone is registered in the returned ``ImageEncoderWrapper``. No
    weights are loaded twice.

    Parameters
    ----------
    device :
        Target device for the encoder (e.g. ``"cuda"``, ``"cuda:7"``,
        ``"cpu"``). The transform's preprocess step runs on CPU in DataLoader
        workers regardless of this setting.
    hub :
        HuggingFace snapshot identifier (default: ``"lion-ai/MedImageInsights"``).
        The snapshot is downloaded to the HuggingFace cache on first call.
    output_keys :
        Keys to keep in each sample dict. Defaults to ``{"img", "cls"}``.
        Pass ``{"img", "mask"}`` or ``{"img", "cls", "mask"}`` for
        segmentation — the mask is resized to the same spatial resolution as
        the image (480 × 480 by default) using nearest-neighbour interpolation.

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
    image_encoder : ImageEncoderWrapper
        Classification mode (default): ``forward(x) -> Tensor[B, 1024]``
        — DaViT global embedding, projected and L2-normalised.

        Segmentation mode (``"mask"`` in *output_keys*):
        ``forward(x) -> Tensor[B, 2048, H, W]`` — spatial feature map
        captured from the last DaViT block (``image_encoder.blocks[3][0]``)
        via a per-call forward hook. For 480 × 480 input: ``H = W = 15``
        (DaViT final stride 32 → 480 / 32 = 15).
    text_encoder : callable
        ``(texts: list[str]) -> Tensor[T, 1024]`` — L2-normalised text
        embeddings on CPU. MedImageInsights encodes raw strings; no external
        tokeniser is required. Pass to
        ``ZeroShotEvaluator(text_encoder=...)``.
    tokenizer : None
        Placeholder for API consistency with other vision-language backbone
        factories. MedImageInsights does not require an external tokeniser.

    Example
    -------
    ::

        from radharmony.evaluator.backbones import make_medimageinsights
        from radharmony.dataset import VinDrCXRTrainDataset

        transform, encoder, text_encoder, _ = make_medimageinsights(device="cuda:0")

        ds = VinDrCXRTrainDataset(
            base_image_dir="/data/vindr/train/",
            transform=transform,
            cache_dir="/tmp/cache/medimageinsights/",
            output_cls=True,
        )
    """
    # ── Lazy imports (optional dependency) ────────────────────────────────────
    import os
    import sys
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    from huggingface_hub import snapshot_download  # type: ignore[import-untyped]

    # ── Download / locate snapshot ────────────────────────────────────────────
    # snapshot_download is idempotent; on subsequent calls it returns the cached
    # path without re-downloading.
    model_dir = snapshot_download(hub)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    from medimageinsightmodel import MedImageInsight  # type: ignore[import-untyped]

    model_weights_dir = os.path.join(model_dir, _WEIGHTS_SUBDIR)
    classifier = MedImageInsight(
        model_dir=model_weights_dir,
        vision_model_name="medimageinsigt-v1.0.0.pt",
        language_model_name="language_model.pth",
    )
    classifier.load_model()

    # ── Image loader: MONAI → PIL RGB (CPU; runs in DataLoader workers) ───────
    _loader = mt.Compose([
        mt.LoadImage(image_only=True, ensure_channel_first=True),
        mt.Lambda(squeeze_frame_dim),
        mt.Transpose(indices=[0, 2, 1]),   # MONAI (C,W,H) → (C,H,W)
        mt.Lambda(to_pil_rgb),             # min-max → uint8 → PIL RGB
    ])

    # ── Mask loader: same spatial steps, no intensity normalization ───────────
    _loader_mask = mt.Compose([
        mt.LoadImage(image_only=True, ensure_channel_first=True),
        mt.Lambda(squeeze_frame_dim),
        mt.Transpose(indices=[0, 2, 1]),
        mt.Lambda(to_3channel),            # keep float, replicate to 3-channel
    ])

    # ── Preprocess closure: path → Tensor[3, 480, 480], ImageNet-normalised ──
    _preprocess_chain = classifier.preprocess  # torchvision Compose

    def _preprocess(path) -> torch.Tensor:
        return _preprocess_chain(_loader(str(path)))

    # ── Mask preprocess: mirror spatial transform with NEAREST interpolation ──
    # Extract T.Resize from the model's own preprocess chain so the mask always
    # undergoes identical spatial transforms as the image, even if config.yaml
    # changes the target resolution.
    _spatial_transforms = [
        T.Resize(t.size, interpolation=InterpolationMode.NEAREST, antialias=False)
        for t in _preprocess_chain.transforms
        if isinstance(t, T.Resize)
    ]

    def _preprocess_mask(path) -> torch.Tensor:
        mask = _loader_mask(str(path))     # [3, H, W] float, class-index values
        for t in _spatial_transforms:
            mask = t(mask)
        return mask[:1]                    # [1, 480, 480]

    # ── Model call ────────────────────────────────────────────────────────────
    _seg_mode = output_keys is not None and "mask" in output_keys
    _backbone = classifier.model           # nn.Module (DaViT + projection head)

    if _seg_mode:
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            # Hook the last DaViT block to capture [B, H*W, C] before
            # norms → mean-pool → avgpool → [B, 2048].
            # The hook is registered and removed per call to avoid side-effects.
            final_feat: dict[str, torch.Tensor] = {}

            def _hook(mod, inp, out) -> None:
                t = out[0] if isinstance(out, (tuple, list)) else out
                final_feat["f"] = t

            handle = m.image_encoder.blocks[3][0].register_forward_hook(_hook)
            try:
                m.encode_image(x)
            finally:
                handle.remove()

            spatial = final_feat["f"]          # [B, H*W, C]
            B, HW, C = spatial.shape
            H = W = int(math.sqrt(HW))
            return spatial.permute(0, 2, 1).reshape(B, C, H, W)  # [B, 2048, H, W]
    else:
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            return m.encode_image(x)           # [B, 1024], L2-normalised

    # ── Text encoder for zero-shot ────────────────────────────────────────────
    def _text_encoder(texts: list[str]) -> torch.Tensor:
        with torch.no_grad():
            result = classifier.encode(images=None, texts=texts)
        emb = result["text_embeddings"]        # numpy or tensor [T, 1024]
        return torch.as_tensor(emb).float()

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
    image_encoder = ImageEncoderWrapper.from_custom(_backbone, model_call=_model_call).to(device)

    return transform, image_encoder, _text_encoder, None
