"""SigLIP 2 backbone recipe (general-domain vision-language model).

Uses ``google/siglip2-so400m-patch16-384`` by default. SigLIP 2 adds a
multi-positive contrastive objective and multilingual training on top of
SigLIP v1 — stronger zero-shot classification and retrieval at the cost of
a slightly larger eval input (384 × 384, patch 16 → 24 × 24 grid).

Accepts any format supported by MONAI's ``LoadImage`` (PNG, JPEG, DICOM,
NIfTI, …). Images are min-max normalized to uint8 and passed through
``AutoProcessor`` (resize to ``image_size × image_size`` and
``(0.5, 0.5, 0.5)`` normalization handled by the processor).

Requires the ``siglip2`` extra::

    uv pip install -e ".[siglip2]"
"""

# ──────────────────────────────────────────────────────────────────────────────
# Implementation notes
# ──────────────────────────────────────────────────────────────────────────────
# Image side uses **Option B**: ``m.vision_model(pixel_values=x).pooler_output``.
# Avoids the transformers 5.x ``get_image_features`` quirk (returns a
# ``BaseModelOutputWithPooling`` object, not a tensor) and lets segmentation
# mode share the same vision-model call site.
#
# SigLIP / SigLIP 2 use an attention-pool head and have **no CLS token**:
# ``last_hidden_state`` is already pure patch tokens. Do NOT slice ``[:, 1:]``
# — that would drop a real patch.
#
# The patch grid is read dynamically from
# ``model.config.vision_config.image_size // model.config.vision_config.patch_size``
# so this recipe survives variant swaps (so400m-patch14-384, siglip-base-patch16-224, …).
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any, Callable

import torch
import monai.transforms as mt

from radharmony.evaluator.transforms import EncoderPreprocessTransform, RadiologyEncoderTransform
from radharmony.evaluator.wrappers import ImageEncoderWrapper
from ._utils import squeeze_frame_dim, to_3channel, normalize_to_uint8, patches_to_spatial


_SIGLIP2_HUB = "google/siglip2-so400m-patch16-384"


def make_siglip2(
    device: str = "cuda",
    hub: str = _SIGLIP2_HUB,
    output_keys: set[str] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[mt.Compose, ImageEncoderWrapper, Callable[[torch.Tensor], torch.Tensor], Any]:
    """Return ``(transform, image_encoder, text_encoder, processor)`` for SigLIP 2.

    A single ``SiglipModel`` instance is created and shared: the processor is
    captured in the transform's preprocess closure, the vision tower is used
    by the returned ``ImageEncoderWrapper``, and the text tower is captured
    in the text-encoder closure. No weights are loaded twice.

    Parameters
    ----------
    device :
        Target device for the encoder (e.g. ``"cuda"``, ``"cuda:7"``,
        ``"cpu"``). The transform's preprocess step runs on CPU in
        DataLoader workers regardless of this setting.
    hub :
        HuggingFace model identifier. Defaults to
        ``"google/siglip2-so400m-patch16-384"`` (~1.1B params, 1152-d,
        24 × 24 patch grid at 384 px input).
    output_keys :
        Keys to keep in each sample dict. Defaults to ``{"img", "cls"}``.
        Pass ``{"img", "mask"}`` or ``{"img", "cls", "mask"}`` for
        segmentation — the mask is processed through the same spatial
        pipeline as the image (processor resize + crop to
        ``image_size × image_size``) using nearest-neighbour interpolation.

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
    image_encoder : ImageEncoderWrapper
        Classification mode (default): ``forward(x) -> Tensor[B, 1152]``
        — the attention-pooled embedding from ``vision_model.pooler_output``.
        For non-default variants ``D`` matches ``vision_config.hidden_size``
        (e.g. 768 for ``siglip2-base-patch16-224``).

        Segmentation mode (``"mask"`` in *output_keys*):
        ``forward(x) -> Tensor[B, D, H, W]`` — spatial patch feature map.
        For the default so400m-patch16-384: ``[B, 1152, 24, 24]``. ``H = W
        = image_size // patch_size`` is read from the model config so
        variant swaps work automatically.
    text_encoder : callable
        ``(token_dict) -> Tensor[T, 1152]`` — encodes tokenized text on
        ``device`` and returns float32 on CPU. Pass to
        ``ZeroShotEvaluator(text_encoder=...)``.
    processor : SiglipProcessor
        Pass to ``ZeroShotEvaluator(tokenizer=...)`` for text tokenization.
        SigLIP 2 uses a 64-token max length by default.

    Example
    -------
    ::

        from radharmony.evaluator.backbones import make_siglip2
        from radharmony.dataset import VinDrCXRTrainDataset

        transform, encoder, text_encoder, processor = make_siglip2(device="cuda:0")

        ds = VinDrCXRTrainDataset(
            base_image_dir="/data/vindr/train/",
            transform=transform,
            cache_dir="/tmp/cache/siglip2/",
            output_cls=True,
        )
    """
    # ── Lazy imports (optional dependency) ────────────────────────────────────
    from transformers import AutoModel, AutoProcessor  # type: ignore[import-untyped]

    # ── Load model (single instance shared by transform + encoders) ───────────
    _model = AutoModel.from_pretrained(hub).to(device).eval()
    _processor = AutoProcessor.from_pretrained(hub)

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
        mt.Lambda(to_3channel),
    ])

    # ── Preprocess closures (processor handles resize + normalize) ────────────
    def _preprocess(path) -> torch.Tensor:
        img = _loader(str(path))
        return _processor(images=img, return_tensors="pt")["pixel_values"][0]

    def _preprocess_mask(path) -> torch.Tensor:
        mask = _loader_mask(str(path))
        out = _processor(
            images=mask, return_tensors="pt",
            do_normalize=False, do_rescale=False, resample=0,
        )
        return out["pixel_values"][0, :1]

    # ── Model call (Option B — direct vision_model access) ────────────────────
    _seg_mode = output_keys is not None and "mask" in output_keys

    if _seg_mode:
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            # SigLIP has no CLS token, so last_hidden_state is pure patches.
            patches = m.vision_model(pixel_values=x).last_hidden_state  # [B, N, D]
            return patches_to_spatial(patches)                          # [B, D, H, W]
    else:
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            return m.vision_model(pixel_values=x).pooler_output         # [B, D]

    # ── Text encoder (Option B parallel — direct text_model access) ───────────
    def _text_encoder(token_dict) -> torch.Tensor:
        with torch.no_grad():
            return _model.text_model(
                **{k: v.to(device) for k, v in token_dict.items()}
            ).pooler_output.float().cpu()

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
    image_encoder = ImageEncoderWrapper.from_custom(_model, model_call=_model_call).to(device)

    return transform, image_encoder, _text_encoder, _processor
