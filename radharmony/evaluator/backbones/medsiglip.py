"""MedSigLIP backbone recipe.

Uses ``google/medsiglip-448`` from HuggingFace. Accepts any format supported
by MONAI's ``LoadImage`` (PNG, JPEG, DICOM, NIfTI, …). Images are min-max
normalized to uint8 and passed through ``AutoProcessor`` (resize to 448 × 448
and [-1, 1] normalization handled by the processor).

Requires the ``medsiglip`` extra::

    uv pip install -e ".[medsiglip]"
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import monai.transforms as mt

from radharmony.evaluator.transforms import EncoderPreprocessTransform, RadiologyEncoderTransform
from radharmony.evaluator.wrappers import ImageEncoderWrapper
from ._utils import squeeze_frame_dim, to_3channel, normalize_to_uint8, patches_to_spatial

_MEDSIGLIP_HUB = "google/medsiglip-448"


def make_medsiglip(
    device: str = "cuda",
    hub: str = _MEDSIGLIP_HUB,
    output_keys: set[str] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[mt.Compose, ImageEncoderWrapper, Callable[[torch.Tensor], torch.Tensor], Any]:
    """Return ``(transform, image_encoder, text_encoder, processor)`` for MedSigLIP.

    A single ``SiglipModel`` instance is created and shared: the processor is
    captured in the transform's preprocess closure and the backbone is
    registered in the returned ``ImageEncoderWrapper``. No weights are loaded
    twice.

    Parameters
    ----------
    device :
        Target device for the encoder (e.g. ``"cuda"``, ``"cuda:7"``,
        ``"cpu"``). The transform's preprocess step runs on CPU in
        DataLoader workers regardless of this setting.
    hub :
        HuggingFace model identifier.
    output_keys :
        Keys to keep in each sample dict. Defaults to ``{"img", "cls"}``.
        Pass ``{"img", "mask"}`` or ``{"img", "cls", "mask"}`` for
        segmentation — the mask is processed through the same spatial
        pipeline as the image (processor resize + crop to 448 × 448)
        using nearest-neighbour interpolation.

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
        — the pooled CLS embedding.

        Segmentation mode (``"mask"`` in *output_keys*):
        ``forward(x) -> Tensor[B, 1152, H, W]`` — spatial patch feature
        map, where ``H = W = 32`` for 448 px input (14-px patches).
    text_encoder : callable
        ``(token_dict) -> Tensor[T, 512]`` — encodes tokenized text on
        ``device`` and returns float32 on CPU. Pass to
        ``ZeroShotEvaluator(text_encoder=...)``.
    processor : SiglipProcessor
        Pass to ``ZeroShotEvaluator(tokenizer=...)`` for text tokenization.
        Maximum text length is 64 tokens.

    Example
    -------
    ::

        from radharmony.evaluator.backbones import make_medsiglip
        from radharmony.dataset import VinDrCXRTrainDataset

        transform, encoder, text_encoder, processor = make_medsiglip(device="cuda:0")

        ds = VinDrCXRTrainDataset(
            base_image_dir="/data/vindr/train/",
            transform=transform,
            cache_dir="/tmp/cache/medsiglip/",
            output_cls=True,
        )
    """
    from transformers import AutoModel, AutoProcessor  # type: ignore[import-untyped]

    _model = AutoModel.from_pretrained(hub).to(device).eval()
    _processor = AutoProcessor.from_pretrained(hub)

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

    _seg_mode = output_keys is not None and "mask" in output_keys

    if _seg_mode:
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            # SigLIP has no CLS token (uses attention-pool head); last_hidden_state
            # is already pure patch tokens — do NOT slice [:, 1:].
            patches = m.vision_model(pixel_values=x).last_hidden_state
            return patches_to_spatial(patches)  # [B, 1152, H, W]
    else:
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            return m.vision_model(pixel_values=x).pooler_output  # [B, 1152]

    def _text_encoder(token_dict) -> torch.Tensor:
        with torch.no_grad():
            return _model.text_model(
                **{k: v.to(device) for k, v in token_dict.items()}
            ).pooler_output.float().cpu()

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
