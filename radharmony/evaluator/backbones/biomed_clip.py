"""BiomedCLIP backbone recipe.

Uses ``microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`` from the
HuggingFace Hub via OpenCLIP. Accepts any format supported by MONAI's
``LoadImage`` (PNG, JPEG, DICOM, NIfTI, …). Images are min-max normalized
to [0, 255], converted to PIL RGB, then passed through OpenCLIP's
``preprocess_val``.

Requires the ``biomed`` extra::

    uv pip install -e ".[biomed]"
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import monai.transforms as mt

from radharmony.evaluator.transforms import EncoderPreprocessTransform, RadiologyEncoderTransform
from radharmony.evaluator.wrappers import ImageEncoderWrapper
from ._utils import squeeze_frame_dim, to_3channel, to_pil_rgb, patches_to_spatial

_BIOMED_HUB = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def make_biomed_clip(
    device: str = "cuda",
    hub: str = _BIOMED_HUB,
    output_keys: set[str] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[mt.Compose, ImageEncoderWrapper, Callable[[torch.Tensor], torch.Tensor], Any]:
    """Return ``(transform, image_encoder, text_encoder, tokenizer)`` for BiomedCLIP.

    A single model instance is created and shared across all returned
    objects — no weights are loaded twice.

    Parameters
    ----------
    device :
        Target device for the encoder and text encoder (e.g. ``"cuda"``,
        ``"cuda:7"``, ``"cpu"``). The transform's preprocess step runs on
        CPU in DataLoader workers.
    hub :
        OpenCLIP hub identifier. Defaults to the BiomedCLIP checkpoint on
        the HuggingFace Hub.
    output_keys :
        Keys to keep in each sample dict. Defaults to ``{"img", "cls"}``.
        Pass ``{"img", "mask"}`` or ``{"img", "cls", "mask"}`` for
        segmentation — the mask is processed through the same spatial
        pipeline as the image (OpenCLIP Resize + CenterCrop to 224 × 224)
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
        MONAI transform for any format supported by ``LoadImage``.
    image_encoder : ImageEncoderWrapper
        Classification mode (default): ``forward(x) -> Tensor[B, 512]``
        — the projected image CLS embedding.

        Segmentation mode (``"mask"`` in *output_keys*):
        ``forward(x) -> Tensor[B, 768, H, W]`` — pre-projection ViT patch
        feature map (trunk hidden dim), where ``H = W = 14`` for 224 px
        input (16-px patches).
    text_encoder : callable
        ``(tokens: Tensor[T, L]) -> Tensor[T, 512]`` — encodes tokenized
        text on ``device`` and returns float32 on CPU. Pass to
        ``ZeroShotEvaluator(text_encoder=...)``.
    tokenizer : open_clip tokenizer
        Pass to ``ZeroShotEvaluator(tokenizer=...)``.

    Example
    -------
    ::

        from radharmony.evaluator.backbones import make_biomed_clip
        from radharmony.dataset import VinDrCXRTrainDataset

        transform, encoder, text_encoder, tokenizer = make_biomed_clip(device="cuda:0")

        ds = VinDrCXRTrainDataset(
            base_image_dir="/data/vindr/train/",
            transform=transform,
            cache_dir="/tmp/cache/biomed/",
            output_cls=True,
        )
    """
    import open_clip  # type: ignore[import-untyped]

    biomed_model, _, preprocess_val = open_clip.create_model_and_transforms(hub)
    biomed_model = biomed_model.to(device).eval()
    biomed_tokenizer = open_clip.get_tokenizer(hub)

    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    _loader_to_pil = mt.Compose([
        mt.LoadImage(image_only=True, ensure_channel_first=True),
        mt.Lambda(squeeze_frame_dim),
        mt.Transpose(indices=[0, 2, 1]),
        mt.Lambda(to_pil_rgb),
    ])
    # Mask loader: tensor path (no PIL conversion / no value normalization).
    _loader_mask = mt.Compose([
        mt.LoadImage(image_only=True, ensure_channel_first=True),
        mt.Lambda(squeeze_frame_dim),
        mt.Transpose(indices=[0, 2, 1]),
        mt.Lambda(to_3channel),
    ])
    # Spatial-only transforms extracted from preprocess_val (Resize + CenterCrop),
    # applied to the mask with NEAREST interpolation instead of BICUBIC.
    _spatial_transforms = [
        T.Resize(t.size, interpolation=InterpolationMode.NEAREST)
        if isinstance(t, T.Resize)
        else T.CenterCrop(t.size)
        for t in preprocess_val.transforms
        if isinstance(t, (T.Resize, T.CenterCrop))
    ]

    _seg_mode = output_keys is not None and "mask" in output_keys

    def _preprocess(path) -> torch.Tensor:
        return preprocess_val(_loader_to_pil(str(path)))

    def _preprocess_mask(path) -> torch.Tensor:
        mask = _loader_mask(str(path))  # [3, H, W] float32, class-index values
        for t in _spatial_transforms:
            mask = t(mask)
        return mask[:1]  # [1, H, W]

    def _text_encoder(tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return biomed_model.encode_text(tokens.to(device)).float().cpu()

    if _seg_mode:
        # Return pre-projection patch tokens from the timm ViT trunk.
        # trunk.num_prefix_tokens skips CLS (and any register tokens).
        # trunk.forward_features returns [B, num_prefix_tokens+N_patches, D].
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            trunk = m.visual.trunk
            all_tokens = trunk.forward_features(x)
            patches = all_tokens[:, trunk.num_prefix_tokens:]  # [B, N, 768]
            return patches_to_spatial(patches)  # [B, 768, H, W]
    else:
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            return m.encode_image(x)  # [B, 512]

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
    image_encoder = ImageEncoderWrapper.from_custom(
        biomed_model,
        model_call=_model_call,
    ).to(device)

    return transform, image_encoder, _text_encoder, biomed_tokenizer
