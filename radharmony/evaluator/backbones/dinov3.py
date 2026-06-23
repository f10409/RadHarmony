"""DINOv3 backbone recipe (HuggingFace).

Uses ``facebook/dinov3-vitb16-pretrain-lvd1689m`` by default (ViT-B/16,
hidden size 768, 16-px patches, 4 register tokens). Accepts any format
supported by MONAI's ``LoadImage`` (PNG, JPEG, DICOM, NIfTI, …). Images
are min-max normalized to uint8 and passed through ``AutoImageProcessor``
(resize + ImageNet-style normalization handled by the processor).

DINOv3's token sequence is ``[CLS, reg_0, …, reg_{R-1}, patch_0, …, patch_{N-1}]``.
The number of register tokens ``R`` is read from ``model.config.num_register_tokens``
so the recipe works across variants. Register tokens act as model scratch
space and are skipped in segmentation mode (patches only).

Requires the ``dinov3`` extra::

    uv pip install -e ".[dinov3]"
"""

from __future__ import annotations

import torch
import monai.transforms as mt

from radharmony.evaluator.transforms import EncoderPreprocessTransform, RadiologyEncoderTransform
from radharmony.evaluator.wrappers import ImageEncoderWrapper
from ._utils import squeeze_frame_dim, to_3channel, normalize_to_uint8, patches_to_spatial

_DINOV3_HUB = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def make_dinov3(
    device: str = "cuda",
    hub: str = _DINOV3_HUB,
    output_keys: set[str] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[mt.Compose, ImageEncoderWrapper]:
    """Return ``(transform, encoder)`` for DINOv3.

    A single HF model + processor pair is created and shared: the processor
    is captured in the transform's preprocess closure and the backbone is
    registered in the returned ``ImageEncoderWrapper``. No weights are
    loaded twice.

    Parameters
    ----------
    device :
        Target device for the encoder (e.g. ``"cuda"``, ``"cuda:7"``,
        ``"cpu"``). The transform's preprocess step runs on CPU in
        DataLoader workers regardless of this setting.
    hub :
        HuggingFace model identifier. Defaults to the ViT-B/16 LVD-1689M
        pretrain. Other variants (ViT-L, ViT-H, ViT-7B) work with the same
        recipe — only the embedding dim and input size change.
    output_keys :
        Keys to keep in each sample dict. Defaults to ``{"img", "cls"}``.
        Pass ``{"img", "mask"}`` or ``{"img", "cls", "mask"}`` for
        segmentation — the mask is processed through the same spatial
        pipeline as the image (processor resize + crop) using
        nearest-neighbour interpolation.

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
        Classification mode (default): ``forward(x) -> Tensor[B, D]`` —
        the CLS-token / pooler output (``D = 768`` for ViT-B/16).

        Segmentation mode (``"mask"`` in *output_keys*):
        ``forward(x) -> Tensor[B, D, H, W]`` — spatial patch feature map,
        where ``H = W = 14`` for 224 px input (16-px patches). Register
        tokens are skipped.

    Example
    -------
    ::

        from radharmony.evaluator.backbones import make_dinov3
        from radharmony.dataset import VinDrCXRTrainDataset

        transform, encoder = make_dinov3(device="cuda:0")

        ds = VinDrCXRTrainDataset(
            base_image_dir="/data/vindr/train/",
            transform=transform,
            cache_dir="/tmp/cache/dinov3/",
            output_cls=True,
        )
    """
    from transformers import AutoModel, AutoImageProcessor  # type: ignore[import-untyped]

    _model = AutoModel.from_pretrained(hub).to(device).eval()
    _processor = AutoImageProcessor.from_pretrained(hub)
    _n_register = int(getattr(_model.config, "num_register_tokens", 0))
    _skip = 1 + _n_register  # CLS + register tokens

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
        return out["pixel_values"][0, :1]  # [1, H, W]

    _seg_mode = output_keys is not None and "mask" in output_keys

    if _seg_mode:
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            # Token layout: [CLS, reg_0, ..., reg_{R-1}, patch_0, ...]
            patches = m(pixel_values=x).last_hidden_state[:, _skip:]
            return patches_to_spatial(patches)  # [B, D, H, W]
    else:
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            return m(pixel_values=x).pooler_output  # [B, D]

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
    encoder = ImageEncoderWrapper.from_custom(_model, model_call=_model_call).to(device)

    return transform, encoder
