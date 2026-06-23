"""RAD-DINO backbone recipe for pre-processed PNGs.

Accepts any format supported by MONAI's ``LoadImage`` (PNG, JPEG,
DICOM, NIfTI, …). The loader applies min-max normalization to uint8
regardless of input bit depth or format before passing the result
through the RAD-DINO processor. Images should be resized so that the
shortest side is 518 px before or after loading.

Requires the ``raddino`` extra::

    uv pip install -e ".[raddino]"
"""

from __future__ import annotations

import torch
import monai.transforms as mt

from radharmony.evaluator.transforms import EncoderPreprocessTransform, RadiologyEncoderTransform
from radharmony.evaluator.wrappers import ImageEncoderWrapper
from ._utils import squeeze_frame_dim, to_3channel, normalize_to_uint8, patches_to_spatial


def make_raddino(
    device: str = "cuda",
    output_keys: set[str] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[mt.Compose, ImageEncoderWrapper]:
    """Return ``(transform, encoder)`` for RAD-DINO on pre-processed PNGs.

    A single ``RadDino`` instance is created and shared: its processor is
    captured in the transform's preprocess closure, and its backbone is
    registered in the returned ``ImageEncoderWrapper``. No weights are
    loaded twice.

    Parameters
    ----------
    device :
        Target device for the encoder (e.g. ``"cuda"``, ``"cuda:7"``,
        ``"cpu"``). The transform's preprocess step runs on CPU in
        DataLoader workers regardless of this setting.
    output_keys :
        Keys to keep in each sample dict. Defaults to ``{"img", "cls"}``.
        Pass ``{"img", "mask"}`` or ``{"img", "cls", "mask"}`` for
        segmentation — the mask is automatically processed through the
        same spatial pipeline as the image (resize + center-crop to
        518 × 518) using nearest-neighbour interpolation.

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
        Classification mode (default): ``forward(x) -> Tensor[B, 768]``
        — the CLS-token embedding.

        Segmentation mode (``"mask"`` in *output_keys*):
        ``forward(x) -> Tensor[B, 768, H, W]`` — spatial patch feature
        map, where ``H = W = 37`` for 518 px input (14-px patches).

    Example
    -------
    ::

        from radharmony.evaluator.backbones import make_raddino
        from radharmony.dataset import VinDrCXRTrainDataset, SIIMACRPTXDataset

        # Classification
        transform, encoder = make_raddino(device="cuda:0")
        ds = VinDrCXRTrainDataset(
            base_image_dir="/data/vindr/train/",
            transform=transform,
            cache_dir="/tmp/cache/raddino/",
            output_cls=True,
        )

        # Segmentation
        transform, encoder = make_raddino(
            device="cuda:0",
            output_keys={"img", "mask"},
        )
        ds = SIIMACRPTXDataset(
            base_image_dir="/data/siim/",
            transform=transform,
            output_seg=True,
        )
    """
    from rad_dino import RadDino  # type: ignore[import-untyped]
    from transformers.feature_extraction_utils import BatchFeature  # type: ignore[import-untyped]

    _model = RadDino()

    _loader = mt.Compose([
        mt.LoadImage(image_only=True, ensure_channel_first=True),
        mt.Lambda(squeeze_frame_dim),
        mt.Transpose(indices=[0, 2, 1]),
        mt.Lambda(to_3channel),
        mt.Lambda(normalize_to_uint8),
    ])
    # Mask loader: same spatial steps, no uint8 normalization (preserves class indices).
    _loader_mask = mt.Compose([
        mt.LoadImage(image_only=True, ensure_channel_first=True),
        mt.Lambda(squeeze_frame_dim),
        mt.Transpose(indices=[0, 2, 1]),
        mt.Lambda(to_3channel),
    ])

    def _preprocess(path) -> torch.Tensor:
        img = _loader(str(path))
        return _model.processor(images=img, return_tensors="pt")["pixel_values"][0]

    def _preprocess_mask(path) -> torch.Tensor:
        # Same spatial pipeline as the image processor (resize + center-crop),
        # nearest-neighbour interpolation, no intensity normalization.
        mask = _loader_mask(str(path))
        out = _model.processor(
            images=mask, return_tensors="pt",
            do_normalize=False, do_rescale=False, resample=0,
        )
        return out["pixel_values"][0, :1]  # [1, H, W] — class indices

    _seg_mode = output_keys is not None and "mask" in output_keys

    if _seg_mode:
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            _, patch_tokens = m.encode(BatchFeature({"pixel_values": x}))
            return patches_to_spatial(patch_tokens)  # [B, 768, H, W]
    else:
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            cls_token, _ = m.encode(BatchFeature({"pixel_values": x}))
            return cls_token  # [B, 768]

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
