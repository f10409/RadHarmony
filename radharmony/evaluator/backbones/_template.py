"""<ModelName> backbone recipe for <input format, e.g. "pre-processed PNGs" or "DICOM inputs">.

<One or two sentences on the expected input format and any pre-processing
assumptions — e.g. bit depth, shortest-side resize, stored format.>

Requires the ``<extra_name>`` extra::

    uv pip install -e ".[<extra_name>]"
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
# 3. Image preprocessing (resize, normalization, channel expansion) lives
#    entirely in the MONAI transform. ImageEncoderWrapper never does it.
#
# 4. For vision-language models return (transform, image_encoder, text_encoder,
#    tokenizer).  For vision-only models return (transform, encoder).
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any, Callable

import torch
import monai.transforms as mt

from radharmony.evaluator.transforms import EncoderPreprocessTransform, RadiologyEncoderTransform
from radharmony.evaluator.wrappers import ImageEncoderWrapper
# from ._utils import squeeze_frame_dim, to_3channel, normalize_to_uint8, patches_to_spatial

# Module-level constants (hub IDs, default sizes, etc.)
# _MODEL_HUB = "org/model-name"


def make_<model_name>(
    device: str = "cuda",
    output_keys: set[str] | None = None,
    # add model-specific kwargs here, e.g.:
    # hub: str = _MODEL_HUB,
) -> tuple[mt.Compose, ImageEncoderWrapper]:
    # For vision-language models change the return type to:
    # -> tuple[mt.Compose, ImageEncoderWrapper, Callable[[torch.Tensor], torch.Tensor], Any]:
    """Return ``(transform, encoder)`` for <ModelName>.

    A single <ModelClass> instance is created and shared: its processor is
    captured in the transform's preprocess closure and its backbone is
    registered in the returned ``ImageEncoderWrapper``. No weights are loaded
    twice.

    Parameters
    ----------
    device :
        Target device for the encoder (e.g. ``"cuda"``, ``"cuda:7"``,
        ``"cpu"``). The transform's preprocess step runs on CPU in DataLoader
        workers regardless of this setting.
    output_keys :
        Keys to keep in each sample dict. Defaults to ``{"img", "cls"}``.
        Pass ``{"img", "mask"}`` or ``{"img", "cls", "mask"}`` for
        segmentation tasks — the mask is automatically loaded and resized to
        ``<INPUT_SIZE> × <INPUT_SIZE>`` (this model's input resolution) with
        nearest-neighbour interpolation.

    Returns
    -------
    transform : monai.transforms.Compose
        MONAI transform suitable for passing to a RadHarmony dataset via the
        ``transform=`` kwarg.
    encoder : ImageEncoderWrapper
        ``forward(x) -> Tensor[B, <D>]`` — the <description of embedding>.

    Example
    -------
    ::

        from radharmony.evaluator.backbones import make_<model_name>
        from radharmony.dataset import VinDrCXRTrainDataset

        transform, encoder = make_<model_name>(device="cuda:0")

        ds = VinDrCXRTrainDataset(
            base_image_dir="/data/vindr/train/",
            transform=transform,
            cache_dir="/tmp/cache/<model_name>/",
            output_cls=True,
        )
    """
    # ── Lazy imports (optional dependency) ────────────────────────────────────
    from some_package import SomeModel  # type: ignore[import-untyped]
    # from transformers import AutoModel, AutoImageProcessor  # HF example

    # ── Load model (single instance shared by transform + encoder) ────────────
    _model = SomeModel()
    # _model = AutoModel.from_pretrained(_MODEL_HUB)

    # ── Helper transforms (pure functions, no model state) ────────────────────

    def _squeeze_frame_dim(x: torch.Tensor) -> torch.Tensor:
        # MONAI LoadImage stores 2D images as (C, W, H, 1) when the file has a
        # singleton frame dimension; drop it to get (C, W, H).
        return x.squeeze(-1) if x.ndim == 4 and x.shape[-1] == 1 else x

    # Add further helpers as needed, e.g.:
    # def _to_3channel(x: torch.Tensor) -> torch.Tensor: ...
    # def _to_pil_rgb(arr: torch.Tensor) -> "PIL.Image.Image": ...

    # ── Image loader compose (CPU; runs in DataLoader workers) ────────────────
    _loader = mt.Compose([
        mt.LoadImage(image_only=True, ensure_channel_first=True),
        mt.Lambda(_squeeze_frame_dim),
        mt.Transpose(indices=[0, 2, 1]),  # MONAI (C,W,H) → (C,H,W)
        # add backbone-specific steps here
    ])

    # ── Preprocess closure: path → Tensor[C, H, W] ────────────────────────────
    def _preprocess(path) -> torch.Tensor:
        img = _loader(str(path))
        # Call the model's processor / preprocess_val / AutoImageProcessor here.
        # Must return a single Tensor[C, H, W] (no batch dimension).
        return _model.processor(images=img, return_tensors="pt")["pixel_values"][0]

    # ── Model call closure: switches on segmentation vs classification mode ────
    _seg_mode = output_keys is not None and "mask" in output_keys

    if _seg_mode:
        # Return a spatial patch feature map [B, D, H, W] for the segmentation head.
        # Skip CLS (index 0) and reshape the patch sequence to a 2-D grid.
        # HF ViT example:
        #   def _model_call(m, x):
        #       patches = m(pixel_values=x).last_hidden_state[:, 1:]  # [B, N, D]
        #       return patches_to_spatial(patches)                     # [B, D, H, W]
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError
    else:
        # Return a global [B, D] embedding for the classification head.
        # Examples:
        #   HF ViT:   return m(pixel_values=x).last_hidden_state[:, 0]
        #   OpenCLIP: return m.encode_image(x)
        #   RAD-DINO: cls, _ = m.encode(BatchFeature({"pixel_values": x})); return cls
        #   timm:     return m(x)   (num_classes=0)
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

    # ── Mask preprocess closure (mirrors _preprocess spatially, no normalization) ─
    # For HF-processor-based models:
    #   def _preprocess_mask(path) -> torch.Tensor:
    #       mask = _loader_mask(str(path))   # same as _loader but no normalize_to_uint8
    #       out = _model.processor(
    #           images=mask, return_tensors="pt",
    #           do_normalize=False, do_rescale=False, resample=0,
    #       )
    #       return out["pixel_values"][0, :1]   # [1, H, W]
    #
    # For OpenCLIP-based models: extract Resize+CenterCrop from preprocess_val with
    # InterpolationMode.NEAREST and apply them to the raw tensor loader output.

    # ── Assemble transform + encoder ──────────────────────────────────────────
    pp = EncoderPreprocessTransform.from_custom(keys=["img"], preprocess=_preprocess)
    transform = (
        RadiologyEncoderTransform(
            preprocess=pp,
            output_keys=output_keys,
            mask_preprocess=_preprocess_mask,
        )
        .get_transform()
    )
    encoder = ImageEncoderWrapper.from_custom(_model, model_call=_model_call).to(device)

    return transform, encoder

    # ── Vision-language extension ──────────────────────────────────────────────
    # If your model also has a text tower, define a text encoder closure and
    # return it alongside the tokenizer:
    #
    #   def _text_encoder(tokens: torch.Tensor) -> torch.Tensor:
    #       with torch.no_grad():
    #           return _model.encode_text(tokens.to(device)).float().cpu()
    #
    #   tokenizer = get_tokenizer(...)
    #   return transform, encoder, _text_encoder, tokenizer
