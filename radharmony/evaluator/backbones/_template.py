"""MyModel backbone recipe (Hugging Face). A template: copy it to ``<name>.py`` and fill it in.

Write here: the model, the input format, the input size, the resize, the crop, the mean and std, and the
feature. Example: "ViT-B/16, 224 x 224 input, ImageNet mean and std, the CLS token after the final norm".

Requires the ``my_model`` extra::

    uv pip install -e ".[my_model]"
"""

# ── Rules for a recipe ───────────────────────────────────────────────────────
# 1. Import the model packages inside the factory. Then the package imports with no extra.
# 2. Load the model one time. The transform and the encoder share it.
# 3. Do all image steps in the transform. The encoder only runs the model.
# 4. An image-only model returns (transform, encoder). A text-and-image model returns
#    (transform, image_encoder, text_encoder, tokenizer). See the end of this file.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import monai.transforms as mt
import torch

from radharmony.evaluator.transforms import EncoderPreprocessTransform, RadiologyEncoderTransform
from radharmony.evaluator.wrappers import ImageEncoderWrapper

from ._utils import display_255, load_image, patches_to_spatial, squeeze_frame_dim, to_3channel, to_uint8

_MY_MODEL_HUB = "org/my-model"


def make_my_model(
    device: str = "cuda",
    hub: str = _MY_MODEL_HUB,
    output_keys: set[str] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[mt.Compose, ImageEncoderWrapper]:
    """Return ``(transform, encoder)`` for MyModel.

    One model and one processor are loaded. The transform and the encoder share them.

    Parameters
    ----------
    device :
        The device of the encoder, for example ``"cuda"``, ``"cuda:1"`` or ``"cpu"``. The transform runs on
        the CPU, in the DataLoader workers.
    hub :
        The Hugging Face model name. Keep this argument only when the model family has more than one variant.
    output_keys :
        The keys of each sample. ``None`` gives ``{"img", "cls"}``. With ``"mask"``, the encoder gives patch
        features, and the transform gives the mask the same resize and crop as the image.
    dtype :
        The dtype of the image tensor. ``torch.bfloat16`` (the default) matches the autocast of the
        evaluators. ``torch.float32`` is for an encoder that cannot autocast.

    Returns
    -------
    transform : monai.transforms.Compose
        Give it to a RadHarmony dataset as ``transform=``.
    encoder : ImageEncoderWrapper
        Classification mode: ``forward(x) -> Tensor[B, D]``, the <feature>, ``D = <D>``.
        Segmentation mode (``"mask"`` in ``output_keys``): ``forward(x) -> Tensor[B, D, G, G]``, with
        ``G = <input size> / <patch size>``.
    """
    # ── Lazy imports (the model packages of the extra) ───────────────────────
    from transformers import AutoImageProcessor, AutoModel  # type: ignore[import-untyped]

    # ── Load the model one time ──────────────────────────────────────────────
    _model = AutoModel.from_pretrained(hub).to(device).eval()
    _processor = AutoImageProcessor.from_pretrained(hub)
    # Register tokens come after the CLS token. Read their number from the config (0 when there are none).
    _skip = 1 + int(getattr(_model.config, "num_register_tokens", 0))

    # ── Image and mask loaders (CPU, in the DataLoader workers) ──────────────
    _loader = mt.Compose([
        load_image(),  # a palette or alpha PNG reads as the image it shows (D-13)
        mt.Lambda(squeeze_frame_dim),
        mt.Transpose(indices=[0, 2, 1]),  # MONAI (C, W, H) to (C, H, W)
        mt.Lambda(display_255),  # 0 to 255, as the dataset load shows the image (D-13, D-27)
        mt.Lambda(to_3channel),
        mt.Lambda(to_uint8),
    ])
    # A mask keeps mt.LoadImage and has no scaling: a palette mask has its class numbers in the index (L-012).
    _loader_mask = mt.Compose([
        mt.LoadImage(image_only=True, ensure_channel_first=True),
        mt.Lambda(squeeze_frame_dim),
        mt.Transpose(indices=[0, 2, 1]),
        mt.Lambda(to_3channel),
    ])

    # ── Preprocess: path to Tensor[C, H, W] ──────────────────────────────────
    def _preprocess(path) -> torch.Tensor:
        return _processor(images=_loader(str(path)), return_tensors="pt")["pixel_values"][0]

    # The same resize and crop as the image, with nearest-neighbor interpolation and no normalization.
    # For an OpenCLIP or torchvision model, take the Resize and the CenterCrop of its image steps with
    # InterpolationMode.NEAREST instead.
    def _preprocess_mask(path) -> torch.Tensor:
        out = _processor(
            images=_loader_mask(str(path)), return_tensors="pt",
            do_normalize=False, do_rescale=False, resample=0,
        )
        return out["pixel_values"][0, :1]  # [1, H, W], class numbers

    # ── Model call: classification or segmentation ───────────────────────────
    _seg_mode = output_keys is not None and "mask" in output_keys

    if _seg_mode:
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            patches = m(pixel_values=x).last_hidden_state[:, _skip:]  # skip the CLS and register tokens
            return patches_to_spatial(patches)  # [B, D, G, G]
    else:
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            # Use the feature of the authors' code (D-16): CLS token, mean of the patch tokens, or pooler.
            return m(pixel_values=x).last_hidden_state[:, 0]  # [B, D]

    # ── Assemble the transform and the encoder ───────────────────────────────
    # processor= lets the transform warn when the processor crops or pads a box (D-28).
    pp = EncoderPreprocessTransform.from_custom(keys=["img"], preprocess=_preprocess, processor=_processor)
    transform = RadiologyEncoderTransform(
        preprocess=pp,
        output_keys=output_keys,
        mask_preprocess=_preprocess_mask,
        dtype=dtype,
    ).get_transform()
    encoder = ImageEncoderWrapper.from_custom(_model, model_call=_model_call).to(device)
    return transform, encoder

    # ── Text-and-image models ────────────────────────────────────────────────
    # Also return a text encoder and its tokenizer. The text encoder gives a float32 tensor on the CPU.
    # See chexagent.py. Example:
    #
    #   def _text_encoder(token_dict) -> torch.Tensor:
    #       with torch.no_grad():
    #           out = _model.text_model(**{k: v.to(device) for k, v in token_dict.items()})
    #           return out.pooler_output.float().cpu()
    #
    #   return transform, encoder, _text_encoder, _processor
