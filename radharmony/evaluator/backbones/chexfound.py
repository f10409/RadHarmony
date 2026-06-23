"""CheXFound backbone recipe (ViT-Large/16, iBOT-style self-supervised pre-training).

Accepts any format supported by MONAI's ``LoadImage`` (PNG, JPEG, DICOM,
NIfTI, …). The loader applies min-max normalization to uint8 before the
torchvision preprocessing pipeline:
Resize(512, BICUBIC) → CenterCrop(512) → Normalize(ImageNet).

CheXFound weights must be manually downloaded from Google Drive and the
repository cloned locally:

1. Clone the repository::

       git clone https://github.com/RPIDIAL/CheXFound <chexfound_dir>

2. Download ``teacher_checkpoint.pth`` from Google Drive:
   https://drive.google.com/drive/folders/1GX2BWbujuVABtVpSZ4PTBykGULzrw806
   Place at: ``<chexfound_dir>/weights/teacher_checkpoint.pth``

Requires the ``chexfound`` extra::

    uv pip install -e ".[chexfound]"
"""

# ──────────────────────────────────────────────────────────────────────────────
# Implementation notes
# ──────────────────────────────────────────────────────────────────────────────
# CheXFound uses a DINOv2/iBOT-style ViT-L/16 backbone saved from 4-shard FSDP
# training. The teacher checkpoint stores blocks as backbone.blocks.GROUP.IDX.*
# where GROUP is the FSDP shard (0–3) and IDX is the absolute block index
# (0–23). The official CheXFound loading code drops the GROUP dimension to
# produce flat blocks.ABS_IDX.* keys, which match block_chunks=0 (flat
# nn.ModuleList). See:
#   chexfound/scripts/chexfound_with_glori.py → load_pretrained_weights
#
# Key stripping logic:
#   backbone.blocks.GROUP.IDX.xxx  →  blocks.IDX.xxx  (keep ls[1], skip ls[2])
#   backbone.<other>               →  <other>          (strip backbone. prefix)
#   <non-backbone>                 →  unchanged        (dino_head / ibot_head — loaded strict=False)
#
# model(x) returns the CLS token [B, 1024] directly (DINOv2-style forward).
# get_intermediate_layers strips CLS + register tokens from output;
# use return_class_token=True to retrieve patch tokens and CLS separately.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sys

import torch
import monai.transforms as mt

from radharmony.evaluator.transforms import EncoderPreprocessTransform, RadiologyEncoderTransform
from radharmony.evaluator.wrappers import ImageEncoderWrapper
from ._utils import patches_to_spatial

_INPUT_SIZE = 512
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Default: <repo_root>/third_party_models/CheXFound
_DEFAULT_CHEXFOUND_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party_models", "CheXFound")
)


def make_chexfound(
    chexfound_dir: str = _DEFAULT_CHEXFOUND_DIR,
    checkpoint_path: str | None = None,
    device: str = "cuda",
    output_keys: set[str] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[mt.Compose, ImageEncoderWrapper]:
    """Return ``(transform, encoder)`` for CheXFound.

    A single ``vit_large`` instance is created and its weights loaded once.
    The same model object is captured in both the transform's preprocess closure
    (CPU; runs in DataLoader workers) and the ``ImageEncoderWrapper`` (on
    *device*). No weights are loaded twice.

    Parameters
    ----------
    chexfound_dir :
        Path to the cloned CheXFound repository. Defaults to
        ``<repo_root>/third_party_models/CheXFound``.
    checkpoint_path :
        Path to ``teacher_checkpoint.pth``. Defaults to
        ``<chexfound_dir>/weights/teacher_checkpoint.pth``.
    device :
        Target device for the encoder (e.g. ``"cuda"``, ``"cuda:2"``,
        ``"cpu"``). The transform's preprocess step always runs on CPU.
    output_keys :
        Keys to keep in each sample dict. Defaults to ``{"img", "cls"}``.
        Pass ``{"img", "mask"}`` or ``{"img", "cls", "mask"}`` for
        segmentation — the mask is spatially aligned to the image
        (NEAREST resize + center-crop to 512 × 512) without normalization.

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
        Classification mode (default): ``forward(x) -> Tensor[B, 1024]``
        — the CLS-token embedding from the ViT-L/16 backbone.

        Segmentation mode (``"mask"`` in *output_keys*):
        ``forward(x) -> Tensor[B, 1024, 32, 32]`` — spatial patch feature
        map (512 px / 16 px patches = 32 × 32 grid).

    Example
    -------
    ::

        from radharmony.evaluator.backbones import make_chexfound
        from radharmony.dataset import VinDrCXRTrainDataset

        transform, encoder = make_chexfound(device="cuda:0")

        ds = VinDrCXRTrainDataset(
            base_image_dir="/data/vindr/train/",
            transform=transform,
            cache_dir="/tmp/cache/chexfound/",
            output_cls=True,
        )
    """
    # ── Lazy imports (optional deps + local repo) ─────────────────────────────
    from torchvision import transforms as tv_transforms  # type: ignore[import-untyped]
    from torchvision.transforms.functional import InterpolationMode  # type: ignore[import-untyped]

    chexfound_dir = os.path.abspath(chexfound_dir)
    if not os.path.isdir(chexfound_dir):
        raise FileNotFoundError(
            f"CheXFound directory not found: {chexfound_dir}\n"
            "Clone with: git clone https://github.com/RPIDIAL/CheXFound <path>"
        )
    if chexfound_dir not in sys.path:
        sys.path.insert(0, chexfound_dir)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(chexfound_dir, "weights", "teacher_checkpoint.pth")
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"CheXFound checkpoint not found: {checkpoint_path}\n"
            "Download from: https://drive.google.com/drive/folders/1GX2BWbujuVABtVpSZ4PTBykGULzrw806"
        )

    from chexfound.models.vision_transformer import vit_large  # type: ignore[import-untyped]

    # ── Build and load model ──────────────────────────────────────────────────
    # block_chunks=0 → flat nn.ModuleList (keys: blocks.0.* … blocks.23.*)
    # This matches the output of the official FSDP key stripping below.
    # init_values=1e-5 → enables LayerScale; without it the trained `ls1.gamma`/
    # `ls2.gamma` weights (48 tensors in the teacher checkpoint, magnitudes up
    # to ~0.16) load into nn.Identity placeholders and are silently dropped,
    # so every block's residual contribution is left unscaled. Matches the
    # `layerscale: 1.0e-05` field in chexfound/configs/ssl_default_config.yaml.
    model = vit_large(
        patch_size=16,
        num_register_tokens=4,
        img_size=_INPUT_SIZE,
        ffn_layer="swiglufused",
        block_chunks=0,
        init_values=1.0e-05,
    )

    # Official CheXFound key stripping (from chexfound_with_glori.py):
    # backbone.blocks.GROUP.ABS_IDX.xxx → blocks.ABS_IDX.xxx
    # backbone.<other>                  → <other>
    # dino_head.* / ibot_head.*         → unchanged (ignored via strict=False)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["teacher"]
    new_state_dict = {}
    for k, v in raw.items():
        if k.startswith("backbone"):
            ls = k.split(".")
            if "blocks" in k:
                new_k = ".".join([ls[1], *ls[3:]])  # drop group index
            else:
                new_k = ".".join(ls[1:])
        else:
            new_k = k
        new_state_dict[new_k] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # ── MONAI loader (handles PNG, JPEG, DICOM, NIfTI, …) ────────────────────
    from ._utils import squeeze_frame_dim, to_3channel, normalize_to_uint8

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

    # ── Image transform (ImageNet norm, BICUBIC resize) ───────────────────────
    # ToTensor() is omitted — _loader already produces a torch.Tensor.
    # Divide by 255 to bring uint8 → float [0, 1] before Normalize.
    _tv_transform = tv_transforms.Compose([
        tv_transforms.Resize(_INPUT_SIZE, interpolation=InterpolationMode.BICUBIC),
        tv_transforms.CenterCrop(_INPUT_SIZE),
        tv_transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    def _preprocess(path) -> torch.Tensor:
        img = _loader(str(path)).float() / 255.0  # [3, H, W] float in [0, 1]
        return _tv_transform(img)  # [3, 512, 512]

    # ── Mask transform (NEAREST resize, no intensity normalization) ───────────
    _tv_transform_mask = tv_transforms.Compose([
        tv_transforms.Resize(_INPUT_SIZE, interpolation=InterpolationMode.NEAREST),
        tv_transforms.CenterCrop(_INPUT_SIZE),
    ])

    def _preprocess_mask(path) -> torch.Tensor:
        mask = _loader_mask(str(path))[:1].float()  # [1, H, W]
        return _tv_transform_mask(mask)  # [1, 512, 512]

    # ── Model call ────────────────────────────────────────────────────────────
    _seg_mode = output_keys is not None and "mask" in output_keys

    if _seg_mode:
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            # get_intermediate_layers strips CLS + register tokens by default.
            # return_class_token=True returns (patch_tokens, cls_token) per layer.
            patches, _ = m.get_intermediate_layers(x, n=1, return_class_token=True)[0]
            return patches_to_spatial(patches)  # [B, 1024, 32, 32]
    else:
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            return m(x)  # [B, 1024] — CLS token

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
