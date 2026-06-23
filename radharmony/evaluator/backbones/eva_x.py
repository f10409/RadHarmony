"""EVA-X backbone recipe (CXR-pretrained ViT).

Loads an EVA-X ViT pretrained with masked image modeling on a merged 520k-CXR
corpus. The model code lives in the vendored submodule at
``third_party_models/EVA-X/`` (cloned from https://github.com/hustvl/EVA-X);
weights are pulled from the ``MapleF/eva_x`` HuggingFace repo on first use.

Three variants are supported (selectable via ``variant=``):

==========  ==========  ==========
variant     embed dim   params
==========  ==========  ==========
``tiny``    192         ~6M
``small``   384         ~22M
``base``    768         ~86M
==========  ==========  ==========

Input: 224×224 RGB, normalized with EVA-X's own CXR-pretraining stats
(``mean=0.49185243`` / ``std=0.28509309`` per channel) and the timm-style
``crop_pct=224/256`` eval pipeline (resize 256, center-crop 224, BICUBIC).

Before first use, initialise the submodule::

    git submodule update --init third_party_models/EVA-X

Requires the ``eva_x`` extra::

    uv pip install -e ".[eva_x]"
"""

# ──────────────────────────────────────────────────────────────────────────────
# Implementation notes
# ──────────────────────────────────────────────────────────────────────────────
# 1. ALL backbone imports are LAZY (inside the factory body) so this module can
#    be imported even when the optional dependency is not installed.
#
# 2. A SINGLE model instance is created and shared between the transform's
#    preprocess closure and the encoder — no weights are loaded twice.
#
# 3. Image preprocessing (resize/crop/normalize) lives entirely in the MONAI
#    transform. ImageEncoderWrapper never does it.
#
# 4. The EVA-X checkpoint pickles numpy types alongside the weights. Since
#    PyTorch 2.6 defaults to ``weights_only=True``, we register the specific
#    numpy types EVA-X actually stores via ``add_safe_globals`` rather than
#    flipping to ``weights_only=False``.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import torch
import monai.transforms as mt

from radharmony.evaluator.transforms import EncoderPreprocessTransform, RadiologyEncoderTransform
from radharmony.evaluator.wrappers import ImageEncoderWrapper
from ._utils import squeeze_frame_dim, to_3channel, normalize_to_uint8, patches_to_spatial


_EVA_X_REPO_DEFAULT = Path(__file__).resolve().parents[3] / "third_party_models" / "EVA-X"
_HF_REPO = "MapleF/eva_x"

# EVA-X CXR-pretraining stats (per the upstream build_transform).
_EVA_X_MEAN = (0.49185243,) * 3
_EVA_X_STD = (0.28509309,) * 3

# timm eval pipeline for input_size <= 224.
_INPUT_SIZE = 224
_CROP_PCT = 224 / 256
_RESIZE_SIZE = int(_INPUT_SIZE / _CROP_PCT)   # 256

_EMBED_DIM = {"tiny": 192, "small": 384, "base": 768}


def make_eva_x(
    device: str = "cuda",
    variant: str = "base",
    output_keys: set[str] | None = None,
    dtype: torch.dtype = torch.bfloat16,
    repo_path: str | Path | None = None,
) -> tuple[mt.Compose, ImageEncoderWrapper]:
    """Return ``(transform, encoder)`` for EVA-X.

    A single EVA-X instance is created and shared: the preprocessing pipeline
    is captured in the transform's closure and the backbone is registered in
    the returned ``ImageEncoderWrapper``. No weights are loaded twice.

    Parameters
    ----------
    device :
        Target device for the encoder (e.g. ``"cuda"``, ``"cuda:7"``,
        ``"cpu"``). The transform's preprocess step runs on CPU in DataLoader
        workers regardless of this setting.
    variant :
        ``"tiny"`` (192-d, ~6M params), ``"small"`` (384-d, ~22M), or
        ``"base"`` (768-d, ~86M).
    output_keys :
        Keys to keep in each sample dict. Defaults to ``{"img", "cls"}``.
        Pass ``{"img", "mask"}`` or ``{"img", "cls", "mask"}`` for
        segmentation — the mask is processed through the same spatial
        pipeline as the image (resize + center-crop to 224 × 224) using
        nearest-neighbour interpolation.
    repo_path :
        Path to the EVA-X submodule. Defaults to ``third_party_models/EVA-X/``
        relative to the radharmony repo root; override only if the submodule
        was cloned elsewhere.

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
        the CLS-token embedding. ``D ∈ {192, 384, 768}`` depending on
        *variant*.

        Segmentation mode (``"mask"`` in *output_keys*):
        ``forward(x) -> Tensor[B, D, 14, 14]`` — spatial patch feature map.

    Example
    -------
    ::

        from radharmony.evaluator.backbones import make_eva_x
        from radharmony.dataset import VinDrCXRTrainDataset

        transform, encoder = make_eva_x(device="cuda:0", variant="base")

        ds = VinDrCXRTrainDataset(
            base_image_dir="/data/vindr/train/",
            transform=transform,
            cache_dir="/tmp/cache/eva_x_base/",
            output_cls=True,
        )
    """
    if variant not in _EMBED_DIM:
        raise ValueError(
            f"variant must be one of {sorted(_EMBED_DIM)}, got {variant!r}"
        )

    # ── Lazy imports (optional dependency) ────────────────────────────────────
    import numpy as np  # noqa: F401  — registered below
    from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]
    from torchvision import transforms as T
    from torchvision.transforms.functional import InterpolationMode

    # Make the vendored EVA-X submodule importable.
    _repo = Path(repo_path) if repo_path is not None else _EVA_X_REPO_DEFAULT
    if not _repo.exists():
        raise FileNotFoundError(
            f"EVA-X submodule not found at {_repo}. Initialise it with:\n"
            f"  git submodule update --init third_party_models/EVA-X"
        )
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))

    from eva_x import (  # type: ignore[import-untyped]
        eva_x_tiny_patch16,
        eva_x_small_patch16,
        eva_x_base_patch16,
    )
    _factory = {
        "tiny": eva_x_tiny_patch16,
        "small": eva_x_small_patch16,
        "base": eva_x_base_patch16,
    }[variant]

    # EVA-X checkpoints store numpy scalars + dtype objects alongside tensors;
    # allowlist them so PyTorch 2.6's default weights_only=True accepts them.
    # The (obj, "qualified_name") form is needed for numpy.core.* because
    # NumPy 2 split numpy.core → numpy._core while the pickle stream still
    # references the legacy path.
    torch.serialization.add_safe_globals([
        (np.core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        np.ndarray,
        np.dtypes.Float64DType,
        np.dtypes.Float32DType,
        np.dtypes.Int64DType,
        np.dtypes.Int32DType,
        np.dtypes.UInt8DType,
        np.dtypes.BoolDType,
    ])

    # ── Download checkpoint + build model ─────────────────────────────────────
    _ckpt_path = hf_hub_download(
        repo_id=_HF_REPO,
        filename=f"eva_x_{variant}_patch16_merged520k_mim.pt",
    )
    _model = _factory(pretrained=_ckpt_path).to(device).eval()

    # ── Image loader (CPU; runs in DataLoader workers) ────────────────────────
    # MONAI handles path → uint8 [3, H, W] tensor; torchvision handles the
    # canonical EVA-X eval pipeline (resize/crop/normalize).
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

    _post = T.Compose([
        T.Resize(_RESIZE_SIZE, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(_INPUT_SIZE),
        T.ConvertImageDtype(torch.float32),   # uint8 → float32 / 255
        T.Normalize(mean=_EVA_X_MEAN, std=_EVA_X_STD),
    ])
    _post_mask = T.Compose([
        T.Resize(_RESIZE_SIZE, interpolation=InterpolationMode.NEAREST),
        T.CenterCrop(_INPUT_SIZE),
    ])

    def _preprocess(path) -> torch.Tensor:
        img = _loader(str(path))      # uint8 [3, H, W]
        return _post(img)             # float32 [3, 224, 224]

    def _preprocess_mask(path) -> torch.Tensor:
        mask = _loader_mask(str(path))
        return _post_mask(mask)[:1]   # [1, 224, 224] — class indices

    # ── Model call closure (classification vs segmentation) ───────────────────
    _seg_mode = output_keys is not None and "mask" in output_keys

    if _seg_mode:
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            patches = m.forward_features(x)[:, 1:]   # skip CLS at index 0
            return patches_to_spatial(patches)        # [B, D, 14, 14]
    else:
        def _model_call(m, x: torch.Tensor) -> torch.Tensor:
            return m.forward_features(x)[:, 0]        # CLS token, [B, D]

    # ── Assemble ──────────────────────────────────────────────────────────────
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
