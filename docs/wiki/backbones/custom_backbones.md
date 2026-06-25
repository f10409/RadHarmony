# Custom Backbones

This page explains how to plug any vision model into the RadHarmony evaluator.
The evaluator's encoder contract is intentionally minimal — any callable with
the signature `forward(x: Tensor) -> Tensor[B, D]` qualifies. This page covers
two progressive levels of integration:

1. **Quick wiring** — wrap an existing model in a few lines using
   `ImageEncoderWrapper` and `EncoderPreprocessTransform`.
2. **Reusable recipe** — package the wiring as a `make_<model>()` factory
   following the `backbones/_template.py` pattern so it can be shared or
   contributed upstream.

---

## The two-piece pattern

RadHarmony splits model integration into two independent pieces:

| Piece | Where | Responsibility |
|-------|-------|----------------|
| **Preprocessing transform** | Dataset side (MONAI, DataLoader workers) | Read the image file, resize, normalize, return `Tensor[C, H, W]` |
| **Encoder wrapper** | Model side (GPU) | Route `forward(x)` to the backbone, extract the pooled feature vector |

Keeping them separate means preprocessing runs in CPU workers with the correct
reader and statistics for each backbone — no silent double-normalization.

---

## `ImageEncoderWrapper`

`ImageEncoderWrapper` is a thin `nn.Module` that registers a backbone as a
submodule and routes its forward through optional `model_call` / `pool`
hooks. This makes `.cuda()`, `state_dict()`, and `parameters()` propagate
correctly — a plain `lambda` over an `nn.Module` silently breaks all three:

- `encoder.to("cuda")` moves the backbone weights with it.
- `encoder.state_dict()` / `.parameters()` see the backbone — required for
  `FinetuneEvaluator` when `freeze_backbone=False`, and for saving a fine-tuned
  checkpoint.
- `copy.deepcopy(encoder)` correctly duplicates the backbone.

A closure over a loose `nn.Module` silently fails all three.

### HuggingFace (the common case)

```python
from radharmony.evaluator import ImageEncoderWrapper
from transformers import AutoModel

model = AutoModel.from_pretrained("microsoft/rad-dino")

# pool="cls" (default) | "mean" | "pooler" | callable(out) -> Tensor[B, D]
encoder = ImageEncoderWrapper.from_huggingface(model, pool="cls").cuda()
```

`from_huggingface` routes the forward through `model(pixel_values=x)`.

### Other ecosystems — `from_custom`

```python
# timm: num_classes=0 → backbone already returns (B, D); no hooks needed
encoder = ImageEncoderWrapper.from_custom(
    timm.create_model("resnet50", pretrained=True, num_classes=0)
).cuda()

# OpenCLIP / BiomedCLIP
encoder = ImageEncoderWrapper.from_custom(
    clip_model, model_call=lambda m, x: m.encode_image(x)
).cuda()

# Bespoke — point model_call / pool at whatever returns features
encoder = ImageEncoderWrapper.from_custom(
    my_model,
    model_call=lambda m, x: m(x, return_dict=True),
    pool=lambda out: out["features"][:, 0, :],
).cuda()
```

See the [`_model_call` patterns table](#common-_model_call-patterns) at the
bottom for more.

---

## `EncoderPreprocessTransform`

`EncoderPreprocessTransform` is a MONAI `MapTransform` that calls your
preprocessing function on each configured key. Plug it into any dataset's
`transform=` argument. The base class does **no** conversion or reading — the
callable owns the full `sample value → Tensor[C, H, W]` contract.

Why hand the whole contract to the callable: each encoder was pretrained with a
specific reader (`PIL.Image.open` for HF ViTs; a DICOM reader for some CXR
models) and a specific preprocessor. Forcing a generic tensor↔PIL round-trip or
a shared MONAI reader silently perturbs pixel statistics — letting the encoder's
own machinery run end-to-end avoids that.

### HuggingFace processor

```python
from radharmony.evaluator import EncoderPreprocessTransform

pp = EncoderPreprocessTransform.from_huggingface(
    keys=["img"],
    processor="microsoft/rad-dino",
    reader="pil",   # see the reader table below
)
```

`reader` options for `from_huggingface`:

| Value | What it does |
|---|---|
| `"pil"` (default) | `PIL.Image.open(path).convert("RGB")` — matches HF pretraining |
| `"monai"` | `mt.LoadImage(ensure_channel_first=True)` + `mt.Transpose([0, 2, 1])` — DICOM / NIfTI via MONAI, fixing the `(C, W, H)` axis order MONAI writes for 2D |
| `callable` | Your own `(path) -> image-like` |
| `None` | Upstream (`LoadImaged`) already loaded the image; pass through |

`reader="monai"` is 2D-only; 3D volumes need a different transpose.

### Custom preprocessor

```python
def my_preprocess(path) -> torch.Tensor:
    # owns the full contract: read → resize → normalize → Tensor[C, H, W]
    ...

pp = EncoderPreprocessTransform.from_custom(keys=["img"], preprocess=my_preprocess)
```

---

## `RadiologyEncoderTransform`

`RadiologyEncoderTransform` is a fluent builder that wraps an
`EncoderPreprocessTransform` and adds the same augmentation hooks
(`with_flip`, etc.) as the standard `RadiologyTransform2D`.

```python
from radharmony.evaluator import EncoderPreprocessTransform, RadiologyEncoderTransform

pp = EncoderPreprocessTransform.from_huggingface(
    keys=["img"], processor="microsoft/rad-dino",
)

# Classification — keep img + cls, add horizontal flip augmentation
transform = (
    RadiologyEncoderTransform(preprocess=pp, output_keys={"img", "cls"})
    .with_flip(spatial_axis=1)
    .get_transform()
)

# Pass to any dataset
from radharmony.dataset import VinDrCXRTrainDataset

ds = VinDrCXRTrainDataset(
    base_image_dir="/data/VinDr-CXR/",
    transform=transform,
    output_cls=True,
    cache_dir=None,
)
```

The final `ToTensorD` defaults to `dtype=torch.bfloat16` so the dataset emits
bf16 tensors — bridged transparently to fp32 encoder weights by the
`BaseClsEvaluator` / `BaseSegEvaluator` autocast helper. Pass
`RadiologyEncoderTransform(..., dtype=torch.float32)` to skip autocast for an
encoder that cannot autocast. Every built-in `make_*` factory forwards a
`dtype=` kwarg to this builder, so the same override works at the recipe
level (e.g. `make_raddino(dtype=torch.float32)`).

---

## Writing a reusable backbone recipe

For models you use repeatedly, package the wiring as a factory function
following [`radharmony/evaluator/backbones/_template.py`](https://github.com/f10409/RadHarmony/blob/main/radharmony/evaluator/backbones/_template.py).
Three design rules:

1. **Lazy imports** — `import` heavy deps inside the factory body so the
   module imports even without the optional dep installed.
2. **One model instance** — share it between the transform's preprocess
   closure and the encoder wrapper; never load weights twice.
3. **No preprocessing in the wrapper** — all reading/resizing/normalizing
   lives in the MONAI transform.

### Skeleton

```python
# radharmony/evaluator/backbones/my_model.py
import torch
import monai.transforms as mt
from radharmony.evaluator.transforms import EncoderPreprocessTransform, RadiologyEncoderTransform
from radharmony.evaluator.wrappers import ImageEncoderWrapper
from ._utils import squeeze_frame_dim, to_3channel, normalize_to_uint8

_MODEL_HUB = "org/my-model"


def make_my_model(device: str = "cuda"):
    """Return (transform, encoder) for MyModel."""
    from transformers import AutoModel, AutoImageProcessor  # lazy

    _model = AutoModel.from_pretrained(_MODEL_HUB)
    _processor = AutoImageProcessor.from_pretrained(_MODEL_HUB)

    _loader = mt.Compose([
        mt.LoadImage(image_only=True, ensure_channel_first=True),
        mt.Lambda(squeeze_frame_dim),
        mt.Transpose(indices=[0, 2, 1]),
        mt.Lambda(to_3channel),
        mt.Lambda(normalize_to_uint8),
    ])

    def _preprocess(path) -> torch.Tensor:
        return _processor(images=_loader(str(path)), return_tensors="pt")["pixel_values"][0]

    def _model_call(m, x):
        return m(pixel_values=x).last_hidden_state[:, 0]   # CLS token

    pp = EncoderPreprocessTransform.from_custom(keys=["img"], preprocess=_preprocess)
    transform = RadiologyEncoderTransform(preprocess=pp, output_keys={"img", "cls"}).get_transform()
    encoder = ImageEncoderWrapper.from_custom(_model, model_call=_model_call).to(device)
    return transform, encoder
```

Register in `backbones/__init__.py`:

```python
from .my_model import make_my_model
__all__ = [..., "make_my_model"]
```

For **vision-language** (zero-shot capable) models, return a 4-tuple
`(transform, image_encoder, text_encoder, tokenizer)` so `ZeroShotEvaluator`
can use it directly — see [`chexagent.py`](https://github.com/f10409/RadHarmony/blob/main/radharmony/evaluator/backbones/chexagent.py)
as a worked example.

---

## Loader helpers (`backbones._utils`)

These pure tensor functions are shared across all built-in backbone recipes
and available for reuse in your own:

```python
from radharmony.evaluator.backbones._utils import (
    squeeze_frame_dim,    # (C,W,H,1) → (C,W,H)  — drops MONAI's frame dim
    to_3channel,          # (1,H,W) → (3,H,W)     — grayscale → RGB repeat
    normalize_to_uint8,   # float tensor → uint8 [0, 255], min-max scaled
    patches_to_spatial,   # [B,N,D] → [B,D,H,W]   — ViT patches to feature map
    to_pil_rgb,           # (1,H,W) float → PIL RGB  — for PIL-expecting processors
)
```

---

## Common `_model_call` patterns

| Backbone family | `_model_call` |
|----------------|---------------|
| HF ViT (CLS token) | `lambda m, x: m(pixel_values=x).last_hidden_state[:, 0]` |
| HF SigLIP / CLIP | `lambda m, x: m(pixel_values=x).pooler_output` |
| OpenCLIP / BiomedCLIP | `lambda m, x: m.encode_image(x)` |
| timm (`num_classes=0`) | `None` (model already returns `(B, D)`) |
| RAD-DINO | `lambda m, x: m.encode(BatchFeature({"pixel_values": x}))[0]` |
| SSL backbone (MAE, SimCLR) | `lambda m, x: m.backbone(x)` or whichever attribute holds the extractor |
