# SigLIP2

Google's general-domain SigLIP2 — `google/siglip2-so400m-patch16-384`.
General-domain (not medical) but a useful zero-shot / contrastive baseline.

| Embed dim | Input size | Returns | Extra |
|-----------|------------|---------|-------|
| 1152 | 384×384 | `(transform, image_encoder, text_encoder, processor)` | `siglip2` |

SigLIP-So-400m/16 (16-px patches). Preprocessed via the HuggingFace
`AutoProcessor` (SigLIP normalization, BICUBIC resize to 384×384).

## Install

```bash
uv pip install -e ".[siglip2]"
```

Weights are auto-downloaded from HuggingFace on first call.

## Usage

```python
from radharmony.evaluator.backbones import make_siglip2
from radharmony.dataset import VinDrCXRTrainDataset

transform, image_encoder, text_encoder, processor = make_siglip2(device="cuda:0")

ds = VinDrCXRTrainDataset(
    base_image_dir="/data/vindr/train/",
    transform=transform,
    cache_dir="/tmp/cache/siglip2/",
    output_cls=True,
)
```

Vision-only evaluators ignore `text_encoder` / `processor`; zero-shot needs
all four.

## Segmentation mode

```python
transform, image_encoder, *_ = make_siglip2(device="cuda:0", output_keys={"img", "mask"})
# image_encoder(x) -> Tensor[B, 1152, 24, 24]   (384 / 16 = 24)
```

SigLIP has no CLS token, so `last_hidden_state` is pure patches — the recipe
reshapes them directly into the spatial map. The patch grid is read
dynamically from `model.config.vision_config.{image_size, patch_size}` so
the recipe survives a hub-identifier swap to a different SigLIP2 size.
