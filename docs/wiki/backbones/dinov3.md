# DINOv3

Meta's general-domain DINOv3 ViT-Base —
`facebook/dinov3-vitb16-pretrain-lvd1689m`. Not medical-specific, but a
strong general-domain baseline for comparison.

| Embed dim | Input size | Returns | Extra |
|-----------|------------|---------|-------|
| 768 | 224×224 | `(transform, encoder)` | `dinov3` |

ViT-B/16 (16-px patches) with **4 register tokens** in addition to the CLS
token. Register tokens are an architectural choice from DINOv3 — they
participate in attention but don't correspond to spatial positions, and are
skipped in segmentation mode (patches only).

## Install

```bash
uv pip install -e ".[dinov3]"
```

Weights are auto-downloaded from HuggingFace on first call.

## Usage

```python
from radharmony.evaluator.backbones import make_dinov3
from radharmony.dataset import VinDrCXRTrainDataset

transform, encoder = make_dinov3(device="cuda:0")

ds = VinDrCXRTrainDataset(
    base_image_dir="/data/vindr/train/",
    transform=transform,
    cache_dir="/tmp/cache/dinov3/",
    output_cls=True,
)
```

## Segmentation mode

```python
transform, encoder = make_dinov3(device="cuda:0", output_keys={"img", "mask"})
# forward(x) -> Tensor[B, 768, 14, 14]   (224 / 16 = 14)
```

The recipe slices off CLS + 4 register tokens from `last_hidden_state` before
reshaping into the spatial map.
