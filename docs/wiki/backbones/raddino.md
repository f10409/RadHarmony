# RAD-DINO

Microsoft's chest-X-ray DINOv2 ViT — `microsoft/rad-dino` on HuggingFace.

| Embed dim | Input size | Returns | Extra |
|-----------|------------|---------|-------|
| 768 | 518×518 | `(transform, encoder)` | `raddino` |

ViT-B/14 (14-px patches). Preprocessing follows the RAD-DINO processor:
8-bit grayscale → 3-channel uint8 → resize 518 → ImageNet normalize.

## Install

```bash
uv pip install -e ".[raddino]"
```

Weights are auto-downloaded from HuggingFace on first call.

## Usage

```python
from radharmony.evaluator.backbones import make_raddino
from radharmony.dataset import VinDrCXRTrainDataset

transform, encoder = make_raddino(device="cuda:0")

ds = VinDrCXRTrainDataset(
    base_image_dir="/data/vindr/train/",
    transform=transform,
    cache_dir="/tmp/cache/raddino/",
    output_cls=True,
)
```

## Segmentation mode

```python
transform, encoder = make_raddino(device="cuda:0", output_keys={"img", "mask"})
# forward(x) -> Tensor[B, 768, 37, 37]   (518 / 14 = 37)
```
