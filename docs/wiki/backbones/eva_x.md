# EVA-X

ViT pretrained with masked image modeling on a merged 520k-CXR corpus
([hustvl/EVA-X](https://github.com/hustvl/EVA-X)). Model code lives in the
vendored submodule at `third_party_models/EVA-X/`; weights are pulled from
the `MapleF/eva_x` HuggingFace repo on first use.

| Variant | Embed dim | Params |
|---------|-----------|--------|
| `tiny` | 192 | ~6M |
| `small` | 384 | ~22M |
| `base` (default) | 768 | ~86M |

| Input size | Returns | Extra |
|------------|---------|-------|
| 224×224 (CXR norm, **not** ImageNet) | `(transform, encoder)` | `eva_x` + git submodule |

Preprocessing: resize 256, center-crop 224 (BICUBIC), normalize with EVA-X's
own CXR-pretraining stats (`mean=0.49185243` / `std=0.28509309` per channel) —
the canonical timm-style `crop_pct=224/256` eval pipeline.

## Install

```bash
uv pip install -e ".[eva_x]"
git submodule update --init third_party_models/EVA-X
```

Weights are auto-downloaded from HuggingFace on first call.

## Override default paths

| Kwarg | Default | Purpose |
|-------|---------|---------|
| `repo_path=` | `third_party_models/EVA-X/` | Path to the cloned EVA-X submodule. The recipe adds this to `sys.path` to import the model code. |

```python
from radharmony.evaluator.backbones import make_eva_x

transform, encoder = make_eva_x(
    variant="base",
    repo_path="/shared/repos/EVA-X",
    device="cuda:0",
)
```

## Usage

```python
from radharmony.evaluator.backbones import make_eva_x
from radharmony.dataset import VinDrCXRTrainDataset

transform, encoder = make_eva_x(variant="base", device="cuda:0")

ds = VinDrCXRTrainDataset(
    base_image_dir="/data/vindr/train/",
    transform=transform,
    cache_dir="/tmp/cache/eva_x/",
    output_cls=True,
)
```

## Segmentation mode

```python
transform, encoder = make_eva_x(device="cuda:0", output_keys={"img", "mask"})
# forward(x) -> Tensor[B, D, 14, 14]   (224 / 16 = 14)
```
