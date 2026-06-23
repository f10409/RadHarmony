# Ark+

Vision-only chest-X-ray foundation model — a bare
`timm.models.swin_transformer.SwinTransformer` (Swin-L/192 @ 768 px, window 12)
pretrained on six CXR datasets via cyclic multi-task learning.

| Embed dim | Input size | Returns | Extra |
|-----------|------------|---------|-------|
| 1536 | 768×768 | `(transform, encoder)` | `ark_plus` + manual checkpoint + side-loaded `timm==0.5.4` |

## Install

```bash
uv pip install -e ".[ark_plus]"
```

Weights are not auto-downloadable. Three manual steps:

1. Clone the upstream repo (only needed for reference; the recipe does not
   import from it):

   ```bash
   git clone https://github.com/jlianglab/Ark third_party_models/Ark
   ```

2. Request weights at
   [https://forms.gle/qkoDGXNiKRPTDdCe8](https://forms.gle/qkoDGXNiKRPTDdCe8)
   (Google Form) or [https://www.wjx.cn/vm/OvwfYFx.aspx](https://www.wjx.cn/vm/OvwfYFx.aspx) (WeChat).

3. Place `Ark6_swinLarge768_ep50.pth.tar` at
   `third_party_models/Ark/Ark_Plus/Ark6_swinLarge768_ep50.pth.tar`.

On first call the recipe side-installs `timm==0.5.4` into
`third_party_models/Ark/timm-054/` via `uv pip install --target --no-deps` — the
0.5.4 Swin layout doesn't match modern timm (`downsample` is offset by one
layer index in newer versions). The legacy timm is used **only** during model
construction; the rest of the Python session keeps the venv's timm, so EVA-X
and other timm-based backbones still work.

## Override default paths

Both the checkpoint and the legacy-timm install dir can be redirected — useful
when weights live on shared storage, or when you want a fast local cache for
the side-install.

| Kwarg | Default | Purpose |
|-------|---------|---------|
| `checkpoint_path=` | `third_party_models/Ark/Ark_Plus/Ark6_swinLarge768_ep50.pth.tar` | Full path to the `.pth.tar` weight file. |
| `legacy_timm_dir=` | `third_party_models/Ark/timm-054/` | Where `timm==0.5.4` gets side-installed on first call. Point at a fast local cache (e.g. `/tmp`, `/path/to/cache…`) to avoid re-installing per user. |

```python
from radharmony.evaluator.backbones import make_ark_plus

transform, encoder = make_ark_plus(
    checkpoint_path="/path/to/models/Ark_Plus/Ark6_swinLarge768_ep50.pth.tar",
    legacy_timm_dir="/path/to/cache/ark-timm-054",
    device="cuda:0",
)
```

## Usage

```python
from radharmony.evaluator.backbones import make_ark_plus
from radharmony.dataset import VinDrCXRTrainDataset

transform, encoder = make_ark_plus(device="cuda:0")

ds = VinDrCXRTrainDataset(
    base_image_dir="/data/vindr/train/",
    transform=transform,
    cache_dir="/tmp/cache/ark_plus/",
    output_cls=True,
)
```

## Segmentation mode

```python
transform, encoder = make_ark_plus(device="cuda:0", output_keys={"img", "mask"})
# forward(x) -> Tensor[B, 1536, 24, 24]   (stride 32 -> 768/32 = 24)
```

Captured from the final `model.norm` LayerNorm via a forward hook on the Swin
hierarchical features.
