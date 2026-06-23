# Medical MAE

ViT pretrained with Masked Autoencoder (MAE) on a 0.3M / 0.5M chest-X-ray
corpus by [lambert-x/medical_mae](https://github.com/lambert-x/medical_mae).

| Variant | Embed dim | Pretrain corpus |
|---------|-----------|-----------------|
| `small` | 384 | 0.3M CXR |
| `base` (default) | 768 | 0.5M CXR |

| Input size | Returns | Extra |
|------------|---------|-------|
| 224×224 (CXR norm, **not** ImageNet) | `(transform, encoder)` | `medical_mae` + cloned repo + manual checkpoint + side-loaded `timm==0.4.12` |

Preprocessing follows the canonical
`Augmentation(normalize="chestx-ray").get_augmentation("full_224", "val")`
pipeline: resize 256, center-crop 224, normalize with
`mean=std=[0.5056, 0.252]*3` (CXR statistics).

## Install

```bash
uv pip install -e ".[medical_mae]"
```

Two manual steps:

1. Clone the upstream repo for `models_vit` and `util.pos_embed`:

   ```bash
   git clone https://github.com/lambert-x/medical_mae third_party_models/medical_mae
   ```

2. Download a pretrained checkpoint from the Google Drive link in the repo's
   README and place it under `third_party_models/medical_mae/`:

   - ViT-Base: `vit-b_CXR_0.5M_mae.pth`
   - ViT-Small: `vit_small_patch16_CXR_0.3M_mae_pretrain.pth`

On first call the recipe side-installs `timm==0.4.12` into
`third_party_models/medical_mae/timm-0412/` via
`uv pip install --target --no-deps` — the upstream `VisionTransformer`
subclass overrides `forward_features(self, x)`, which matches the timm 0.4.x
parent signature but not modern timm's
`forward_features(x, attn_mask=..., is_causal=...)`.

## Override default paths

| Kwarg | Default | Purpose |
|-------|---------|---------|
| `mae_dir=` | `third_party_models/medical_mae/` | Root of the cloned repo. The recipe adds this to `sys.path` to import `models_vit` and `util.pos_embed`. |
| `checkpoint_path=` | `<mae_dir>/vit-b_CXR_0.5M_mae.pth` (or `<mae_dir>/vit_small_patch16_CXR_0.3M_mae_pretrain.pth` for `variant="small"`) | Explicit path to the `.pth` weight file. |
| `legacy_timm_dir=` | `<mae_dir>/timm-0412/` | Where `timm==0.4.12` gets side-installed on first call. Point at a fast local cache. |

```python
from radharmony.evaluator.backbones import make_medical_mae

transform, encoder = make_medical_mae(
    variant="base",
    mae_dir="/shared/repos/medical_mae",
    checkpoint_path="/path/to/models/medical_mae/vit-b_CXR_0.5M_mae.pth",
    legacy_timm_dir="/path/to/cache/mae-timm-0412",
    device="cuda:0",
)
```

## Usage

```python
from radharmony.evaluator.backbones import make_medical_mae
from radharmony.dataset import VinDrCXRTrainDataset

transform, encoder = make_medical_mae(variant="base", device="cuda:0")

ds = VinDrCXRTrainDataset(
    base_image_dir="/data/vindr/train/",
    transform=transform,
    cache_dir="/tmp/cache/medical_mae/",
    output_cls=True,
)
```

## Segmentation mode

```python
transform, encoder = make_medical_mae(device="cuda:0", output_keys={"img", "mask"})
# forward(x) -> Tensor[B, D, 14, 14]   (224 / 16 = 14)
```

Default mode uses the CLS token (`global_pool=False`) with the pretrained
`self.norm`. `global_pool=True` is intended for fine-tuning, not zero-shot
features — it warm-starts `fc_norm` from a default LayerNorm init.
