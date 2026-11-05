# Medical MAE

| | |
|---|---|
| Recipe | `make_medical_mae()`, returns `(transform, encoder)` |
| Model card | [lambert-x/medical_mae](https://github.com/lambert-x/medical_mae) |
| Input size | 224 × 224 (bilinear resize of the short side to 256, then center crop) |
| Normalization | mean 0.5056 and std 0.252 on each channel, from the authors' `Augmentation(normalize="chestx-ray")` |
| Feature | CLS token after the final norm, 768 values (base) or 384 (small) |
| Install extra | `medical_mae`, a clone of the authors' repo and a checkpoint |
| Segmentation features | `Tensor[B, D, 14, 14]`, with D = 768 or 384 |

<!-- claim: id=medical_mae.paper.n_images source="Xiao et al., WACV 2023" expect="266,340" -->
<!-- claim: id=medical_mae.readme.base_data source="Xiao et al., medical_mae README 2023" expect="0.5M" -->
Medical MAE is a ViT trained with a masked autoencoder (MAE) on chest X-rays, from Johns Hopkins University
(Xiao et al., WACV 2023). ViT-Small was trained on 266,340 chest X-rays, and ViT-Base on 0.5M (the authors'
README). Use it as a chest X-ray ViT that was trained with images only.

## Quick use

This block encodes one [Montgomery](../datasets/cxr-boxes-masks/montgomery_cxr.md) image.

<!-- block: run=gpu timeout=600 -->
```python
import torch
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator.backbones import make_medical_mae

transform, encoder = make_medical_mae(variant="base", device="cuda")
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_cls=True,
    cache_dir=None,
)
img = ds.get_datasets()[0]["img"]  # Tensor[3, 224, 224], torch.bfloat16
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    feature = encoder(img[None].to("cuda"))
print(feature.shape)  # torch.Size([1, 768])
```

## How the input is prepared

1. Load: the shared loader gives a 3-channel `uint8` image in 0 to 255
   ([How a recipe loads an image](index.md#how-a-recipe-loads-an-image)).
2. Scale: the values are divided by 255.
3. Resize: the short side goes to 256 with bilinear interpolation (the torchvision default).
4. Crop: a center crop of 224 × 224.
5. Mean and std: 0.5056 and 0.252 on each channel. These are the authors' chest X-ray values, not the ImageNet
   values.
6. Dtype: the transform casts the image to `dtype`.

Steps 3 to 5 are `Augmentation(normalize="chestx-ray").get_augmentation("full_224", "val")` of the authors
(`util/dataloader_med.py`, used in `util/datasets.py`).

## Segmentation mode

<!-- block: run=gpu timeout=600 -->
```python
transform, encoder = make_medical_mae(variant="base", device="cuda", output_keys={"img", "mask"})
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    patches = encoder(img[None].to("cuda"))  # img from Quick use
print(patches.shape)  # torch.Size([1, 768, 14, 14])
```

The patch features are the patch tokens after the final norm (`self.norm`), without the CLS token. A forward
hook on `norm` takes them. The patch size is 16, so the grid is 224 / 16 = 14 on each side. See
[Segmentation mode](index.md#segmentation-mode) for the shared rules.

## Feature

The feature is the CLS token after the pretrained final norm (`norm`). The recipe builds the model with
`global_pool=False`, the default of the authors' linear probe (`main_linprobe_chestxray.py`) (D-16). With
`global_pool=True`, the model uses a new `fc_norm` with no pretrained weights.

## Weights and code

Do these 2 steps by hand before the first call:

1. Clone the authors' repo. The recipe imports `models_vit` and `util.pos_embed` from it.

   <!-- block: run=skip reason="clones a repo" -->
   ```bash
   git clone https://github.com/lambert-x/medical_mae third_party_models/medical_mae
   ```

2. Download a pretrained checkpoint from the Google Drive link in the README of the repo. Put it in the clone
   folder: `vit-b_CXR_0.5M_mae.pth` (base) or `vit_small_patch16_CXR_0.3M_mae_pretrain.pth` (small).

The model code of the authors needs `timm==0.4.12`. Its `forward_features(self, x)` does not match the newer
`timm`, which calls `forward_features(x, attn_mask=..., is_causal=...)`. On the first call, the recipe
installs `timm==0.4.12` into `legacy_timm_dir`
([Weights and code locations](index.md#weights-and-code-locations)).

## Arguments

<!-- args: radharmony.evaluator.backbones.make_medical_mae -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `variant` | `str` | `"base"` | `"base"` or `"small"`. Any other value raises `ValueError` |
| `checkpoint_path` | `str` | `None` | The checkpoint file. `None` gives the file of `variant` in `mae_dir`. A missing file raises `FileNotFoundError` |
| `mae_dir` | `str` | `None` | The folder of the clone. `None` gives `third_party_models/medical_mae/` at the repo root. A missing folder raises `FileNotFoundError` |
| `legacy_timm_dir` | `str` | `None` | The folder for `timm==0.4.12`. `None` gives `<mae_dir>/timm-0412/` |
| `device` | `str` | `"cuda"` | The device of the encoder, for example `"cuda:1"` or `"cpu"` |
| `output_keys` | `set[str]` | `None` | The keys of each sample. `None` gives `{"img", "cls"}`. With `"mask"`, the encoder gives patch features ([Segmentation mode](index.md#segmentation-mode)) |
| `dtype` | `torch.dtype` | `torch.bfloat16` | The dtype of the image tensor ([Output dtype](index.md#output-dtype)) |

## Notes


- The recipe loads the checkpoint as `main_linprobe_chestxray.py` does. It drops the head weights when their
  shape does not match the model, and loads with `strict=False`. The decoder keys of MAE are not used.
