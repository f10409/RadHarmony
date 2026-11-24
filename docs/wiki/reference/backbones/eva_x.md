# EVA-X

| | |
|---|---|
| Recipe | `make_eva_x()`, returns `(transform, encoder)` |
| Model card | [hustvl/EVA-X](https://github.com/hustvl/EVA-X) (weights: [MapleF/eva_x](https://huggingface.co/MapleF/eva_x)) |
| Input size | 224 × 224 (bicubic resize of the short side to 256, then center crop) |
| Normalization | mean 0.49185243 and std 0.28509309 on each channel, from `build_transform()` of the EVA-X repo |
| Feature | Mean of the patch tokens, then `fc_norm`, 768 values (base), 384 (small) or 192 (tiny) |
| Install extra | `eva_x` and a clone of the authors' repo |
| Segmentation features | `Tensor[B, D, 14, 14]`, with D = 768, 384 or 192 |

<!-- claim: id=eva_x.paper.data source="Yao et al., npj Digital Medicine 2025" expect="Merged520k" -->
EVA-X is a family of ViTs from HUST VL (Yao et al., npj Digital Medicine 2025). It was trained with masked
image modeling on Merged520k, a merged set of public chest X-ray datasets. It comes in 3 sizes (tiny, small
and base). Use it as a chest X-ray ViT that was trained with images only.

## Quick use

This block encodes one [Montgomery](../datasets/cxr-boxes-masks/montgomery_cxr.md) image.

<!-- block: run=gpu timeout=600 -->
```python
import torch
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator.backbones import make_eva_x

transform, encoder = make_eva_x(variant="base", device="cuda")
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
2. Resize: the short side goes to 256 with bicubic interpolation (`crop_pct = 224 / 256`).
3. Crop: a center crop of 224 × 224.
4. Scale: the values are divided by 255.
5. Mean and std: 0.49185243 and 0.28509309 on each channel. These are the EVA-X values, not the ImageNet
   values.
6. Dtype: the transform casts the image to `dtype`.

Steps 2 to 5 are `build_transform()` in `classification/utils/datasets.py` of the EVA-X repo. These are its
eval steps for an input size of 224.

## Segmentation mode

<!-- block: run=gpu timeout=600 -->
```python
transform, encoder = make_eva_x(variant="base", device="cuda", output_keys={"img", "mask"})
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    patches = encoder(img[None].to("cuda"))  # img from Quick use
print(patches.shape)  # torch.Size([1, 768, 14, 14])
```

The patch features are the patch tokens of `forward_features()`, without the CLS token. The patch size is 16,
so the grid is 224 / 16 = 14 on each side. The patch features have no final norm. The model moves the
pretrained `norm` weights to `fc_norm`, and `fc_norm` acts on the pooled feature only. See
[Segmentation mode](index.md#segmentation-mode) for the shared rules.

## Feature

The feature is the mean of the patch tokens, then the final norm `fc_norm`
(`forward_head(forward_features(x), pre_logits=True)`). The fine-tune scripts of the authors use this pool
(`--use_mean_pooling` in `classification/train_files/eva_x/`) (D-16). The model has no final norm for the CLS
token, so the recipe does not use the CLS token.

## Weights and code

The recipe imports the model code from a clone of the EVA-X repo. Clone it before the first call:

<!-- block: run=skip reason="clones a repo" -->
```bash
git clone https://github.com/hustvl/EVA-X.git third_party_models/EVA-X
```

The weights are the file `eva_x_<variant>_patch16_merged520k_mim.pt` of `MapleF/eva_x` on Hugging Face. The
tests used commit `f08e647` of the clone.

<!-- claim: id=eva_x.paper.params_tiny source="Yao et al., npj Digital Medicine 2025" expect="6M" -->
<!-- claim: id=eva_x.paper.params_small source="Yao et al., npj Digital Medicine 2025" expect="22M" -->
<!-- claim: id=eva_x.paper.params_base source="Yao et al., npj Digital Medicine 2025" expect="86M" -->
| Variant | Feature size | Parameters (Yao et al.) |
|---|---|---|
| `tiny` | 192 | 6M |
| `small` | 384 | 22M |
| `base` | 768 | 86M |

## Arguments

<!-- args: radharmony.evaluator.backbones.make_eva_x -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `device` | `str` | `"cuda"` | The device of the encoder, for example `"cuda:1"` or `"cpu"` |
| `variant` | `str` | `"base"` | `"tiny"`, `"small"` or `"base"`. Any other value raises `ValueError` |
| `output_keys` | `set[str]` | `None` | The keys of each sample. `None` gives `{"img", "cls"}`. With `"mask"`, the encoder gives patch features ([Segmentation mode](index.md#segmentation-mode)) |
| `dtype` | `torch.dtype` | `torch.bfloat16` | The dtype of the image tensor ([Output dtype](index.md#output-dtype)) |
| `repo_path` | `str` or `Path` | `None` | The folder of the EVA-X clone. `None` gives `third_party_models/EVA-X/` at the repo root. A missing folder raises `FileNotFoundError` |

## Notes

- The first call downloads the weights from Hugging Face.

- Before 2026-10-28, the recipe gave the CLS token. So results from before that date used another feature.
- When it loads, the model code prints the missing keys `head.weight` and `head.bias`. The recipe does not use
  the head.
