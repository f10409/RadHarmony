# DINOv3

| | |
|---|---|
| Recipe | `make_dinov3()`, returns `(transform, encoder)` |
| Model card | [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) |
| Input size | 224 × 224 (bilinear resize of the whole image, no crop) |
| Normalization | ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225), from the DINOv3 processor |
| Feature | CLS token (`pooler_output`), 768 values |
| Install extra | `dinov3` |
| Segmentation features | `Tensor[B, 768, 14, 14]` |

DINOv3 is a ViT-B/16 from Meta (Siméoni et al., arXiv 2025), trained with self-supervised learning on
LVD-1689M, a set of web images. It is not a medical model. Use it as a strong general-domain baseline next to
the chest X-ray models.

## Quick use

This block encodes one [Montgomery](../datasets/cxr-boxes-masks/montgomery_cxr.md) image.

<!-- block: run=gpu timeout=600 -->
```python
import torch
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator.backbones import make_dinov3

transform, encoder = make_dinov3(device="cuda")
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
2. Resize: the whole image goes to 224 × 224 with bilinear interpolation (`size` 224 × 224, `resample=2` in
   `preprocessor_config.json` of the model card). The resize does not keep the aspect ratio. The processor
   does not crop.
3. Scale: the values are divided by 255.
4. Mean and std: the ImageNet values.
5. Dtype: the transform casts the image to `dtype`.

Steps 2 to 4 are the `AutoImageProcessor` of the model card. Its settings come from
`preprocessor_config.json`.

## Segmentation mode

<!-- block: run=gpu timeout=600 -->
```python
transform, encoder = make_dinov3(device="cuda", output_keys={"img", "mask"})
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    patches = encoder(img[None].to("cuda"))  # img from Quick use
print(patches.shape)  # torch.Size([1, 768, 14, 14])
```

The token sequence of DINOv3 is the CLS token, 4 register tokens, then the patch tokens. The recipe drops the
CLS and register tokens of `last_hidden_state` and keeps the patch tokens. It reads the number of register
tokens from `config.num_register_tokens`. The register tokens do not match a patch of the image. The patch
size is 16, so the grid is 224 / 16 = 14 on each side. See [Segmentation mode](index.md#segmentation-mode) for
the shared rules.

## Arguments

<!-- args: radharmony.evaluator.backbones.make_dinov3 -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `device` | `str` | `"cuda"` | The device of the encoder, for example `"cuda:1"` or `"cpu"` |
| `hub` | `str` | `"facebook/dinov3-vitb16-pretrain-lvd1689m"` | The Hugging Face name of the model |
| `output_keys` | `set[str]` | `None` | The keys of each sample. `None` gives `{"img", "cls"}`. With `"mask"`, the encoder gives patch features ([Segmentation mode](index.md#segmentation-mode)) |
| `dtype` | `torch.dtype` | `torch.bfloat16` | The dtype of the image tensor ([Output dtype](index.md#output-dtype)) |

## Notes

- The first call downloads the weights from Hugging Face.
- The Hugging Face model is gated. Accept its license on the model card, then run `hf auth login` before the
  first call.
- Another DINOv3 ViT model (`hub`) works with the same recipe. It gives its own feature size.
