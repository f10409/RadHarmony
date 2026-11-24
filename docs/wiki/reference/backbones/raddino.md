# RAD-DINO

| | |
|---|---|
| Recipe | `make_raddino()`, returns `(transform, encoder)` |
| Model card | [microsoft/rad-dino](https://huggingface.co/microsoft/rad-dino) |
| Input size | 518 × 518 (bicubic resize of the short side to 518, then center crop) |
| Normalization | mean 0.5307 and std 0.2583 on each channel, from the RAD-DINO processor |
| Feature | CLS token after the final norm, 768 values |
| Install extra | `raddino` |
| Segmentation features | `Tensor[B, 768, 37, 37]` |

RAD-DINO is a ViT-B/14 from Microsoft (Pérez-García et al., Nature Machine Intelligence 2025). It was trained
with DINOv2 self-supervised learning on chest X-rays from MIMIC-CXR, CheXpert, NIH-CXR, PadChest and BRAX. It
used the images only, with no labels and no text. Use it for frozen features in chest X-ray classification and
segmentation.

## Quick use

This block encodes one [Montgomery](../datasets/cxr-boxes-masks/montgomery_cxr.md) image.

<!-- block: run=gpu timeout=600 -->
```python
import torch
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator.backbones import make_raddino

transform, encoder = make_raddino(device="cuda")
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_cls=True,
    cache_dir=None,
)
img = ds.get_datasets()[0]["img"]  # Tensor[3, 518, 518], torch.bfloat16
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    feature = encoder(img[None].to("cuda"))
print(feature.shape)  # torch.Size([1, 768])
```

## How the input is prepared

1. Load: the shared loader gives a 3-channel `uint8` image in 0 to 255
   ([How a recipe loads an image](index.md#how-a-recipe-loads-an-image)).
2. Resize: the short side goes to 518 with bicubic interpolation (`size.shortest_edge=518`, `resample=3` in
   `preprocessor_config.json` of `microsoft/rad-dino`).
3. Crop: a center crop of 518 × 518.
4. Scale: the values are divided by 255.
5. Mean and std: 0.5307 and 0.2583 on each channel. These are the RAD-DINO values, not the ImageNet values.
6. Dtype: the transform casts the image to `dtype`.

Steps 2 to 5 are the RAD-DINO processor, as the `rad-dino` package loads it (`use_fast=False`, the PIL version
of `BitImageProcessor`).

## Segmentation mode

<!-- block: run=gpu timeout=600 -->
```python
transform, encoder = make_raddino(device="cuda", output_keys={"img", "mask"})
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    patches = encoder(img[None].to("cuda"))  # img from Quick use
print(patches.shape)  # torch.Size([1, 768, 37, 37])
```

The patch features are the patch tokens of the last block, after the final norm. The patch size is 14, so the
grid is 518 / 14 = 37 on each side. See [Segmentation mode](index.md#segmentation-mode) for the shared rules.

## Arguments

<!-- args: radharmony.evaluator.backbones.make_raddino -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `device` | `str` | `"cuda"` | The device of the encoder, for example `"cuda:1"` or `"cpu"` |
| `output_keys` | `set[str]` | `None` | The keys of each sample. `None` gives `{"img", "cls"}`. With `"mask"`, the encoder gives patch features ([Segmentation mode](index.md#segmentation-mode)) |
| `dtype` | `torch.dtype` | `torch.bfloat16` | The dtype of the image tensor ([Output dtype](index.md#output-dtype)) |

## Notes

- The first call downloads the weights from Hugging Face. The `rad-dino` package does the download.
- The feature is `last_hidden_state[:, 0]`, as in `RadDino.encode()` of the `rad-dino` package. The model card
  example uses this package.
