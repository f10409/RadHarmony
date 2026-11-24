# MedImageInsight

| | |
|---|---|
| Recipe | `make_medimageinsights()`, returns `(transform, encoder, text_encoder, None)` |
| Model card | [lion-ai/MedImageInsights](https://huggingface.co/lion-ai/MedImageInsights) |
| Input size | 480 × 480 (bicubic resize of the whole image, no crop) |
| Normalization | ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225), from `config.yaml` of the model |
| Feature | Mean pool of the last DaViT stage, LayerNorm, projection and L2 normalization, 1024 values |
| Install extra | `medimageinsights` |
| Segmentation features | `Tensor[B, 2048, 15, 15]` |

MedImageInsight is an image and text embedding model from Microsoft (Codella et al., arXiv 2024), with a
DaViT (Dual Attention Vision Transformer) image encoder. Microsoft trained it with contrastive learning on
medical images of many kinds and their text. `lion-ai/MedImageInsights` (with an "s") is a copy of the
Microsoft weights with simpler loading code. Use it for zero-shot classification and as a general medical
image encoder.

## Quick use

This block encodes one [Montgomery](../datasets/cxr-boxes-masks/montgomery_cxr.md) image and one prompt.

<!-- block: run=gpu timeout=600 -->
```python
import torch
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator.backbones import make_medimageinsights

transform, encoder, text_encoder, _ = make_medimageinsights(device="cuda")
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_cls=True,
    cache_dir=None,
)
img = ds.get_datasets()[0]["img"]  # Tensor[3, 480, 480], torch.bfloat16
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    feature = encoder(img[None].to("cuda"))
print(feature.shape)  # torch.Size([1, 1024])

print(text_encoder(["tuberculosis"]).shape)  # torch.Size([1, 1024]), float32 on the CPU
```

## How the input is prepared

1. Load: the shared loader gives a PIL RGB image in 0 to 255
   ([How a recipe loads an image](index.md#how-a-recipe-loads-an-image)).
2. Resize: the whole image goes to 480 × 480 with bicubic interpolation. `config.yaml` sets
   `TEST.CENTER_CROP: false`, so there is no crop, and the resize changes the aspect ratio of the image.
3. Scale: the values are divided by 255.
4. Mean and std: the ImageNet values of `config.yaml` (see At a glance).
5. Dtype: the transform casts the image to `dtype`.

Steps 2 to 4 are the `preprocess` chain of the model (`build_transforms` of its own code, for test data)
(D-16).

## Segmentation mode

<!-- block: run=gpu timeout=600 -->
```python
transform, encoder, *_ = make_medimageinsights(device="cuda", output_keys={"img", "mask"})
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    patches = encoder(img[None].to("cuda"))  # img from Quick use
print(patches.shape)  # torch.Size([1, 2048, 15, 15])
```

The patch features are the tokens of the last DaViT block, before the mean pool and the projection. So
they have 2048 values (the width of the last stage), not 1024. A forward hook on
`image_encoder.blocks[3][0]` takes them, and the recipe removes the hook after each call. The 4 DaViT stages
reduce the size by 32 in total, so the grid is 480 / 32 = 15 on each side. See
[Segmentation mode](index.md#segmentation-mode) for the shared rules.

## Text encoder

- Output: `text_encoder(texts)` takes a `list[str]` and gives `Tensor[T, 1024]`, float32, on the CPU. It
  is `encode` of the model code: the feature of the end token, a projection, and L2 normalization.
- Tokenizer: the model code has its own CLIP tokenizer (BPE, lowercase). It pads and cuts each prompt to
  77 tokens. So the recipe returns `None` in place of a tokenizer. This keeps the same 4 items as the
  other image and text recipes.
- Prompt form: the model card passes plain labels, for example `"normal"` and `"Pneumonia"`.

Give the text encoder to the evaluator with no tokenizer: `ZeroShotEvaluator(encoder, text_encoder, ...)`.
Then the evaluator gives the prompt strings to the text encoder. See [Zero-shot](../evaluators/zero_shot.md).

## Arguments

<!-- args: radharmony.evaluator.backbones.make_medimageinsights -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `device` | `str` | `"cuda"` | The device of the encoder and the text encoder, for example `"cuda:1"` or `"cpu"` |
| `hub` | `str` | `"lion-ai/MedImageInsights"` | The Hugging Face name of the model |
| `output_keys` | `set[str]` | `None` | The keys of each sample. `None` gives `{"img", "cls"}`. With `"mask"`, the encoder gives patch features ([Segmentation mode](index.md#segmentation-mode)) |
| `dtype` | `torch.dtype` | `torch.bfloat16` | The dtype of the image tensor ([Output dtype](index.md#output-dtype)) |

## Notes

- The first call downloads the weights from Hugging Face. The recipe adds the downloaded folder to
  `sys.path` and imports `medimageinsightmodel` from it. The weights are in the `2024.09.27` folder.
- The model code leaves the model in train mode after loading. The recipe sets eval mode, so the
  drop-path layers are off.
- The model code prints `Model loaded successfully on device: cuda`, and PyTorch warns about
  `use_reentrant` of `torch.utils.checkpoint`. Both are harmless.
- One model gives the image encoder and the text encoder. The recipe does not load the weights 2 times.
- Vision-only evaluators use only the transform and the encoder.
