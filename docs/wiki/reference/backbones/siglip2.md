# SigLIP 2

| | |
|---|---|
| Recipe | `make_siglip2()`, returns `(transform, encoder, text_encoder, processor)` |
| Model card | [google/siglip2-so400m-patch16-384](https://huggingface.co/google/siglip2-so400m-patch16-384) |
| Input size | 384 × 384 (bilinear resize of the whole image, no crop) |
| Normalization | mean 0.5 and std 0.5 on each channel, from the SigLIP 2 processor |
| Feature | Pooled output of the attention-pool head (`pooler_output`), 1152 values |
| Install extra | `siglip2` |
| Segmentation features | `Tensor[B, 1152, 24, 24]` |

SigLIP 2 is a general-domain image and text model from Google (Tschannen et al., arXiv 2025). Google trained
it on WebLI, a large set of web images with their text, not on medical data. Its text encoder is
multilingual. Use it as a general-domain baseline for zero-shot classification and for probes.

## Quick use

This block encodes one [Montgomery](../datasets/cxr-boxes-masks/montgomery_cxr.md) image and one prompt.

<!-- block: run=gpu timeout=600 -->
```python
import torch
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator.backbones import make_siglip2

transform, encoder, text_encoder, processor = make_siglip2(device="cuda")
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_cls=True,
    cache_dir=None,
)
img = ds.get_datasets()[0]["img"]  # Tensor[3, 384, 384], torch.bfloat16
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    feature = encoder(img[None].to("cuda"))
print(feature.shape)  # torch.Size([1, 1152])

tokens = processor(text=["This is a photo of a chest X-ray with tuberculosis."], padding="max_length",
                   return_tensors="pt")
print(text_encoder(tokens).shape)  # torch.Size([1, 1152]), float32 on the CPU
```

## How the input is prepared

1. Load: the shared loader gives a 3-channel `uint8` image in 0 to 255
   ([How a recipe loads an image](index.md#how-a-recipe-loads-an-image)).
2. Resize: the whole image goes to 384 × 384 with bilinear interpolation (`resample=2`). There is no crop,
   so the resize changes the aspect ratio of the image.
3. Scale: the values are divided by 255.
4. Mean and std: 0.5 and 0.5 on each channel, so the values go to -1 to 1.
5. Dtype: the transform casts the image to `dtype`.

Steps 2 to 4 are the `AutoProcessor` of the model (`preprocessor_config.json`).

## Segmentation mode

<!-- block: run=gpu timeout=600 -->
```python
transform, encoder, *_ = make_siglip2(device="cuda", output_keys={"img", "mask"})
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    patches = encoder(img[None].to("cuda"))  # img from Quick use
print(patches.shape)  # torch.Size([1, 1152, 24, 24])
```

The patch features are the last hidden state of the vision model. SigLIP has no CLS token, so all tokens
are patch tokens. The patch size is 16, so the grid is 384 / 16 = 24 on each side. See
[Segmentation mode](index.md#segmentation-mode) for the shared rules.

## Text encoder

- Output: `text_encoder(tokens)` gives `Tensor[T, 1152]`, float32, on the CPU. It is the `pooler_output`
  of the text model, the same as `get_text_features` of Transformers. The features are not
  L2-normalized. `ZeroShotEvaluator` normalizes them.
- Tokenizer: the returned `processor` (a `SiglipProcessor` with a Gemma tokenizer). The model trained
  with 64 tokens, so pad each prompt to 64 tokens with `padding="max_length"`. `ZeroShotEvaluator` does
  this for a SigLIP processor.
- Lowercase: SigLIP 2 trained with lowercase text. The tokenizer of the Hugging Face model keeps capital
  letters, so the recipe adds a lowercase step to the processor. The `Siglip2Tokenizer` of Transformers has
  the same step.
- Prompt form: the Transformers docs use `"This is a photo of {label}."`. The zero-shot pipeline uses the
  same template.

Give both to the evaluator: `ZeroShotEvaluator(encoder, text_encoder, tokenizer=processor, ...)`. See
[Zero-shot](../evaluators/zero_shot.md).

## Arguments

<!-- args: radharmony.evaluator.backbones.make_siglip2 -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `device` | `str` | `"cuda"` | The device of the encoder and the text encoder, for example `"cuda:1"` or `"cpu"` |
| `hub` | `str` | `"google/siglip2-so400m-patch16-384"` | The Hugging Face name of the model |
| `output_keys` | `set[str]` | `None` | The keys of each sample. `None` gives `{"img", "cls"}`. With `"mask"`, the encoder gives patch features ([Segmentation mode](index.md#segmentation-mode)) |
| `dtype` | `torch.dtype` | `torch.bfloat16` | The dtype of the image tensor ([Output dtype](index.md#output-dtype)) |

## Notes

- The first call downloads the weights from Hugging Face.
- One model gives the image encoder and the text encoder. The recipe does not load the weights 2 times.
- Another `hub` model gives other sizes. The feature size is `vision_config.hidden_size`, for example 768
  for `google/siglip2-base-patch16-224`. The patch grid is the input size divided by the patch size.
- Before 2026-10-28, the recipe did not make the text lowercase. Zero-shot results made before that date
  have different text features. The image features did not change. In one run on 202 frontal images of the
  CheXpert validation split (5 labels), the zero-shot macro AUROC went from 0.717 to 0.727.
- The processor does not cut a long prompt. A prompt of more than 64 tokens raises `ValueError` in the
  text model. `ZeroShotEvaluator` cuts it to 64 tokens and warns (D-52).
- Vision-only evaluators use only the transform and the encoder.
