# BiomedCLIP

| | |
|---|---|
| Recipe | `make_biomed_clip()`, returns `(transform, encoder, text_encoder, tokenizer)` |
| Model card | [microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) |
| Input size | 224 × 224 (bicubic resize of the short side to 224, then center crop) |
| Normalization | CLIP mean (0.48145466, 0.4578275, 0.40821073) and std (0.26862954, 0.26130258, 0.27577711), from `open_clip_config.json` of the model |
| Feature | CLS token after the linear projection, 512 values |
| Install extra | `biomed` |
| Segmentation features | `Tensor[B, 768, 14, 14]` |

<!-- claim: id=biomed_clip.paper.pairs source="Zhang et al., NEJM AI 2025" expect="15 million" -->
BiomedCLIP is a CLIP model from Microsoft, with a ViT-B/16 image encoder and a PubMedBERT text encoder.
Microsoft trained it on PMC-15M, 15 million figure and caption pairs from PubMed Central articles. The
data covers many kinds of biomedical images, not only chest X-rays. Use it for zero-shot classification
and as a general biomedical image encoder.

## Quick use

This block encodes one [Montgomery](../datasets/cxr-boxes-masks/montgomery_cxr.md) image and one prompt.

<!-- block: run=gpu timeout=600 -->
```python
import torch
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator.backbones import make_biomed_clip

transform, encoder, text_encoder, tokenizer = make_biomed_clip(device="cuda")
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_cls=True,
    cache_dir=None,
)
img = ds.get_datasets()[0]["img"]  # Tensor[3, 224, 224], torch.bfloat16
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    feature = encoder(img[None].to("cuda"))
print(feature.shape)  # torch.Size([1, 512])

tokens = tokenizer(["this is a photo of chest X-ray with tuberculosis"])  # Tensor[1, 256]
print(text_encoder(tokens).shape)  # torch.Size([1, 512]), float32 on the CPU
```

## How the input is prepared

1. Load: the shared loader gives a PIL RGB image in 0 to 255
   ([How a recipe loads an image](index.md#how-a-recipe-loads-an-image)).
2. Resize: the short side goes to 224 with bicubic interpolation.
3. Crop: a center crop of 224 × 224.
4. Scale: the values are divided by 255.
5. Mean and std: the CLIP values of `open_clip_config.json` (see At a glance).
6. Dtype: the transform casts the image to `dtype`.

Steps 2 to 5 are the OpenCLIP `preprocess_val` of the model. It is the same transform that
`open_clip.create_model_from_pretrained` gives on the model card (D-16).

## Segmentation mode

<!-- block: run=gpu timeout=600 -->
```python
transform, encoder, *_ = make_biomed_clip(device="cuda", output_keys={"img", "mask"})
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    patches = encoder(img[None].to("cuda"))  # img from Quick use
print(patches.shape)  # torch.Size([1, 768, 14, 14])
```

The patch features are the patch tokens of the last ViT block, with no CLS token and no projection. So
they have 768 values (the ViT width), not 512. The patch size is 16, so the grid is 224 / 16 = 14 on each
side. See [Segmentation mode](index.md#segmentation-mode) for the shared rules.

## Text encoder

- Output: `text_encoder(tokens)` gives `Tensor[T, 512]`, float32, on the CPU. It is `encode_text` of the
  model: the CLS token of PubMedBERT and an MLP projection. The features are not L2-normalized.
  `ZeroShotEvaluator` normalizes them.
- Tokenizer: the OpenCLIP tokenizer of the model, with the BiomedBERT WordPiece vocabulary. It makes the
  text lowercase. It pads and cuts each prompt to 256 tokens, so it gives `Tensor[T, 256]`.
- Prompt form: the model card puts `"this is a photo of "` before each label, for example
  `"this is a photo of chest X-ray"`.

Give both to the evaluator: `ZeroShotEvaluator(encoder, text_encoder, tokenizer=tokenizer, ...)`. See
[Zero-shot](../evaluators/zero_shot.md).

## Arguments

<!-- args: radharmony.evaluator.backbones.make_biomed_clip -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `device` | `str` | `"cuda"` | The device of the encoder and the text encoder, for example `"cuda:1"` or `"cpu"` |
| `hub` | `str` | `"hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"` | The OpenCLIP name of the model, with the `hf-hub:` prefix |
| `output_keys` | `set[str]` | `None` | The keys of each sample. `None` gives `{"img", "cls"}`. With `"mask"`, the encoder gives patch features ([Segmentation mode](index.md#segmentation-mode)) |
| `dtype` | `torch.dtype` | `torch.bfloat16` | The dtype of the image tensor ([Output dtype](index.md#output-dtype)) |

## Notes

- The first call downloads the weights from Hugging Face.
- One model gives the image encoder and the text encoder. The recipe does not load the weights 2 times.
- Vision-only evaluators use only the transform and the encoder.
