# MedSigLIP

| | |
|---|---|
| Recipe | `make_medsiglip()`, returns `(transform, encoder, text_encoder, processor)` |
| Model card | [google/medsiglip-448](https://huggingface.co/google/medsiglip-448) |
| Input size | 448 × 448 (bicubic resize of the whole image, no crop) |
| Normalization | mean 0.5 and std 0.5 on each channel, from the MedSigLIP processor |
| Feature | Pooled output of the attention-pool head (`pooler_output`), 1152 values |
| Install extra | `medsiglip` |
| Segmentation features | `Tensor[B, 1152, 32, 32]` |

MedSigLIP is a SigLIP model from Google, based on SigLIP-400M. Google trained it on medical image and text
pairs: chest X-rays, skin, eye and pathology images, and CT and MRI slices, with their reports. It is the
image encoder of MedGemma (Sellergren et al., arXiv 2025). Use it for zero-shot classification, for probes
with little data, and for image search.

## Quick use

This block encodes one [Montgomery](../datasets/cxr-boxes-masks/montgomery_cxr.md) image and one prompt.

<!-- block: run=gpu timeout=600 -->
```python
import torch
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator.backbones import make_medsiglip

transform, encoder, text_encoder, processor = make_medsiglip(device="cuda")
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_cls=True,
    cache_dir=None,
)
img = ds.get_datasets()[0]["img"]  # Tensor[3, 448, 448], torch.bfloat16
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    feature = encoder(img[None].to("cuda"))
print(feature.shape)  # torch.Size([1, 1152])

tokens = processor(text=["a chest X-ray with tuberculosis"], padding="max_length", return_tensors="pt")
print(text_encoder(tokens).shape)  # torch.Size([1, 1152]), float32 on the CPU
```

## How the input is prepared

1. Load: the shared loader gives a 3-channel `uint8` image in 0 to 255
   ([How a recipe loads an image](index.md#how-a-recipe-loads-an-image)).
2. Resize: the whole image goes to 448 × 448 with bicubic interpolation (`resample=3`). There is no crop,
   so the resize changes the aspect ratio of the image.
3. Scale: the values are divided by 255.
4. Mean and std: 0.5 and 0.5 on each channel, so the values go to -1 to 1.
5. Dtype: the transform casts the image to `dtype`.

Steps 2 to 4 are the `AutoProcessor` of the model (`preprocessor_config.json`).

The model card gives 2 ways to resize. It recommends `tf.image.resize` (bilinear, no antialias) to reproduce
the results of its evals, as the Big Vision code does. The other way is the resize of the Transformers
processor, which the card also allows. The recipe uses the processor (D-16). So a MedSigLIP score can differ a
little from the scores of the model card.

## Segmentation mode

<!-- block: run=gpu timeout=600 -->
```python
transform, encoder, *_ = make_medsiglip(device="cuda", output_keys={"img", "mask"})
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    patches = encoder(img[None].to("cuda"))  # img from Quick use
print(patches.shape)  # torch.Size([1, 1152, 32, 32])
```

The patch features are the last hidden state of the vision model. SigLIP has no CLS token, so all tokens
are patch tokens. The patch size is 14, so the grid is 448 / 14 = 32 on each side. See
[Segmentation mode](index.md#segmentation-mode) for the shared rules.

## Text encoder

- Output: `text_encoder(tokens)` gives `Tensor[T, 1152]`, float32, on the CPU. It is the `pooler_output`
  of the text model, the same as `get_text_features` of Transformers. The features are not
  L2-normalized. `ZeroShotEvaluator` normalizes them.
- Tokenizer: the returned `processor` (a `SiglipProcessor`). Its SentencePiece tokenizer makes the text
  lowercase. The model trained with 64 tokens, so pad each prompt to 64 tokens with
  `padding="max_length"`. `ZeroShotEvaluator` does this for a SigLIP processor.
- Prompt form: the model card uses short phrases, for example `"a photo of an arm with a rash"`.

Give both to the evaluator: `ZeroShotEvaluator(encoder, text_encoder, tokenizer=processor, ...)`. See
[Zero-shot](../evaluators/zero_shot.md).

## Arguments

<!-- args: radharmony.evaluator.backbones.make_medsiglip -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `device` | `str` | `"cuda"` | The device of the encoder and the text encoder, for example `"cuda:1"` or `"cpu"` |
| `hub` | `str` | `"google/medsiglip-448"` | The Hugging Face name of the model |
| `output_keys` | `set[str]` | `None` | The keys of each sample. `None` gives `{"img", "cls"}`. With `"mask"`, the encoder gives patch features ([Segmentation mode](index.md#segmentation-mode)) |
| `dtype` | `torch.dtype` | `torch.bfloat16` | The dtype of the image tensor ([Output dtype](index.md#output-dtype)) |

## Notes

- The first call downloads the weights from Hugging Face.
- The Hugging Face model is gated. Accept its license (the Health AI Developer Foundations terms of use)
  on the model card, then run `hf auth login` before the first call.
- One model gives the image encoder and the text encoder. The recipe does not load the weights 2 times.
- The processor does not cut a long prompt. A prompt of more than 64 tokens raises `ValueError` in the
  text model. `ZeroShotEvaluator` cuts it to 64 tokens and warns (D-52).
- Vision-only evaluators use only the transform and the encoder.
