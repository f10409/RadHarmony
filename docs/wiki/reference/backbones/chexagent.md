# CheXagent (XraySigLIP)

| | |
|---|---|
| Recipe | `make_chexagent()`, returns `(transform, encoder, text_encoder, processor)` |
| Model card | [StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli](https://huggingface.co/StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli) |
| Input size | 512 × 512 (bicubic resize of the whole image, no crop) |
| Normalization | mean 0.5 and std 0.5 on each channel, from the XraySigLIP processor |
| Feature | Pooled output of the attention-pool head (`pooler_output`), 1024 values |
| Install extra | `chexagent` |
| Segmentation features | `Tensor[B, 1024, 32, 32]` |

XraySigLIP is a SigLIP image and text model for chest X-rays from Stanford AIMI. It is the image encoder
of the CheXagent-2 report generator (Chen et al., arXiv 2024). The name says that it is a SigLIP ViT-L/16
that started from WebLI weights at 384 × 384. Use it for zero-shot classification and probes on chest X-rays.

## Quick use

This block encodes one [Montgomery](../datasets/cxr-boxes-masks/montgomery_cxr.md) image and one prompt.

<!-- block: run=gpu timeout=600 -->
```python
import torch
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator.backbones import make_chexagent

transform, encoder, text_encoder, processor = make_chexagent(device="cuda")
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_cls=True,
    cache_dir=None,
)
img = ds.get_datasets()[0]["img"]  # Tensor[3, 512, 512], torch.bfloat16
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    feature = encoder(img[None].to("cuda"))
print(feature.shape)  # torch.Size([1, 1024])

tokens = processor(text=["tuberculosis"], padding="max_length", return_tensors="pt")  # 256 tokens
print(text_encoder(tokens).shape)  # torch.Size([1, 1024]), float32 on the CPU
```

## How the input is prepared

1. Load: the shared loader gives a 3-channel `uint8` image in 0 to 255
   ([How a recipe loads an image](index.md#how-a-recipe-loads-an-image)).
2. Resize: the whole image goes to 512 × 512 with bicubic interpolation (`resample=3`). There is no crop,
   so the resize changes the aspect ratio of the image.
3. Scale: the values are divided by 255.
4. Mean and std: 0.5 and 0.5 on each channel, so the values go to -1 to 1.
5. Dtype: the transform casts the image to `dtype`.

Steps 2 to 4 are the `AutoProcessor` of the model (`preprocessor_config.json`).

The name of the model says 384, but its processor resizes to 512. The CheXagent-2 model code also resizes
to 512 × 512 (bicubic) with the mean and std of this processor (D-16).

## Segmentation mode

<!-- block: run=gpu timeout=600 -->
```python
transform, encoder, *_ = make_chexagent(device="cuda", output_keys={"img", "mask"})
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    patches = encoder(img[None].to("cuda"))  # img from Quick use
print(patches.shape)  # torch.Size([1, 1024, 32, 32])
```

The patch features are the last hidden state of the vision model. SigLIP has no CLS token, so all tokens
are patch tokens. The patch size is 16, so the grid is 512 / 16 = 32 on each side. See
[Segmentation mode](index.md#segmentation-mode) for the shared rules.

## Text encoder

- Output: `text_encoder(tokens)` gives `Tensor[T, 1024]`, float32, on the CPU. It is the `pooler_output`
  of the text model, the same as `get_text_features` of Transformers. The features are not
  L2-normalized. `ZeroShotEvaluator` normalizes them.
- Tokenizer: the returned `processor` (a `SiglipProcessor`). Its SentencePiece tokenizer makes the text
  lowercase.
- Text length: the tokenizer of the model pads to 64 tokens, but the text model has 256 positions. The
  recipe sets the tokenizer length to 256. SigLIP takes the feature from the last position, so the padding
  length changes the feature. Pad each prompt with `padding="max_length"`. `ZeroShotEvaluator` does this
  for a SigLIP processor.
- Prompt form: the model card gives no template. Its example passes plain labels to the Transformers
  zero-shot pipeline, and this pipeline uses the template `"This is a photo of {}."`.

Give both to the evaluator: `ZeroShotEvaluator(encoder, text_encoder, tokenizer=processor, ...)`. See
[Zero-shot](../evaluators/zero_shot.md).

## Arguments

<!-- args: radharmony.evaluator.backbones.make_chexagent -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `device` | `str` | `"cuda"` | The device of the encoder and the text encoder, for example `"cuda:1"` or `"cpu"` |
| `hub` | `str` | `"StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"` | The Hugging Face name of the model |
| `output_keys` | `set[str]` | `None` | The keys of each sample. `None` gives `{"img", "cls"}`. With `"mask"`, the encoder gives patch features ([Segmentation mode](index.md#segmentation-mode)) |
| `dtype` | `torch.dtype` | `torch.bfloat16` | The dtype of the image tensor ([Output dtype](index.md#output-dtype)) |

## Notes

- The first call downloads the weights from Hugging Face.
- A note in the recipe code gives one VinDr-CXR zero-shot run: the macro AUROC was 0.488 with 64 tokens and
  0.772 with 256 tokens.
- The model card gives no facts about the training data.
- For report generation, see [CheXagent-2 (generator)](chexagent_gen.md).
- One model gives the image encoder and the text encoder. The recipe does not load the weights 2 times.
- The processor does not cut a long prompt. A prompt of more than 256 tokens raises `ValueError` in the
  text model. `ZeroShotEvaluator` cuts it to 256 tokens and warns (D-52).
- Vision-only evaluators use only the transform and the encoder.
