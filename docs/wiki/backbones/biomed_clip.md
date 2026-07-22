# BiomedCLIP

Microsoft's BiomedCLIP — `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
via OpenCLIP. Vision-language model with PubMedBERT text tower; the image
tower is a ViT-B/16 trained on biomedical image-caption pairs.

| Embed dim | Input size | Returns | Extra |
|-----------|------------|---------|-------|
| 512 | 224×224 | `(transform, image_encoder, text_encoder, tokenizer)` | `biomed` |

ViT-B/16 (16-px patches). Resize to 224×224 with the OpenCLIP preprocess
pipeline (BiomedCLIP's own image mean/std).

## Install

```bash
uv pip install -e ".[biomed]"
```

Weights are auto-downloaded from HuggingFace on first call.

## Usage

```python
from radharmony.evaluator.backbones import make_biomed_clip
from radharmony.dataset import VinDrCXRTrainDataset

transform, image_encoder, text_encoder, tokenizer = make_biomed_clip(device="cuda:0", output_keys={"img", "cls"})

ds = VinDrCXRTrainDataset(
    base_image_dir="/data/vindr/train/",
    transform=transform,
    cache_dir="/tmp/cache/biomed_clip/",
    output_cls=True,
)
```

Vision-only evaluators ignore `text_encoder` / `tokenizer`; zero-shot needs
all four.

## Segmentation mode

```python
from radharmony.dataset import SIIMACRPTXTrainDataset

transform, image_encoder, *_ = make_biomed_clip(device="cuda:0", output_keys={"img", "mask"})
# image_encoder(x) -> Tensor[B, 768, 14, 14]   (224 / 16 = 14)

ds = SIIMACRPTXTrainDataset(
    base_image_dir="/data/siim-acr-ptx/dicom-images-train/",
    csv_path="/data/siim-acr-ptx/train-rle.csv",
    transform=transform,
    mask_output_dir="/tmp/cache/siim_ptx_masks/",
    output_mask=True,
    cache_dir="/tmp/cache/biomed_clip_seg/",
)
```

Segmentation mode strips the prefix tokens from the OpenCLIP trunk's
`forward_features` output. The patch-feature dim is **768** (the ViT's
hidden size), not 512 — 512 is the projected image-embed dim used for
contrastive matching with text.
