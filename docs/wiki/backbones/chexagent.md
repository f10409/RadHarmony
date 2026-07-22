# CheXagent

Stanford's CheXagent / XraySigLIP image tower —
`StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli`. SigLIP-style
vision-language model trained on chest-X-ray report pairs.

| Embed dim | Input size | Returns | Extra |
|-----------|------------|---------|-------|
| 1024 | 512×512 | `(transform, image_encoder, text_encoder, processor)` | `chexagent` |

ViT-L/16 (16-px patches). Preprocessed via the HuggingFace `AutoProcessor`
(SigLIP normalization, BICUBIC resize to 512×512).

## Install

```bash
uv pip install -e ".[chexagent]"
```

Weights are auto-downloaded from HuggingFace on first call.

## Usage

```python
from radharmony.evaluator.backbones import make_chexagent
from radharmony.dataset import VinDrCXRTrainDataset

transform, image_encoder, text_encoder, processor = make_chexagent(device="cuda:0", output_keys={"img", "cls"})

ds = VinDrCXRTrainDataset(
    base_image_dir="/data/vindr/train/",
    transform=transform,
    cache_dir="/tmp/cache/chexagent/",
    output_cls=True,
)
```

Vision-only evaluators ignore `text_encoder` / `processor`; zero-shot needs
all four.

## Segmentation mode

```python
from radharmony.dataset import SIIMACRPTXTrainDataset

transform, image_encoder, *_ = make_chexagent(device="cuda:0", output_keys={"img", "mask"})
# image_encoder(x) -> Tensor[B, 1024, 32, 32]   (512 / 16 = 32)

ds = SIIMACRPTXTrainDataset(
    base_image_dir="/data/siim-acr-ptx/dicom-images-train/",
    csv_path="/data/siim-acr-ptx/train-rle.csv",
    transform=transform,
    mask_output_dir="/tmp/cache/siim_ptx_masks/",
    output_mask=True,
    cache_dir="/tmp/cache/chexagent_seg/",
)
```
