# MedSigLIP

Google's medical SigLIP — `google/medsiglip-448`. SigLIP-So-400m image tower
+ multilingual text tower, fine-tuned on a medical-image corpus.

| Embed dim | Input size | Returns | Extra |
|-----------|------------|---------|-------|
| 1152 | 448×448 | `(transform, image_encoder, text_encoder, processor)` | `medsiglip` |

SigLIP-So-400m/14 (14-px patches). Preprocessed via the HuggingFace
`AutoProcessor` (SigLIP normalization, BICUBIC resize to 448×448).

## Install

```bash
uv pip install -e ".[medsiglip]"
```

Weights are auto-downloaded from HuggingFace on first call.

## Usage

```python
from radharmony.evaluator.backbones import make_medsiglip
from radharmony.dataset import VinDrCXRTrainDataset

transform, image_encoder, text_encoder, processor = make_medsiglip(device="cuda:0", output_keys={"img", "cls"})

ds = VinDrCXRTrainDataset(
    base_image_dir="/data/vindr/train/",
    transform=transform,
    cache_dir="/tmp/cache/medsiglip/",
    output_cls=True,
)
```

Vision-only evaluators ignore `text_encoder` / `processor`; zero-shot needs
all four.

## Segmentation mode

```python
from radharmony.dataset import SIIMACRPTXTrainDataset

transform, image_encoder, *_ = make_medsiglip(device="cuda:0", output_keys={"img", "mask"})
# image_encoder(x) -> Tensor[B, 1152, 32, 32]   (448 / 14 = 32)

ds = SIIMACRPTXTrainDataset(
    base_image_dir="/data/siim-acr-ptx/dicom-images-train/",
    csv_path="/data/siim-acr-ptx/train-rle.csv",
    transform=transform,
    mask_output_dir="/tmp/cache/siim_ptx_masks/",
    output_mask=True,
    cache_dir="/tmp/cache/medsiglip_seg/",
)
```
