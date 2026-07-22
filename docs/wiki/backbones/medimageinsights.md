# MedImageInsights

`lion-ai/MedImageInsights` — multi-modal medical-image foundation model. The
image tower is a DaViT (Dual Attention Vision Transformer); the projected
image-embed dim is 1024, but the dense patch features (used for segmentation)
are 2048-dim.

| Embed dim (cls) | Input size | Returns | Extra |
|-----------------|------------|---------|-------|
| 1024 | 480×480 | `(transform, image_encoder, text_encoder, None)` | `medimageinsights` |

The text-side tuple slot is `None` because MedImageInsights bundles the
tokenizer inside `text_encoder`; passing `None` for the fourth slot keeps the
4-tuple shape uniform with other vision-language recipes.

## Install

```bash
uv pip install -e ".[medimageinsights]"
```

Weights are auto-downloaded from HuggingFace on first call.

## Usage

```python
from radharmony.evaluator.backbones import make_medimageinsights
from radharmony.dataset import VinDrCXRTrainDataset

transform, image_encoder, text_encoder, _ = make_medimageinsights(device="cuda:0", output_keys={"img", "cls"})

ds = VinDrCXRTrainDataset(
    base_image_dir="/data/vindr/train/",
    transform=transform,
    cache_dir="/tmp/cache/medimageinsights/",
    output_cls=True,
)
```

Vision-only evaluators ignore `text_encoder`; zero-shot needs all four.

## Segmentation mode

```python
from radharmony.dataset import SIIMACRPTXTrainDataset

transform, image_encoder, *_ = make_medimageinsights(device="cuda:0", output_keys={"img", "mask"})
# image_encoder(x) -> Tensor[B, 2048, 15, 15]   (480-px input -> 15x15 grid at the final stage)

ds = SIIMACRPTXTrainDataset(
    base_image_dir="/data/siim-acr-ptx/dicom-images-train/",
    csv_path="/data/siim-acr-ptx/train-rle.csv",
    transform=transform,
    mask_output_dir="/tmp/cache/siim_ptx_masks/",
    output_mask=True,
    cache_dir="/tmp/cache/medimageinsights_seg/",
)
```

Note the **dense-feature dim is 2048**, not 1024 — 1024 is the projected
image-embed dim used for contrastive matching with text. The 2048-dim dense
features come from the DaViT trunk's final stage and are captured via a
per-call forward hook.
