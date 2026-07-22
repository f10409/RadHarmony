# CheXFound

ViT-Large/16 backbone pretrained with iBOT-style self-supervised learning by
[RPIDIAL/CheXFound](https://github.com/RPIDIAL/CheXFound). Implementation
lives in the cloned repo; the recipe loads the teacher checkpoint and
re-routes the FSDP-shard key layout to flat `blocks.IDX.*` keys.

| Embed dim | Input size | Returns | Extra |
|-----------|------------|---------|-------|
| 1024 | 512×512 | `(transform, encoder)` | `chexfound` + cloned repo + manual checkpoint |

Preprocessing: MONAI `LoadImage` (supports PNG, JPEG, DICOM, NIfTI, …),
min-max normalize to uint8, then `Resize(512, BICUBIC) → CenterCrop(512) →
Normalize(ImageNet)`.

## Install

```bash
uv pip install -e ".[chexfound]"
```

Two manual steps:

1. Clone the upstream repository:

   ```bash
   git clone https://github.com/RPIDIAL/CheXFound third_party_models/CheXFound
   ```

2. Download `teacher_checkpoint.pth` from
   [Google Drive](https://drive.google.com/drive/folders/1GX2BWbujuVABtVpSZ4PTBykGULzrw806).
   Place at `third_party_models/CheXFound/weights/teacher_checkpoint.pth`.

## Override default paths

| Kwarg | Default | Purpose |
|-------|---------|---------|
| `chexfound_dir=` | `third_party_models/CheXFound/` | Root of the cloned repo. The recipe adds this to `sys.path` to import the model code. |
| `checkpoint_path=` | `<chexfound_dir>/weights/teacher_checkpoint.pth` | Explicit path to the teacher `.pth` weight file. |

```python
from radharmony.evaluator.backbones import make_chexfound

transform, encoder = make_chexfound(
    chexfound_dir="/shared/repos/CheXFound",
    checkpoint_path="/path/to/models/CheXFound/teacher_checkpoint.pth",
    device="cuda:0",
)
```

## Usage

```python
from radharmony.evaluator.backbones import make_chexfound
from radharmony.dataset import VinDrCXRTrainDataset

transform, encoder = make_chexfound(device="cuda:0", output_keys={"img", "cls"})

ds = VinDrCXRTrainDataset(
    base_image_dir="/data/vindr/train/",
    transform=transform,
    cache_dir="/tmp/cache/chexfound/",
    output_cls=True,
)
```

## Segmentation mode

```python
from radharmony.dataset import SIIMACRPTXTrainDataset

transform, encoder = make_chexfound(device="cuda:0", output_keys={"img", "mask"})
# forward(x) -> Tensor[B, 1024, 32, 32]   (512 / 16 = 32)

ds = SIIMACRPTXTrainDataset(
    base_image_dir="/data/siim-acr-ptx/dicom-images-train/",
    csv_path="/data/siim-acr-ptx/train-rle.csv",
    transform=transform,
    mask_output_dir="/tmp/cache/siim_ptx_masks/",
    output_mask=True,
    cache_dir="/tmp/cache/chexfound_seg/",
)
```

`get_intermediate_layers` strips CLS + register tokens; the recipe uses
`return_class_token=True` to retrieve patch tokens for the segmentation path.
