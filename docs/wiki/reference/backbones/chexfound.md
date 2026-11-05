# CheXFound

| | |
|---|---|
| Recipe | `make_chexfound()`, returns `(transform, encoder)` |
| Model card | [RPIDIAL/CheXFound](https://github.com/RPIDIAL/CheXFound) |
| Input size | 512 × 512 (bicubic resize of the short side to 512, then center crop) |
| Normalization | Each channel goes to [0, 1] with its minimum and maximum. Then the ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225), from the authors' eval transform |
| Feature | CLS tokens of the last 4 blocks, joined, 4096 values |
| Install extra | `chexfound`, a clone of the authors' repo and a checkpoint |
| Segmentation features | `Tensor[B, 1024, 32, 32]` |

<!-- claim: id=chexfound.paper.n_images source="Yang et al., IEEE Transactions on Medical Imaging 2025" expect="1 million" -->
CheXFound is a ViT-L/16 with 4 register tokens. It comes from RPIDIAL (Yang et al., IEEE Transactions on
Medical Imaging 2025). It was trained with iBOT-style self-supervised learning (as in DINOv2) on up to 1
million public chest X-rays, at 512 × 512. Use it as a large chest X-ray ViT that was trained with images
only.

## Quick use

This block encodes one [Montgomery](../datasets/cxr-boxes-masks/montgomery_cxr.md) image.

<!-- block: run=gpu timeout=600 -->
```python
import torch
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator.backbones import make_chexfound

transform, encoder = make_chexfound(device="cuda")
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_cls=True,
    cache_dir=None,
)
img = ds.get_datasets()[0]["img"]  # Tensor[3, 512, 512], torch.bfloat16
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    feature = encoder(img[None].to("cuda"))
print(feature.shape)  # torch.Size([1, 4096])
```

## How the input is prepared

1. Load: the shared loader gives a 3-channel `uint8` image in 0 to 255
   ([How a recipe loads an image](index.md#how-a-recipe-loads-an-image)).
2. Scale: the values are divided by 255.
3. Resize: the short side goes to 512 with bicubic interpolation.
4. Crop: a center crop of 512 × 512.
5. Min-max: the image is clipped to [0, 1]. Then each channel is scaled to [0, 1] with its minimum and
   maximum.
6. Mean and std: the ImageNet values.
7. Dtype: the transform casts the image to `dtype`.

Steps 3 to 6 are `make_classification_eval_transform()` in `chexfound/data/transforms.py`.
`chexfound/eval/classification/linear_glori.py` calls it with `resize_size=crop_size=image_size`, and the
authors' eval commands set `--image-size 512` (README and `scripts/eval/run_eval_shenzhen.sh`). A bicubic
resize can give values below 0 or above 255. The authors resize a `uint8` PIL image, so PIL clips them to 0 to
255. Step 5 does the same clip, then runs the authors' `RescaleImage`.

## Segmentation mode

<!-- block: run=gpu timeout=600 -->
```python
transform, encoder = make_chexfound(device="cuda", output_keys={"img", "mask"})
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    patches = encoder(img[None].to("cuda"))  # img from Quick use
print(patches.shape)  # torch.Size([1, 1024, 32, 32])
```

The patch features are the patch tokens of the last block, after the final norm. They come from
`get_intermediate_layers(x, n=1, return_class_token=True)`, which drops the CLS and register tokens. The patch
size is 16, so the grid is 512 / 16 = 32 on each side. See [Segmentation mode](index.md#segmentation-mode) for
the shared rules.

## Feature

The classification feature joins the normed CLS tokens of the last 4 blocks, in block order (4 × 1024 = 4096
values). The authors' linear probe uses this feature: `n_last_blocks=[4]` and `avgpools=[False]` in
`chexfound/eval/classification/linear_glori.py`, with `create_linear_input()` (D-16).

## Weights and code

Do these 2 steps by hand before the first call:

1. Clone the authors' repo. The recipe imports the model code from it.

   <!-- block: run=skip reason="clones a repo" -->
   ```bash
   git clone https://github.com/RPIDIAL/CheXFound third_party_models/CheXFound
   ```

2. Download `teacher_checkpoint.pth` from the
   [Google Drive folder](https://drive.google.com/drive/folders/1GX2BWbujuVABtVpSZ4PTBykGULzrw806) of the
   authors. Put it at `third_party_models/CheXFound/weights/teacher_checkpoint.pth`, or pass
   `checkpoint_path`.

The checkpoint comes from FSDP training, with the block keys in 4 groups (`backbone.blocks.<group>.<index>`).
The recipe renames them to `blocks.<index>`, as `notebooks/chexfound_example.ipynb` of the authors does. It
builds the model with LayerScale (`init_values=1e-5`), so the trained LayerScale weights load.

## Arguments

<!-- args: radharmony.evaluator.backbones.make_chexfound -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `chexfound_dir` | `str` | `None` | The folder of the clone. `None` gives `third_party_models/CheXFound/` at the repo root. A missing folder raises `FileNotFoundError` |
| `checkpoint_path` | `str` | `None` | The checkpoint file. `None` gives `<chexfound_dir>/weights/teacher_checkpoint.pth`. A missing file raises `FileNotFoundError` |
| `device` | `str` | `"cuda"` | The device of the encoder, for example `"cuda:1"` or `"cpu"` |
| `output_keys` | `set[str]` | `None` | The keys of each sample. `None` gives `{"img", "cls"}`. With `"mask"`, the encoder gives patch features ([Segmentation mode](index.md#segmentation-mode)) |
| `dtype` | `torch.dtype` | `torch.bfloat16` | The dtype of the image tensor ([Output dtype](index.md#output-dtype)) |

## Notes

- Before 2026-11-01, the feature was the last CLS token (1024 values). Before 2026-11-03, the input had no
  step 5. An `embedding_cache` file from before 2026-11-03 raises `ValueError` (D-41). Delete the file. The
  next run makes a new one.
- The plain forward of the model gives only the last CLS token (1024 values). The recipe calls
  `get_intermediate_layers()` instead, to get the 4 CLS tokens.
