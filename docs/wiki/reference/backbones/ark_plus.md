# Ark+

| | |
|---|---|
| Recipe | `make_ark_plus()`, returns `(transform, encoder)` |
| Model card | [jlianglab/Ark](https://github.com/jlianglab/Ark) (folder `Ark_Plus`) |
| Input size | 768 × 768 (bilinear resize to 896 × 896, then center crop) |
| Normalization | ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225), from the authors' fine-tune code |
| Feature | Projector output of the pooled Swin-L feature, 1376 values |
| Install extra | `ark_plus` and a checkpoint that you request from the authors |
| Segmentation features | `Tensor[B, 1536, 24, 24]` |

Ark+ is a Swin-L chest X-ray model from the jlianglab group (Ma et al., Nature 2025). It learned from the
labels of 6 public chest X-ray datasets: MIMIC-CXR, CheXpert, ChestX-ray14, RSNA Pneumonia, VinDr-CXR and
Shenzhen. The training is cyclic multi-task learning with a student and a teacher. Use it as a chest X-ray
encoder trained with expert labels.

## Quick use

This block encodes one [Montgomery](../datasets/cxr-boxes-masks/montgomery_cxr.md) image.

<!-- block: run=gpu timeout=600 -->
```python
import torch
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator.backbones import make_ark_plus

transform, encoder = make_ark_plus(device="cuda")
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_cls=True,
    cache_dir=None,
)
img = ds.get_datasets()[0]["img"]  # Tensor[3, 768, 768], torch.bfloat16
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    feature = encoder(img[None].to("cuda"))
print(feature.shape)  # torch.Size([1, 1376])
```

## How the input is prepared

1. Load: the shared loader gives a 3-channel `uint8` image in 0 to 255
   ([How a recipe loads an image](index.md#how-a-recipe-loads-an-image)).
2. Scale: the values are divided by 255.
3. Resize: the whole image goes to 896 × 896 with bilinear interpolation (the torchvision default). The resize
   does not keep the aspect ratio.
4. Crop: a center crop of 768 × 768.
5. Mean and std: the ImageNet values.
6. Dtype: the transform casts the image to `dtype`.

Steps 3 to 5 are the `"valid"` mode of `build_transform_classification()` in `Finetuning/dataloader.py`, with
`--img_size 896 --input_size 768` from the Ark+ README. The authors' pretraining code resizes with PIL
(bicubic) instead. The recipe uses the fine-tune steps (D-16). The `"test"` mode of the authors averages 10
crops. The recipe uses the one center crop.

## Segmentation mode

<!-- block: run=gpu timeout=600 -->
```python
transform, encoder = make_ark_plus(device="cuda", output_keys={"img", "mask"})
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    patches = encoder(img[None].to("cuda"))  # img from Quick use
print(patches.shape)  # torch.Size([1, 1536, 24, 24])
```

Swin is a hierarchical model. A forward hook on the final `model.norm` takes the feature map before the pool.
The last stage has a stride of 32, so the grid is 768 / 32 = 24 on each side. The patch features do not use
the projector. See [Segmentation mode](index.md#segmentation-mode) for the shared rules.

## Feature

The classification feature is the output of the projector (one Linear layer, 1536 to 1376) on the pooled
`forward_features()` output. The authors' linear probe uses this feature:
`generate_embeddings(x, after_proj=True)` in `Pretraining/models.py`, and `emb_dim=1376` in
`Linearprobing/main_linearprobing.py` (D-16). The recipe loads the projector from the checkpoint. A checkpoint
with no projector raises `KeyError`.

## Weights

The weights do not download by themselves:

1. Request them with the [Google Form](https://forms.gle/qkoDGXNiKRPTDdCe8) or the
   [WeChat form](https://www.wjx.cn/vm/OvwfYFx.aspx).
2. Put `Ark6_swinLarge768_ep50.pth.tar` at `third_party_models/Ark/Ark_Plus/Ark6_swinLarge768_ep50.pth.tar`,
   or pass `checkpoint_path`.

The recipe does not import code from the Ark repo, so a clone is optional. The checkpoint was saved with
`timm==0.5.4`, and the Swin layout of a newer `timm` is different (`downsample` is one layer index off). So on
the first call, the recipe installs `timm==0.5.4` into `legacy_timm_dir`
([Weights and code locations](index.md#weights-and-code-locations)).

## Arguments

<!-- args: radharmony.evaluator.backbones.make_ark_plus -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `checkpoint_path` | `str` | `None` | The checkpoint file. `None` gives `third_party_models/Ark/Ark_Plus/Ark6_swinLarge768_ep50.pth.tar` at the repo root. A missing file raises `FileNotFoundError` |
| `legacy_timm_dir` | `str` | `None` | The folder for `timm==0.5.4`. `None` gives `third_party_models/Ark/timm-054/` at the repo root |
| `device` | `str` | `"cuda"` | The device of the encoder, for example `"cuda:1"` or `"cpu"` |
| `output_keys` | `set[str]` | `None` | The keys of each sample. `None` gives `{"img", "cls"}`. With `"mask"`, the encoder gives patch features ([Segmentation mode](index.md#segmentation-mode)) |
| `dtype` | `torch.dtype` | `torch.bfloat16` | The dtype of the image tensor ([Output dtype](index.md#output-dtype)) |

## Notes

- Ark+ learned from the labels of its 6 training datasets. So a score on one of these datasets is not a test
  on new data. For example, Ark+ saw the Shenzhen TB labels.
- The recipe builds a plain `timm` `SwinTransformer` (patch 4, window 12, embed size 192, depths 2, 2, 18, 2).
  It loads `checkpoint["teacher"]` with `strict=False`. It drops the `attn_mask` buffers and the task heads.
- Before 2026-11-03, the feature was `forward_features()` (1536 values). An `embedding_cache` file from before
  that date raises `ValueError` (D-41). Delete the file. The next run makes a new one.
- Before 2026-10-28, the recipe resized to 768 × 768 with no crop. So results from before that date used
  another input.
- Under autocast, the feature is `torch.bfloat16`, because the last step is a Linear layer.
