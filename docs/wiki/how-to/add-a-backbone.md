# Add a backbone

An evaluator needs 2 pieces from a backbone: a dataset transform and an encoder. This page wraps a Hugging
Face model in these 2 pieces and runs a probe with them. Then it shows how to make a reusable backbone
recipe, a `make_<name>()` function. To add a recipe to the repo with all its files, use the
[add-backbone skill](https://github.com/f10409/RadHarmony/blob/main/.claude/skills/add-backbone/SKILL.md).

Before you start, install the `raddino` extra, and
[get the Montgomery County CXR data](get-the-data.md#montgomery-county-cxr) for step 3.

## 1. Wrap the model

The 2 pieces have separate jobs:

| Piece | Where it runs | What it does |
|---|---|---|
| Transform (`RadiologyEncoderTransform`) | CPU, in the DataLoader workers | Reads the image file, resizes and normalizes it, and gives `Tensor[C, H, W]` |
| Encoder (`ImageEncoderWrapper`) | `device` (GPU) | Runs the model and gives the feature, `Tensor[B, D]` ([Encoder contract](../reference/evaluators/index.md#encoder-contract)) |

So each model reads and normalizes its images with its own code. No other step normalizes them again
(D-16). This step makes the encoder, and step 2 makes the transform.

`ImageEncoderWrapper` is an `nn.Module` that holds the model. So `.to("cuda")`, `state_dict()`,
`parameters()` and `copy.deepcopy()` see the model weights. A plain function around a model does not.
`FinetuneEvaluator` raises `TypeError` for an encoder that is not an `nn.Module`.

For a Hugging Face model, `from_huggingface` calls `model(pixel_values=x)`. `pool` picks the feature:
`"cls"` (the default, the first token), `"mean"` (the mean of the tokens), `"pooler"` (`pooler_output`) or
a function `(output) -> Tensor[B, D]`.

<!-- block: run=gpu timeout=600 -->
```python
from transformers import AutoModel

from radharmony.evaluator import ImageEncoderWrapper

model = AutoModel.from_pretrained("microsoft/rad-dino")
encoder = ImageEncoderWrapper.from_huggingface(model, pool="cls").to("cuda")
```

For other models, `from_custom` takes `model_call(model, x)` and `pool(output)`. Both are optional.

<!-- block: run=skip reason="clip_model and my_model are your own models" -->
```python
import timm

# timm with num_classes=0 gives (B, D), so it needs no function.
encoder = ImageEncoderWrapper.from_custom(timm.create_model("resnet50", pretrained=True, num_classes=0))

# OpenCLIP or BiomedCLIP
encoder = ImageEncoderWrapper.from_custom(clip_model, model_call=lambda m, x: m.encode_image(x))

# Any model: model_call runs it, and pool takes the feature from its output.
encoder = ImageEncoderWrapper.from_custom(
    my_model,
    model_call=lambda m, x: m(x, return_dict=True),
    pool=lambda out: out["features"][:, 0, :],
)
```

Common `model_call` functions:

| Model family | `model_call` |
|---|---|
| Hugging Face ViT (CLS token) | `lambda m, x: m(pixel_values=x).last_hidden_state[:, 0]` |
| Hugging Face SigLIP or CLIP | `lambda m, x: m(pixel_values=x).pooler_output` |
| OpenCLIP or BiomedCLIP | `lambda m, x: m.encode_image(x)` |
| timm (`num_classes=0`) | `None`, the model gives `(B, D)` |
| RAD-DINO (`rad-dino` package) | `lambda m, x: m.encode(BatchFeature({"pixel_values": x}))[0]` |

## 2. Build the transform

`EncoderPreprocessTransform.from_huggingface` runs the Hugging Face image processor of the model on each
image. `reader` sets how it opens the image file:

| `reader` | What it does |
|---|---|
| `"pil"` (default) | `PIL.Image.open(path)` as RGB, as in most Hugging Face pretraining. PIL cannot open DICOM |
| `"monai"` | MONAI `LoadImage` and a transpose to `(C, H, W)`. It reads DICOM and NIfTI. Only for 2-D images |
| a function | Your own `(path) -> image` |
| `None` | No read. A step before it has loaded the image |

Both readers give values from 0 to 255, because the processor divides by 255. An 8-bit image keeps its
values. A reader scales a 12-bit or 16-bit image with its minimum and maximum, for example a 16-bit PadChest
PNG.
A reader never clips (D-27). `"monai"` gives a DICOM file the HighBit fix and the window (VOI LUT) of a
MONOCHROME image (D-13). It reads a palette or RGBA PNG as the RGB image it shows, and an LA PNG as gray.

`RadiologyEncoderTransform` puts this step in a MONAI transform for the dataset. `output_keys` sets the
keys of each sample. It has the same augmentation steps as `RadiologyTransform2D`, for example
`.with_flip(spatial_axis=1)`.

<!-- block: run=gpu timeout=600 -->
```python
from radharmony.evaluator import EncoderPreprocessTransform, RadiologyEncoderTransform

pp = EncoderPreprocessTransform.from_huggingface(keys=["img"], processor="microsoft/rad-dino", reader="pil")
transform = RadiologyEncoderTransform(preprocess=pp, output_keys={"img", "cls"}).get_transform()
```

For a model with no Hugging Face processor, write a function that does all steps, from the path to
`Tensor[C, H, W]`:

<!-- block: run=skip reason="my_preprocess is your own function" -->
```python
pp = EncoderPreprocessTransform.from_custom(keys=["img"], preprocess=my_preprocess)
```

Notes on the transform:

- The image tensor is `torch.bfloat16` by default. The evaluators run the encoder under autocast, so
  an encoder with float32 weights takes the bfloat16 image. For an encoder that cannot autocast, pass
  `RadiologyEncoderTransform(..., dtype=torch.float32)`. Each built-in encoder recipe also takes
  `dtype=`. The labels (`cls`, `reg`, `mask`) are always float32.
- A box (`"bbox"` in `output_keys`) does not follow the crop or the pad of the processor (D-28). So the
  transform warns when the processor does more than resize. A custom `preprocess` always warns, unless
  you pass the Hugging Face processor that it calls: `from_custom(..., processor=processor)`.

## 3. Run a probe with the 2 pieces

Give the transform to the dataset and the encoder to the evaluator. This block uses the `encoder` and the
`transform` of steps 1 and 2, on [Montgomery](../reference/datasets/cxr-boxes-masks/montgomery_cxr.md).

<!-- block: run=gpu timeout=600 -->
```python
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator import LinearProbeEvaluator

ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_cls=True,
    cache_dir=None,
)
df = LinearProbeEvaluator(encoder, dataset=ds, n_folds=5).evaluate()
print(round(df.loc[df["label"] == "macro_average", "auroc"].mean(), 3))  # 0.972
```

The `make_raddino` recipe gives 0.971 on the same folds ([Run a probe](run-a-probe.md)). It loads the same
Hugging Face weights with the `rad-dino` package.

## 4. Make a reusable recipe

A recipe is one file in `radharmony/evaluator/backbones/`. Start from
[`_template.py`](https://github.com/f10409/RadHarmony/blob/main/radharmony/evaluator/backbones/_template.py).
Follow 3 rules:

1. Import the model packages inside the function. Then `import radharmony` works without the extra.
2. Load the model one time. The transform and the encoder share it.
3. Do all image steps in the transform. The encoder only runs the model.

<!-- block: run=skip reason="a template for a new file in radharmony/evaluator/backbones/" -->
```python
# radharmony/evaluator/backbones/my_model.py
import monai.transforms as mt
import torch

from radharmony.evaluator.backbones._utils import (
    display_255, load_image, squeeze_frame_dim, to_3channel, to_uint8,
)
from radharmony.evaluator.transforms import EncoderPreprocessTransform, RadiologyEncoderTransform
from radharmony.evaluator.wrappers import ImageEncoderWrapper

_MODEL_HUB = "org/my-model"


def make_my_model(device: str = "cuda"):
    """Return (transform, encoder) for MyModel."""
    from transformers import AutoImageProcessor, AutoModel  # inside the function (rule 1)

    _model = AutoModel.from_pretrained(_MODEL_HUB)
    _processor = AutoImageProcessor.from_pretrained(_MODEL_HUB)

    _loader = mt.Compose([
        load_image(),  # a palette or alpha PNG reads as shown (D-13)
        mt.Lambda(squeeze_frame_dim),
        mt.Transpose(indices=[0, 2, 1]),
        mt.Lambda(display_255),  # DICOM window, 8-bit kept, others scaled to 0..255 (D-13, D-27)
        mt.Lambda(to_3channel),
        mt.Lambda(to_uint8),
    ])

    def _preprocess(path) -> torch.Tensor:
        return _processor(images=_loader(str(path)), return_tensors="pt")["pixel_values"][0]

    def _model_call(m, x):
        return m(pixel_values=x).last_hidden_state[:, 0]  # CLS token

    pp = EncoderPreprocessTransform.from_custom(keys=["img"], preprocess=_preprocess, processor=_processor)
    transform = RadiologyEncoderTransform(preprocess=pp, output_keys={"img", "cls"}).get_transform()
    encoder = ImageEncoderWrapper.from_custom(_model, model_call=_model_call).to(device)
    return transform, encoder
```

The loader functions of `backbones/_utils.py`, which all built-in recipes use:

| Function | What it does |
|---|---|
| `load_image()` | The MONAI image loader. A palette or alpha PNG reads as the image it shows |
| `squeeze_frame_dim` | `(C, W, H, 1)` to `(C, W, H)`: removes the frame dimension of MONAI |
| `display_255` | The image as the dataset load shows it, float in 0 to 255 (D-13, D-27) |
| `to_3channel` | `(1, H, W)` to `(3, H, W)` |
| `to_uint8` | 0 to 255 to `uint8`, rounded, with no stretch |
| `patches_to_spatial` | `[B, N, D]` to `[B, D, H, W]`: ViT patch tokens to a feature map |
| `to_pil_rgb` | `(C, H, W)` float to a PIL RGB image, for a PIL processor |

Two more cases:

- Segmentation: when `output_keys` has `"mask"`, the encoder gives patch features (use
  `patches_to_spatial`), and the transform needs `mask_preprocess`. It gives the mask the same resize and
  crop as the image, with nearest-neighbor interpolation and no normalization. `_template.py` shows both.
- Text and image: return `(transform, image_encoder, text_encoder, tokenizer)`. Then
  [`ZeroShotEvaluator`](../reference/evaluators/zero_shot.md) can use the recipe. See
  [`chexagent.py`](https://github.com/f10409/RadHarmony/blob/main/radharmony/evaluator/backbones/chexagent.py).

## 5. Register the recipe

Add the recipe to `radharmony/evaluator/backbones/__init__.py`:

<!-- block: run=skip reason="lines to add to radharmony/evaluator/backbones/__init__.py" -->
```python
from .my_model import make_my_model
__all__ = [..., "make_my_model"]
```

A recipe in the repo also needs these changes. The add-backbone skill lists each step.

- An install extra in `pyproject.toml`.
- A page in `docs/wiki/reference/backbones/`, and a line in the `mkdocs.yml` nav.
- A row in the [Backbones](../reference/backbones/index.md) table, and in the [Extras](../get-started/install.md#extras) table.

## Check

- The probe of step 3 gives a result table with an AUROC above 0.5.
- The feature has the size that the model card gives. A shape check is not enough: a wrong crop or mean
  still gives the right shape. Run the recipe and the authors' own code on a few real images. The cosine
  similarity between the 2 features must be near 1 (D-16).

## See also

- [Backbones](../reference/backbones/index.md): the built-in recipes, and how a recipe loads an image.
- [Evaluators](../reference/evaluators/index.md#encoder-contract): what each evaluator gives to the
  encoder, and what it expects back.
- [Compare backbones](compare-backbones.md): run your recipe next to the built-in ones.
