# Backbone Recipes

Drop-in factory functions in `radharmony.evaluator.backbones` that return a
ready-to-use `(transform, encoder, …)` tuple for every supported foundation
model. Each recipe handles its own input preprocessing, channel expansion,
normalisation, and `forward(x) -> Tensor[B, D]` routing — so you can pair any
backbone with any evaluator in three lines.

```python
from radharmony.evaluator.backbones import make_raddino, make_biomed_clip

transform, encoder = make_raddino(device="cuda")
transform, encoder, text_enc, tokenizer = make_biomed_clip(device="cuda")
```

Install the corresponding extra first: `uv pip install -e ".[<extra>]"`.

## Recipes

| Factory | Embed dim | Input size | Returns | Extra |
|---------|-----------|------------|---------|-------|
| [`make_raddino`](raddino.md) | 768 | 518×518 | `(transform, encoder)` | `raddino` |
| [`make_biomed_clip`](biomed_clip.md) | 512 | 448×448 | `(transform, encoder, text_encoder, tokenizer)` | `biomed` |
| [`make_chexagent`](chexagent.md) | 1024 | 512×512 | `(transform, encoder, text_encoder, processor)` | `chexagent` |
| [`make_medsiglip`](medsiglip.md) | 1152 | 448×448 | `(transform, encoder, text_encoder, processor)` | `medsiglip` |
| [`make_medimageinsights`](medimageinsights.md) | 1024 | 480×480 | `(transform, encoder, text_encoder, None)` | `medimageinsights` |
| [`make_chexfound`](chexfound.md) | 1024 | 512×512 | `(transform, encoder)` | `chexfound` + cloned repo + manual ckpt |
| [`make_dinov3`](dinov3.md) | 768 | 224×224 | `(transform, encoder)` | `dinov3` |
| [`make_eva_x`](eva_x.md) | 192 / 384 / 768 | 224×224 | `(transform, encoder)` | `eva_x` + submodule |
| [`make_ark_plus`](ark_plus.md) | 1536 | 768×768 | `(transform, encoder)` | `ark_plus` + manual ckpt + side-loaded `timm==0.5.4` |
| [`make_medical_mae`](medical_mae.md) | 768 / 384 | 224×224 | `(transform, encoder)` | `medical_mae` + cloned repo + manual ckpt + side-loaded `timm==0.4.12` |
| [`make_siglip2`](siglip2.md) | 1152 | 384×384 | `(transform, image_encoder, text_encoder, processor)` | `siglip2` |

## Segmentation mode

Every recipe accepts `output_keys={"img", "mask"}` to switch the encoder into
dense-feature mode — `forward(x)` then returns `Tensor[B, D, H, W]` patch
features, ready for `LinearProbeSegEvaluator` /
`ConvProbeSegEvaluator` / `UPerNetSegEvaluator`.

```python
transform, encoder = make_raddino(device="cuda", output_keys={"img", "mask"})
```

## Overriding the output dtype

Every recipe also accepts `dtype=` (forwarded to `RadiologyEncoderTransform`)
to set the final tensor dtype. Defaults to `torch.bfloat16`, matching the
autocast bridge in `BaseClsEvaluator` / `BaseSegEvaluator`. Pass
`torch.float32` when pairing with an encoder that cannot autocast.

```python
import torch
transform, encoder = make_raddino(device="cuda", dtype=torch.float32)
```

## Pointing at custom checkpoint / repo locations

Four recipes need files that can't be fetched from a public hub — manual
checkpoints and/or a cloned upstream repo. They default to
`<repo_root>/third_party_models/<name>/`, but each factory accepts kwargs to
point at a different location (useful when weights live on shared storage,
or when you want to keep `third_party_models/` out of the repo).

| Factory | Default location | Override kwargs |
|---------|------------------|-----------------|
| [`make_ark_plus`](ark_plus.md) | `third_party_models/Ark/Ark_Plus/Ark6_swinLarge768_ep50.pth.tar` | `checkpoint_path=`, `legacy_timm_dir=` |
| [`make_medical_mae`](medical_mae.md) | `third_party_models/medical_mae/<variant>.pth` | `checkpoint_path=`, `mae_dir=`, `legacy_timm_dir=` |
| [`make_eva_x`](eva_x.md) | `third_party_models/EVA-X/` | `repo_path=` |
| [`make_chexfound`](chexfound.md) | `third_party_models/CheXFound/weights/teacher_checkpoint.pth` | `chexfound_dir=`, `checkpoint_path=` |

```python
transform, encoder = make_ark_plus(
    checkpoint_path="/path/to/models/Ark_Plus/Ark6_swinLarge768_ep50.pth.tar",
    legacy_timm_dir="/path/to/cache/ark-timm-054",
    device="cuda:0",
)
```

For `ark_plus` and `medical_mae`, `legacy_timm_dir` is where the recipe
side-installs the legacy `timm` version via
`uv pip install --target --no-deps` on first call. Point it at a fast local
cache to avoid re-installing across users and to survive a
`third_party_models/` cleanup.

## Authoring a new recipe

See [Custom Backbones](custom_backbones.md) for the recipe template,
`ImageEncoderWrapper` patterns (HuggingFace / timm / OpenCLIP / bespoke),
preprocessing transforms, and the loader-helper utilities.
