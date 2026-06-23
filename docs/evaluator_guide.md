# RadHarmony Evaluator Guide

The `radharmony.evaluator` package takes any RadHarmony dataset and any
`nn.Module` image encoder and produces standardized downstream-task
results. It ships six classification evaluators — linear probe, k-NN
probe, SVM probe, prototype probe, zero-shot, and minimal fine-tune — plus
one segmentation evaluator (`LinearProbeSegEvaluator`, a 1×1 Conv2d head
over frozen dense features). The `language/` namespace remains reserved
for phase 2+.

## Table of contents

1. [The encoder contract](#the-encoder-contract)
2. [Backbone recipes](#backbone-recipes)
3. [Wrapping common architectures](#wrapping-common-architectures)
4. [Split modes: k-fold vs fixed-split](#split-modes-k-fold-vs-fixed-split)
5. [Output schema](#output-schema)
6. [LinearProbeEvaluator](#linearprobeevaluator)
7. [KNNProbeEvaluator](#knnprobeevaluator)
8. [SVMProbeEvaluator](#svmprobeevaluator)
9. [PrototypeProbeEvaluator](#prototypeprobeevaluator)
10. [ZeroShotEvaluator](#zeroshotevaluator)
11. [FinetuneEvaluator](#finetuneevaluator)
12. [LinearProbeSegEvaluator](#linearprobesegevaluator)
13. [Registry](#registry)

## The encoder contract

Every classification evaluator accepts an `image_encoder`. That's any
`nn.Module` or callable with the signature:

```python
image_encoder(imgs: Tensor[B, ...]) -> Tensor[B, D]
```

The evaluator does **not** reach into model internals. It does not
assume a specific input shape, normalization, or output-object type.
The user is responsible for wrapping whatever model they have so it
returns a pooled feature vector per image.

For zero-shot, a second argument is required: `text_encoder(list[str]) -> Tensor[T, D]`
(and an optional `tokenizer`).

## Backbone recipes

`radharmony.evaluator.backbones` ships drop-in factory functions that return
ready-to-use `(transform, encoder, ...)` tuples — preprocessing and forward
routing wired up for each supported model.

| Factory | Embed dim | Input size | Returns | Extra |
|---------|-----------|------------|---------|-------|
| `make_raddino(device)` | 768 | 518×518 | `(transform, encoder)` | `raddino` |
| `make_biomed_clip(device)` | 512 | 448×448 | `(transform, encoder, text_encoder, tokenizer)` | `biomed` |
| `make_chexagent(device)` | 1024 | 512×512 | `(transform, encoder, text_encoder, processor)` | `chexagent` |
| `make_medsiglip(device)` | 1152 | 448×448 | `(transform, encoder, text_encoder, processor)` | `medsiglip` |
| `make_medimageinsights(device)` | 1024 | 480×480 | `(transform, encoder, text_encoder, None)` | `medimageinsights` |
| `make_chexfound(device)` | 1024 | 512×512 | `(transform, encoder)` | `chexfound` |
| `make_dinov3(device)` | 768 | 224×224 | `(transform, encoder)` | `dinov3` |
| `make_eva_x(device, variant)` | 192 / 384 / 768 | 224×224 | `(transform, encoder)` | `eva_x` + submodule |
| `make_ark_plus(device, checkpoint_path)` | 1536 | 768×768 | `(transform, encoder)` | `ark_plus` + manual ckpt + side-loaded `timm==0.5.4` |
| `make_medical_mae(variant, device)` | 768 / 384 | 224×224 | `(transform, encoder)` | `medical_mae` + cloned repo + manual ckpt + side-loaded `timm==0.4.12` |
| `make_siglip2(device)` | 1152 | 384×384 | `(transform, image_encoder, text_encoder, processor)` | `siglip2` |

```python
from radharmony.evaluator.backbones import make_raddino, make_biomed_clip

transform, encoder = make_raddino(device="cuda")
transform, encoder, text_enc, tokenizer = make_biomed_clip(device="cuda")
```

Install the corresponding extra before use, e.g. `uv pip install -e ".[raddino]"`.

Every recipe also accepts `output_keys={"img", "mask"}` to switch the encoder
into segmentation mode — `forward(x)` then returns dense patch features as
`Tensor[B, D, H, W]`, suitable for `LinearProbeSegEvaluator`.

Recipes also accept `dtype=` (forwarded to `RadiologyEncoderTransform`) to
set the final tensor dtype — defaults to `torch.bfloat16` to match the
autocast bridge in `BaseClsEvaluator` / `BaseSegEvaluator`. Pass
`torch.float32` for an encoder that cannot autocast.

To wrap your own model or author a new recipe see the
[Custom Backbones](wiki/backbones/custom_backbones.md) wiki page; per-recipe
defaults (embed dims, input sizes, override knobs for manual-setup models)
live under [Backbones](wiki/backbones/index.md).

## Wrapping common architectures

```python
# HuggingFace ViT (e.g. RAD-DINO, DINOv2) — take the CLS token
from transformers import AutoModel
model = AutoModel.from_pretrained("microsoft/rad-dino")
image_encoder = lambda x: model(pixel_values=x).last_hidden_state[:, 0]

# timm CNN (e.g. ResNet, DenseNet) — num_classes=0 returns (B, D) directly
import timm
image_encoder = timm.create_model("resnet50", pretrained=True, num_classes=0)

# OpenCLIP / BiomedCLIP — the model already exposes encode_image
import open_clip
model, _, _ = open_clip.create_model_and_transforms("ViT-B-16")
image_encoder = model.encode_image

# Self-supervised wrappers (LeJEPA, MAE, SimCLR)
image_encoder = my_ssl_model.backbone  # whichever attribute holds the feature extractor
```

**Image preprocessing lives on the transform side, not the encoder.**
Each backbone has its own pretraining recipe — ImageNet norm for most
HF ViTs, `(0.5, 0.5, 0.5)` for BiomedCLIP, raw `[-1, 1]` for a CXR-SSL
model — so channel expansion, resize, and normalization belong in a
MONAI transform that feeds the dataset. Use
[`EncoderPreprocessTransform`](#encoderpreprocesstransform--drop-a-backbones-preprocessor-into-a-monai-pipeline)
below to wrap any HuggingFace / timm / OpenCLIP preprocessor in one line.

## `EncoderPreprocessTransform` — drop a backbone's preprocessor into a MONAI pipeline

`EncoderPreprocessTransform` is a thin MONAI `MapTransform` that calls
an encoder-specific `preprocess` callable on each configured key. The
base class does **no** conversion or reading — the callable owns the
full `sample value → Tensor[C, H, W]` contract.

Why this shape: each encoder was pretrained with a specific reader
(`PIL.Image.open` for HF ViTs; a DICOM reader for some CXR models) and
a specific preprocessor. Forcing a generic tensor↔PIL round-trip or a
shared MONAI reader silently perturbs pixel statistics. Letting the
encoder's own machinery run end-to-end avoids that.

```python
from radharmony.evaluator import EncoderPreprocessTransform
from monai.transforms import Compose

# HuggingFace — defaults to reader="pil" so HF's pretraining reader runs here.
pp = EncoderPreprocessTransform.from_huggingface(
    keys=["img"], processor="microsoft/rad-dino",
)

# Pre-built processor instance + explicit DICOM/NIfTI reader.
# Override the resize/crop size by passing it to from_pretrained — the
# schema (shortest_edge vs height/width) varies by model family, so we
# don't wrap it behind a convenience kwarg.
from transformers import AutoImageProcessor
pp = EncoderPreprocessTransform.from_huggingface(
    keys=["img"],
    processor=AutoImageProcessor.from_pretrained(
        "microsoft/rad-dino", size={"shortest_edge": 518},
    ),
    reader="monai",   # mt.LoadImage + (C, W, H)→(C, H, W) transpose
)

# Keep MONAI's upstream LoadImaged instead — disable the built-in reader.
pp = EncoderPreprocessTransform.from_huggingface(
    keys=["img"], processor="microsoft/rad-dino", reader=None,
)

# Custom callable — base constructor does no conversion. You own the contract.
# preprocess receives whatever is at sample["img"] (path / tensor / PIL) and
# is responsible for reading, resizing, and normalizing.
def my_preprocess(x):
    ...  # x is whatever upstream placed at sample["img"] (path / tensor / PIL)
    return tensor_chw
pp = EncoderPreprocessTransform(keys=["img"], preprocess=my_preprocess)

# Plug into a RadHarmony transform pipeline
transform = Compose([..., pp, ...])
train_ds = VinDrCXRTrainDataset(base_image_dir=..., transform=transform, output_cls=True)
```

`reader` options for `from_huggingface`:

| Value | What it does |
|---|---|
| `"pil"` (default) | `PIL.Image.open(path).convert("RGB")` — matches HF pretraining |
| `"monai"` | `mt.LoadImage(ensure_channel_first=True)` + `mt.Transpose([0, 2, 1])` — DICOM / NIfTI via MONAI, fixing the `(C, W, H)` axis order MONAI writes for 2D |
| `callable` | Your own `(path) -> image-like` |
| `None` | Upstream (`LoadImaged`) already loaded the image; pass through |

`reader="monai"` is 2D-only; 3D volumes need a different transpose.

## `ImageEncoderWrapper` — keep the backbone registered

`ImageEncoderWrapper` is a thin `nn.Module` adapter that wraps a
backbone into the evaluator's `forward(x) -> Tensor[B, D]` contract.
It does **no image preprocessing** — it only (a) keeps the backbone as
a registered submodule, (b) routes the forward through a
caller-supplied call signature, and (c) applies a pooling function.

Why `nn.Module` instead of a `lambda`:

- `encoder.to("cuda")` moves the backbone weights with it.
- `encoder.state_dict()` / `.parameters()` see the backbone — required
  for `FinetuneEvaluator` when `freeze_backbone=False`, and for saving
  a fine-tuned checkpoint.
- `copy.deepcopy(encoder)` correctly duplicates the backbone.

A closure over a loose `nn.Module` silently fails all three.

```python
from radharmony.evaluator import ImageEncoderWrapper
from transformers import AutoModel

# HuggingFace ViT — CLS-pool by default; pass pool="mean" for mean-pooling
encoder = ImageEncoderWrapper.from_huggingface(
    AutoModel.from_pretrained("microsoft/rad-dino")
).cuda()

# timm CNN (num_classes=0 → already returns (B, D); no pooling needed)
import timm
encoder = ImageEncoderWrapper.from_custom(
    timm.create_model("resnet50", pretrained=True, num_classes=0),
).cuda()

# OpenCLIP / BiomedCLIP — route through .encode_image
import open_clip
clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16")
encoder = ImageEncoderWrapper.from_custom(
    clip_model, model_call=lambda m, x: m.encode_image(x),
).cuda()

# SSL backbone that already returns (B, D) — bare wrapper for submodule registration
encoder = ImageEncoderWrapper.from_custom(my_ssl_model.backbone).cuda()
```

For a bespoke backbone that needs a custom call signature and pool:

```python
encoder = ImageEncoderWrapper.from_custom(
    model=my_model,
    model_call=lambda m, x: m.encode(x, return_dict=False)[0],
    pool=lambda out: out[:, 0, :],
)
```

## Split modes: k-fold vs fixed-split

The evaluator requires **exactly one** of two modes:

```python
# k-fold mode: one dataset, KFold over pooled embeddings, variance from folds
ev = LinearProbeEvaluator(image_encoder, dataset=ds, n_folds=5)

# fixed-split mode: separate train + test sets, variance from seeds × bootstrap
ev = LinearProbeEvaluator(
    image_encoder,
    train_dataset=train_ds,
    test_dataset=test_ds,
    n_seeds=5,              # train-subsample replicates
    n_bootstrap=100,        # test-row resamples per seed
    n_train_samples=[500, 1500, 5000],
)
```

- In **k-fold mode**, `n_seeds` and `n_bootstrap` are ignored (fold index is the variance axis).
- In **fixed-split mode**, variance is composed: `n_seeds` replicates × `n_bootstrap` resamples per seed. Final rows = `labels × seeds × (1 + n_bootstrap) × n_train_points`.

## Output schema

Every classification evaluator's `evaluate()` returns a `pd.DataFrame`
with these columns:

| Column | Meaning |
|---|---|
| `label` | Per-label row, plus `macro_average` appended per variance group. |
| `n_train` | Training-set size for this row (sweep via `n_train_samples`). `-1` when `n_train_samples is None` (full train pool used) or for zero-shot. |
| `fold` | `0..n_folds-1` in k-fold mode, `-1` in fixed-split. |
| `seed` | `base_seed..base_seed+n_seeds-1` in fixed-split, `-1` in k-fold. |
| `bootstrap` | `-1` for the point estimate; `0..n_bootstrap-1` for resamples. |
| `auroc, auprc` | Threshold-free. |
| `f1, accuracy, balanced_accuracy, tpr, tnr, ppv, npv, mcc, threshold` | Threshold-based; threshold picked via `threshold_strategy`. |

`save_results(output_dir)` writes `results.csv` (per-row) and
`results_summary.csv` (per-label mean / std / 95% CI).

Threshold strategies: `"youden"` (default, argmax TPR−FPR),
`"f1"` (argmax F1 on the PR curve), or `"fixed:<value>"`.

## LinearProbeEvaluator

```python
from radharmony.evaluator import LinearProbeEvaluator

ev = LinearProbeEvaluator(
    image_encoder,
    train_dataset=train_ds,
    test_dataset=test_ds,
    n_seeds=5,
    n_train_samples=[1500],
    n_bootstrap=100,
    labels=["pneumothorax", "cardiomegaly"],   # optional subset
    output_dir="outputs/linprobe",
)
df = ev.evaluate()
ev.save_results(df)
```

Defaults for the underlying `sklearn.linear_model.LogisticRegression`:
`class_weight="balanced", C=1.0, max_iter=10000, solver="lbfgs"`.
Override via `probe_kwargs={...}`.

## KNNProbeEvaluator

```python
from radharmony.evaluator import KNNProbeEvaluator

ev = KNNProbeEvaluator(
    image_encoder,
    dataset=train_ds,
    n_folds=5,
    k=20,
    metric="cosine",
)
df = ev.evaluate()
```

Embeddings are L2-normalized before the nearest-neighbor search by default
(`l2_normalize=True` is forced on at the `BaseClsEvaluator` boundary), so
`metric="cosine"` operates on unit vectors. Pass `l2_normalize=False` to
feed raw embeddings into the chosen `metric`. Per-label probability is the
mean of the `k` neighbors' labels.

## SVMProbeEvaluator

Per-label `LinearSVC` on frozen encoder features. The decision function
(not a sigmoid probability) is used as the ranking score, so AUROC and
AUPRC are reliable without calibration.

```python
from radharmony.evaluator import SVMProbeEvaluator

ev = SVMProbeEvaluator(
    image_encoder,
    dataset=ds,
    n_folds=5,
    svm_kwargs={"C": 0.1, "class_weight": "balanced"},  # merged with defaults
)
df = ev.evaluate()
```

Defaults: `class_weight="balanced", C=1.0, max_iter=10000, dual=False`
(primal — faster when `n_samples > n_features`). Features are
L2-normalized by default (`l2_normalize=True` is forced on at the
`BaseClsEvaluator` boundary) so the L2 penalty isn't dominated by outlier
embedding dimensions. `threshold_strategy="fixed:<float>"` is not
meaningful for SVM decision values (not in [0, 1]); use `"youden"` or
`"f1"` instead.

## PrototypeProbeEvaluator

Nearest-centroid classifier: compute the mean L2-normalised embedding of
positive and negative training examples per label, then score each test
embedding as `cos(x, pos_centroid) − cos(x, neg_centroid)`. No
optimisation required.

```python
from radharmony.evaluator import PrototypeProbeEvaluator

ev = PrototypeProbeEvaluator(
    image_encoder,
    dataset=ds,
    n_folds=5,
    n_train_samples=[200, 500, 2000],
)
df = ev.evaluate()
```

Useful as a fast, parameter-free baseline — fitting is a single pass
over training embeddings with no hyperparameters to tune.

## ZeroShotEvaluator

```python
from radharmony.evaluator import ZeroShotEvaluator
import open_clip

model, _, _ = open_clip.create_model_and_transforms("ViT-B-16")
tokenizer = open_clip.get_tokenizer("ViT-B-16")

ev = ZeroShotEvaluator(
    image_encoder=lambda x: model.encode_image(x),
    text_encoder=lambda toks: model.encode_text(toks),
    tokenizer=tokenizer,
    test_dataset=test_ds,
    prompts={
        "pneumothorax": ["a chest X-ray showing pneumothorax", "pneumothorax present"],
        "cardiomegaly": ["a chest X-ray showing cardiomegaly"],
    },
    negative_prompts={
        "pneumothorax": ["a normal chest X-ray"],
    },
    n_bootstrap=100,
)
df = ev.evaluate()
```

Zero-shot uses the test set only. `n_seeds` is ignored; `n_bootstrap`
still applies for test-row CIs.

## FinetuneEvaluator

Deliberately minimal — single-GPU, BCE loss, AdamW + cosine schedule.
Heavy training workflows (DDP, WandB, mixed-precision) stay outside
radharmony.

```python
from radharmony.evaluator import FinetuneEvaluator

ev = FinetuneEvaluator(
    image_encoder,
    train_dataset=train_ds,
    test_dataset=test_ds,
    n_seeds=3,
    n_train_samples=[1500],
    epochs=20,
    lr=1e-4,
    freeze_backbone=False,   # True → head-only training
    loss="bce",
)
df = ev.evaluate()
```

Train a deployment model once and reuse it (skips the cross-validated sweep):

```python
ev.fit()                       # or pass store_final_model=True to evaluate()
ev.save_model("runs/ft.pt")    # encoder saved too when freeze_backbone=False

ev2 = FinetuneEvaluator(image_encoder, dataset=ds, freeze_backbone=False)
ev2.load_model("runs/ft.pt")
preds = ev2.predict(test_ds)   # [{image_id, y_true, prob}, ...]
```

## LinearProbeSegEvaluator

Frozen-feature semantic segmentation probe — a 1×1 `nn.Conv2d` head with
bilinear upsample, trained on dense patch features produced by a
segmentation-mode backbone. Requires the dataset to emit `output_mask=True`
and the backbone to be built with `output_keys={"img", "mask"}` (so
`forward(x)` returns `Tensor[B, D, H, W]` instead of `Tensor[B, D]`).

```python
from radharmony.evaluator import LinearProbeSegEvaluator
from radharmony.evaluator.backbones import make_raddino

transform, encoder = make_raddino(device="cuda", output_keys={"img", "mask"})

ev = LinearProbeSegEvaluator(
    encoder,
    dataset=train_ds,                 # output_mask=True required
    num_classes=2,                    # background + foreground
    epochs=20,
    lr=1e-3,
    n_folds=5,
)
df = ev.evaluate()
ev.save_results(df)
```

Output schema mirrors the classification evaluators (per-(class, fold, …)
rows + `label='macro_average'`). Metric panel: `dice`, `iou`, `pixel_acc`,
`tpr`, `tnr`, `ppv`, `npv`. When `save_predictions=True` and `output_dir`
is set, per-image GT / Pred / Prob PNGs are written under
`<output_dir>/{GT,Pred,Prob}/<run-id>/`.

Save the trained head and reuse it for inference — only the head is persisted
(the frozen backbone is rebuilt from its factory). The same API
(`fit` / `save_head` / `load_head` / `predict`) is inherited by
`ConvProbeSegEvaluator` and `UPerNetSegEvaluator`.

```python
ev.fit()                       # or pass store_final_model=True to evaluate()
ev.save_head("runs/seg_head.pt")

ev2 = LinearProbeSegEvaluator(encoder, dataset=train_ds, num_classes=2)
ev2.load_head("runs/seg_head.pt")
preds = ev2.predict(train_ds, indices=[0, 1, 2], return_images=True)
```

## Registry

Register custom evaluators alongside the built-ins:

```python
from radharmony.evaluator import register_evaluator, BaseClsEvaluator

@register_evaluator("my_probe")
class MyProbeEvaluator(BaseClsEvaluator):
    def evaluate(self):
        ...

from radharmony.evaluator import list_evaluators, resolve_evaluator
print(list_evaluators())   # includes 'my_probe'
cls = resolve_evaluator("my_probe")
```
