# Evaluator API

`radharmony.evaluator` takes any RadHarmony dataset and any image encoder and
produces standardized downstream-task results — DataFrame output with AUROC,
AUPRC, F1, and eight other per-label metrics.

Six classification evaluators plus three segmentation evaluators are available:

| Evaluator | Registry key | Mode | Summary |
|-----------|-------------|------|---------|
| [LinearProbeEvaluator](linear_probe.md) | `"linear_probe"` | k-fold / fixed-split | Logistic regression on frozen features |
| [KNNProbeEvaluator](knn_probe.md) | `"knn_probe"` | k-fold / fixed-split | Soft-vote k-NN on L2-normalised features |
| [SVMProbeEvaluator](svm_probe.md) | `"svm_probe"` | k-fold / fixed-split | `LinearSVC` on frozen features |
| [PrototypeProbeEvaluator](prototype_probe.md) | `"prototype_probe"` | k-fold / fixed-split | Nearest-centroid; no optimization |
| [ZeroShotEvaluator](zero_shot.md) | `"zero_shot"` | test-only | Cosine similarity to text prompts |
| [FinetuneEvaluator](finetune.md) | `"finetune"` | k-fold / fixed-split | Single-GPU end-to-end fine-tune |
| [LinearProbeSegEvaluator](linear_probe_seg.md) | `"linear_probe_seg"` | k-fold / fixed-split | 1×1 Conv2d head over frozen dense features (requires segmentation-mode backbone + `output_mask=True` dataset) |
| [ConvProbeSegEvaluator](conv_probe_seg.md) | `"conv_probe_seg"` | k-fold / fixed-split | `1×1 → LN2d → 3×3 → LN2d` conv-block head over frozen dense features |
| [UPerNetSegEvaluator](upernet_seg.md) | `"upernet_seg"` | k-fold / fixed-split | Simple Feature Pyramid + HF `UperNetHead` (PSP + FPN fuse) over frozen dense features |

For a hands-on walkthrough see
[`notebooks/tutorials/evaluator_tutorial.ipynb`](https://github.com/f10409/RadHarmony/blob/main/notebooks/tutorials/evaluator_tutorial.ipynb).

---

## Quick start

Use a built-in backbone recipe — the fastest path:

```python
from radharmony.evaluator.backbones import make_raddino
from radharmony.evaluator import LinearProbeEvaluator
from radharmony.dataset import VinDrCXRTrainDataset, VinDrCXRTestDataset

transform, encoder = make_raddino(device="cuda")

train_ds = VinDrCXRTrainDataset(base_image_dir="/data/VinDr-CXR/", transform=transform, output_cls=True, cache_dir=None)
test_ds  = VinDrCXRTestDataset (base_image_dir="/data/VinDr-CXR/", transform=transform, output_cls=True, cache_dir=None)

ev = LinearProbeEvaluator(
    encoder,
    train_dataset=train_ds,
    test_dataset=test_ds,
    n_seeds=3,
    n_train_samples=[500, 1500],
    n_bootstrap=100,
    output_dir="outputs/linprobe",
)
df = ev.evaluate()
ev.save_results(df)
```

Or wire up transform and encoder manually:

```python
from transformers import AutoModel
from radharmony.evaluator import ImageEncoderWrapper, LinearProbeEvaluator
from radharmony.evaluator import EncoderPreprocessTransform, RadiologyEncoderTransform

pp = EncoderPreprocessTransform.from_huggingface(
    keys=["img"], processor="microsoft/rad-dino",
)
transform = RadiologyEncoderTransform(preprocess=pp, output_keys={"img", "cls"}).get_transform()

encoder = ImageEncoderWrapper.from_huggingface(
    AutoModel.from_pretrained("microsoft/rad-dino")
).cuda()
```

For backbone recipes see [Backbones](../backbones/index.md). For custom backbones and recipe authoring see [Custom Backbones](../backbones/custom_backbones.md).

---

## Encoder contract

Every classification evaluator accepts an `image_encoder` — any `nn.Module` or
callable with the signature:

```text
image_encoder(imgs: Tensor[B, ...]) -> Tensor[B, D]
```

The evaluator does not inspect model internals or enforce a specific input shape.
Users are responsible for wrapping their backbone to return a pooled feature
vector per image. Use `ImageEncoderWrapper` to handle the most common patterns —
see [Custom Backbones](../backbones/custom_backbones.md) for wiring examples
covering HuggingFace, timm, OpenCLIP, and bespoke backbones.

For zero-shot a second argument is also required:
`text_encoder(list[str]) -> Tensor[T, D]`.

---

## Backbone recipes

Drop-in factory functions in `radharmony.evaluator.backbones` cover the
common foundation models — `make_raddino`, `make_biomed_clip`,
`make_chexagent`, `make_medsiglip`, `make_medimageinsights`, `make_chexfound`,
`make_dinov3`, `make_eva_x`, `make_ark_plus`, `make_medical_mae`,
`make_siglip2`. Each returns a `(transform, encoder, …)` tuple ready to pair
with the dataset and the evaluator below.

```python
from radharmony.evaluator.backbones import make_raddino, make_biomed_clip

transform, encoder = make_raddino(device="cuda")
transform, encoder, text_enc, tokenizer = make_biomed_clip(device="cuda")
```

Every recipe accepts `output_keys={"img", "mask"}` to switch the encoder
into segmentation mode — `forward(x)` then returns dense patch features as
`Tensor[B, D, H, W]`, ready for `LinearProbeSegEvaluator` /
`ConvProbeSegEvaluator` / `UPerNetSegEvaluator`.

See [Backbones](../backbones/index.md) for the full recipe table, per-model
pages with embed dims and override knobs, and the
[Custom Backbones](../backbones/custom_backbones.md) recipe-authoring guide.

---

## Split modes

Every evaluator requires exactly one of two modes:

```python
# k-fold mode: one dataset, GroupKFold over pooled embeddings
ev = LinearProbeEvaluator(image_encoder, dataset=ds, n_folds=5)

# fixed-split mode: separate train + test, variance from seeds × bootstrap
ev = LinearProbeEvaluator(
    image_encoder,
    train_dataset=train_ds,
    test_dataset=test_ds,
    n_seeds=5,
    n_bootstrap=100,
    n_train_samples=[500, 1500, 5000],
)
```

Splits are **patient-level** — `GroupKFold` on `patient_id` ensures no patient
appears in both train and val folds. In fixed-split mode, variance is composed
from `n_seeds` train-subsample replicates × `n_bootstrap` test-row resamples, so
the result table has `labels × seeds × (1 + n_bootstrap) × n_train_points` rows
(the `+1` is the point estimate alongside each set of bootstrap resamples).

---

## Output schema

`evaluate()` returns a `pd.DataFrame` with these columns:

| Column | Meaning |
|--------|---------|
| `label` | Per-label row, plus `macro_average` appended per variance group |
| `n_train` | Training-set size; `-1` when `n_train_samples is None` (full train pool used) or for zero-shot |
| `fold` | `0..n_folds-1` in k-fold, `-1` in fixed-split |
| `seed` | Seed index in fixed-split, `-1` in k-fold |
| `bootstrap` | `-1` = point estimate; `0..n_bootstrap-1` = resample |
| `auroc`, `auprc` | Threshold-free |
| `f1`, `accuracy`, `balanced_accuracy`, `tpr`, `tnr`, `ppv`, `npv`, `mcc`, `threshold` | Threshold-based; threshold chosen via `threshold_strategy` |

`save_results(df, output_dir)` writes `results.csv` (per-row) and
`results_summary.csv` (per-label mean / std / 95 % CI).

Threshold strategies: `"youden"` (default), `"f1"`, or `"fixed:<float>"`.

---

## Common constructor arguments

All classification evaluators inherit these arguments from `BaseClsEvaluator`:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `image_encoder` | `nn.Module` / callable | — | `forward(imgs) → Tensor[B, D]` |
| `dataset` | dataset | `None` | k-fold mode (mutually exclusive with `train_dataset`/`test_dataset`) |
| `train_dataset` | dataset | `None` | Fixed-split mode train pool |
| `test_dataset` | dataset | `None` | Fixed-split mode test set |
| `labels` | `list[str]` | `None` | Subset of `LABEL_COLS` to evaluate; default = all |
| `device` | `str` | `"cuda"` | `"cuda"`, `"cuda:N"`, or `"cpu"` |
| `batch_size` | `int` | `64` | Inference DataLoader batch size |
| `num_workers` | `int` | `4` | Inference DataLoader workers |
| `autocast_dtype` | `torch.dtype` | `torch.bfloat16` | Inference autocast dtype; `None` disables |
| `embedding_cache` | `str` | `None` | Path prefix to pickle cached embeddings (caches store **raw** features; normalization is reapplied after load) |
| `l2_normalize` | `bool` | `False` | L2-normalize features at the cache boundary. `KNNProbeEvaluator` and `SVMProbeEvaluator` force this to `True` by default |
| `output_dir` | `str` | `None` | Directory for CSV output |
| `n_seeds` | `int` | `1` | Fixed-split only: train-subsample replicates |
| `base_seed` | `int` | `0` | RNG seed base |
| `n_bootstrap` | `int` | `0` | Fixed-split only: test-row bootstrap resamples |
| `bootstrap_seed` | `int` | `0` | RNG seed base for test-row resampling |
| `threshold_strategy` | `str` | `"youden"` | `"youden"`, `"f1"`, or `"fixed:<float>"` |

---

## Registry

```python
from radharmony.evaluator import register_evaluator, resolve_evaluator, list_evaluators
from radharmony.evaluator import BaseClsEvaluator

@register_evaluator("my_probe")
class MyProbeEvaluator(BaseClsEvaluator):
    def evaluate(self):
        ...

print(list_evaluators())   # includes 'my_probe'
cls = resolve_evaluator("my_probe")
```
