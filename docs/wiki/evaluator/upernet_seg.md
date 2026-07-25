# UPerNetSegEvaluator

Registry key: `"upernet_seg"`.

Wraps HuggingFace's `UperNetHead` (PSP + FPN top-down fuse) on top of a
small **Simple Feature Pyramid** adapter that synthesises strides
`{4, 8, 16, 32}` from a single-scale ViT feature map (stride 16). Only
the SFP + UPerNet head are trainable; the backbone stays frozen.

The heaviest of the three segmentation probes (~9.8M params at `D=768`,
`hidden_size=256`) — useful when the linear / conv-block heads
underfit. Sits above [`ConvProbeSegEvaluator`](conv_probe_seg.md) on
the head-capacity spectrum.

---

## Encoder + dataset contract

Identical to [`LinearProbeSegEvaluator`](linear_probe_seg.md):

- **Encoder** must return `Tensor[B, D, H', W']` (dense patch features).
  Every RadHarmony `make_*` recipe supports this via
  `output_keys={"img", "mask"}`:

  ```python
  from radharmony.evaluator.backbones import make_raddino
  transform, encoder = make_raddino(device="cuda", output_keys={"img", "mask"})
  ```

- **Dataset** must emit a `"mask"` key per sample (`output_mask=True`).

---

## Usage

```python
from radharmony.evaluator import UPerNetSegEvaluator
from radharmony.evaluator.backbones import make_raddino
from radharmony.dataset import SIIMACRPTXDataset

transform, encoder = make_raddino(device="cuda", output_keys={"img", "mask"})

ds = SIIMACRPTXDataset(
    base_image_dir="/data/SIIM-ACR-PTX/",
    transform=transform,
    output_cls=True,
    output_mask=True,
    mask_output_dir="/scratch/ptx_masks/",
)

ev = UPerNetSegEvaluator(
    encoder,
    dataset=ds,                  # k-fold mode
    num_classes=2,
    n_folds=5,
    epochs=20,
    lr=1e-3,
    upernet_hidden_size=256,
    upernet_pool_scales=(1, 2, 3, 6),
    output_dir="outputs/upernet_seg",
)
df = ev.evaluate()
ev.save_results(df)
```

---

## Constructor arguments

Specific to `UPerNetSegEvaluator`:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `upernet_hidden_size` | `int` | `256` | UPerNet fuse channels |
| `upernet_pool_scales` | `tuple[int, ...]` | `(1, 2, 3, 6)` | PSP pool grid scales |

### Training arguments (shared with `LinearProbeSegEvaluator`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `n_folds` | `int` | `5` | Number of folds in k-fold mode; ignored in fixed-split mode |
| `n_train_samples` | `list[int]` \| `None` | `None` | Training-set size sweep; `None` = full train pool as a single point |
| `epochs` | `int` | `20` | Number of training epochs for the segmentation head |
| `lr` | `float` | `1e-3` | AdamW learning rate |
| `weight_decay` | `float` | `0.05` | AdamW weight decay |
| `class_weights` | `list[float]` \| `None` | `None` | Per-class weights for `nn.CrossEntropyLoss` — useful for class-imbalanced foreground (e.g. small lesions) |
| `save_predictions` | `bool` | `False` | Write per-image GT / Pred / Prob PNGs under `<output_dir>/{GT,Pred,Prob}/<run-id>/` |
| `early_stop_metric` | `str` | `"dice"` | Per-epoch val metric tracked for early stopping — any key in the seg metric panel; macro-averaged over the foreground classes |
| `early_stop_patience` | `int` | `5` | Stop training after this many epochs without improvement; best-epoch head state is restored |
| `val_fraction` | `float` | `0.1` | Fraction of the train subsample carved off as an inner val set for early stopping; `0.0` disables |
| `store_final_model` | `bool` | `False` | After `evaluate()`, also train a deployment head on all data → `final_head_` |

### Shared arguments (inherited from `BaseSegEvaluator`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `image_encoder` | `nn.Module` / callable | — | Frozen encoder; segmentation mode returns `Tensor[B, D, H, W]` |
| `dataset` | dataset | `None` | k-fold mode (mutually exclusive with `train_dataset`/`test_dataset`) |
| `train_dataset` | dataset | `None` | Fixed-split mode train pool |
| `test_dataset` | dataset | `None` | Fixed-split mode test set |
| `num_classes` | `int` | `2` | Number of segmentation classes, including background |
| `device` | `str` | `"cuda"` | `"cuda"`, `"cuda:N"`, or `"cpu"` |
| `batch_size` | `int` | `8` | DataLoader batch size (use `>= 2`; see Caveats) |
| `num_workers` | `int` | `4` | DataLoader workers |
| `autocast_dtype` | `torch.dtype` | `torch.bfloat16` | Encoder-forward autocast dtype; `None` disables |
| `output_dir` | `str` | `None` | Directory for CSV output |
| `n_seeds` | `int` | `1` | Fixed-split only: train-subsample replicates |
| `base_seed` | `int` | `0` | RNG seed base |
| `n_bootstrap` | `int` | `0` | Fixed-split only: test-image bootstrap resamples |
| `bootstrap_seed` | `int` | `0` | RNG seed base for test-image resampling |
| `threshold_strategy` | `str` | `"youden"` | Threshold selection strategy for binary metrics |

---

## Output schema

Identical to [`LinearProbeSegEvaluator`](linear_probe_seg.md) — same
`dice / iou / pixel_acc / tpr / tnr / ppv / npv` metric panel, same
per-row + summary CSV layout via `save_results`.

---

## Caveats

- `UperNetHead` is imported from
  `transformers.models.upernet.modeling_upernet` (a private module path).
  HF has kept this stable across the 4.x and 5.x lines but it is not part
  of the public `transformers` re-exports.
- The PSP module contains BatchNorm with a pool-to-(1,1) branch, which
  raises `ValueError` in training mode at batch size 1. Use
  `batch_size >= 2` (default `8` in `BaseSegEvaluator`).

---

## Notes

- Same training loop, split modes, metrics, and prediction-dump behaviour
  as [`LinearProbeSegEvaluator`](linear_probe_seg.md); only `_make_head`
  is overridden.
- The saving / reuse API (`fit` / `save_head` / `load_head` / `predict`, plus
  `store_final_model`) is inherited — see
  [Saving and reuse](linear_probe_seg.md#saving-and-reuse). `save_head` records
  `upernet_hidden_size` and `upernet_pool_scales` in the checkpoint so
  `load_head` rebuilds the matching head.
- Encoder weights are frozen (`requires_grad_(False)`); only the SFP +
  UPerNet head are updated. Optimizer: `AdamW` + `CosineAnnealingLR`.
- `bfloat16` autocast is used for the encoder forward pass; the head runs
  in fp32 for numerical stability.
- For SIIM-ACR-PTX / Montgomery / Shenzhen workflows see
  [`notebooks/evaluator/fm_comparison_seg.ipynb`](https://github.com/f10409/RadHarmony/blob/main/notebooks/evaluator/fm_comparison_seg.ipynb)
  and [`notebooks/evaluator/evaluator_backbone_verify_seg.ipynb`](https://github.com/f10409/RadHarmony/blob/main/notebooks/evaluator/evaluator_backbone_verify_seg.ipynb).
