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

`UPerNetSegEvaluator` inherits every argument from
[`LinearProbeSegEvaluator`](linear_probe_seg.md) (`n_folds`,
`n_train_samples`, `epochs`, `lr`, `weight_decay`, `class_weights`,
`save_predictions`, `early_stop_metric`, `early_stop_patience`,
`val_fraction`, `store_final_model`) and `BaseSegEvaluator` (`dataset`,
`train_dataset`, `test_dataset`, `num_classes`, `device`, `batch_size`,
`num_workers`, `autocast_dtype`, `output_dir`, `n_seeds`, `base_seed`,
`n_bootstrap`, `bootstrap_seed`, `threshold_strategy`). Extra knobs:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `upernet_hidden_size` | `int` | `256` | UPerNet fuse channels |
| `upernet_pool_scales` | `tuple[int, ...]` | `(1, 2, 3, 6)` | PSP pool grid scales |

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
