# ConvProbeSegEvaluator

Registry key: `"conv_probe_seg"`.

A natural step up from [`LinearProbeSegEvaluator`](linear_probe_seg.md):
instead of a single `1×1` conv, the head is a small convolutional block
that mixes channels and spatial context before the per-pixel classifier.

```
Sequential(
    Conv2d(D, hidden, kernel_size=1, bias=False),
    LayerNorm2d(hidden),
    Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
    LayerNorm2d(hidden),
)
```

followed by a `1×1` classifier and a bilinear upsample to the mask size.
`LayerNorm2d` is channel-wise layer norm at each spatial position
(canonical SAM / ViTDet impl) — works at any batch size, unlike BN.

Sits between [`LinearProbeSegEvaluator`](linear_probe_seg.md) (~150 params)
and [`UPerNetSegEvaluator`](upernet_seg.md) (~9.8M params) on the
head-capacity spectrum (~0.79M params at `D=768`, `hidden_size=256`).

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
from radharmony.evaluator import ConvProbeSegEvaluator
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

ev = ConvProbeSegEvaluator(
    encoder,
    dataset=ds,                  # k-fold mode
    num_classes=2,
    n_folds=5,
    epochs=20,
    lr=1e-3,
    conv_hidden_size=256,
    output_dir="outputs/conv_probe_seg",
)
df = ev.evaluate()
ev.save_results(df)
```

---

## Constructor arguments

`ConvProbeSegEvaluator` inherits every argument from
[`LinearProbeSegEvaluator`](linear_probe_seg.md) (`n_folds`,
`n_train_samples`, `epochs`, `lr`, `weight_decay`, `class_weights`,
`save_predictions`, `early_stop_metric`, `early_stop_patience`,
`val_fraction`, `store_final_model`) and `BaseSegEvaluator` (`dataset`,
`train_dataset`, `test_dataset`, `num_classes`, `device`, `batch_size`,
`num_workers`, `autocast_dtype`, `output_dir`, `n_seeds`, `base_seed`,
`n_bootstrap`, `bootstrap_seed`, `threshold_strategy`). The only extra
knob is:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `conv_hidden_size` | `int` | `256` | Neck channel width — controls head capacity |

---

## Output schema

Identical to [`LinearProbeSegEvaluator`](linear_probe_seg.md) — same
`dice / iou / pixel_acc / tpr / tnr / ppv / npv` metric panel, same
per-row + summary CSV layout via `save_results`.

---

## Notes

- Same training loop, split modes, metrics, and prediction-dump behaviour
  as [`LinearProbeSegEvaluator`](linear_probe_seg.md); only `_make_head`
  is overridden.
- The saving / reuse API (`fit` / `save_head` / `load_head` / `predict`, plus
  `store_final_model`) is inherited — see
  [Saving and reuse](linear_probe_seg.md#saving-and-reuse). `save_head` records
  `conv_hidden_size` in the checkpoint so `load_head` rebuilds the matching
  conv-block head.
- Encoder weights are frozen (`requires_grad_(False)`); only the conv-block
  head is updated. Optimizer: `AdamW` + `CosineAnnealingLR`.
- `bfloat16` autocast is used for the encoder forward pass; the head runs
  in fp32 for numerical stability.
- For SIIM-ACR-PTX / Montgomery / Shenzhen workflows see
  [`notebooks/evaluator/fm_comparison_seg.ipynb`](https://github.com/f10409/RadHarmony/blob/main/notebooks/evaluator/fm_comparison_seg.ipynb)
  and [`notebooks/evaluator/evaluator_backbone_verify_seg.ipynb`](https://github.com/f10409/RadHarmony/blob/main/notebooks/evaluator/evaluator_backbone_verify_seg.ipynb).
