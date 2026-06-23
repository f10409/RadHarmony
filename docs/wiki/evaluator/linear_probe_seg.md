# LinearProbeSegEvaluator

Registry key: `"linear_probe_seg"`.

Trains a 1×1 `nn.Conv2d` head with bilinear upsampling on the frozen dense
patch features produced by a segmentation-mode backbone. The head is the
only trainable component, matching the "linear probe" protocol used for
classification — but applied to semantic segmentation.

---

## Encoder + dataset contract

Two requirements that differ from the classification evaluators:

- **Encoder** must return `Tensor[B, D, H', W']` (dense patch features),
  not the pooled `Tensor[B, D]`. Every RadHarmony `make_*` recipe supports
  this via `output_keys={"img", "mask"}`:

  ```python
  from radharmony.evaluator.backbones import make_raddino
  transform, encoder = make_raddino(device="cuda", output_keys={"img", "mask"})
  ```

- **Dataset** must emit a `"mask"` key per sample (`output_mask=True`).
  The mask shape can be `[B, 1, H, W]`, `[B, H, W]`, or `[B, C, H, W]`
  one-hot — all are normalized to `[B, H, W]` int64 class indices
  internally.

If either contract is violated, `evaluate()` raises with a message pointing
at the right knob (see [base.py:112-122](../../../radharmony/evaluator/segmentation/base.py)).

---

## Usage

```python
from radharmony.evaluator import LinearProbeSegEvaluator
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

ev = LinearProbeSegEvaluator(
    encoder,
    dataset=ds,                  # k-fold mode
    num_classes=2,               # background + foreground
    n_folds=5,
    epochs=20,
    lr=1e-3,
    output_dir="outputs/linprobe_seg",
)
df = ev.evaluate()
ev.save_results(df)
```

Fixed-split mode:

```python
ev = LinearProbeSegEvaluator(
    encoder,
    train_dataset=train_ds,
    test_dataset=test_ds,
    num_classes=2,
    epochs=20,
    n_seeds=3,
    n_bootstrap=100,
)
df = ev.evaluate()
```

---

## Constructor arguments

These arguments are specific to `LinearProbeSegEvaluator`. The full set of
shared arguments (`dataset`, `train_dataset`, `test_dataset`,
`num_classes`, `device`, `batch_size`, `num_workers`, `autocast_dtype`,
`output_dir`, `n_seeds`, `base_seed`, `n_bootstrap`, `bootstrap_seed`,
`threshold_strategy`) comes from `BaseSegEvaluator`.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `n_folds` | `int` | `5` | Number of folds in k-fold mode; ignored in fixed-split mode |
| `n_train_samples` | `list[int]` \| `None` | `None` | Training-set size sweep; `None` = full train pool as a single point |
| `epochs` | `int` | `20` | Number of training epochs for the 1×1 Conv2d head |
| `lr` | `float` | `1e-3` | AdamW learning rate |
| `weight_decay` | `float` | `0.05` | AdamW weight decay |
| `class_weights` | `list[float]` \| `None` | `None` | Per-class weights for `nn.CrossEntropyLoss` — useful for class-imbalanced foreground (e.g. small lesions) |
| `save_predictions` | `bool` | `False` | Write per-image GT / Pred / Prob PNGs under `<output_dir>/{GT,Pred,Prob}/<run-id>/` |
| `early_stop_metric` | `str` | `"dice"` | Per-epoch val metric tracked for early stopping — any key in the seg metric panel (`dice`, `iou`, `pixel_acc`, `tpr`, `tnr`, `ppv`, `npv`); macro-averaged over the foreground classes |
| `early_stop_patience` | `int` | `5` | Stop training after this many epochs without improvement on `early_stop_metric`; best-epoch head state is restored at the end |
| `val_fraction` | `float` | `0.1` | Fraction of the train subsample carved off as an inner val set for early stopping. Set to `0.0` to disable. |
| `store_final_model` | `bool` | `False` | After `evaluate()`, also train a deployment head on all data → `final_head_` (see [Saving and reuse](#saving-and-reuse)) |

Shared `BaseSegEvaluator` defaults worth noting: `num_classes=2`,
`batch_size=8`, `num_workers=4`, `autocast_dtype=torch.bfloat16`.

---

## Output schema

`evaluate()` returns a `pd.DataFrame` with one row per
`(class, fold|seed, [bootstrap])`. Macro rows (`label='macro_average'`)
aggregate the foreground classes per variance group. Metric panel:

| Column | Meaning |
|--------|---------|
| `dice` | Dice coefficient (`2·TP / (2·TP + FP + FN)`) |
| `iou` | Intersection-over-union (`TP / (TP + FP + FN)`) |
| `pixel_acc` | Pixel accuracy |
| `tpr` | Sensitivity / recall |
| `tnr` | Specificity |
| `ppv` | Precision |
| `npv` | Negative predictive value |

`save_results(df, output_dir)` writes `results.csv` (per-row) and
`results_summary.csv` (per-class mean / std / 95 % CI), matching the
classification evaluators' on-disk shape.

---

## Saving and reuse

Train a single deployment head once, persist it, and reuse it for inference —
without re-running the cross-validated `evaluate()`. Only the trainable head is
saved; the frozen backbone is rebuilt from its factory at load time, so the
checkpoint is small.

```python
ev = LinearProbeSegEvaluator(encoder, dataset=ds, num_classes=2, epochs=20)
ev.fit()                            # trains on all data (minus a patient-grouped
                                    # inner-val slice for early stopping)
ev.save_head("runs/seg_head.pt")

# later — construct with the SAME backbone recipe and head args (num_classes, …):
transform, encoder = make_raddino(device="cuda", output_keys={"img", "mask"})
ev2 = LinearProbeSegEvaluator(encoder, dataset=ds, num_classes=2)
ev2.load_head("runs/seg_head.pt")
preds = ev2.predict(ds, indices=[0, 1, 2], return_images=True)
```

| Method | Behaviour |
|--------|-----------|
| `fit(dataset=None)` | Train a deployment head on all data (defaults to `dataset` / `train_dataset`); sets `final_head_` and returns it. Same patient-grouped inner-val early-stopping loop as `evaluate()`. |
| `save_head(path=None)` | Save the head state dict + shape / `num_classes` / `head_config` metadata. Defaults to `<output_dir>/final_head.pt`. Returns the path. |
| `load_head(path)` | Rebuild the head via `_make_head` from the checkpoint's shape metadata and reload weights; warns on a `num_classes` / `head_config` mismatch. |
| `predict(dataset, *, head=None, indices=None, return_images=False)` | Run the head over `dataset`, returning per-image `{image_id, y_true[H,W], y_pred[H,W], prob[C,H,W]}`. `return_images=True` adds a grayscale `image[H,W]` (min-max normalized, resized to the mask grid) for overlays; `indices` restricts to a subset; `head` defaults to `final_head_`. |
| `inner_val_indices(dataset=None)` | Dataset indices of the patient-grouped inner-val slice `fit()` holds out for early stopping. Deterministic (`base_seed` + patient grouping), so it reproduces the exact slice without retraining — works even after `load_head()`. Empty array when `val_fraction` disables the slice. Useful for visualizing held-out predictions (see [`fm_comparison_seg.ipynb`](https://github.com/f10409/RadHarmony/blob/main/notebooks/evaluator/fm_comparison_seg.ipynb)). |

Setting `store_final_model=True` runs `fit()` automatically at the end of
`evaluate()`, so the benchmark and the saved head come from one call.

---

## Notes

- The head is `nn.Conv2d(in_channels=D, out_channels=num_classes, kernel_size=1)`
  followed by `F.interpolate(..., mode="bilinear")` to the mask resolution.
- Encoder weights are frozen (`requires_grad_(False)`); only the conv head
  is updated. Optimizer: `AdamW` + `CosineAnnealingLR`.
- **Early stopping** is on by default. The val signal is the macro
  foreground `early_stop_metric` (default `dice`) recomputed each epoch;
  the best-epoch head state is restored before scoring. In **both** k-fold
  and fixed-split modes the val set is a `val_fraction` slice carved from
  the train subsample — the held-out fold (k-fold) and the test set
  (fixed-split) are used only for scoring, never for early-stopping
  selection. Set `val_fraction=0.0` to disable. With
  `early_stop_patience` consecutive no-improvement epochs, training
  exits early.
- Single-channel masks with arbitrary value ranges (`{0,1}`, `{0,255}`,
  etc.) are rescaled to `[0, num_classes-1]` automatically — see
  [linear_probe.py:95-118](../../../radharmony/evaluator/segmentation/linear_probe.py).
- `bfloat16` autocast is used for the encoder forward pass; the head runs
  in fp32 for numerical stability.
- For SIIM-ACR-PTX / Montgomery / Shenzhen workflows see
  [`notebooks/evaluator/fm_comparison_seg.ipynb`](https://github.com/f10409/RadHarmony/blob/main/notebooks/evaluator/fm_comparison_seg.ipynb)
  and [`notebooks/evaluator/evaluator_backbone_verify_seg.ipynb`](https://github.com/f10409/RadHarmony/blob/main/notebooks/evaluator/evaluator_backbone_verify_seg.ipynb).
