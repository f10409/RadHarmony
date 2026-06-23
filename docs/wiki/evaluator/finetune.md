# FinetuneEvaluator

Registry key: `"finetune"`.

Minimal single-GPU end-to-end fine-tune: BCE loss, AdamW optimizer, cosine
learning-rate schedule, and optional early stopping. The same split-mode and
variance-estimation machinery as the probe evaluators applies here — results
are directly comparable across evaluator types.

Heavy training workflows (DDP, WandB, mixed-precision, MONAI Ignite engines)
are intentionally out of scope.

---

## Usage

```python
from radharmony.evaluator import FinetuneEvaluator

ev = FinetuneEvaluator(
    image_encoder,
    train_dataset=train_ds,
    test_dataset=test_ds,
    n_seeds=3,
    n_train_samples=[1500],
    n_bootstrap=100,
    epochs=20,
    lr=1e-4,
    freeze_backbone=False,
    loss="bce",
    output_dir="outputs/finetune",
)
df = ev.evaluate()
ev.save_results(df)
```

Head-only fine-tune (freeze the backbone):

```python
ev = FinetuneEvaluator(
    image_encoder,
    train_dataset=train_ds,
    test_dataset=test_ds,
    epochs=30,
    lr=1e-3,
    freeze_backbone=True,
)
```

k-fold mode:

```python
ev = FinetuneEvaluator(image_encoder, dataset=ds, n_folds=5, epochs=20)
```

---

## Constructor arguments

These arguments are specific to `FinetuneEvaluator`. For the full set of
shared arguments (`device`, `batch_size`, `n_seeds`, `n_bootstrap`, etc.) see
[Common constructor arguments](index.md#common-constructor-arguments).

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `n_folds` | `int` | `5` | Number of folds in k-fold mode |
| `n_train_samples` | `list[int]` \| `None` | `None` | Training-set size sweep |
| `epochs` | `int` | `20` | Training epochs per seed/fold |
| `lr` | `float` | `1e-4` | Learning rate for the classification head (and backbone when `freeze_backbone=False`) |
| `backbone_lr_scale` | `float` | `0.1` | Multiplier applied to `lr` for backbone parameters when `freeze_backbone=False` |
| `weight_decay` | `float` | `0.05` | AdamW weight decay |
| `freeze_backbone` | `bool` | `False` | If `True`, only the classification head is trained |
| `head` | `nn.Module` \| `None` | `None` | Custom classification head; default = `nn.Linear(D, n_labels)` |
| `loss` | `str` | `"bce"` | `"bce"` (binary cross-entropy per label) or `"ce"` (cross-entropy) |
| `early_stop_metric` | `str` | `"auroc"` | Metric monitored for early stopping |
| `early_stop_patience` | `int` | `3` | Epochs without improvement before stopping |
| `val_fraction` | `float` | `0.1` | Fraction of the train subsample carved off (patient-grouped) as an inner val set for early stopping; `0.0` disables |
| `store_final_model` | `bool` | `False` | After `evaluate()`, also train a deployment model on all data → `final_encoder_` / `final_head_` (see [Saving and reuse](#saving-and-reuse)) |

---

## Saving and reuse

Train a single deployment model (encoder + head) once, persist it, and reuse it
for inference — without re-running the cross-validated `evaluate()`:

```python
ev = FinetuneEvaluator(image_encoder, dataset=ds, epochs=20, freeze_backbone=True)
ev.fit()                            # trains on all data (minus a patient-grouped
                                    # inner-val slice for early stopping)
ev.save_model("runs/ft.pt")

# later — construct with the SAME encoder factory, head=, and freeze_backbone=:
ev2 = FinetuneEvaluator(image_encoder, dataset=ds, freeze_backbone=True)
ev2.load_model("runs/ft.pt")
preds = ev2.predict(test_ds)        # [{image_id, y_true[L], prob[L]}, ...]
```

| Method | Behaviour |
|--------|-----------|
| `fit(dataset=None)` | Train a deployment model on all data (defaults to `dataset` / `train_dataset`); sets `final_encoder_` / `final_head_` and returns the head. Uses the same patient-grouped inner-val early-stopping loop as `evaluate()`. |
| `save_model(path=None)` | Save the head weights + shape/label metadata; the **encoder** weights are saved too **only when `freeze_backbone=False`** (a frozen backbone is rebuilt unchanged from its factory). Defaults to `<output_dir>/final_model.pt`. Returns the path. |
| `load_model(path)` | Rebuild the head from the checkpoint's shape metadata and reload weights; warns on a `freeze_backbone` mismatch. Construct the evaluator with the same `image_encoder` factory, `head=`, and `freeze_backbone=`. |
| `predict(dataset, *, encoder=None, head=None, indices=None)` | Run the trained model over `dataset`, returning per-image `{image_id, y_true, prob}` (sigmoid probabilities over `final_labels_`). `indices` restricts to a subset; `encoder` / `head` default to `final_encoder_` / `final_head_`. |

Setting `store_final_model=True` runs `fit()` automatically at the end of
`evaluate()`, so the benchmark and the saved model come from one call.

---

## Notes

- When `freeze_backbone=False`, `backbone_lr_scale=0.1` applies a 10× lower
  learning rate to backbone parameters. This prevents catastrophic forgetting
  while still allowing the backbone to adapt.
- Bootstrap resamples reuse the fitted model's test predictions — no retraining
  per bootstrap draw.
- `loss="bce"` treats each label as an independent binary classification task,
  which is appropriate for multi-label CXR datasets. Use `loss="ce"` for
  single-label datasets.
- For very small `n_train_samples`, `freeze_backbone=True` is usually better
  because there are too few examples to fine-tune the backbone without
  overfitting.
