# PrototypeProbeEvaluator

Registry key: `"prototype_probe"`.

Nearest-centroid classifier on L2-normalised encoder features. For each label
it computes the mean embedding of positive and negative training examples, then
scores each test image as:

```
score = cos(x, pos_centroid) − cos(x, neg_centroid)
```

No optimization is required — fitting is a single pass over training embeddings
with no hyperparameters to tune. Use it as a fast, parameter-free baseline to
check whether features carry label-relevant signal before running heavier probes.

---

## Usage

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

Fixed-split mode:

```python
ev = PrototypeProbeEvaluator(
    image_encoder,
    train_dataset=train_ds,
    test_dataset=test_ds,
    n_seeds=3,
    n_train_samples=[200, 500, 2000],
    n_bootstrap=100,
)
df = ev.evaluate()
```

---

## Constructor arguments

These arguments are specific to `PrototypeProbeEvaluator`. For the full set of
shared arguments (`device`, `batch_size`, `n_seeds`, `n_bootstrap`, etc.) see
[Common constructor arguments](index.md#common-constructor-arguments).

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `n_folds` | `int` | `5` | Number of folds in k-fold mode |
| `n_train_samples` | `list[int]` \| `None` | `None` | Training-set size sweep; `None` = full train pool |
| `store_final_model` | `bool` | `False` | Save the centroid vectors to `output_dir` |

---

## Notes

- Because the only computation is mean embedding and cosine similarity, the
  prototype probe is very fast — useful for quick iteration during backbone
  selection.
- Performance degrades gracefully with fewer training examples, making the
  n-train sweep especially informative here.
- Labels with zero positives or zero negatives in a training fold contribute
  `NaN` for that fold and are excluded from the macro average.
