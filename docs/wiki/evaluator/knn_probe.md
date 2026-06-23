# KNNProbeEvaluator

Registry key: `"knn_probe"`.

Per-label soft-vote k-nearest-neighbour classifier on L2-normalised encoder
features. No training loop — the only computation is a k-NN index build and
lookup. Useful as a fast, parameter-light probe and as a sanity check that
features cluster meaningfully before fitting a logistic regression.

---

## Usage

```python
from radharmony.evaluator import KNNProbeEvaluator

ev = KNNProbeEvaluator(
    image_encoder,
    dataset=ds,
    n_folds=5,
    k=20,
    metric="cosine",
)
df = ev.evaluate()
```

Fixed-split mode:

```python
ev = KNNProbeEvaluator(
    image_encoder,
    train_dataset=train_ds,
    test_dataset=test_ds,
    k=20,
    n_seeds=3,
    n_train_samples=[500, 1500],
    n_bootstrap=100,
)
df = ev.evaluate()
```

---

## Constructor arguments

These arguments are specific to `KNNProbeEvaluator`. For the full set of
shared arguments (`device`, `batch_size`, `n_seeds`, `n_bootstrap`, etc.) see
[Common constructor arguments](index.md#common-constructor-arguments).

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `k` | `int` | `20` | Number of neighbours |
| `metric` | `str` | `"cosine"` | Distance metric passed to `sklearn.neighbors.NearestNeighbors` |
| `n_folds` | `int` | `5` | Number of folds in k-fold mode |
| `n_train_samples` | `list[int]` \| `None` | `None` | Training-set size sweep |
| `store_final_model` | `bool` | `False` | Save the fitted index to `output_dir` |

---

## Notes

- Features are L2-normalised by default before the nearest-neighbour search
  (`l2_normalize=True` is forced on at the `BaseClsEvaluator` boundary), so
  `metric="cosine"` operates on unit vectors and is equivalent to a maximum
  inner-product search. Pass `l2_normalize=False` to feed raw embeddings into
  the chosen `metric`.
- Per-label probability = mean of the `k` neighbours' binary labels. For
  multi-label datasets each label is treated independently.
- `k` has a large effect on performance: sweep `k=[5, 10, 20, 50]` if you want
  to tune it.
