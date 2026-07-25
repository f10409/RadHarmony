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

Specific to `KNNProbeEvaluator`:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `k` | `int` | `20` | Number of neighbours |
| `metric` | `str` | `"cosine"` | Distance metric passed to `sklearn.neighbors.NearestNeighbors` |
| `n_folds` | `int` | `5` | Number of folds in k-fold mode |
| `n_train_samples` | `list[int]` \| `None` | `None` | Training-set size sweep |
| `store_final_model` | `bool` | `False` | Save the fitted index to `output_dir` |

### Shared arguments (inherited from `BaseClsEvaluator`)

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
| `l2_normalize` | `bool` | `False` | L2-normalize features at the cache boundary. `KNNProbeEvaluator` forces this on (effective default `True`); pass `l2_normalize=False` to feed raw embeddings |
| `output_dir` | `str` | `None` | Directory for CSV output |
| `n_seeds` | `int` | `1` | Fixed-split only: train-subsample replicates |
| `base_seed` | `int` | `0` | RNG seed base |
| `n_bootstrap` | `int` | `0` | Fixed-split only: test-row bootstrap resamples |
| `bootstrap_seed` | `int` | `0` | RNG seed base for test-row resampling |
| `threshold_strategy` | `str` | `"youden"` | `"youden"`, `"f1"`, or `"fixed:<float>"` |

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
