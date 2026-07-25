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

Specific to `PrototypeProbeEvaluator`:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `n_folds` | `int` | `5` | Number of folds in k-fold mode |
| `n_train_samples` | `list[int]` \| `None` | `None` | Training-set size sweep; `None` = full train pool |
| `store_final_model` | `bool` | `False` | Save the centroid vectors to `output_dir` |

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
| `l2_normalize` | `bool` | `False` | L2-normalize features at the cache boundary |
| `output_dir` | `str` | `None` | Directory for CSV output |
| `n_seeds` | `int` | `1` | Fixed-split only: train-subsample replicates |
| `base_seed` | `int` | `0` | RNG seed base |
| `n_bootstrap` | `int` | `0` | Fixed-split only: test-row bootstrap resamples |
| `bootstrap_seed` | `int` | `0` | RNG seed base for test-row resampling |
| `threshold_strategy` | `str` | `"youden"` | `"youden"`, `"f1"`, or `"fixed:<float>"` |

---

## Notes

- Because the only computation is mean embedding and cosine similarity, the
  prototype probe is very fast — useful for quick iteration during backbone
  selection.
- Performance degrades gracefully with fewer training examples, making the
  n-train sweep especially informative here.
- Labels with zero positives or zero negatives in a training fold contribute
  `NaN` for that fold and are excluded from the macro average.
