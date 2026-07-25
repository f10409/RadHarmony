# LinearProbeEvaluator

Registry key: `"linear_probe"`.

Fits a per-label `sklearn.linear_model.LogisticRegression` on frozen encoder
features. The simplest and most widely reported evaluation protocol for
frozen foundation models.

---

## Usage

```python
from radharmony.evaluator import LinearProbeEvaluator

ev = LinearProbeEvaluator(
    image_encoder,
    train_dataset=train_ds,
    test_dataset=test_ds,
    n_seeds=5,
    n_train_samples=[500, 1500, 5000],
    n_bootstrap=100,
    labels=["pneumothorax", "cardiomegaly"],   # optional subset
    output_dir="outputs/linprobe",
)
df = ev.evaluate()
ev.save_results(df)
```

k-fold mode:

```python
ev = LinearProbeEvaluator(image_encoder, dataset=ds, n_folds=5, n_train_samples=[500, 1500])
df = ev.evaluate()
```

---

## Constructor arguments

Specific to `LinearProbeEvaluator`:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `n_folds` | `int` | `5` | Number of folds in k-fold mode; ignored in fixed-split mode |
| `n_train_samples` | `list[int]` \| `None` | `None` | Training-set size sweep; `None` = full train pool as a single point |
| `probe_cls` | `type` | `LogisticRegression` | Drop-in replacement sklearn estimator |
| `probe_kwargs` | `dict` \| `None` | `None` | Kwargs merged into `probe_cls(...)` |
| `store_final_model` | `bool` | `False` | Save the fitted probe to `output_dir` after the last fold/seed |

Default `probe_kwargs` when `None`:

```python
{"class_weight": "balanced", "C": 1.0, "max_iter": 10000, "solver": "lbfgs"}
```

Override any key via `probe_kwargs={"C": 0.1, "max_iter": 5000}` — the dict is
merged with the defaults, so unspecified keys keep their values.

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
| `l2_normalize` | `bool` | `False` | L2-normalize features at the cache boundary. `KNNProbeEvaluator` and `SVMProbeEvaluator` force this to `True` by default |
| `output_dir` | `str` | `None` | Directory for CSV output |
| `n_seeds` | `int` | `1` | Fixed-split only: train-subsample replicates |
| `base_seed` | `int` | `0` | RNG seed base |
| `n_bootstrap` | `int` | `0` | Fixed-split only: test-row bootstrap resamples |
| `bootstrap_seed` | `int` | `0` | RNG seed base for test-row resampling |
| `threshold_strategy` | `str` | `"youden"` | `"youden"`, `"f1"`, or `"fixed:<float>"` |

---

## Notes

- Features are extracted once per split and cached in memory (or on disk via
  `embedding_cache`). The n-train sweep and seed loop operate on the cached
  embeddings, so extraction time is paid only once.
- `class_weight="balanced"` compensates for the label imbalance common in CXR
  datasets. Disable with `probe_kwargs={"class_weight": None}` if you have a
  balanced training set.
- For very large train pools (`n_train > 50 000`) consider `solver="saga"` for
  faster convergence.
