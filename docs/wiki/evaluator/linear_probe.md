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

These arguments are specific to `LinearProbeEvaluator`. For the full set of
shared arguments (`device`, `batch_size`, `n_seeds`, `n_bootstrap`, etc.) see
[Common constructor arguments](index.md#common-constructor-arguments).

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
