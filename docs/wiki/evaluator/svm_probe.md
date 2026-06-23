# SVMProbeEvaluator

Registry key: `"svm_probe"`.

Per-label `sklearn.svm.LinearSVC` on frozen encoder features. The raw decision
function (not a calibrated probability) is used as the ranking score, so AUROC
and AUPRC are reliable without Platt scaling or isotonic regression. Often
outperforms logistic regression on small training sets.

---

## Usage

```python
from radharmony.evaluator import SVMProbeEvaluator

ev = SVMProbeEvaluator(
    image_encoder,
    dataset=ds,
    n_folds=5,
    svm_kwargs={"C": 0.1, "class_weight": "balanced"},
)
df = ev.evaluate()
```

Fixed-split mode:

```python
ev = SVMProbeEvaluator(
    image_encoder,
    train_dataset=train_ds,
    test_dataset=test_ds,
    n_seeds=3,
    n_train_samples=[500, 1500],
    n_bootstrap=100,
)
df = ev.evaluate()
```

---

## Constructor arguments

These arguments are specific to `SVMProbeEvaluator`. For the full set of
shared arguments (`device`, `batch_size`, `n_seeds`, `n_bootstrap`, etc.) see
[Common constructor arguments](index.md#common-constructor-arguments).

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `n_folds` | `int` | `5` | Number of folds in k-fold mode |
| `n_train_samples` | `list[int]` \| `None` | `None` | Training-set size sweep |
| `svm_kwargs` | `dict` \| `None` | `None` | Kwargs merged into `LinearSVC(...)` |
| `store_final_model` | `bool` | `False` | Save the fitted SVM to `output_dir` |

Default `svm_kwargs` when `None`:

```python
{"class_weight": "balanced", "C": 1.0, "max_iter": 10000, "dual": False}
```

`dual=False` selects the primal formulation, which is much faster than the
dual when `n_samples > n_features` — typical for CXR embedding probes.

Features are L2-normalized by default (`l2_normalize=True` is forced on at
the `BaseClsEvaluator` boundary) because `LinearSVC`'s L2 penalty assumes
features are on a similar scale; foundation-model embeddings have outlier
dimensions that otherwise dominate the optimizer. Pass
`l2_normalize=False` if you have already standardised features upstream.

---

## Notes

- `threshold_strategy="fixed:<float>"` is not meaningful for SVM decision values
  (they are not in `[0, 1]`). Use `"youden"` or `"f1"` instead.
- Because `LinearSVC` does not output calibrated probabilities, threshold-based
  metrics (F1, TPR, TNR, …) are computed after selecting the threshold that
  maximises the chosen strategy on the same evaluation set — treat them as
  approximate for the SVM.
- AUROC and AUPRC are threshold-free and fully reliable.
