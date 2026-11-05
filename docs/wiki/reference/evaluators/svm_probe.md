# SVMProbeEvaluator

| | |
|---|---|
| Registry key | `svm_probe` |
| Task | Classification, frozen features |
| Dataset outputs it needs | `cls` (`output_cls=True`) |
| Split modes | k-fold (`dataset=`), fixed split (`train_dataset=`, `test_dataset=`) |
| Encoder it needs | a pooled feature vector, `Tensor[B, D]` |
| Output | The result table, one row for each label and run. See [Output](#output) |

The SVM probe fits one linear SVM (`sklearn.svm.LinearSVC`) for each label on the frozen features of an
encoder. The score of an image is the SVM decision value, not a probability. Use it as a second linear
probe next to the [linear probe](linear_probe.md). It has the squared hinge loss of `LinearSVC` in place
of the logistic loss.

## Quick use

This block runs the SVM probe on [Montgomery](../datasets/cxr-boxes-masks/montgomery_cxr.md) (138 images)
with RAD-DINO, with 5 folds.

<!-- block: run=gpu timeout=600 -->
```python
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator import SVMProbeEvaluator
from radharmony.evaluator.backbones import make_raddino

transform, encoder = make_raddino(device="cuda")
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_cls=True,
    cache_dir=None,
)

ev = SVMProbeEvaluator(encoder, dataset=ds, n_folds=5, output_dir="outputs/svm_probe")
df = ev.evaluate()
ev.save_results(df)  # writes results.csv and results_summary.csv
print(df.columns.tolist())
# ['n_train', 'fold', 'seed', 'bootstrap', 'label', 'auroc', 'auprc', 'f1', ...]
```

## How it works

- Features: the evaluator runs the encoder one time over each dataset. It L2-normalizes the features by
  default (D-17). The L2 penalty of `LinearSVC` needs features of the same scale. The
  [embedding cache](index.md#embedding-cache) keeps the raw features.
- Probe: one `LinearSVC` for each label. The default arguments are `class_weight="balanced"`, `C=1.0`,
  `max_iter=10000` and `dual=False`. `svm_kwargs` changes some of them, and the others stay. `dual=False`
  is the fast form when there are more train rows than feature values.
- Score: the decision value of the SVM. AUROC and AUPRC need only the order of the scores, so they need
  no probability.
- Labels: each label fits only on the rows where it is 0 or 1. A NaN label is not a negative (D-18). A
  label with only one class gets NaN metrics.
- Threshold: the probe holds out a validation part of the train patients and does not fit on it. It
  picks the threshold on the decision values there ([Threshold](index.md#threshold), D-29).

## Arguments

<!-- args: radharmony.evaluator.SVMProbeEvaluator -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `n_folds` | `int` | `5` | The number of folds in k-fold mode. Fixed-split mode does not use it |
| `n_train_samples` | `list[int]` | `None` | The train sizes in images, one run for each ([Split modes](index.md#split-modes)). `None` uses all train images |
| `svm_kwargs` | `dict` | `None` | Arguments for `LinearSVC`. Each key replaces the same default key |
| `store_final_model` | `bool` | `False` | After `evaluate()`, also fit one SVM for each label on all train images. They go to `final_models_`, not to a file |
| `val_fraction` | `float` | `0.1` | The part of the train patients for the threshold. The probe does not fit on it. `0.0` gives NaN threshold metrics, except with `"fixed:<float>"` |

### Shared arguments (inherited from BaseClsEvaluator)

| Argument | Type | Default | Description |
|---|---|---|---|
| `image_encoder` | `nn.Module` or callable | required | The encoder. It returns `Tensor[B, D]` ([Encoder contract](index.md#encoder-contract)) |
| `dataset` | dataset | `None` | K-fold mode: the one dataset ([Split modes](index.md#split-modes)) |
| `train_dataset` | dataset | `None` | Fixed-split mode: the train dataset |
| `test_dataset` | dataset | `None` | Fixed-split mode: the test dataset |
| `labels` | `list[str]` | `None` | The labels to evaluate. `None` gives all labels. The match ignores case. An unknown name raises `KeyError` |
| `device` | `str` | `"cuda"` | `"cuda"`, `"cuda:N"` or `"cpu"` |
| `batch_size` | `int` | `64` | The batch size of the feature extraction |
| `num_workers` | `int` | `4` | The number of DataLoader workers |
| `autocast_dtype` | `torch.dtype` | `torch.bfloat16` | The autocast type of the encoder. `None` gives no autocast |
| `embedding_cache` | `str` | `None` | The path of the feature file ([Embedding cache](index.md#embedding-cache)) |
| `l2_normalize` | `bool` | `False` | L2-normalize the features. This class sets `True` when you do not pass it (D-17) |
| `output_dir` | `str` | `None` | The folder for `save_results()` |
| `n_seeds` | `int` | `1` | Fixed-split mode: the number of train samples, each with its own seed ([Split modes](index.md#split-modes)) |
| `base_seed` | `int` | `0` | The first seed of the train samples and of the validation part |
| `n_bootstrap` | `int` | `0` | Fixed-split mode: the number of bootstrap samples of the test set ([Split modes](index.md#split-modes)) |
| `bootstrap_seed` | `int` | `0` | The first seed of the bootstrap samples |
| `threshold_strategy` | `str` | `"youden"` | `"youden"`, `"f1"` or `"fixed:<float>"` ([Threshold](index.md#threshold)) |

## Output

`evaluate()` returns the [result table](index.md#result-table) with these columns: `n_train`, `fold`,
`seed`, `bootstrap`, `label`, `auroc`, `auprc`, `f1`, `accuracy`, `balanced_accuracy`, `tpr`, `tnr`,
`ppv`, `npv`, `mcc` and `threshold`. `save_results(df)` writes `results.csv` and `results_summary.csv` to
`output_dir`. The `threshold` column is a decision value, so it can be below 0 or above 1.

With `store_final_model=True`, `final_models_` maps each label to its fitted `LinearSVC`, or to `None` for
a label with only one class.

## Notes

- `threshold_strategy="fixed:0"` is the decision boundary of the SVM: it gives the class of
  `LinearSVC.predict`. Another fixed value is a value on the decision scale, not a probability.
- `l2_normalize=False` gives the raw features to the SVM. Use it only for features that you scaled
  yourself.
