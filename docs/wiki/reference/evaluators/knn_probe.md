# KNNProbeEvaluator

| | |
|---|---|
| Registry key | `knn_probe` |
| Task | Classification, frozen features |
| Dataset outputs it needs | `cls` (`output_cls=True`) |
| Split modes | k-fold (`dataset=`), fixed split (`train_dataset=`, `test_dataset=`) |
| Encoder it needs | a pooled feature vector, `Tensor[B, D]` |
| Output | The result table, one row for each label and run. See [Output](#output) |

The kNN probe scores each test image by its `k` nearest train images in feature space. The score of a
label is the mean label of these neighbors. The probe has no training loop, only a neighbor search. Use
it as a fast check that images with the same label have close features.

## Quick use

This block runs the kNN probe on [Montgomery](../datasets/cxr-boxes-masks/montgomery_cxr.md) (138 images)
with RAD-DINO, with 5 folds.

<!-- block: run=gpu timeout=600 -->
```python
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator import KNNProbeEvaluator
from radharmony.evaluator.backbones import make_raddino

transform, encoder = make_raddino(device="cuda")
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_cls=True,
    cache_dir=None,
)

ev = KNNProbeEvaluator(encoder, dataset=ds, n_folds=5, k=20, output_dir="outputs/knn_probe")
df = ev.evaluate()
ev.save_results(df)  # writes results.csv and results_summary.csv
print(df.columns.tolist())
# ['n_train', 'fold', 'seed', 'bootstrap', 'label', 'auroc', 'auprc', 'f1', ...]
```

## How it works

- Features: the evaluator runs the encoder one time over each dataset. It L2-normalizes the features by
  default (D-17). So the default `metric="cosine"` works on unit vectors. The
  [embedding cache](index.md#embedding-cache) keeps the raw features.
- Neighbors: `sklearn.neighbors.NearestNeighbors` with `metric` finds the `k` nearest train rows. With
  fewer than `k` train rows, it uses all of them.
- Score: the score of a label is the mean of its 0 or 1 values over the `k` neighbors. For example, 6
  positives in 20 neighbors give 0.3. Each label is scored alone.
- Labels: each label searches only the train rows where it is 0 or 1. A row with a NaN label is not a
  neighbor for that label (D-18).
- Threshold: the probe holds out a validation part of the train patients and does not search it. It
  picks the threshold there ([Threshold](index.md#threshold), D-29).

## Arguments

<!-- args: radharmony.evaluator.KNNProbeEvaluator -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `k` | `int` | `20` | The number of neighbors |
| `metric` | `str` | `"cosine"` | The distance of `sklearn.neighbors.NearestNeighbors` |
| `n_folds` | `int` | `5` | The number of folds in k-fold mode. Fixed-split mode does not use it |
| `n_train_samples` | `list[int]` | `None` | The train sizes in images, one run for each ([Split modes](index.md#split-modes)). `None` uses all train images |
| `store_final_model` | `bool` | `False` | After `evaluate()`, also fit the neighbor index on all train images. It goes to `final_models_`, not to a file |
| `val_fraction` | `float` | `0.1` | The part of the train patients for the threshold. The probe does not search it. `0.0` gives NaN threshold metrics, except with `"fixed:<float>"` |

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
`output_dir`.

With `store_final_model=True`, `final_models_` maps each label to a dict with 2 keys. `"nn"` is the fitted
`NearestNeighbors` over the rows where the label is 0 or 1. `"y_train"` holds the labels of these rows. A
label with no such row maps to `None`. The score of new features `x` is
`y_train[nn.kneighbors(x)[1]].mean(axis=1)`.

## Notes

- With unit vectors, `metric="cosine"` gives the same neighbors as a largest inner-product search. With
  `l2_normalize=False`, `metric` works on the raw features.
- The score has only `k + 1` possible values, so many test images can have the same score.
