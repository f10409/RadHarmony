# PrototypeProbeEvaluator

| | |
|---|---|
| Registry key | `prototype_probe` |
| Task | Classification, frozen features |
| Dataset outputs it needs | `cls` (`output_cls=True`) |
| Split modes | k-fold (`dataset=`), fixed split (`train_dataset=`, `test_dataset=`) |
| Encoder it needs | a pooled feature vector, `Tensor[B, D]` |
| Output | The result table, one row for each label and run. See [Output](#output) |

The prototype probe is a nearest-centroid classifier. For each label, it takes the mean feature of the
positive train images and of the negative train images. It scores a test image by its cosine to these 2
centroids. It has no optimization and no setting to tune, so it is a fast baseline.

## Quick use

This block runs the prototype probe on [Montgomery](../datasets/cxr-boxes-masks/montgomery_cxr.md) (138
images) with RAD-DINO, with 5 folds.

<!-- block: run=gpu timeout=600 -->
```python
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator import PrototypeProbeEvaluator
from radharmony.evaluator.backbones import make_raddino

transform, encoder = make_raddino(device="cuda")
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_cls=True,
    cache_dir=None,
)

ev = PrototypeProbeEvaluator(encoder, dataset=ds, n_folds=5, output_dir="outputs/prototype_probe")
df = ev.evaluate()
ev.save_results(df)  # writes results.csv and results_summary.csv
print(df.columns.tolist())
# ['n_train', 'fold', 'seed', 'bootstrap', 'label', 'auroc', 'auprc', 'f1', ...]
```

## How it works

- Features: the evaluator runs the encoder one time over each dataset. It L2-normalizes the features by
  default (D-17). The [embedding cache](index.md#embedding-cache) keeps the raw features.
- Centroids: for each label, `c_pos` is the mean feature of the train rows with label 1, and `c_neg` of
  the rows with label 0. Both centroids get L2 norm 1. A row with a NaN label is in neither centroid
  (D-18).
- Score: `cos(x, c_pos) - cos(x, c_neg)` for a test feature `x`. A score above 0 means that `x` is closer
  to the positive centroid.
- Missing class: a label with no positive or no negative train row gets NaN metrics in that run. The
  macro average leaves it out.
- Threshold: the probe holds out a validation part of the train patients. The centroids do not use it.
  The probe picks the threshold there ([Threshold](index.md#threshold), D-29).

## Arguments

<!-- args: radharmony.evaluator.PrototypeProbeEvaluator -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `n_folds` | `int` | `5` | The number of folds in k-fold mode. Fixed-split mode does not use it |
| `n_train_samples` | `list[int]` | `None` | The train sizes in images, one run for each ([Split modes](index.md#split-modes)). `None` uses all train images |
| `store_final_model` | `bool` | `False` | After `evaluate()`, also compute the centroids on all train images. They go to `final_models_`, not to a file |
| `val_fraction` | `float` | `0.1` | The part of the train patients for the threshold. The centroids do not use it. `0.0` gives NaN threshold metrics, except with `"fixed:<float>"` |

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
| `l2_normalize` | `bool` | `False` | L2-normalize the features before the centroids. This class sets `True` when you do not pass it (D-17) |
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
`output_dir`. The score and the threshold are in [-2, 2].

With `store_final_model=True`, `final_models_` maps each label to its `(c_pos, c_neg)` pair of unit
vectors, or to `None` for a label with only one class.

## Notes

- With `l2_normalize=True` (the default), each centroid is the mean of unit vectors. With
  `l2_normalize=False`, it is the mean of the raw features. The score is a cosine in both cases.
- `threshold_strategy="fixed:0"` means "closer to the positive centroid".
