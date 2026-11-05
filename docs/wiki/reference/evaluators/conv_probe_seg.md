# ConvProbeSegEvaluator

| | |
|---|---|
| Registry key | `conv_probe_seg` |
| Task | Segmentation, frozen patch features, small convolution head |
| Dataset outputs it needs | `mask` (`output_mask=True`) |
| Split modes | k-fold (`dataset=`), fixed split (`train_dataset=`, `test_dataset=`) |
| Encoder it needs | a map of patch features, `Tensor[B, D, H', W']`, from `make_<name>(output_keys={"img", "mask"})` |
| Output | The result table, one row for each class and run. See [Output](#output) |

The conv probe is the [segmentation linear probe](linear_probe_seg.md) with a larger head. The head mixes
the feature channels and the nearby patches before it gives a class score for each patch. Use it when the
1×1 head is too simple and the [UPerNet probe](upernet_seg.md) head is too large. The training, the
scoring and the output are the same as in the linear probe.

## Quick use

This block segments the lungs on Montgomery (138 images) with RAD-DINO, 2 folds and 3 epochs.

<!-- block: run=gpu timeout=600 -->
```python
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator import ConvProbeSegEvaluator
from radharmony.evaluator.backbones import make_raddino

transform, encoder = make_raddino(device="cuda", output_keys={"img", "mask"})
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_mask=True,
    mask_output_dir="/data/montgomery_lung_masks/",  # any folder you can write to
    cache_dir=None,
)

ev = ConvProbeSegEvaluator(
    encoder,
    dataset=ds,
    num_classes=2,  # background and lung
    conv_hidden_size=256,
    n_folds=2,
    epochs=3,
    num_workers=8,  # 8 workers read the large PNG files faster
    output_dir="outputs/conv_probe_seg",
)
df = ev.evaluate()
ev.save_results(df)  # writes results.csv and results_summary.csv
print(df.columns.tolist())
# ['n_train', 'fold', 'seed', 'bootstrap', 'label', 'dice', 'iou', 'pixel_acc', 'tpr', 'tnr', 'ppv', 'npv']
```

## How it works

The head is:

```text
Conv2d(D, hidden, kernel_size=1, bias=False)
LayerNorm2d(hidden)
Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False)
LayerNorm2d(hidden)
Conv2d(hidden, num_classes, kernel_size=1)
bilinear resize to the mask size
```

- Head: `hidden` is `conv_hidden_size`. For `D=768`, `hidden=256` and 2 classes, the head has about 0.79
  million weights.
- Norm: `LayerNorm2d` normalizes the channels at each patch. It works with any batch size.
- Training: for the training loop, early stopping, the splits, the masks and the metrics, see
  [How it works](linear_probe_seg.md#how-it-works) and [Metrics](linear_probe_seg.md#metrics) on the
  linear probe page.

## Save and reuse

`fit()`, `save_head()`, `load_head()`, `predict()`, `inner_val_indices()` and `store_final_model` work as
in the linear probe. See [Save and reuse](linear_probe_seg.md#save-and-reuse). `save_head()` keeps
`conv_hidden_size` in the file. `load_head()` warns when the evaluator has another `conv_hidden_size`.

## Arguments

<!-- args: radharmony.evaluator.ConvProbeSegEvaluator -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `conv_hidden_size` | `int` | `256` | The number of channels of the 2 convolutions before the classifier. It sets the size of the head |

### Shared arguments (inherited from LinearProbeSegEvaluator)

| Argument | Type | Default | Description |
|---|---|---|---|
| `n_folds` | `int` | `5` | The number of folds in k-fold mode. Fixed-split mode does not use it |
| `n_train_samples` | `list[int]` | `None` | The train sizes in images, one run for each ([Split modes](index.md#split-modes)). `None` uses all train images |
| `epochs` | `int` | `20` | The largest number of epochs of one run |
| `lr` | `float` | `0.001` | The learning rate of the head |
| `weight_decay` | `float` | `0.05` | The AdamW weight decay |
| `class_weights` | `list[float]` | `None` | A weight for each class in the cross-entropy loss, for example for a small foreground. `None` gives the same weight to all classes |
| `save_predictions` | `bool` | `False` | Write the PNGs `GT`, `Pred` and `Prob` of each test image to `<output_dir>/{GT,Pred,Prob}/<run-id>/`. The file name is the `image_id` with `__` for each `/` (D-46). It needs `output_dir` |
| `store_final_model` | `bool` | `False` | After `evaluate()`, also run `fit()` ([Save and reuse](#save-and-reuse)) |
| `early_stop_metric` | `str` | `"dice"` | The metric of the validation part for early stopping: `dice`, `iou`, `pixel_acc`, `tpr`, `tnr`, `ppv` or `npv` |
| `early_stop_patience` | `int` | `5` | The number of epochs with no better value before the run stops |
| `val_fraction` | `float` | `0.1` | The part of the train patients for early stopping. The head does not train on it. `0.0` gives no early stopping |
| `image_encoder` | `nn.Module` or callable | required | The encoder. It returns `Tensor[B, D, H', W']` ([Encoder contract](index.md#encoder-contract)) |
| `dataset` | dataset | `None` | K-fold mode: the one dataset ([Split modes](index.md#split-modes)) |
| `train_dataset` | dataset | `None` | Fixed-split mode: the train dataset |
| `test_dataset` | dataset | `None` | Fixed-split mode: the test dataset |
| `num_classes` | `int` | `2` | The number of classes, with background. `2` is background and one foreground class |
| `device` | `str` | `"cuda"` | `"cuda"`, `"cuda:N"` or `"cpu"` |
| `batch_size` | `int` | `8` | The batch size of training and of prediction |
| `num_workers` | `int` | `4` | The number of DataLoader workers |
| `autocast_dtype` | `torch.dtype` | `torch.bfloat16` | The autocast type of the encoder. `None` gives no autocast |
| `output_dir` | `str` | `None` | The folder for `save_results()`, `save_head()` and the PNGs of `save_predictions` |
| `n_seeds` | `int` | `1` | Fixed-split mode: the number of train samples, each with its own seed ([Split modes](index.md#split-modes)) |
| `base_seed` | `int` | `0` | The first seed of the train samples, the validation part and the training |
| `n_bootstrap` | `int` | `0` | Fixed-split mode: the number of bootstrap samples of the test set ([Split modes](index.md#split-modes)) |
| `bootstrap_seed` | `int` | `0` | The first seed of the bootstrap samples |
| `threshold_strategy` | `str` | `"youden"` | Not used. The head gives the class with the highest score for each pixel. A value other than `"youden"` warns (D-52) |

## Output

`evaluate()` returns the [result table](index.md#result-table), with the same rows and columns as the
linear probe ([Output](linear_probe_seg.md#output)). The Quick use run gave these columns: `n_train`,
`fold`, `seed`, `bootstrap`, `label`, `dice`, `iou`, `pixel_acc`, `tpr`, `tnr`, `ppv` and `npv`.
`save_results(df)` writes `results.csv` and `results_summary.csv` to `output_dir`.

## Notes

- The Quick use run took about 5 minutes on a shared GPU, as the linear probe does. Most of the time goes
  to the image reads, not to the head.
