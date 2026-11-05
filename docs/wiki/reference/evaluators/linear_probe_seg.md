# LinearProbeSegEvaluator

| | |
|---|---|
| Registry key | `linear_probe_seg` |
| Task | Segmentation, frozen patch features, 1×1 convolution head |
| Dataset outputs it needs | `mask` (`output_mask=True`) |
| Split modes | k-fold (`dataset=`), fixed split (`train_dataset=`, `test_dataset=`) |
| Encoder it needs | a map of patch features, `Tensor[B, D, H', W']`, from `make_<name>(output_keys={"img", "mask"})` |
| Output | The result table, one row for each class and run. See [Output](#output) |

The segmentation linear probe trains a 1×1 convolution on the frozen patch features of an encoder. The
head gives a class score for each patch, and a bilinear resize takes the scores to the mask size. The score
shows how much location information the patch features hold. This page also has the shared facts of the
[conv probe](conv_probe_seg.md) and the [UPerNet probe](upernet_seg.md). They train and score in the same
way, with a larger head.

## Quick use

This block segments the lungs on Montgomery (138 images) with RAD-DINO, 2 folds and 3 epochs.

<!-- block: run=gpu timeout=600 -->
```python
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator import LinearProbeSegEvaluator
from radharmony.evaluator.backbones import make_raddino

transform, encoder = make_raddino(device="cuda", output_keys={"img", "mask"})
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_mask=True,
    mask_output_dir="/data/montgomery_lung_masks/",  # any folder you can write to
    cache_dir=None,
)

ev = LinearProbeSegEvaluator(
    encoder,
    dataset=ds,
    num_classes=2,  # background and lung
    n_folds=2,
    epochs=3,
    num_workers=8,  # 8 workers read the large PNG files faster
    output_dir="outputs/linear_probe_seg",
)
df = ev.evaluate()
ev.save_results(df)  # writes results.csv and results_summary.csv
print(df.columns.tolist())
# ['n_train', 'fold', 'seed', 'bootstrap', 'label', 'dice', 'iou', 'pixel_acc', 'tpr', 'tnr', 'ppv', 'npv']
```

`make_raddino` resizes each image and mask to 518×518. With `cache_dir=None`, the run reads and resizes
every image again in each epoch. On a shared GPU, this run took about 5 minutes.

## How it works

- Features: the encoder stays frozen. The head gets the raw patch features, with no L2 norm (D-17). The
  encoder runs in `autocast_dtype`, and the head runs in float32.
- Head: it trains with AdamW, a cosine learning-rate schedule and a cross-entropy loss. `class_weights`
  sets a weight for each class in the loss.
- Splits: the folds, the validation part and the bootstrap group the images by patient
  ([Split modes](index.md#split-modes)).
- Fixed split: it needs 2 datasets with masks, for example `QaTaCOV19TrainDataset` and
  `QaTaCOV19TestDataset`. The SIIM-ACR test split has no masks, so it does not work as `test_dataset`.
- Early stopping: the evaluator takes `val_fraction` of the train patients as a validation part. It never
  uses the test fold or the test set to train or to stop (D-18). After each epoch, it scores
  `early_stop_metric` on the validation part, as a mean over the classes other than background. It keeps
  the head of the best epoch, and it stops after `early_stop_patience` epochs with no better score. When
  the validation part has no foreground pixel, the epoch gets no score and does not count.
- Train size: `n_train_samples` sets how many images the evaluator draws from the train part (D-40). It
  includes the validation part.
- Masks (D-14): with `num_classes=2`, every value above 0 is foreground, for example 1 or 255. With more
  classes, the mask holds the class indices 0 to `num_classes - 1`, and a larger value raises `ValueError`.
  A mask can be `[B, 1, H, W]`, `[B, H, W]` or one-hot `[B, C, H, W]`. A one-hot mask gives its class
  indices with `argmax`.

## Metrics

The evaluator computes 7 metrics for each class and each test image. Then it takes the mean over the images
with `np.nanmean`.

| Metric | Meaning |
|---|---|
| `dice` | Dice: `2·TP / (2·TP + FP + FN)` |
| `iou` | Intersection over union: `TP / (TP + FP + FN)` |
| `pixel_acc` | Pixel accuracy: `(TP + TN) / N` |
| `tpr` | Sensitivity (recall) |
| `tnr` | Specificity |
| `ppv` | Precision |
| `npv` | Negative predictive value |

An image with no ground-truth pixel of a class:

- An empty prediction gives NaN `dice` and `iou`, so the image does not count for that class.
- Any predicted pixel gives `dice = iou = 0`.

This rule is not an option of MONAI `DiceMetric`. MONAI gives NaN in both cases (`ignore_empty=True`), or
1 and 0 (`ignore_empty=False`). On a dataset with many empty images (for example SIIM-ACR), compare Dice
only with results that use the same rule.

## Save and reuse

`fit()` trains one head on all images of the dataset, minus the validation part for early stopping. It
does not run the folds. `save_head()` writes only the head, so the file is small. To use it again, build
the encoder with the same backbone recipe, and the evaluator with the same head arguments. This block uses
`ev`, `encoder` and `ds` of [Quick use](#quick-use).

<!-- block: run=gpu timeout=600 -->
```python
ev.fit()
path = ev.save_head("outputs/linear_probe_seg/head.pt")

ev2 = LinearProbeSegEvaluator(encoder, dataset=ds, num_classes=2)
ev2.load_head(path)
preds = ev2.predict(ds, indices=[0, 1], return_images=True)
print(sorted(preds[0]))  # ['image', 'image_id', 'prob', 'y_pred', 'y_true']
```

| Method | What it does |
|---|---|
| `fit(dataset=None)` | Trains a head on all images of `dataset` (default: `dataset` or `train_dataset` of the evaluator), with the same early stopping. It sets `final_head_` and returns it |
| `save_head(path=None)` | Writes `final_head_` with its shape, `num_classes` and head settings. The default path is `<output_dir>/final_head.pt`. It returns the path |
| `load_head(path)` | Builds the head again from the file and loads its weights. It warns when `num_classes` or the head settings differ from the evaluator |
| `predict(dataset, *, head=None, indices=None, return_images=False)` | Runs the head (default: `final_head_`) on `dataset`, or on the rows in `indices`. It returns one dict for each image: `image_id`, `y_true` `[H, W]`, `y_pred` `[H, W]` and `prob` `[C, H, W]`. `return_images=True` adds `image`, a gray `[H, W]` image in [0, 1] at the mask size, for overlays |
| `inner_val_indices(dataset=None)` | Returns the dataset rows of the validation part that `fit()` keeps out. It needs no training, so it also works after `load_head()`. It is empty when `val_fraction=0` |

`image_id` is the `image_path` with no file ending (D-46). `store_final_model=True` runs `fit()` at the end
of `evaluate()`, so one call gives the result table and the head. The images of `inner_val_indices()` did
not train the head, but they chose the best epoch. So their predictions look a little better than the
predictions on new images.

## Arguments

<!-- args: radharmony.evaluator.LinearProbeSegEvaluator -->
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

### Shared arguments (inherited from BaseSegEvaluator)

| Argument | Type | Default | Description |
|---|---|---|---|
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

`evaluate()` returns the [result table](index.md#result-table), one row for each class and run. A run is
one fold or one seed, for each `n_train` value. Each run also gets a `macro_average` row. In fixed-split
mode, each bootstrap sample of a run gets its own rows. The columns of the Quick use run:

| Column | Meaning |
|---|---|
| `n_train` | The train size of `n_train_samples`. `-1` means all train images |
| `fold` | The fold, from 0. `-1` in fixed-split mode |
| `seed` | The seed of the train sample. `-1` in k-fold mode |
| `bootstrap` | `-1` for the point estimate. `0` to `n_bootstrap - 1` for a bootstrap sample |
| `label` | `class_0` (background), `class_1`, ..., or `macro_average` |
| `dice`, `iou`, `pixel_acc`, `tpr`, `tnr`, `ppv`, `npv` | The metrics. See [Metrics](#metrics) |

`macro_average` is the mean over the classes other than `class_0`. `save_results(df)` writes
`results.csv` and `results_summary.csv` to `output_dir`. The summary has one row for each label and
`n_train`. For each metric, it gives `<metric>_mean`, `<metric>_n`, `<metric>_std`, `<metric>_ci_lo` and
`<metric>_ci_hi` (D-19).

## Notes

- The head is `nn.Conv2d(D, num_classes, kernel_size=1)` and a bilinear resize to the mask size. For
  `D=768` and 2 classes, it has 1538 weights.
- An encoder that gives no 4-D map raises `ValueError`. A dataset with no `mask` raises `KeyError`.
- Each run seeds torch from `base_seed`, so the same `base_seed` gives the same result.
- The evaluator reads the mask size from the last dimension of the first batch. It expects square masks, as
  the backbone recipes give.
- The dataset transform is the same for the train, validation and test rows of a k-fold run. So a random
  step, for example `with_flip`, also changes the validation and test images. It draws again in each
  epoch, and the evaluator does not warn.
- For more examples with SIIM-ACR, Montgomery and Shenzhen, see
  [`notebooks/evaluator/fm_comparison_seg.ipynb`](https://github.com/f10409/RadHarmony/blob/main/notebooks/evaluator/fm_comparison_seg.ipynb).
