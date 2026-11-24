# FinetuneEvaluator

| | |
|---|---|
| Registry key | `finetune` |
| Task | Classification, trains the encoder and a head, or the head only |
| Dataset outputs it needs | `cls` (`output_cls=True`) |
| Split modes | k-fold (`dataset=`), fixed split (`train_dataset=`, `test_dataset=`) |
| Encoder it needs | a pooled feature vector, `Tensor[B, D]`. It must be an `nn.Module`, also with `freeze_backbone=True` |
| Output | The result table, one row for each label and run. See [Output](#output) |

The fine-tune evaluator trains a classification head on the encoder, and by default the encoder too. It
uses one GPU, AdamW, a cosine learning-rate schedule and early stopping. It has the same split modes and
result table as the probes, so you can compare the results. It has no multi-GPU training and no
experiment logger.

## Quick use

This block fine-tunes RAD-DINO on [Montgomery](../datasets/cxr-boxes-masks/montgomery_cxr.md) (138 images),
with 3 folds, 3 epochs and a batch of 8 images.

<!-- block: run=gpu timeout=600 -->
```python
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator import FinetuneEvaluator
from radharmony.evaluator.backbones import make_raddino

transform, encoder = make_raddino(device="cuda")
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_cls=True,
    cache_dir=None,
)

ev = FinetuneEvaluator(encoder, dataset=ds, n_folds=3, epochs=3, batch_size=8, output_dir="outputs/finetune")
df = ev.evaluate()
ev.save_results(df)  # writes results.csv and results_summary.csv
print(df.columns.tolist())
# ['n_train', 'fold', 'seed', 'bootstrap', 'label', 'auroc', 'auprc', 'f1', ...]
```

## How it works

- Model: the head is `nn.Linear(D, number of labels)`, or a copy of `head`. The evaluator finds `D` from
  one batch.
- Encoder: with `freeze_backbone=False`, each fold or seed trains its own copy of the encoder, so the runs
  do not share weights. The encoder learning rate is `lr * backbone_lr_scale`. With
  `freeze_backbone=True`, only the head trains.
- Training: AdamW with `weight_decay`, and a cosine schedule over `epochs`. The loss is one sigmoid
  cross-entropy for each label (`loss="bce"`), or a softmax cross-entropy over the labels (`loss="ce"`).
- Labels: a NaN label adds no loss (D-18). With `loss="ce"`, the evaluator drops a row with a NaN label.
- Early stopping: the evaluator holds out a validation part of the train patients, and the model does not
  train on it. After each epoch, the evaluator scores this part with `early_stop_metric`. It takes the mean
  over the labels with 2 or more positives and 2 or more negatives in the validation part. It stops
  after `early_stop_patience` epochs with no better value, and keeps the weights of the best epoch. An
  epoch with no value, for example with no positive in the validation part, does not count.
- Threshold: after training, the evaluator picks the threshold on the same validation part
  ([Threshold](index.md#threshold), D-29).
- Bootstrap: the bootstrap samples reuse the test predictions of the trained model. They do not train
  again.

## Save and reuse

`fit()` trains one model on all train data, with no benchmark. `save_model()` writes it, and `predict()` runs
it. This block uses `encoder` and `ds` of [Quick use](#quick-use).

<!-- block: run=gpu timeout=600 -->
```python
ev = FinetuneEvaluator(encoder, dataset=ds, epochs=3, batch_size=8, freeze_backbone=True)
ev.fit()  # trains on all images, minus a validation part for early stopping
path = ev.save_model("outputs/finetune/final_model.pt")

# Later: make the evaluator with the same encoder, head= and freeze_backbone=.
ev2 = FinetuneEvaluator(encoder, dataset=ds, freeze_backbone=True)
ev2.load_model(path)
preds = ev2.predict(ds)
print(sorted(preds[0]))  # ['image_id', 'prob', 'y_true']
```

| Method | What it does |
|---|---|
| `fit(dataset=None)` | Trains a model on all of `dataset` (default: `dataset` or `train_dataset` of the evaluator), with the same early stopping as `evaluate()`. It sets `final_encoder_` and `final_head_`, and returns the head |
| `save_model(path=None)` | Writes the head weights, the labels, `freeze_backbone`, `l2_normalize` and `loss`. It writes the encoder weights only with `freeze_backbone=False`. The default path is `<output_dir>/final_model.pt`. It returns the path |
| `load_model(path)` | Builds the head again from the file and loads the weights. It warns when `freeze_backbone` differs from the file. It takes `l2_normalize` and `loss` from the file, with a warning when they differ |
| `predict(dataset, *, encoder=None, head=None, indices=None)` | Runs the model over `dataset`. It returns one dict for each image, with `image_id`, `y_true` and `prob`. `indices` limits the run to some rows |

- `image_id` is the `image_path` with no file ending (D-46).
- `prob` is a sigmoid for each label, or a softmax over the labels with `loss="ce"`.
- `predict()` takes the label columns of the dataset by the names in `final_labels_`. So the dataset can
  have more labels. A missing label raises `ValueError`.
- `store_final_model=True` runs `fit()` at the end of `evaluate()`, on `dataset` or `train_dataset`.

## Arguments

<!-- args: radharmony.evaluator.FinetuneEvaluator -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `n_folds` | `int` | `5` | The number of folds in k-fold mode. Fixed-split mode does not use it |
| `n_train_samples` | `list[int]` | `None` | The train sizes in images, one run for each ([Split modes](index.md#split-modes)). `None` uses all train images |
| `epochs` | `int` | `20` | The largest number of epochs of one run |
| `lr` | `float` | `0.0001` | The learning rate of the head |
| `backbone_lr_scale` | `float` | `0.1` | With `freeze_backbone=False`, the encoder learning rate is `lr` times this value |
| `weight_decay` | `float` | `0.05` | The AdamW weight decay |
| `freeze_backbone` | `bool` | `False` | `True` trains only the head |
| `head` | `nn.Module` | `None` | Your own head. Each run trains a copy. `None` gives `nn.Linear(D, number of labels)` |
| `loss` | `str` | `"bce"` | `"bce"`: one sigmoid for each label. `"ce"`: a softmax over 2 or more label columns, for one class for each image. `"ce"` with 1 label column raises `ValueError` (D-47) |
| `early_stop_metric` | `str` | `"auroc"` | The metric of the validation part for early stopping, for example `"auroc"` or `"auprc"` |
| `early_stop_patience` | `int` | `3` | The number of epochs with no better value before the run stops |
| `val_fraction` | `float` | `0.1` | The part of the train patients for early stopping and the threshold. The model does not train on it. `0.0` gives no early stopping, and NaN threshold metrics except with `"fixed:<float>"` |
| `store_final_model` | `bool` | `False` | After `evaluate()`, also run `fit()` ([Save and reuse](#save-and-reuse)) |

### Shared arguments (inherited from BaseClsEvaluator)

| Argument | Type | Default | Description |
|---|---|---|---|
| `image_encoder` | `nn.Module` | required | The encoder. It returns `Tensor[B, D]` ([Encoder contract](index.md#encoder-contract)). Another type raises `TypeError` at once (D-52) |
| `dataset` | dataset | `None` | K-fold mode: the one dataset ([Split modes](index.md#split-modes)) |
| `train_dataset` | dataset | `None` | Fixed-split mode: the train dataset |
| `test_dataset` | dataset | `None` | Fixed-split mode: the test dataset |
| `labels` | `list[str]` | `None` | The labels to train and evaluate. `None` gives all labels. The match ignores case. An unknown name raises `KeyError` |
| `device` | `str` | `"cuda"` | `"cuda"`, `"cuda:N"` or `"cpu"` |
| `batch_size` | `int` | `64` | The batch size of training and of prediction |
| `num_workers` | `int` | `4` | The number of DataLoader workers |
| `autocast_dtype` | `torch.dtype` | `torch.bfloat16` | The autocast type of the encoder, also in training. `None` gives no autocast |
| `embedding_cache` | `str` | `None` | Not used. This class runs the encoder on each batch |
| `l2_normalize` | `bool` | `False` | L2-normalize the features before the head. With `freeze_backbone=True`, this class sets `True` when you do not pass it (D-17) |
| `output_dir` | `str` | `None` | The folder for `save_results()` and `save_model()` |
| `n_seeds` | `int` | `1` | Fixed-split mode: the number of train samples, each with its own seed ([Split modes](index.md#split-modes)) |
| `base_seed` | `int` | `0` | The first seed of the train samples, the validation part and the training |
| `n_bootstrap` | `int` | `0` | Fixed-split mode: the number of bootstrap samples of the test set ([Split modes](index.md#split-modes)) |
| `bootstrap_seed` | `int` | `0` | The first seed of the bootstrap samples |
| `threshold_strategy` | `str` | `"youden"` | `"youden"`, `"f1"` or `"fixed:<float>"` ([Threshold](index.md#threshold)) |

## Output

`evaluate()` returns the [result table](index.md#result-table) with these columns: `n_train`, `fold`,
`seed`, `bootstrap`, `label`, `auroc`, `auprc`, `f1`, `accuracy`, `balanced_accuracy`, `tpr`, `tnr`,
`ppv`, `npv`, `mcc` and `threshold`. `save_results(df)` writes `results.csv` and `results_summary.csv` to
`output_dir`.

With `store_final_model=True`, the evaluator also sets `final_encoder_`, `final_head_` and
`final_labels_` ([Save and reuse](#save-and-reuse)).

## Notes

- Use `loss="bce"` for a dataset with 1 label column, for example Montgomery or Shenzhen TB. Use
  `loss="ce"` only when each image has exactly 1 of 2 or more label columns.
- The dataset transform is the same for the train, validation and test rows of a k-fold run. So a random
  step, for example `with_flip`, also changes the validation and test images. It draws again in each
  epoch, and the evaluator does not warn.
- With `freeze_backbone=True`, the fine-tune evaluator still runs the encoder on each batch of each
  epoch. A [linear probe](linear_probe.md) runs it only one time.
