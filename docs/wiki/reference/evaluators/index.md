# Evaluators

An evaluator takes a RadHarmony dataset and an encoder, and gives a result table (a `pd.DataFrame`). It
is in `radharmony.evaluator`. This page lists the 11 evaluators and the facts that they share.

| Class | Registry key | Task | Page |
|---|---|---|---|
| `LinearProbeEvaluator` | `linear_probe` | Classification, frozen features | [Linear probe](linear_probe.md) |
| `KNNProbeEvaluator` | `knn_probe` | Classification, frozen features | [kNN probe](knn_probe.md) |
| `SVMProbeEvaluator` | `svm_probe` | Classification, frozen features | [SVM probe](svm_probe.md) |
| `PrototypeProbeEvaluator` | `prototype_probe` | Classification, frozen features | [Prototype probe](prototype_probe.md) |
| `ZeroShotEvaluator` | `zero_shot` | Classification with text prompts, no training | [Zero-shot](zero_shot.md) |
| `FinetuneEvaluator` | `finetune` | Classification, trains the encoder and a head, or the head only | [Fine-tune](finetune.md) |
| `LinearProbeSegEvaluator` | `linear_probe_seg` | Segmentation, frozen patch features, 1×1 convolution head | [Linear probe (segmentation)](linear_probe_seg.md) |
| `ConvProbeSegEvaluator` | `conv_probe_seg` | Segmentation, frozen patch features, small convolution head | [Conv probe (segmentation)](conv_probe_seg.md) |
| `UPerNetSegEvaluator` | `upernet_seg` | Segmentation, frozen patch features, UPerNet head | [UPerNet (segmentation)](upernet_seg.md) |
| `ReportGenerationEvaluator` | `report_generation` | Report generation, scores generated reports | [Report generation](report_generation.md) |
| `VQAEvaluator` | `vqa` | Visual question answering, scores generated answers | [Visual question answering](vqa.md) |

For a walkthrough, see
[`notebooks/tutorials/evaluator_tutorial.ipynb`](https://github.com/f10409/RadHarmony/blob/main/notebooks/tutorials/evaluator_tutorial.ipynb).

## Quick start

To run your first probe and read its result table, see [Run a probe](../../how-to/run-a-probe.md).

## Encoder contract

The evaluators do not look inside the encoder. They only call it:

| Evaluators | The encoder gets | The encoder returns |
|---|---|---|
| Classification | `imgs: Tensor[B, ...]` | a pooled feature vector, `Tensor[B, D]` |
| `ZeroShotEvaluator` (text encoder) | the output of `tokenizer`, or a `list[str]` with no `tokenizer` | a text feature, `Tensor[T, D]` |
| Segmentation | `imgs: Tensor[B, ...]` | patch features, `Tensor[B, D, H', W']` |
| `ReportGenerationEvaluator` | the `img` batch (a `list[str]` of image paths with the CheXagent, MAIRA-2 and MedGemma recipes). With `use_indication=True`, also the indications | a `list[str]` of reports, one for each image |
| `VQAEvaluator` | the `img` batch and the `list[str]` of questions | a `list[str]` of answers, one for each image |

The input shape is the output of the dataset transform. `ImageEncoderWrapper` wraps a Hugging Face, timm
or OpenCLIP model in this contract. See [Add a backbone](../../how-to/add-a-backbone.md).

## Backbone recipes

A backbone recipe is a `make_<name>()` function in `radharmony.evaluator.backbones`. It returns the
dataset transform and the encoder, and a text encoder for a vision-language model. Give the transform to
the dataset and the encoder to the evaluator. Pass `output_keys={"img", "mask"}` to the recipe. Then the
encoder gives patch features for the segmentation evaluators. See [Backbones](../backbones/index.md) for
the list of recipes.

## Split modes

`ZeroShotEvaluator`, `ReportGenerationEvaluator` and `VQAEvaluator` use only test data. The other evaluators need exactly
one of 2 split modes. Any other mix raises `ValueError`.

<!-- block: run=skip reason="needs the encoder and the datasets ds, train_ds and test_ds" -->
```python
# k-fold mode: one dataset. GroupKFold makes the folds.
ev = LinearProbeEvaluator(encoder, dataset=ds, n_folds=5)

# fixed-split mode: a train dataset and a test dataset.
ev = LinearProbeEvaluator(encoder, train_dataset=train_ds, test_dataset=test_ds, n_seeds=5, n_bootstrap=100)
```

- Patients: the folds, the validation part and the bootstrap group the images by `patient_id`. All images
  of a patient go to the same part. With no `patient_id` column, each image is its own patient. A NaN
  `patient_id` raises `ValueError` (D-32).
- K-fold mode: the folds give the variance. The evaluator ignores `n_seeds` and `n_bootstrap`, and warns
  if you set them (D-19).
- Fixed-split mode: the variance comes from `n_seeds` train samples and `n_bootstrap` bootstrap samples of
  the test set. Each bootstrap sample draws test patients with replacement and keeps all their images
  (D-32).
- `n_train_samples` (for example `[50, 100]`) gives one run for each train size. It counts images
  (D-40), and it includes the validation part (D-29). A size larger than the train data uses all rows,
  and the result keeps the size that you asked for.

## Embedding cache

The frozen-feature evaluators (the linear, kNN, SVM and prototype probes, and zero-shot) extract the
features one time, before they fit. With `embedding_cache="<path>"`, they also write the raw features to
a file. A later run with the same path reads the file, for any `labels=` list (D-41).

- Files: `<path>.pkl` in k-fold mode. `<path>.train.pkl` and `<path>.test.pkl` in fixed-split mode.
  `<path>.test.pkl` for zero-shot.
- Labels: a run always reads the labels from the dataset, not from the file.
- Checks: a file from other images, another `base_image_dir`, another encoder, transform or
  `autocast_dtype` raises `ValueError`. A file from before 2026-11-03 raises `ValueError` too. Delete the
  file to make it again.
- Random steps: use a transform with no random step, for example no `with_flip`. A random step gives one
  random draw for each image, so these evaluators warn (D-41).
- L2 norm: the file keeps raw features. The probe normalizes them after it reads them (D-17).

`FinetuneEvaluator` does not use the cache ([Fine-tune](finetune.md#arguments)).

## Result table

`evaluate()` of a classification evaluator returns one row for each label and run, with these columns:

| Column | Meaning |
|---|---|
| `n_train` | The train size of `n_train_samples`. `-1` when `n_train_samples=None` (all train rows), and in zero-shot |
| `fold` | The fold, from 0. `-1` in fixed-split mode and in zero-shot |
| `seed` | The seed of the train sample, `base_seed` plus the seed index. `-1` in k-fold mode and in zero-shot |
| `bootstrap` | `-1` for the point estimate. `0` to `n_bootstrap - 1` for a bootstrap sample |
| `label` | The label name, or `macro_average` |
| `auroc`, `auprc` | They use no threshold |
| `f1`, `accuracy`, `balanced_accuracy`, `tpr`, `tnr`, `ppv`, `npv`, `mcc` | They use the threshold. See [Threshold](#threshold) |
| `threshold` | The threshold of the label |

Each run also gets a `macro_average` row, also for a dataset with 1 label. It is the mean of the label
rows, and it skips NaN values. So a fixed-split run has `(labels + 1) * n_seeds * (1 + n_bootstrap)` rows
for each train size.

A label with fewer than 2 positives or 2 negatives in the test rows of a run gets NaN in all metrics,
also in AUROC. Rows with a NaN label are not scored.

`save_results(df)` writes 2 files to `output_dir`:

- `results.csv`: the result table.
- `results_summary.csv`: one row for each label and `n_train`. For each metric, it gives `<metric>_mean`
  (the mean of the point estimates) and `<metric>_n` (the number of runs with a value). It also gives
  `<metric>_std`, `<metric>_ci_lo` and `<metric>_ci_hi` (the 95% CI) over the bootstrap rows. With no
  bootstrap rows, they come from the folds or the seeds. With fewer than 2 values, they are NaN (D-19).
  `<metric>_std` is the population std (`np.nanstd`, which divides by n), as scikit-learn gives. The sample
  std (pandas `.std()`) is a little bigger: over 5 folds, about 12%.

When a metric of a label is NaN in some runs, for example the F1 of a rare label, `save_results()` warns one
time.

### Threshold

`threshold_strategy` is `"youden"` (the default), `"f1"` or `"fixed:<float>"`. The evaluators that train
(the linear, kNN, SVM and prototype probes, and fine-tune) pick the threshold on validation data, not on
the test labels (D-29):

1. The probe holds out `val_fraction` (default `0.1`) of the train patients. It does not fit on this
   validation part.
2. The probe scores the validation part and picks one threshold for each label with `threshold_strategy`.
3. It uses this threshold on the test set and on each bootstrap sample.

So `n_train` counts the validation part too. The threshold metrics are NaN when the validation part has
fewer than 2 positives or 2 negatives of a label, or with `val_fraction=0.0`. `"fixed:<float>"` needs no
validation data. AUROC and AUPRC use no threshold.

`ZeroShotEvaluator` has no train data, so its threshold metrics are NaN, except with
`threshold_strategy="fixed:<float>"` (see [Zero-shot](zero_shot.md#how-it-works)).

## Registry

Each evaluator has a registry key. `resolve_evaluator` gives the class of a key. To add your own
evaluator, register it:

<!-- block: run=ci -->
```python
from radharmony.evaluator import BaseClsEvaluator, list_evaluators, register_evaluator, resolve_evaluator


@register_evaluator("my_probe")
class MyProbeEvaluator(BaseClsEvaluator):
    def evaluate(self):
        ...


print("my_probe" in list_evaluators())  # True
print(resolve_evaluator("linear_probe").__name__)  # LinearProbeEvaluator
```
