# Run a probe

A probe scores the frozen features of an encoder on the labels of a dataset. This page runs the linear probe
on [Montgomery County CXR](../reference/datasets/cxr-boxes-masks/montgomery_cxr.md) (138 images, 1 label:
`tuberculosis`) with RAD-DINO. Then it runs the kNN, SVM and prototype probes and zero-shot on the same
features. [Your first benchmark](../get-started/first-benchmark.md) explains the same linear probe run step by
step.

Before you start, install the `raddino` and `biomed` extras, and
[get the Montgomery data](get-the-data.md#montgomery-county-cxr).

## 1. Build the encoder and the dataset

The backbone recipe gives the transform and the encoder. Give the transform to the dataset.
`output_cls=True` adds the labels to each sample.

<!-- block: run=gpu timeout=600 -->
```python
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator.backbones import make_raddino

transform, encoder = make_raddino(device="cuda")
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_cls=True,
    cache_dir=None,
)
print(len(ds.get_harmonized_df()), ds.LABEL_COLS)  # 138 ['tuberculosis']
```

## 2. Run the linear probe

`n_folds=5` gives 5 folds, grouped by patient. `embedding_cache` writes the features to a file, so the
next probes do not run the encoder again.

<!-- block: run=gpu timeout=600 -->
```python
from radharmony.evaluator import LinearProbeEvaluator

ev = LinearProbeEvaluator(
    encoder,
    dataset=ds,
    n_folds=5,
    embedding_cache="outputs/cache/montgomery_raddino",
    output_dir="outputs/linear_probe",
)
df = ev.evaluate()
ev.save_results(df)  # writes results.csv and results_summary.csv
print(df.shape)  # (10, 16)
```

## 3. Read the result table

The table has one row for each label and fold, and a `macro_average` row for each fold. Montgomery has 1
label, so 5 folds give 10 rows. `auroc` and `auprc` use no threshold. `f1` and the other threshold metrics
use the threshold of the validation part ([Threshold](../reference/evaluators/index.md#threshold)).

<!-- block: run=gpu timeout=600 -->
```python
macro = df[df["label"] == "macro_average"]
print(macro[["fold", "auroc", "f1"]].round(3))
#    fold  auroc     f1
# 5     0  1.000  1.000
# 6     1  0.974  0.909
# 7     2  0.995  0.957
# 8     3  1.000  1.000
# 9     4  0.886  0.533
print(round(macro["auroc"].mean(), 3))  # 0.971, the mean AUROC of the 5 folds
```

`results_summary.csv` gives the mean, the std and the 95% CI of each metric over the folds. See
[Your first benchmark: Read the summary](../get-started/first-benchmark.md#6-read-the-summary) and
[Result table](../reference/evaluators/index.md#result-table).

## 4. Switch to another probe

The kNN, SVM and prototype probes take the same arguments. With the same `embedding_cache` path, they
read the features from the file.

<!-- block: run=gpu timeout=600 -->
```python
from radharmony.evaluator import KNNProbeEvaluator, PrototypeProbeEvaluator, SVMProbeEvaluator

for probe in (KNNProbeEvaluator, SVMProbeEvaluator, PrototypeProbeEvaluator):
    ev = probe(encoder, dataset=ds, n_folds=5, embedding_cache="outputs/cache/montgomery_raddino")
    df = ev.evaluate()
    macro = df[df["label"] == "macro_average"]
    print(probe.__name__, round(macro["auroc"].mean(), 3))
# KNNProbeEvaluator 0.968
# SVMProbeEvaluator 0.983
# PrototypeProbeEvaluator 0.959
```

Each probe has its own arguments, for example `k` of the kNN probe. See the
[evaluator list](../reference/evaluators/index.md).

## 5. Run zero-shot

Zero-shot needs a text-and-image recipe and one text prompt for each label. It does not train, so it uses
`test_dataset=` and no folds. Give it its own cache path, named after the recipe. A cache file from a
different encoder raises `ValueError`.

<!-- block: run=gpu timeout=600 -->
```python
from radharmony.evaluator import ZeroShotEvaluator
from radharmony.evaluator.backbones import make_biomed_clip

transform, encoder, text_encoder, tokenizer = make_biomed_clip(device="cuda")
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_cls=True,
    cache_dir=None,
)
ev = ZeroShotEvaluator(
    encoder,
    text_encoder,
    prompts={"tuberculosis": ["a chest X-ray with tuberculosis"]},
    negative_prompts={"tuberculosis": ["a normal chest X-ray"]},
    tokenizer=tokenizer,
    test_dataset=ds,
    threshold_strategy="fixed:0",  # without it, F1 and the other threshold metrics are NaN
    embedding_cache="outputs/cache/montgomery_biomed_clip",
)
df = ev.evaluate()
print(df.loc[df["label"] == "macro_average", "auroc"].round(3).tolist())  # [0.9]
```

Zero-shot has no validation part to pick a threshold. With negative prompts, `"fixed:0"` takes a score
above 0 as positive ([Zero-shot](../reference/evaluators/zero_shot.md)).

## Check

- `outputs/linear_probe/` has `results.csv` and `results_summary.csv`.
- `outputs/cache/` has `montgomery_raddino.pkl` and `montgomery_biomed_clip.test.pkl`. Zero-shot adds
  `.test` to the name, because it uses `test_dataset=`.
- The AUROC of each probe is above 0.5. Montgomery has only 138 images, so the AUROC changes a lot from
  one fold to the next.

## See also

- [Evaluators](../reference/evaluators/index.md): the split modes, the embedding cache and the result table.
- [Linear probe](../reference/evaluators/linear_probe.md) and the pages of the other probes: all arguments.
- [Compare backbones](compare-backbones.md): run one probe with 2 or more backbone recipes.
- [Run segmentation](run-segmentation.md): the probes for masks.
