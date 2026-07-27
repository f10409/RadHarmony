# Datathon26

Event-specific datasets and evaluators for the **datathon26** chest-X-ray
benchmark. In this event, model inference does **not** run locally — studies are
submitted to an external *reportbench* service, which returns pre-computed
results. The service runs two tasks — **embed** (per-image feature `.npz`) and
**report** (per-study text) — and the `radharmony.datathon26` submodule turns
those into three scoring tasks: **classification** and **segmentation** (both
over the embed features) and **report generation**. It replaces the model
forward pass with the returned results — datasets serve the embeddings / patch
maps / reports in place of image tensors, and evaluators score them by reusing
the standard RadHarmony probe, segmentation, and generative machinery unchanged.

Importing the package registers everything under `datathon26_*` keys:

```python
import radharmony.datathon26  # registers datasets + evaluators
```

A runnable, self-contained walkthrough of all paths lives in
[datathon26/demo_datathon_evaluators.ipynb](../../datathon26/demo_datathon_evaluators.ipynb):
it fabricates a tiny batch table and result files so it runs end-to-end without
the service — classification (§3), report generation (§4), and segmentation
(§4b) — then shows how to point the same code at a real run (§5) and how to
submit one yourself (§6).

## Workflow

```
batch preparation           reportbench service              radharmony.datathon26
─────────────────           ───────────────────              ─────────────────────
sample + anonymize   ──►  submit a run              ──►  point a dataset at the
6 CXR datasets            (embed / report task)          results dir + batch CSV
                                                                    │
dataset_{DS1..DS6}.csv    writes under                             ▼
(labels + reports)        _reportbench_out/<model>/         evaluate() → metrics
mapping.csv (private)       <study>/<model>_<stem>.npz
                            <study>/<model>_report.txt
```

1. An internal batch-preparation step samples and anonymizes six harmonized CXR
   datasets (DS1=MIMIC-CXR, DS2=ReXGradient-160K, DS3=VinDr-CXR, DS4=EmoryCXR v2,
   DS5=PadChest, DS6=SIIM-ACR-PTX) into per-study submission folders
   (`DS1_patient_00001_study_00001/`), and writes the anonymized batch tables
   `dataset_{DS1..DS6}.csv` (one row per study, carrying the one-hot finding
   **labels** and the free-text **report** ground truth) plus a private
   `mapping.csv` de-anonymization key (never uploaded). DS6 (SIIM-ACR-PTX) also
   carries a pneumothorax `mask_path`, so it is the dataset the segmentation task
   scores.
2. The submission folder is sent to the reportbench service, which runs one of
   two tasks: **embed** or **report**. (The **embed** task submits without a
   manifest; the **report** task attaches a study-keyed `manifest.json` carrying
   the per-study **indication** when an `indication_col` is given.)
3. The service writes results under `<data_dir>/<run>/_reportbench_out/<model>/`:
      - **embed** task → `<study>/<model>_<stem>.npz` **per image** — an archive
        with `global` `[dim]` (the pooled image vector), `patches`
        `[n_patches, dim]`, and `grid` `(H, W)`.
      - **report** task → `<study>/<model>_report.txt` **per study** — the
        generated free-text report.
4. A datathon26 dataset joins those results to the batch CSV; a datathon26
   evaluator scores them. The one embed/report output feeds all three scoring
   tasks: **classification** reads the `.npz` `global` vector, **segmentation**
   folds its `patches` into a dense map, and **report generation** reads the
   `.txt`. Segmentation additionally requires a ground-truth mask per image
   (bring your own `mask_path`), so it applies only to datasets carrying masks.

## Result-file layout

The **real** reportbench service writes results under a per-model subfolder,
with one embedding `.npz` **per image** and one report **per study**:

```
<results_dir>/<model>/                 # e.g. .../_reportbench_out/model-a/
  DS1_patient_00018_study_00001/
    model-a_pa.npz       # embed task — one <model>_<stem>.npz per image
    model-a_lateral.npz  #   archive: global [dim] + patches [n, dim] + grid (H, W)
    model-a_report.txt   # report task — one <model>_report.txt per study
  ...
```

Point the dataset at the `<results_dir>/<model>` folder and pass a `result_path`
resolver — the `viewwise_embedding_path(model)` (defaults to `.npz`) /
`perstudy_report_path(model)` helpers build the names above. (`_load_npy` also
accepts a bare `.npy` array, and the datasets **default** to the simpler
`<study>/embedding.npy` / `<study>/report.txt` when no resolver is given, which is
what the self-contained notebook demo fabricates.)

The classification/segmentation datasets read from the same `.npz`:
classification takes the `global` vector, segmentation folds `patches` back into
a dense `[dim, H, W]` map (using the stored `grid`, falling back to a square
`√n_patches` grid when absent). Because embeddings are per-image, the
classification/segmentation frame carries **one row per image** (the stem is read
from each row's `image_path`); the report frame is deduplicated to **one row per
study**.

## Datasets

| Registry key | Class | Serves | Key sample fields |
|---|---|---|---|
| `datathon26_embedding` | `EmbeddingResultsDataset` | `.npz` `global` (or `result_path`) | `img` = embedding vector, `cls` = label vector |
| `datathon26_segmentation` | `PatchSegResultsDataset` | `.npz` `patches` → `[D, H, W]` (or `result_path`) | `img` = dense feature map, `mask` = GT mask |
| `datathon26_report` | `ReportResultsDataset` | `report.txt` (or `result_path`) | `img` = predicted report, `report` = reference report |

(`DatathonEmbeddingDataset` / `DatathonReportDataset` are the one-shot *factories*
that submit a job and return an `EmbeddingResultsDataset` / `ReportResultsDataset`
— see "raw studies → dataset" below.)

`EmbeddingResultsDataset` auto-detects label columns as the binary (0/1)
numeric columns in the batch frame, so free-text and demographic columns are
never swept into the `cls` vector. The anonymized `patient_id`
(`DS1_patient_00001`) is preserved so patient-grouped k-fold never splits one
patient's studies across folds.

## Evaluators

All are thin subclasses of the standard evaluators, so **every feature of the
originals is preserved** — patient-grouped k-fold, multi-seed, bootstrap,
threshold strategies, `macro_average`, the embedding cache, (for segmentation)
early stopping + prediction dumps, and (for reports) the full RadEval metric
suite + `generations.csv`.

| Registry key | Class | Base | Scores |
|---|---|---|---|
| `datathon26_linear_probe` | `DatathonLinearProbeEvaluator` | `LinearProbeEvaluator` | embeddings → per-label logistic regression |
| `datathon26_knn_probe` | `DatathonKNNProbeEvaluator` | `KNNProbeEvaluator` | embeddings → k-NN |
| `datathon26_svm_probe` | `DatathonSVMProbeEvaluator` | `SVMProbeEvaluator` | embeddings → SVM |
| `datathon26_prototype_probe` | `DatathonPrototypeProbeEvaluator` | `PrototypeProbeEvaluator` | embeddings → nearest-centroid |
| `datathon26_linear_probe_seg` | `DatathonLinearProbeSegEvaluator` | `LinearProbeSegEvaluator` | patch map → 1×1-conv head → Dice/IoU |
| `datathon26_conv_probe_seg` | `DatathonConvProbeSegEvaluator` | `ConvProbeSegEvaluator` | patch map → conv-block head → Dice/IoU |
| `datathon26_upernet_seg` | `DatathonUPerNetSegEvaluator` | `UPerNetSegEvaluator` | patch map → UPerNet head → Dice/IoU |
| `datathon26_report_generation` | `DatathonReportGenerationEvaluator` | `ReportGenerationEvaluator` | predicted vs reference report → RadEval |

The classification and segmentation evaluators default their `image_encoder` to
an `IdentityEncoder` (pass-through), so the sole encoder call returns the
pre-extracted vectors / patch maps unchanged — the segmentation head (1×1 conv /
conv block / UPerNet) trains straight on the frozen patch features. The report
evaluator uses an `identity_generator` (pass-through) and reads the predicted
report from the dataset instead of calling a VLM.

## Usage

### Classification (embeddings → probe)

```python
import radharmony.datathon26  # noqa: F401  (registers datathon26_*)
from radharmony.datathon26 import (
    EmbeddingResultsDataset, viewwise_embedding_path, DatathonLinearProbeEvaluator,
)

ds = EmbeddingResultsDataset(
    embeddings_dir="/path/to/_reportbench_out/model-a",  # <study>/model-a_<stem>.npz
    csv_path="dataset_DS1.csv",                          # one row per image (labels)
    result_path=viewwise_embedding_path("model-a"),
)

ev = DatathonLinearProbeEvaluator(dataset=ds, n_folds=5)
results = ev.evaluate()   # DataFrame: per-label + macro_average rows (auroc, auprc, f1, ...)
```

Swap in `DatathonKNNProbeEvaluator`, `DatathonSVMProbeEvaluator`, or
`DatathonPrototypeProbeEvaluator` — same call shape.

### Segmentation (patch features → Dice/IoU)

```python
import radharmony.datathon26  # noqa: F401
from radharmony.datathon26 import (
    PatchSegResultsDataset, viewwise_embedding_path, DatathonLinearProbeSegEvaluator,
)

ds = PatchSegResultsDataset(
    embeddings_dir="/path/to/_reportbench_out/model-a",  # <study>/model-a_<stem>.npz
    csv_path="dataset_DS1.csv",                          # one row per image
    result_path=viewwise_embedding_path("model-a"),      # reads the .npz `patches`
    mask_col="mask_path",                                # GT mask per image
    mask_size=224,                                       # output resolution
)

ev = DatathonLinearProbeSegEvaluator(dataset=ds, num_classes=2, n_folds=5)
results = ev.evaluate()   # DataFrame: per-class + macro_average rows (dice, iou, ...)
```

`PatchSegResultsDataset` folds the `.npz` `patches` `[n, dim]` into a dense
`[dim, H, W]` map (via the stored `grid`), and the evaluator puts a frozen-feature
segmentation head on it. Swap in `DatathonConvProbeSegEvaluator` or
`DatathonUPerNetSegEvaluator` for heavier heads — same call shape. Pass
`feat_size=` if the patch grid varies per image.

### Report generation (reference + predicted → RadEval)

```python
import radharmony.datathon26  # noqa: F401
from radharmony.datathon26 import (
    ReportResultsDataset, perstudy_report_path, DatathonReportGenerationEvaluator,
)
from radharmony.evaluator.metrics.language import LIGHT_METRICS

ds = ReportResultsDataset(
    results_dir="/path/to/_reportbench_out/model-a",  # <study>/model-a_report.txt
    csv_path="dataset_DS1.csv",                       # one row per study; 'report' = reference
    result_path=perstudy_report_path("model-a"),
)

ev = DatathonReportGenerationEvaluator(
    dataset=ds,
    metrics=LIGHT_METRICS,               # or the default full 16-metric suite
    ref_section="findings",              # both reference and prediction parsed to this section
)
results = ev.evaluate()                  # DataFrame with the RadEval metric panel
```

Both the reference and the predicted report are parsed to the same
`ref_section` so the comparison is symmetric (findings-vs-findings by default).
Studies with a blank reference or empty generation are dropped before scoring.

## One-shot: raw studies → dataset (`ReportBenchClient`)

The usage above assumes you have already run the studies through the service and
have a results folder. The `DatathonEmbeddingDataset` / `DatathonReportDataset`
factories (+ `ReportBenchClient`) do that submission step for you, so you can go
straight from a harmonized frame and the raw image root — **the same two inputs
an ordinary RadHarmony dataset takes** — to a ready dataset object. They wrap the
`rbclient.py` CLI (`config` → `prepare` → `submit` → `watch`, with an optional
`check`) via subprocess.

```python
import radharmony.datathon26  # noqa: F401
from radharmony.datathon26 import ReportBenchClient, DatathonEmbeddingDataset

client = ReportBenchClient(
    rbclient_path="/path/to/datathon/skill/rbclient.py",
    data_dir="/path/to/datathon/data_deposition/<team_id>",
    api_key="rb_...",                    # from your team INSTRUCTIONS.md
)

ds = DatathonEmbeddingDataset(
    harmonized_df=df,                    # one row per image (labels)
    base_image_dir="/data/mimic",        # where the raw images live
    client=client,
    model="model-a",                     # see client.models()
)
# ds is an EmbeddingResultsDataset pointed at
# <data_dir>/run_embed/_reportbench_out/model-a/<study>/model-a_<stem>.npz
```

`DatathonEmbeddingDataset` here is a **factory function**, not the result class:
it (1) stages every study referenced by `harmonized_df` into the per-study folder
layout the service expects (`<study_id>/<view files>`, grouping multiple views per
study, symlinked by default), (2) drives `rbclient.py` to submit the *embed*
task and `watch` it to completion, then (3) returns an
`EmbeddingResultsDataset` pointed at the `_reportbench_out/<model>` results after
verifying every image produced its `.npz` (it raises with the missing ids
otherwise). `DatathonReportDataset(harmonized_df, base_image_dir, client=..., model=...)`
is the report-task counterpart — same call shape, submitting the *report* task
and returning a `ReportResultsDataset`; pass `indication_col=` to attach a
per-study indication via the job's study-keyed `manifest.json`.

Pass `skip_submit=True` with an existing `results_dir=` to re-score a finished
run without resubmitting. Every path (client script, interpreter, staging /
results dirs), the run name / task, the `.npz` key + extension (`emb_key`,
`emb_ext`), and the column names (`study_col`, `image_col`, `view_col`) are
configurable.
