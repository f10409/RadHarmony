# Datathon26

Event-specific datasets and evaluators for the **datathon26** chest-X-ray
benchmark. In this event, model inference does **not** run locally — studies are
submitted to an external *reportbench* service, which returns per-study results.
The `radharmony.datathon26` submodule replaces the model forward pass with those
pre-computed results: datasets serve the returned embeddings / reports in place
of image tensors, and evaluators score them by reusing the standard RadHarmony
probe and generative machinery unchanged.

Importing the package registers everything under `datathon26_*` keys:

```python
import radharmony.datathon26  # registers datasets + evaluators
```

A runnable, self-contained walkthrough of both paths lives in
[datathon26/demo_datathon_evaluators.ipynb](../../datathon26/demo_datathon_evaluators.ipynb):
it fabricates a tiny batch table and result files so it runs end-to-end without
the service (§1–4), then shows how to point the same code at a real run (§5) and
how to submit one yourself (§6).

## Workflow

```
sample_data.ipynb           reportbench service              radharmony.datathon26
─────────────────           ───────────────────              ─────────────────────
sample + anonymize   ──►  submit run1/ (per-study    ──►  point a dataset at the
5 CXR datasets            image folders)                  results dir + batch CSV
                                                                    │
dataset_{A..E}.csv        writes per study:                        ▼
(labels + reports)          <study>/embedding.npy           evaluate() → metrics
mapping.csv (private)       <study>/report.txt
```

1. `datathon26/sample_data.ipynb` samples and anonymizes the five harmonized CXR
   datasets (A=MIMIC, B=ReXGradient, C=VinDr, D=Emory, E=PadChest) into per-study
   submission folders (`A_patient_00001_study_00001/`), and writes the anonymized
   batch tables `dataset_{A..E}.csv` (one row per study, carrying the one-hot
   finding **labels** and the free-text **report** ground truth) plus a private
   `mapping.csv` de-anonymization key (never uploaded).
2. The submission folder is sent to the reportbench service.
3. The service writes results **per study**, inside each study folder:
      - `embeddings` task → `<study>/embedding.npy` — a NumPy feature vector.
      - `report` task → `<study>/report.txt` — the generated free-text report.
4. A datathon26 dataset joins those results to the batch CSV; a datathon26
   evaluator scores them.

Segmentation is not part of this event (the sampled datasets carry no masks).

## Result-file layout

The **real** reportbench service writes results under a per-model subfolder,
with one embedding **per view** and one report **per study**:

```
<results_dir>/<model>/                 # e.g. .../_reportbench_out/model-a/
  A_patient_00018_study_00001/
    model-a_pa.npy       # embeddings task — one <model>_<view>.npy per view
    model-a_lateral.npy
    model-a_report.txt   # report task — one <model>_report.txt per study
  ...
```

Point the dataset at the `<results_dir>/<model>` folder and pass a `result_path`
resolver — the `viewwise_embedding_path(model)` / `perstudy_report_path(model)`
helpers build the names above. (The datasets **default** to the simpler
`<study>/embedding.npy` / `<study>/report.txt` when no resolver is given, which is
what the self-contained notebook demo fabricates.)

Because embeddings are per-view, the classification frame carries **one row per
image** (the view is read from each row's `image_path` stem); the report frame is
deduplicated to **one row per study**.

## Datasets

| Registry key | Class | Serves | Key sample fields |
|---|---|---|---|
| `datathon26_embedding` | `EmbeddingResultsDataset` | `embedding.npy` (or `result_path`) | `img` = embedding vector, `cls` = label vector |
| `datathon26_report` | `ReportResultsDataset` | `report.txt` (or `result_path`) | `img` = predicted report, `report` = reference report |

(`DatathonEmbeddingDataset` / `DatathonReportDataset` are the one-shot *factories*
that submit a job and return an `EmbeddingResultsDataset` / `ReportResultsDataset`
— see "raw studies → dataset" below.)

`EmbeddingResultsDataset` auto-detects label columns as the binary (0/1)
numeric columns in the batch frame, so free-text and demographic columns are
never swept into the `cls` vector. The anonymized `patient_id`
(`A_patient_00001`) is preserved so patient-grouped k-fold never splits one
patient's studies across folds.

## Evaluators

All are thin subclasses of the standard evaluators, so **every feature of the
originals is preserved** — patient-grouped k-fold, multi-seed, bootstrap,
threshold strategies, `macro_average`, the embedding cache, and (for reports)
the full RadEval metric suite + `generations.csv`.

| Registry key | Class | Base | Scores |
|---|---|---|---|
| `datathon26_linear_probe` | `DatathonLinearProbeEvaluator` | `LinearProbeEvaluator` | embeddings → per-label logistic regression |
| `datathon26_knn_probe` | `DatathonKNNProbeEvaluator` | `KNNProbeEvaluator` | embeddings → k-NN |
| `datathon26_svm_probe` | `DatathonSVMProbeEvaluator` | `SVMProbeEvaluator` | embeddings → SVM |
| `datathon26_prototype_probe` | `DatathonPrototypeProbeEvaluator` | `PrototypeProbeEvaluator` | embeddings → nearest-centroid |
| `datathon26_report_generation` | `DatathonReportGenerationEvaluator` | `ReportGenerationEvaluator` | predicted vs reference report → RadEval |

The classification evaluators default their `image_encoder` to an
`IdentityEncoder` (pass-through), so the sole encoder call in
`extract_embeddings` returns the pre-extracted vectors. The report evaluator
uses an `identity_generator` (pass-through) and reads the predicted report from
the dataset instead of calling a VLM.

## Usage

### Classification (embeddings → probe)

```python
import radharmony.datathon26  # noqa: F401  (registers datathon26_*)
from radharmony.datathon26 import (
    EmbeddingResultsDataset, viewwise_embedding_path, DatathonLinearProbeEvaluator,
)

ds = EmbeddingResultsDataset(
    embeddings_dir="/path/to/_reportbench_out/model-a",  # <study>/model-a_<view>.npy
    csv_path="dataset_A.csv",                            # one row per image (labels)
    result_path=viewwise_embedding_path("model-a"),
)

ev = DatathonLinearProbeEvaluator(dataset=ds, n_folds=5)
results = ev.evaluate()   # DataFrame: per-label + macro_average rows (auroc, auprc, f1, ...)
```

Swap in `DatathonKNNProbeEvaluator`, `DatathonSVMProbeEvaluator`, or
`DatathonPrototypeProbeEvaluator` — same call shape.

### Report generation (reference + predicted → RadEval)

```python
import radharmony.datathon26  # noqa: F401
from radharmony.datathon26 import (
    ReportResultsDataset, perstudy_report_path, DatathonReportGenerationEvaluator,
)
from radharmony.evaluator.metrics.language import LIGHT_METRICS

ds = ReportResultsDataset(
    results_dir="/path/to/_reportbench_out/model-a",  # <study>/model-a_report.txt
    csv_path="dataset_A.csv",                         # one row per study; 'report' = reference
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
`rbclient.py` CLI (`config` → `check` → `prepare` → `submit` → `watch`) via
subprocess.

```python
import radharmony.datathon26  # noqa: F401
from radharmony.datathon26 import ReportBenchClient, DatathonEmbeddingDataset

client = ReportBenchClient(
    rbclient_path="/mnt/NAS4/projects/bkhosra/sharing/datathon/skill/rbclient.py",
    data_dir="/mnt/NAS4/projects/bkhosra/sharing/datathon/data_deposition/<team_id>",
    api_key="rb_...",                    # from your team INSTRUCTIONS.md
)

ds = DatathonEmbeddingDataset(
    harmonized_df=df,                    # one row per study (labels + report)
    base_image_dir="/data/mimic",        # where the raw images live
    client=client,
    model="model-a",                     # see client.models()
)
# ds is an EmbeddingResultsDataset pointed at <data_dir>/run_embeddings/<study>/embedding.npy
```

`DatathonEmbeddingDataset` here is a **factory function**, not the result class:
it (1) stages every study referenced by `harmonized_df` into the per-study folder
layout the service expects (`<study_id>/<view files>`, grouping multiple views per
study, symlinked by default), (2) drives `rbclient.py` to submit the *embeddings*
task and `watch` it to completion, then (3) returns an
`EmbeddingResultsDataset(results_dir, harmonized_df=df)` after verifying every
study produced an `embedding.npy` (it raises with the missing study ids
otherwise). `DatathonReportDataset(harmonized_df, base_image_dir, client=..., model=...)`
is the report-task counterpart — same call shape, submitting the *report* task
and returning a `ReportResultsDataset`.

Pass `skip_submit=True` with an existing `results_dir=` to re-score a finished
run without resubmitting. Every path (client script, interpreter, staging /
results dirs) and the column names (`study_col`, `image_col`) are configurable.
