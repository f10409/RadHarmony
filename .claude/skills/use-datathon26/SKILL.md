---
name: use-datathon26
description: >
  Help a user use the `radharmony.datathon26` package of this repo in code: score the results of the external
  reportbench inference service for its three tasks, with the result datasets (`EmbeddingResultsDataset`,
  `PatchSegResultsDataset`, `ReportResultsDataset`) and the `Datathon*` evaluators, and submit or re-score a
  run with `ReportBenchClient`. Classification and segmentation read the per-image embedding `.npz`. Report
  generation reads the per-study `.txt`. It explains the idea from the wiki and the code, and gives a snippet
  that it checks against the installed package. Use when the user says "how do I use radharmony.datathon26",
  "score reportbench results", "run the datathon classification / segmentation / report evaluator", "read
  back / re-score an embed run", "submit studies to reportbench", names `DatathonEmbeddingDataset`,
  `ReportBenchClient` or `EmbeddingResultsDataset`, or asks "what are the datathon26_* registry keys". For
  general RadHarmony use, use use-radharmony. For the participant notebooks, venvs and basepaths, read
  `datathon26/README.md`. To add a dataset or a backbone, use add-dataset or add-backbone.
---

# Use Datathon26

Help a user score reportbench results with the `radharmony.datathon26` package of this repo. Each time,
explain the one idea that the user needs, give a short correct snippet, and check the snippet against the
installed package first.

This package replaces the local model forward pass with results that an external service already wrote to
disk. Its datasets serve those results in place of image tensors. Its evaluators score them with the
*unchanged* RadHarmony probe, segmentation and generative code and a pass-through encoder.

## When to use this skill

- "How do I use `radharmony.datathon26`?", "score reportbench results", "point a dataset at a
  `_reportbench_out` folder and evaluate it".
- "Run the datathon classification / segmentation / report-generation evaluator."
- "Read back / re-score a finished embed run" (offline, no service call).
- "Submit studies to reportbench from a harmonized frame" (`ReportBenchClient` + factories).
- Any mention of `EmbeddingResultsDataset` / `PatchSegResultsDataset` / `ReportResultsDataset`,
  `Datathon*Evaluator`, `DatathonEmbeddingDataset` / `DatathonReportDataset`, or the `datathon26_*`
  registry keys.

Which skill or file to use:

- The `radharmony.datathon26` API: this skill.
- Participant onboarding (notebook order `0…8`, the three-venv install, `basepaths`, kernels):
  `datathon26/README.md` and `datathon26/0_setup.ipynb`.
- General RadHarmony use (core datasets, evaluators and backbones, for example `CheXpertTrainDataset`,
  `make_raddino`, `LinearProbeEvaluator`): the use-radharmony skill.
- To add a dataset or a backbone to the package: add-dataset or add-backbone.

## Where the facts are

Read the facts in this order. Do not fetch the web when a local copy exists.

1. The wiki page `docs/wiki/reference/datathon26.md` of this repo. It is the main user reference: the
   workflow diagram, the result-file layout, the dataset and evaluator tables, and worked snippets. Its code
   examples match the code.
2. The code in `radharmony/datathon26/`. Read it when the wiki does not answer, or to confirm an exact
   signature or default. Map:
   - `dataset/reportbench.py`: `ReportBenchClient`, the `Datathon*Dataset` factories,
     `_run_reportbench` / `_stage_studies` / `_verify_results`.
   - `dataset/embedding_dataset.py`: `EmbeddingResultsDataset`, `viewwise_embedding_path`, `_load_npy`.
   - `dataset/seg_dataset.py`: `PatchSegResultsDataset`, `_load_patch_map`.
   - `dataset/report_dataset.py`: `ReportResultsDataset`, `perstudy_report_path`.
   - `evaluator/{classification,segmentation,language,_identity}.py`: the `Datathon*Evaluator` subclasses,
     `IdentityEncoder`, `identity_generator`.
3. The live site, `https://f10409.github.io/RadHarmony/reference/datathon26.html`, with `WebFetch`. Use it
   **only** when there is no local repo.

**Get the names from the package, not from memory.** The import of the package runs the `@register_*`
decorators. So import it *first*, then filter the registries:

```bash
python -c "import radharmony.datathon26; from radharmony.registry import list_datasets; print([k for k in list_datasets() if k.startswith('datathon26')])"
python -c "import radharmony.datathon26; from radharmony.evaluator import list_evaluators; print([k for k in list_evaluators() if k.startswith('datathon26')])"
```

Use the Python where `radharmony` is installed. This is plain `python` if it is in the active environment.
Otherwise use the Python of that environment (in this repo, `.venv/bin/python`).

## The main idea

**Inference does not run locally.** The user submits studies to the external reportbench service. The service
runs one of two tasks and writes the results to disk:

- **embed** → one `.npz` **per image**. It holds `global [dim]` (the pooled vector), `patches
  [n_patches, dim]`, and `grid (H, W)`.
- **report** → one `.txt` **per study** (the generated report).

The datathon26 **datasets** serve those results under the sample's `"img"` key, in place of a decoded image.
The datathon26 **evaluators** are thin subclasses. They default their `image_encoder` to a pass-through
`IdentityEncoder` (report: `identity_generator`). So the only encoder/generator call returns the pre-computed
result unchanged, and the standard code scores it.

The evaluators keep all the features of the standard code unchanged:

- patient-grouped k-fold, multi-seed, bootstrap, threshold strategies, `macro_average` and the embedding cache
- early stopping and prediction dumps (segmentation)
- the full RadEval suite and `generations.csv` (report)

**Two outputs give three scoring tasks.** One embed submission feeds classification (it reads `global`) and
segmentation (it folds `patches` into a `[D, H, W]` map). The report submission feeds report generation (it
reads the `.txt`).

## Two layers of the API

Pick the layer from what the user already has:

| Layer | Use when | Symbols |
|---|---|---|
| **Result datasets** | You already have a `_reportbench_out/<model>` results folder. | `EmbeddingResultsDataset` (cls), `PatchSegResultsDataset` (seg), `ReportResultsDataset` (report), and the resolvers `viewwise_embedding_path(model)` / `perstudy_report_path(model)` |
| **Submission factories** | You have raw studies and must run them through the service. | `DatathonEmbeddingDataset`, `DatathonPatchEmbeddingDataset`, `DatathonReportDataset`, and `ReportBenchClient` |

A factory does the submission first and returns the matching result dataset. For example,
`DatathonEmbeddingDataset(...)` returns an `EmbeddingResultsDataset`. The factories are **functions**, not
classes.

**At the datathon event, participants mostly use the result datasets.** The embeddings and reports are
**pre-extracted** and stored on the event storage. So participants score them directly (snippets 1 to 3),
with **no submission**. The factories and `ReportBenchClient` (snippet 4) are only to run your **own**
studies through the service. The event data roots have one subfolder per model (`model-a … model-g`).
Ask the event organizers for the real data root:

- **Embeddings** (classification and segmentation):
  `/path/to/datathon26_cxr_datasets/embeddings/<model>/<study_id>/<model>_<stem>.npz`.
  Pass `embeddings_dir=".../embeddings/<model>"` and `result_path=viewwise_embedding_path("<model>")`.
- **Reports** (report generation):
  `/path/to/datathon26_cxr_datasets/reports/<model>/<study_id>/<model>_report.txt`.
  Pass `results_dir=".../reports/<model>"` and `result_path=perstudy_report_path("<model>")`.
  Reports cover `model-a` and `model-c … model-g`. There is no `model-b`.

The default `<study>/embedding.npy` / `<study>/report.txt` layout is only what the self-contained notebook
demo makes. The real event files are named `<model>_<stem>.npz` / `<model>_report.txt`. So always pass the
matching resolver above.

## Snippets by task

Give the smallest correct snippet, then the wiki section for the options. Point the results and image paths
at the folder of the run on the user's machine. The paths below are placeholders. Every snippet starts with
the import that registers the keys:

```python
import radharmony.datathon26  # noqa: F401  (registers the datathon26_* datasets + evaluators)
```

### 1. Classification (embeddings → probe)

The frame has **one row per image** (the stem comes from the `image_path` of each row).

```python
from radharmony.datathon26 import (
    EmbeddingResultsDataset, viewwise_embedding_path, DatathonLinearProbeEvaluator,
)

ds = EmbeddingResultsDataset(
    embeddings_dir="/path/to/_reportbench_out/model-a",   # <study>/model-a_<stem>.npz
    csv_path="dataset_DS1.csv",                           # one row per image (labels)
    result_path=viewwise_embedding_path("model-a"),       # reads the .npz `global`
)
ev = DatathonLinearProbeEvaluator(dataset=ds, n_folds=5)
results = ev.evaluate()   # DataFrame: per-label + macro_average rows (auroc, auprc, f1, ...)
```

You can use `DatathonKNNProbeEvaluator` / `DatathonSVMProbeEvaluator` / `DatathonPrototypeProbeEvaluator`
in the same way. The dataset auto-detects the labels as the binary 0/1 numeric columns. Pass `label_cols=`
to set them yourself. See the section "Classification" of `docs/wiki/reference/datathon26.md`.

### 2. Segmentation (patch features → Dice/IoU)

This is only for datasets with a ground-truth mask per image. `PatchSegResultsDataset` reads the **same**
`.npz` and folds `patches` into a dense `[dim, H, W]` map.

```python
from radharmony.datathon26 import (
    PatchSegResultsDataset, viewwise_embedding_path, DatathonLinearProbeSegEvaluator,
)

ds = PatchSegResultsDataset(
    embeddings_dir="/path/to/_reportbench_out/model-a",   # <study>/model-a_<stem>.npz
    csv_path="dataset_DS6.csv",                           # one row per image
    result_path=viewwise_embedding_path("model-a"),       # reads the .npz `patches`
    mask_col="mask_path",                                 # GT mask per image
    mask_dir="/path/to/masks_DS6",                        # root for RELATIVE mask paths (None if absolute)
    mask_size=224,                                        # output resolution
)
ev = DatathonLinearProbeSegEvaluator(dataset=ds, num_classes=2, n_folds=5)
results = ev.evaluate()   # DataFrame: per-class + macro_average rows (dice, iou, ...)
```

For heavier heads, use `DatathonConvProbeSegEvaluator` / `DatathonUPerNetSegEvaluator`. Pass `feat_size=`
when the patch grid changes from image to image. Without `feat_size=`, all images of a batch must have the
same grid. See
the section "Segmentation" of the wiki page.

### 3. Report generation (reference + predicted → RadEval)

The frame has **one row per study**. The inline `report` column gives the reference text.

```python
from radharmony.datathon26 import (
    ReportResultsDataset, perstudy_report_path, DatathonReportGenerationEvaluator,
)
from radharmony.evaluator.metrics.language import LIGHT_METRICS  # or the default full suite

ds = ReportResultsDataset(
    results_dir="/path/to/_reportbench_out/model-a",      # <study>/model-a_report.txt
    csv_path="dataset_DS1.csv",                           # one row per study; 'report' = reference
    result_path=perstudy_report_path("model-a"),
)
ev = DatathonReportGenerationEvaluator(
    dataset=ds, metrics=LIGHT_METRICS, ref_section="findings",
)
results = ev.evaluate()   # DataFrame with the RadEval metric panel
```

The evaluator parses the reference and the predicted report to the same `ref_section` (findings against
findings by default). So the comparison is symmetric. It drops blank references and empty generations.
`FULL_METRICS` includes RadGraph, which runs only in the `medgemma` venv (see the use-radharmony skill).

### 4. One-shot submit: raw studies → dataset (needs the live service)

Go straight from a harmonized frame and a raw image root to a scored dataset. The factory stages the studies
and drives the `rbclient.py` CLI (`config` → optional `check` → `prepare` → `submit` → `watch`). Then it
checks that every image produced its `.npz`, and returns the result dataset.

```python
from radharmony.datathon26 import ReportBenchClient, DatathonEmbeddingDataset

client = ReportBenchClient(
    rbclient_path="/path/to/datathon/skill/rbclient.py",
    data_dir="/path/to/datathon/data_deposition/<team_id>",
    api_key="rb_...",                    # from your team INSTRUCTIONS.md (never commit a real key)
)
ds = DatathonEmbeddingDataset(
    harmonized_df=df,                    # one row per image (labels)
    base_image_dir="/data/mimic",        # where the raw images live
    client=client, model="model-a",      # see client.models()
)
# ds is an EmbeddingResultsDataset over <data_dir>/run_embed/_reportbench_out/model-a/...
```

`DatathonReportDataset(df, base_image_dir, client=..., model=...)` is the factory for the report task. Pass
`indication_col=` to attach a per-study indication via the job manifest. This path works only when the
reportbench service is reachable.

### 5. Re-score offline: reuse a finished run (no service call)

When the service is down, or you only want to re-score, skip staging and submit:

```python
ds = DatathonEmbeddingDataset(
    harmonized_df=df, base_image_dir="/data/mimic",
    client=client, model="model-a",
    run_name="run_F_demo",               # the finished run
    skip_submit=True,                    # reuse existing results, no submit/watch
    results_dir="/path/to/_reportbench_out/model-a",   # or let it default from run_name
)
```

You can also build the result dataset directly (`EmbeddingResultsDataset(...)`). This also never calls the
service. Use one of these two ways when the reportbench service is not reachable.

### 6. One embed run for both classification and segmentation

`global` and `patches` are in the **same** `.npz`. So embed **once**, and read the patches back for
segmentation with no second submission:

```python
from radharmony.datathon26 import DatathonPatchEmbeddingDataset
seg_ds = DatathonPatchEmbeddingDataset(
    harmonized_df=df, base_image_dir="/data/mimic", client=client, model="model-a",
    run_name="run_embed", skip_submit=True,            # reuse the embed run from snippet 1/4
    results_dir="/path/to/_reportbench_out/model-a",
    mask_col="mask_path", mask_dir="/path/to/masks", mask_size=224,
)
```

## Result-file layout

The service writes the results under a per-model subfolder:

```
<results_dir>/<model>/                       # for example .../_reportbench_out/model-a/
  DS1_patient_00018_study_00001/
    model-a_pa.npz        # embed: one <model>_<stem>.npz per image
    model-a_lateral.npz   #   archive: global [dim] + patches [n, dim] + grid (H, W)
    model-a_report.txt    # report: one <model>_report.txt per study
```

Point the dataset at `<results_dir>/<model>` and pass a `result_path` resolver.
`viewwise_embedding_path(model)` (`.npz`) and `perstudy_report_path(model)` build the names above. With no
resolver, the datasets **default** to the simpler `<study>/embedding.npy` / `<study>/report.txt`. The
self-contained notebook demo makes files with these names. Classification and segmentation frames have
**one row per image**. The report frame has **one row per study**.

## Check the snippet before you give it

The check is fast: no submit and no training. Confirm that the key is registered and that the kwargs
exist. Use the Python where `radharmony` is installed. `import radharmony.datathon26` loads torch and MONAI,
so the first import takes about 6 s.

```bash
python -c "import radharmony.datathon26, inspect; from radharmony.registry import list_datasets; from radharmony.datathon26 import EmbeddingResultsDataset as C; print('datathon26_embedding' in list_datasets()); print(inspect.signature(C.__init__))"
python -c "import radharmony.datathon26, inspect; from radharmony.evaluator import list_evaluators; from radharmony.datathon26 import DatathonLinearProbeEvaluator as E; print('datathon26_linear_probe' in list_evaluators()); print(inspect.signature(E.__init__))"
```

A kwarg must be in the signature, directly or through `**kwargs` to the base evaluator. If it is not, fix
the snippet. Do not give code that raises `TypeError`. The `Datathon*Evaluator` classes pass `**kwargs` to
their base class (`n_folds`, `n_train_samples`, `n_bootstrap`, `threshold_strategy`, `output_dir`, ...).
Check each such kwarg against the signature of the *base* class.

## Do not

- **Do not commit a real reportbench API key** or MIMIC-derived data (report text, DICOM-derived images).
  The files of this repo go to the public repo, and MIMIC is under the PhysioNet DUA. Use the `rb_...`
  placeholder in every snippet.
- **Do not assume that the service is up.** A submission (snippet 4) needs a reachable reportbench host.
  When the service is down, offer the offline re-score path (snippet 5, or a result dataset alone).
- **Do not write the `datathon26_*` keys from memory.** Get them with `list_datasets()` /
  `list_evaluators()` after the import of the package.
- **Do not pass a real `image_encoder`** to a `Datathon*` evaluator. It defaults to the pass-through
  `IdentityEncoder`. A real encoder encodes the already-extracted vectors again, and so defeats the point
  of the pass-through.
- **Do not mix up the frame granularity.** Classification and segmentation need **one row per image**.
  Report scoring de-duplicates to **one row per study**.
- **Do not set `emb_key` / `emb_ext` to values that differ from the files of the service** (`global` and
  `.npz` by default). Do not pass `emb_key="patches"` to the classification dataset. The segmentation
  dataset reads `patches` with its own loader. Use `emb_ext=".npy"` for bare arrays.
- **Do not edit `radharmony/datathon26/**` or `docs/wiki/reference/datathon26.md`** from this skill. The
  skill points at them. A change to the package or its wiki page is a separate task.
## Answer

1. The idea, in 1 to 3 sentences.
2. The snippet, with the output as a comment when you know it from the wiki or a run.
3. The section of `docs/wiki/reference/datathon26.md` to read next (the URL when there is no clone).
4. The check result, or what you could not check.

## Edit this skill

Edit and commit this file, `.claude/skills/use-datathon26/SKILL.md`. It has no copy in `docs/`.
