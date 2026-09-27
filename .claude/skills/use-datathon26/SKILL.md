---
name: use-datathon26
description: >
  Help a user *use* the `radharmony.datathon26` submodule in code: score results from the external
  *reportbench* inference service across its three tasks — classification and segmentation (over the
  per-image embedding `.npz`) and report generation (over the per-study `.txt`) — with the result
  datasets (`EmbeddingResultsDataset` / `PatchSegResultsDataset` / `ReportResultsDataset`) and the
  `Datathon*` evaluators, and submit or re-score a run with `ReportBenchClient`. Explains the concept
  grounded in the local wiki + source and produces a ready-to-run snippet verified against the
  installed package. Use when the user says "how do I use radharmony.datathon26", "score reportbench
  results", "run the datathon classification / segmentation / report evaluator", "read back / re-score
  an embed run", "submit studies to reportbench", mentions `DatathonEmbeddingDataset` /
  `ReportBenchClient` / `EmbeddingResultsDataset`, or asks "what are the datathon26_* registry keys".
  This is for the datathon26 *API*; for general RadHarmony usage or the participant notebook / venv /
  basepaths onboarding use `use-radharmony`; to *extend* the package use `add-dataset` / `add-backbone`.
---

# Use Datathon26 Skill

Help someone score reportbench results with the **`radharmony.datathon26`** submodule. The goal each
time: **explain the one concept that makes it click, hand over a minimal correct snippet, and verify
it against the installed package before presenting it.** This submodule replaces the local model
forward pass with results an external service already wrote to disk — its datasets serve those
results in place of image tensors, and its evaluators score them by reusing the *unmodified*
RadHarmony probe / segmentation / generative machinery with a pass-through encoder.

## When This Skill Triggers

- "How do I use `radharmony.datathon26`?", "score reportbench results", "point a dataset at a
  `_reportbench_out` folder and evaluate it".
- "Run the datathon classification / segmentation / report-generation evaluator."
- "Read back / re-score a finished embed run" (offline, no service call).
- "Submit studies to reportbench from a harmonized frame" (`ReportBenchClient` + factories).
- Any mention of `EmbeddingResultsDataset` / `PatchSegResultsDataset` / `ReportResultsDataset`,
  `Datathon*Evaluator`, `DatathonEmbeddingDataset` / `DatathonReportDataset`, or the
  `datathon26_*` registry keys.

**Disambiguation.**
- The `radharmony.datathon26` **API** (this file) → here.
- **Participant onboarding** — notebook `0…8` order, the three-venv install, `basepaths`, kernels →
  `use-radharmony` (its `## Datathon26 participants` section).
- **Core, non-datathon** datasets / evaluators / backbones (`CheXpertTrainDataset`, `make_raddino`,
  `LinearProbeEvaluator`) → `use-radharmony`.
- **Extending** the package — adding a dataset or backbone → `add-dataset` / `add-backbone`.

## Grounding — where to read from first

Resolve facts in this order; never fetch the web when a local copy is present:

1. **Local wiki** — [docs/wiki/datathon26.md](docs/wiki/datathon26.md), the authoritative
   user-facing reference (workflow diagram, result-file layout, dataset/evaluator tables, worked
   snippets). It ships with the checkout and its code examples match the source.
2. **Local source** under `radharmony/datathon26/` — drop into the code when the wiki is thin or an
   exact signature / default must be confirmed. Map:
   - `dataset/reportbench.py` — `ReportBenchClient`, the `Datathon*Dataset` factories,
     `_run_reportbench` / `_stage_studies` / `_verify_results`.
   - `dataset/embedding_dataset.py` — `EmbeddingResultsDataset`, `viewwise_embedding_path`, `_load_npy`.
   - `dataset/seg_dataset.py` — `PatchSegResultsDataset`, `_load_patch_map`.
   - `dataset/report_dataset.py` — `ReportResultsDataset`, `perstudy_report_path`.
   - `evaluator/{classification,segmentation,language,_identity}.py` — the `Datathon*Evaluator`
     subclasses, `IdentityEncoder`, `identity_generator`.
3. **Live site** `https://f10409.github.io/RadHarmony/datathon26.html` via `WebFetch` — **only** when
   there is no local repo.

**Names are discovered, never hardcoded from memory.** Importing the submodule runs the
`@register_*` decorators, so import it *first*, then filter the registries:

```bash
python -c "import radharmony.datathon26; from radharmony.registry import list_datasets; print([k for k in list_datasets() if k.startswith('datathon26')])"
python -c "import radharmony.datathon26; from radharmony.evaluator import list_evaluators; print([k for k in list_evaluators() if k.startswith('datathon26')])"
```

Use the interpreter where `radharmony` is installed — plain `python` if it's on the active
environment, otherwise that env's interpreter (this repo uses `.venv/bin/python`).

## Mental model

The one insight that unlocks everything: **inference does not run locally.** Studies are submitted to
the external reportbench service, which runs one of two tasks and writes results to disk:

- **embed** → one `.npz` **per image**, holding `global [dim]` (the pooled vector), `patches
  [n_patches, dim]`, and `grid (H, W)`.
- **report** → one `.txt` **per study** (the generated report).

The datathon26 **datasets** serve those results under the sample's `"img"` key in place of a decoded
image. The datathon26 **evaluators** are thin subclasses that default their `image_encoder` to a
pass-through `IdentityEncoder` (report: `identity_generator`), so the sole encoder/generator call
returns the pre-computed result unchanged and the standard machinery scores it — **patient-grouped
k-fold, multi-seed, bootstrap, threshold strategies, `macro_average`, the embedding cache,
early-stopping + prediction dumps (seg), and the full RadEval suite + `generations.csv` (report) are
all inherited unchanged.**

**Two outputs → three scoring tasks.** One embed submission feeds both classification (reads
`global`) and segmentation (folds `patches` into a `[D, H, W]` map); the report submission feeds
report generation (reads the `.txt`).

## Two layers of the API

Pick the layer by what the user already has:

| Layer | Use when | Symbols |
|---|---|---|
| **Result datasets** | you already have a `_reportbench_out/<model>` results folder | `EmbeddingResultsDataset` (cls), `PatchSegResultsDataset` (seg), `ReportResultsDataset` (report) + resolvers `viewwise_embedding_path(model)` / `perstudy_report_path(model)` |
| **Submission factories** | you have raw studies and need to run them through the service | `DatathonEmbeddingDataset`, `DatathonPatchEmbeddingDataset`, `DatathonReportDataset` + `ReportBenchClient` |

A factory does the submission up front and returns the matching result dataset — so
`DatathonEmbeddingDataset(...)` hands back an `EmbeddingResultsDataset`. (They are factory
**functions**, not classes.)

**In the datathon event, participants mainly use the result datasets.** The embeddings and reports
are **pre-extracted** and staged on NAS, so attendees score them directly (playbooks 1–3) with **no
submission** — the factories + `ReportBenchClient` (playbook 4) are only for running your **own**
studies through the service. Event data roots (one subfolder per model, `model-a … model-g`):

- **Embeddings** (classification + segmentation) —
  `/mnt/NAS4/projects/fli40_sharing/datathon26_cxr_datasets/embeddings/<model>/<study_id>/<model>_<stem>.npz`.
  Pass `embeddings_dir=".../embeddings/<model>"` and `result_path=viewwise_embedding_path("<model>")`.
- **Reports** (report generation) —
  `/mnt/NAS4/projects/fli40_sharing/datathon26_cxr_datasets/reports/<model>/<study_id>/<model>_report.txt`.
  Pass `results_dir=".../reports/<model>"` and `result_path=perstudy_report_path("<model>")`.
  (Reports cover `model-a`, `model-c … model-g` — no `model-b`.)

The default `<study>/embedding.npy` / `<study>/report.txt` layout is only what the self-contained
notebook demo fabricates; the real event files are named `<model>_<stem>.npz` / `<model>_report.txt`,
so always pass the matching resolver above.

## Task playbooks

Give the smallest correct snippet, then the wiki section for options. Point the results/image paths
at wherever the run lives on the user's machine; the paths below are placeholders. Every snippet
starts by registering the keys:

```python
import radharmony.datathon26  # noqa: F401  — registers datathon26_* datasets + evaluators
```

### 1. Classification (embeddings → probe)

Frame is **one row per image** (the stem is read from each row's `image_path`).

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

Swap in `DatathonKNNProbeEvaluator` / `DatathonSVMProbeEvaluator` / `DatathonPrototypeProbeEvaluator`
— same call shape. Labels are auto-detected as the binary 0/1 numeric columns (pass `label_cols=` to
override). See [docs/wiki/datathon26.md](docs/wiki/datathon26.md) "Classification".

### 2. Segmentation (patch features → Dice/IoU)

Only for datasets carrying a ground-truth mask per image. `PatchSegResultsDataset` reads the **same**
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

Swap in `DatathonConvProbeSegEvaluator` / `DatathonUPerNetSegEvaluator` for heavier heads. Pass
`feat_size=` when the patch grid varies per image (else a fixed grid across the batch is required).
See "Segmentation" in the wiki.

### 3. Report generation (reference + predicted → RadEval)

Frame is **one row per study**; the inline `report` column supplies the reference text.

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

Both reference and predicted report are parsed to the same `ref_section` (findings-vs-findings by
default) so the comparison is symmetric; blank references / empty generations are dropped.
`FULL_METRICS` includes RadGraph, which only runs in the `medgemma` venv (see `use-radharmony`).

### 4. One-shot submit — raw studies → dataset (needs the live service)

Go straight from a harmonized frame + raw image root to a scored dataset. The factory stages the
studies, drives the `rbclient.py` CLI (`config` → optional `check` → `prepare` → `submit` →
`watch`), verifies every image produced its `.npz`, then returns the result dataset.

```python
from radharmony.datathon26 import ReportBenchClient, DatathonEmbeddingDataset

client = ReportBenchClient(
    rbclient_path="/path/to/datathon/skill/rbclient.py",
    data_dir="/path/to/datathon/data_deposition/<team_id>",
    api_key="rb_...",                    # from your team INSTRUCTIONS.md — never commit a real key
)
ds = DatathonEmbeddingDataset(
    harmonized_df=df,                    # one row per image (labels)
    base_image_dir="/data/mimic",        # where the raw images live
    client=client, model="model-a",      # see client.models()
)
# ds is an EmbeddingResultsDataset over <data_dir>/run_embed/_reportbench_out/model-a/...
```

`DatathonReportDataset(df, base_image_dir, client=..., model=...)` is the report-task counterpart
(pass `indication_col=` to attach a per-study indication via the job manifest). This path only works
when the reportbench service is reachable.

### 5. Re-score offline — reuse a finished run (no service call)

When the service is down or you just want to re-score, skip staging/submit entirely:

```python
ds = DatathonEmbeddingDataset(
    harmonized_df=df, base_image_dir="/data/mimic",
    client=client, model="model-a",
    run_name="run_F_demo",               # the finished run
    skip_submit=True,                    # reuse existing results, no submit/watch
    results_dir="/path/to/_reportbench_out/model-a",   # or let it default from run_name
)
```

Equivalently, construct the result dataset directly (`EmbeddingResultsDataset(...)`) — that never
touches the service either. Prefer one of these whenever the reportbench service is unreachable.

### 6. One embed run, both classification and segmentation

`global` and `patches` ride in the **same** `.npz`, so embed **once** and read patches back for
segmentation with no second submission:

```python
from radharmony.datathon26 import DatathonPatchEmbeddingDataset
seg_ds = DatathonPatchEmbeddingDataset(
    harmonized_df=df, base_image_dir="/data/mimic", client=client, model="model-a",
    run_name="run_embed", skip_submit=True,            # reuse the embed run from playbook 1/4
    results_dir="/path/to/_reportbench_out/model-a",
    mask_col="mask_path", mask_dir="/path/to/masks", mask_size=224,
)
```

## Result-file layout

The service writes under a per-model subfolder:

```
<results_dir>/<model>/                       # e.g. .../_reportbench_out/model-a/
  DS1_patient_00018_study_00001/
    model-a_pa.npz        # embed — one <model>_<stem>.npz per image
    model-a_lateral.npz   #   archive: global [dim] + patches [n, dim] + grid (H, W)
    model-a_report.txt    # report — one <model>_report.txt per study
```

Point the dataset at `<results_dir>/<model>` and pass a `result_path` resolver —
`viewwise_embedding_path(model)` (`.npz`) / `perstudy_report_path(model)` build the names above.
Without a resolver the datasets **default** to the simpler `<study>/embedding.npy` /
`<study>/report.txt` (what the self-contained notebook demo fabricates). Classification / segmentation
frames carry **one row per image**; the report frame is **one row per study**.

## Verify before you hand it over

Cheap, fast, no submit and no training. Confirm the key is registered and the kwargs exist, using the
interpreter where `radharmony` is installed (`import radharmony.datathon26` pulls torch/MONAI — a
one-time ~6s warmup):

```bash
python -c "import radharmony.datathon26, inspect; from radharmony.registry import list_datasets; from radharmony.datathon26 import EmbeddingResultsDataset as C; print('datathon26_embedding' in list_datasets()); print(inspect.signature(C.__init__))"
python -c "import radharmony.datathon26, inspect; from radharmony.evaluator import list_evaluators; from radharmony.datathon26 import DatathonLinearProbeEvaluator as E; print('datathon26_linear_probe' in list_evaluators()); print(inspect.signature(E.__init__))"
```

If a kwarg you used isn't in the signature (directly or via `**kwargs` onto the base evaluator), fix
the snippet — don't present code that would raise `TypeError`. The `Datathon*Evaluator` classes accept
`**kwargs` that forward to their base (`n_folds`, `n_train_samples`, `n_bootstrap`, `threshold_strategy`,
`output_dir`, ...) — confirm any such kwarg against the *base* class signature.

## What NOT to do

- **Never commit a real reportbench API key** or MIMIC-derived data (report text, DICOM-derived
  images). This is a **public** repo and MIMIC is under the PhysioNet DUA. Use the `rb_...`
  placeholder in any snippet.
- **Don't assume the service is up.** Submission (playbook 4) needs a reachable reportbench host; when
  it's down, offer the offline re-score path (playbook 5 / a bare result dataset).
- **Don't hardcode the `datathon26_*` keys** — discover them via `list_datasets()` /
  `list_evaluators()` after importing the submodule.
- **Don't pass a real `image_encoder`** to a `Datathon*` evaluator. It defaults to `IdentityEncoder`
  (pass-through); a real encoder would re-encode the already-extracted vectors and defeat the point.
- **Don't mismatch the frame granularity** — classification/segmentation need **one row per image**;
  report scoring de-dups to **one row per study**.
- **Don't mismatch `emb_key` / `emb_ext`** to what the service wrote (`global` + `.npz` by default;
  use `emb_key="patches"` only via the seg dataset's own loader, `emb_ext=".npy"` for bare arrays).
- **Don't edit `radharmony/datathon26/**` or `docs/wiki/datathon26.md`** from this skill — it points
  at them; changing the submodule or its wiki is a separate task.
- **For extending** the package (a new dataset/backbone) use `add-dataset` / `add-backbone`; **for
  onboarding** (notebooks, the three venvs, `basepaths`) use `use-radharmony`.

## Editing this skill

The copy in the repo, `.claude/skills/use-datathon26/SKILL.md`, is the one to edit and commit.
There is no `docs/` mirror anymore.
