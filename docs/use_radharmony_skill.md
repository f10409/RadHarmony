---
name: use-radharmony
description: >
  Help a user *use* the RadHarmony package: load and harmonize radiological datasets, build a
  DataLoader, run a probe / zero-shot / fine-tune / segmentation evaluation with a foundation-model
  backbone, use the Gradio app, or work through the datathon26 notebooks. Explains the concept
  grounded in the local wiki + source and produces a ready-to-run snippet verified against the
  installed package. Use when the user says "how do I use RadHarmony", "help me load a dataset",
  "run an evaluation", "harmonize datasets", "use a backbone / probe / zero-shot", "build a
  DataLoader from a RadHarmony dataset", "set up the datathon / basepaths", or "which datasets /
  evaluators are available". This is for USING the package; to EXTEND it (add a new dataset or
  backbone) use the add-dataset / add-backbone skills instead.
---

# Use RadHarmony Skill

Help someone accomplish a task *with* RadHarmony — the dataset + evaluator + backbone library.
The goal each time: **explain the concept briefly, hand over a minimal correct snippet, and
verify it against the installed package before presenting it.** RadHarmony wraps MONAI to give
PyTorch `Dataset` objects with a unified sample-dict interface, plus an evaluator subsystem that
probes foundation-model backbones across those datasets.

## When This Skill Triggers

- "How do I use RadHarmony?", "help me load `<dataset>`", "build a DataLoader from it".
- "Run a linear probe / k-NN / SVM / prototype / zero-shot / fine-tune with `<model>`".
- "How do I harmonize / combine multiple datasets?"
- "Which datasets / evaluators / backbones are available?"
- "How do I launch the app?" / remote access.
- "Set up the datathon", "what does `basepaths` do", datathon26 notebook questions.

**Disambiguation from the maintainer skills.** This skill is for *using* the package
("load / run / evaluate / how do I"). For *extending* it — "add / integrate / scaffold / wire in"
a new dataset or backbone — defer to `add-dataset` / `add-backbone`. "Use a dataset" → here;
"add a dataset" → `add-dataset`.

## Grounding — where to read from first

Resolve facts in this order; never fetch the web when a local copy is present:

1. **Local wiki** (`docs/wiki/*.md`) — the authoritative user-facing reference that ships with
   the checkout. Key pages: [quickstart](docs/wiki/quickstart.md),
   [Dataset API](docs/wiki/api.md),
   [transforms](docs/wiki/transforms.md),
   [Evaluator index](docs/wiki/evaluator/index.md) + per-evaluator pages,
   [Backbones](docs/wiki/backbones/index.md),
   [App](docs/wiki/app.md), [Datathon26](docs/wiki/datathon26.md).
   (Paths are relative to the repo root — read whatever exists under `docs/wiki/`.)
2. **Local source** (`radharmony/**/*.py`) — drop into the code when a wiki page is thin or an
   exact constructor signature, default, `LABEL_COLS`, or registry key must be confirmed. The
   source is the final authority and always matches the installed package.
3. **Live site** `https://f10409.github.io/RadHarmony` via `WebFetch` — **only** when there is no
   local repo. It can lag the user's installed version, so prefer local whenever it exists.

**Names are discovered, never hardcoded from memory** — the catalog changes over time. Run these
with the Python interpreter where `radharmony` is installed — plain `python` if it's on the
active environment, otherwise that env's interpreter (e.g. `<env>/bin/python`, or
`.venv/bin/python` if the project uses a `.venv`). Note the dataset registry only fills once
`radharmony.dataset` is imported (the `@register_dataset` decorators run on import), so import it
*before* calling `list_datasets()`:

```bash
python -c "import radharmony.dataset; from radharmony.registry import list_datasets; print(len(list_datasets())); print(list_datasets())"
python -c "from radharmony.evaluator import list_evaluators; print(list_evaluators())"
python -c "import radharmony.evaluator.backbones as b; print([n for n in b.__all__ if n.startswith('make_')])"
```

`resolve_dataset(name)` / `resolve_evaluator(name)` map a registry key to its class.

## Mental model

- A RadHarmony dataset yields a **sample dict**: always `img`, plus `cls` / `mask` / `bbox` /
  `report` / `reg` **gated by `output_*` flags** you pass to the constructor. Turn on only what
  the task needs (`output_cls=True` for classification, `output_mask=True` for segmentation).
- Constructing a dataset auto-runs its **harmonizer** internally, mapping that source's native
  labels onto RadHarmony's unified schema. `LABEL_COLS` on the class selects which harmonized
  columns become the `cls` tensor.
- `dataset.get_datasets(n_splits=k)` returns **split PyTorch datasets** (k-fold); each is a plain
  `torch.utils.data.Dataset` you drop into a `DataLoader`.
- An **evaluator** pairs a backbone `encoder` (from a `make_*` recipe) with train/test datasets
  and returns a **per-row** metrics **`DataFrame`** (one row per fold or bootstrap; AUROC, AUPRC,
  F1, ...) — aggregate it to `mean [ci_lo, ci_hi]` per playbook 8. The encoder contract is
  `encoder(imgs: Tensor[B, ...]) -> Tensor[B, D]`; zero-shot also needs
  `text_encoder(list[str]) -> Tensor[T, D]`.
- The dataset's `transform` must be the one the backbone recipe returns, so pixel preprocessing
  matches the model's pretraining. Always build the dataset with `transform=` from the recipe.

## Task playbooks

Give the user the smallest correct snippet, then the wiki page to read for options. Point
`base_image_dir` at wherever the dataset lives on the user's own machine (the folder of images);
ask the user for the path if you don't have it. The `/data/...` paths below are placeholders.

**Pick the device, don't hardcode `"cuda"`.** Backbone recipes and evaluators default to
`device="cuda"`; on a CPU-only box that errors, and on a shared box whose GPU is already occupied
it fails with `torch.OutOfMemoryError`. Start every snippet by selecting the device and thread it
through **both** the recipe and the evaluator so they agree:

```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
# If CUDA is present but the card is occupied (OOM on the first forward pass),
# force CPU instead:  device = "cpu"
```

The snippets below use this `device`.

### 1. Load one dataset and build a DataLoader

```python
import torch
from torch.utils.data import DataLoader
from radharmony.dataset import CheXpertTrainDataset

ds = CheXpertTrainDataset(base_image_dir="/data/CheXpert-v1.0/train/", output_cls=True)
train_ds, val_ds = ds.get_datasets(n_splits=5)   # k-fold split
loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
sample = train_ds[0]          # dict: {"img": Tensor[1,224,224], "cls": Tensor[14], ...}
```

Leaf classes (`CheXpertTrainDataset`, `VinDrCXRTrainDataset`, ...) supply their own default
`transform` and auto-discover the CSV; the base `__init__` requires `transform` explicitly.
See [Dataset API](docs/wiki/api.md) and the per-dataset page under
`docs/wiki/datasets/`.

### 2. Harmonize / combine multiple datasets

Every dataset harmonizes to the same unified label schema on construction, so a shared
`LABEL_COLS` makes `cls` tensors line up across sources. To train on one shared finding across
datasets, set `LABEL_COLS` before constructing:

```python
from radharmony.dataset import VinDrCXRTrainDataset, SIIMACRPTXDataset  # names per radharmony.dataset.__all__

VinDrCXRTrainDataset.LABEL_COLS = ["pneumothorax"]
# each dataset now emits a 1-d `cls` for the same finding; concat with torch.utils.data.ConcatDataset
```

Read [Dataset API](docs/wiki/api.md) for the harmonizer/`harmonized_df` flow;
the datathon `1_radharmony_hands_on` notebook is a worked example.

### 3. Classification probe with a backbone (linear / k-NN / SVM / prototype)

```python
from radharmony.evaluator.backbones import make_raddino
from radharmony.evaluator import LinearProbeEvaluator
from radharmony.dataset import VinDrCXRTrainDataset, VinDrCXRTestDataset

transform, encoder = make_raddino(device=device)   # device selected above

train_ds = VinDrCXRTrainDataset(base_image_dir="/data/VinDr-CXR/", transform=transform, output_cls=True, cache_dir=None)
test_ds  = VinDrCXRTestDataset (base_image_dir="/data/VinDr-CXR/", transform=transform, output_cls=True, cache_dir=None)

ev = LinearProbeEvaluator(encoder, train_dataset=train_ds, test_dataset=test_ds, device=device,
                          n_seeds=3, n_train_samples=[500, 1500], n_bootstrap=100,
                          output_dir="outputs/linprobe")
df = ev.evaluate(); ev.save_results(df)
```

`KNNProbeEvaluator` / `SVMProbeEvaluator` / `PrototypeProbeEvaluator` share the same call shape —
swap the class. See [Evaluator index](docs/wiki/evaluator/index.md) and the
per-probe pages. All frozen-feature probes L2-normalize features internally.

### 4. Zero-shot with a vision-language backbone

Use a recipe that returns a **4-tuple** (image encoder + text encoder + tokenizer):

```python
from radharmony.evaluator.backbones import make_biomed_clip
from radharmony.evaluator import ZeroShotEvaluator

transform, img_enc, text_enc, tokenizer = make_biomed_clip(device=device)
prompts = {"pneumothorax": ["a chest x-ray with a collapsed lung"], "no_finding": ["a normal chest x-ray"]}
ev = ZeroShotEvaluator(img_enc, text_enc, prompts, tokenizer=tokenizer, test_dataset=test_ds, device=device)
df = ev.evaluate()
```

See [zero_shot](docs/wiki/evaluator/zero_shot.md).

### 5. Fine-tune and segmentation

- **Fine-tune:** `FinetuneEvaluator(encoder, train_dataset=..., test_dataset=..., epochs=20, lr=1e-4, ...)` —
  single-GPU end-to-end. See [finetune](docs/wiki/evaluator/finetune.md).
- **Segmentation:** switch the recipe to segmentation mode
  (`make_raddino(device=device, output_keys={"img", "mask"})` → dense `Tensor[B, D, H, W]`),
  build the dataset with `output_mask=True`, and pair with `LinearProbeSegEvaluator` /
  `ConvProbeSegEvaluator` / `UPerNetSegEvaluator`. See the `*_seg` evaluator pages.

### 6. Transforms / augmentation

`RadiologyTransform2D` / `RadiologyTransform3D` are fluent builders
(`.with_flip()`, `.with_rotate()`, ...) producing the MONAI transform a dataset consumes. For an
evaluator, use the recipe's transform instead so preprocessing matches the backbone. See
[transforms](docs/wiki/transforms.md).

### 7. The Gradio app

```bash
python app.py       # then open http://localhost:7860
```

Datasets are grouped by modality (CXR / CT / MRI / Radiograph). For SSH-tunnel remote access see
[App](docs/wiki/app.md).

### 8. Confidence intervals on the metrics

Every classification / segmentation evaluator returns a **per-row** DataFrame
(`label, n_train, fold, seed, bootstrap, <metrics>`; point-estimate rows carry `bootstrap == -1`).
There are two ways to attach a 95% CI, matched to how the evaluator was run — quote CIs, not bare
point estimates:

- **k-fold cross-validation → Student-t interval across the folds.** With `n_folds=k` you get one
  point-estimate row per fold, so the CI is the small-sample t-interval over the `k` fold values:
  `mean ± t(0.975, k-1) · s / sqrt(k)`. This is the correct estimator for a 5-fold run; prefer it
  over the built-in `save_results()` summary, whose `_ci_lo/_ci_hi` are a *percentile* CI that is
  coarse over only 5 points.

  ```python
  import numpy as np, pandas as pd
  from scipy import stats

  def kfold_ci(df, metrics=("auroc", "auprc", "f1"), alpha=0.05):
      """Per-label 95% CI from the t-distribution across fold rows."""
      folds = df[df["bootstrap"] == -1]           # one point estimate per fold
      out = []
      for label, sub in folds.groupby("label"):
          row = {"label": label}
          for m in metrics:
              v = sub[m].dropna().to_numpy()
              n = len(v)
              mean = v.mean() if n else np.nan
              if n > 1:
                  sem = v.std(ddof=1) / np.sqrt(n)
                  h = stats.t.ppf(1 - alpha / 2, n - 1) * sem
              else:
                  h = np.nan
              row[f"{m}_mean"], row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = mean, mean - h, mean + h
          out.append(row)
      return pd.DataFrame(out)
  ```

- **Fixed train/test split → bootstrap interval.** Build the evaluator with `train_dataset=` /
  `test_dataset=` (not k-fold) and `n_bootstrap=1000` (optionally `n_seeds>1`). The evaluator then
  emits bootstrap rows, and `save_results()` writes `<metric>_ci_lo/_ci_hi` (2.5 / 97.5 percentile)
  into `results_summary.csv`. **`n_bootstrap` is ignored in k-fold mode** (it warns), so choose one
  mode: k-fold + t-interval, *or* fixed-split + bootstrap — you can't get both from one run.

Present each metric as `mean [lo, hi]`.

### 9. Visualizing the results

Produce **two figures per run**, saved as PNG (`fig.savefig(path, dpi=150, bbox_inches="tight")`) —
**never `plt.show()`**, which is a no-op off-notebook:

**(a) Results plot — `outputs/<task>/results.png`.** Plot the aggregated per-label metrics with
their 95% CI as error bars (a bar or point plot). Feed it the `mean [lo, hi]` table from playbook 8:

```python
import matplotlib.pyplot as plt
summary = kfold_ci(df)                       # or the bootstrap _ci_lo/_ci_hi summary
m = "auroc"                                   # one panel per metric, or subplots
lo = summary[f"{m}_mean"] - summary[f"{m}_ci_lo"]
hi = summary[f"{m}_ci_hi"] - summary[f"{m}_mean"]
fig, ax = plt.subplots(figsize=(max(4, 0.5 * len(summary)), 4))
ax.bar(summary["label"], summary[f"{m}_mean"], yerr=[lo, hi], capsize=4)
ax.set_ylabel(m); ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=45)
fig.savefig("outputs/<task>/results.png", dpi=150, bbox_inches="tight")
```

**(b) Sample predictions overlaid on the input image — `outputs/<task>/samples.png`.** Pair a few
held-out samples with what the model produced and overlay it on the source image. The datathon ships
ready-made helpers in [datathon26/viz_helpers.py](datathon26/viz_helpers.py) — reuse them, then save
the current figure (`plt.gcf().savefig("outputs/<task>/samples.png", dpi=150, bbox_inches="tight")`).

| Task | Helper | Overlay it draws |
|---|---|---|
| classification probe | `show_predictions(probe, ds, image_dir)` | held-out images titled `P(label)`, predicted vs true |
| segmentation probe   | `show_masks(seg_probe, ds)` | image, ground-truth mask, predicted mask (colored overlay) |
| VQA                  | `show_answers(vqa_ds, answerer)` | image + question + reference vs model answer |
| report generation    | `show_reports(gen_ds, generator)` | image beside reference vs generated findings |

Underlying pattern when you're **not** in the datathon (write the few lines yourself, save to disk):

- **Segmentation** exposes it cleanly: `seg_probe.fit(ds)`; `idx = seg_probe.inner_val_indices(ds)[:4]`;
  `panels = seg_probe.predict(ds, indices=idx, return_images=True)` → each panel is a dict with
  `image`, `y_true`, `y_pred`; overlay with `plt.imshow(img, cmap="gray")` +
  `plt.imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=0.5)`.
- **VQA / report:** `td = ds.get_datasets(n_splits=None)`; `samples = [td[i] for i in range(n)]`;
  `preds = answerer([s["img"] for s in samples], [s["question"] for s in samples])` (report gen:
  `generator([s["img"] for s in samples], [None] * n)`); each `s["img"]` is a PNG path.
- **Classification:** read `predict_proba` off the head fitted on one held-out fold's cached
  embeddings, then title each image with `P(label)` vs its true label.

## Datathon26 participants

Grounded in [Datathon26](docs/wiki/datathon26.md) and the `datathon26/` notebooks.

- **Notebook order:** `datathon26/0_setup … 8_build_your_own` — work through them in sequence;
  `demo_datathon_evaluators.ipynb` is the self-contained end-to-end walkthrough.
- **Three environments (one per model family).** `datathon26/radharmony_venv_installation.sh`
  builds three separate venvs, each registered as a Jupyter kernel, because CheXagent-2 and MAIRA-2
  pin an **old `transformers`** that conflicts with the newer one RadEval (RadGraph scoring) needs.
  Pick the kernel that holds your model; use its interpreter (`.venv-<name>/bin/python`) for the
  discovery / verify commands too:

  | venv / kernel | display name | models it holds | RadGraph (`radeval`)? |
  |---|---|---|:--:|
  | `.venv-medgemma` / `medgemma`   | `RadHarmony-(medgemma)`  | MedSigLIP, MedGemma (VQA + report), RAD-DINO, BiomedCLIP | ✅ |
  | `.venv-chexagent` / `chexagent` | `RadHarmony-(chexagent)` | CheXagent-2 (VQA + report) | ❌ |
  | `.venv-maira2` / `maira2`       | `RadHarmony-(maira2)`    | MAIRA-2 (report) | ❌ |

  Only `.venv-medgemma` has `radeval`, so **RadGraph scoring runs only there.** Consequence for
  report generation: MedGemma generates *and* scores inline in one `evaluate()`; CheXagent-2 /
  MAIRA-2 must use the **two-stage split** — `ev.generate_only("pairs.parquet")` in their own
  kernel, then `compute_report_metrics(...)` on the parquet in the `medgemma` kernel.
- **Paths:** participant notebooks run **from `datathon26/`** and do `from basepaths import *`
  directly — a bare import, **no** `import sys` / `sys.path.insert(0, os.path.abspath(".."))`.
  `basepaths.py` exposes `SESSION_DATA`, `MIMIC_DIR`, `MONTGOMERY_DIR`, `VINDR_*`, `SIIM_*`,
  etc., all keyed off a single `_ROOT` so one line flips between the local data root and the
  datathon cloud.
- **Scoring flow (reportbench):** studies are submitted to an external service that returns
  embeddings / reports; `DatathonEmbeddingDataset` / `DatathonReportDataset` +
  `ReportBenchClient` stage, submit, and return scored datasets for the classification /
  segmentation / report-generation tasks.
- **Security:** **never** put a real reportbench API key into a committed notebook, and never
  commit MIMIC-derived data (report text, DICOM-derived images) — it is under the PhysioNet DUA.

## Verify before you hand it over

Cheap, fast, no training and no weight downloads. Confirm imports resolve, kwargs exist in the
signature, and any registry key is real, using the interpreter where `radharmony` is installed
(see the discovery block above for which `python` to use; `import radharmony.evaluator` pulls
torch/MONAI — a one-time ~6s warmup):

```bash
python -c "import inspect; from radharmony.evaluator import LinearProbeEvaluator as C; print(inspect.signature(C.__init__))"
python -c "from radharmony.dataset import CheXpertTrainDataset; from radharmony.registry import list_datasets; print('chexpert_train' in list_datasets())"
```

If a kwarg you used isn't in the signature (directly or via `**base_kwargs` on the base class),
fix the snippet — don't present code that would raise `TypeError`.

## What NOT to do

- **Don't hardcode a stale catalog** of datasets/evaluators/backbones — discover via
  `list_datasets()` / `list_evaluators()` / `backbones.__all__`.
- **Don't fetch the web** when the local repo (wiki + source) is present.
- **Don't invent kwargs or registry keys** — verify against the real signature / registry first.
- **Don't run training, `.evaluate()`, or download model weights** just to "check" a snippet;
  a signature/import smoke check is enough.
- **Don't mismatch transform and backbone** — build the dataset with the recipe's `transform`.
- **Don't report bare point estimates, and don't cross wires on CIs** — quote `mean [lo, hi]`;
  use the t-interval across folds for k-fold (playbook 8), bootstrap `_ci_lo/_ci_hi` for a fixed
  split, and don't set `n_bootstrap` in k-fold mode (it's ignored).
- **Don't call the helper's `plt.show()` in a headless agent** — save the figure to `outputs/`.
- **Don't hardcode `device="cuda"`** — select it from `torch.cuda.is_available()` and pass the
  same `device` to the recipe and the evaluator; fall back to `"cpu"` if the GPU is occupied.
- **Don't commit MIMIC-derived data or a real datathon API key.**
- **Don't extend the package here** — adding a dataset/backbone belongs to `add-dataset` /
  `add-backbone`.

## After editing — sync the skill file

This skill is repo-specific, so mirror it into the public RadHarmony repo (version-controlled
alongside the other skill mirrors):

```bash
cp ~/.claude/skills/use-radharmony/SKILL.md docs/use_radharmony_skill.md
md5sum ~/.claude/skills/use-radharmony/SKILL.md docs/use_radharmony_skill.md
```

If the repo copy was edited instead, sync the other direction. Always confirm both copies match
with `md5sum`.
