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
   the checkout. Key pages: [quickstart](../../RadHarmony/docs/wiki/quickstart.md),
   [Dataset API](../../RadHarmony/docs/wiki/api.md),
   [transforms](../../RadHarmony/docs/wiki/transforms.md),
   [Evaluator index](../../RadHarmony/docs/wiki/evaluator/index.md) + per-evaluator pages,
   [Backbones](../../RadHarmony/docs/wiki/backbones/index.md),
   [App](../../RadHarmony/docs/wiki/app.md), [Datathon26](../../RadHarmony/docs/wiki/datathon26.md).
   (Paths are relative to the repo root — read whatever exists under `docs/wiki/`.)
2. **Local source** (`radharmony/**/*.py`) — drop into the code when a wiki page is thin or an
   exact constructor signature, default, `LABEL_COLS`, or registry key must be confirmed. The
   source is the final authority and always matches the installed package.
3. **Live site** `https://f10409.github.io/RadHarmony` via `WebFetch` — **only** when there is no
   local repo. It can lag the user's installed version, so prefer local whenever it exists.

**Names are discovered, never hardcoded from memory** — the catalog changes over time. Note the
dataset registry only fills once `radharmony.dataset` is imported (the `@register_dataset`
decorators run on import), so import it *before* calling `list_datasets()`:

```bash
.venv/bin/python -c "import radharmony.dataset; from radharmony.registry import list_datasets; print(len(list_datasets())); print(list_datasets())"
.venv/bin/python -c "from radharmony.evaluator import list_evaluators; print(list_evaluators())"
.venv/bin/python -c "import radharmony.evaluator.backbones as b; print([n for n in b.__all__ if n.startswith('make_')])"
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
  and returns a metrics **`DataFrame`** (AUROC, AUPRC, F1, ...). The encoder contract is
  `encoder(imgs: Tensor[B, ...]) -> Tensor[B, D]`; zero-shot also needs
  `text_encoder(list[str]) -> Tensor[T, D]`.
- The dataset's `transform` must be the one the backbone recipe returns, so pixel preprocessing
  matches the model's pretraining. Always build the dataset with `transform=` from the recipe.

## Task playbooks

Give the user the smallest correct snippet, then the wiki page to read for options. Fill real
paths from the [NAS paths](../../RadHarmony/docs/wiki/nas_paths.md) page or ask the user.

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
See [Dataset API](../../RadHarmony/docs/wiki/api.md) and the per-dataset page under
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

Read [Dataset API](../../RadHarmony/docs/wiki/api.md) for the harmonizer/`harmonized_df` flow;
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
swap the class. See [Evaluator index](../../RadHarmony/docs/wiki/evaluator/index.md) and the
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

See [zero_shot](../../RadHarmony/docs/wiki/evaluator/zero_shot.md).

### 5. Fine-tune and segmentation

- **Fine-tune:** `FinetuneEvaluator(encoder, train_dataset=..., test_dataset=..., epochs=20, lr=1e-4, ...)` —
  single-GPU end-to-end. See [finetune](../../RadHarmony/docs/wiki/evaluator/finetune.md).
- **Segmentation:** switch the recipe to segmentation mode
  (`make_raddino(device=device, output_keys={"img", "mask"})` → dense `Tensor[B, D, H, W]`),
  build the dataset with `output_mask=True`, and pair with `LinearProbeSegEvaluator` /
  `ConvProbeSegEvaluator` / `UPerNetSegEvaluator`. See the `*_seg` evaluator pages.

### 6. Transforms / augmentation

`RadiologyTransform2D` / `RadiologyTransform3D` are fluent builders
(`.with_flip()`, `.with_rotate()`, ...) producing the MONAI transform a dataset consumes. For an
evaluator, use the recipe's transform instead so preprocessing matches the backbone. See
[transforms](../../RadHarmony/docs/wiki/transforms.md).

### 7. The Gradio app

```bash
python app.py       # then open http://localhost:7860
```

Datasets are grouped by modality (CXR / CT / MRI / Radiograph). For SSH-tunnel remote access see
[App](../../RadHarmony/docs/wiki/app.md).

## Datathon26 participants

Grounded in [Datathon26](../../RadHarmony/docs/wiki/datathon26.md) and the `datathon26/` notebooks.

- **Notebook order:** `datathon26/0_setup … 8_build_your_own` — work through them in sequence;
  `demo_datathon_evaluators.ipynb` is the self-contained end-to-end walkthrough.
- **Paths:** participant notebooks run **from `datathon26/`** and do `from basepaths import *`
  directly — a bare import, **no** `import sys` / `sys.path.insert(0, os.path.abspath(".."))`.
  `basepaths.py` exposes `SESSION_DATA`, `MIMIC_DIR`, `MONTGOMERY_DIR`, `VINDR_*`, `SIIM_*`,
  etc., all keyed off a single `_ROOT` so one line flips between the NAS and the datathon cloud.
- **Scoring flow (reportbench):** studies are submitted to an external service that returns
  embeddings / reports; `DatathonEmbeddingDataset` / `DatathonReportDataset` +
  `ReportBenchClient` stage, submit, and return scored datasets for the classification /
  segmentation / report-generation tasks.
- **Security:** **never** put a real reportbench API key into a committed notebook, and never
  commit MIMIC-derived data (report text, DICOM-derived images) — it is under the PhysioNet DUA.

## Verify before you hand it over

Cheap, fast, no training and no weight downloads. Confirm imports resolve, kwargs exist in the
signature, and any registry key is real, using the repo venv (`import radharmony.evaluator` pulls
torch/MONAI — a one-time ~6s warmup):

```bash
.venv/bin/python -c "import inspect; from radharmony.evaluator import LinearProbeEvaluator as C; print(inspect.signature(C.__init__))"
.venv/bin/python -c "from radharmony.dataset import CheXpertTrainDataset; from radharmony.registry import list_datasets; print('chexpert_train' in list_datasets())"
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
