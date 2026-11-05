---
name: use-radharmony
description: >
  Help a user use the RadHarmony package: load a dataset, split it, run a probe, zero-shot, fine-tune,
  segmentation or report generation with a backbone recipe, compare backbones, read or plot the result
  table, or start the app. It finds the answer in the wiki, gives a short snippet, and checks the snippet
  first. Use when the user asks "how do I use RadHarmony", "load <dataset>",
  "run a probe", or "which datasets, evaluators or backbones are there". To add a dataset or a backbone, use
  add-dataset or add-backbone. To review or fix the repo, use evaluate or fix.
---

# Use RadHarmony

Answer in the form of step 5. Check every snippet before you give it (step 4).

## Where the facts are

The wiki is the home of the facts. A page `<page>` is in 2 places:

- In a RadHarmony clone: `docs/wiki/<page>.md`. Read this copy first. It matches the code of the clone.
- Online: `https://f10409.github.io/RadHarmony/<page>.html`. Use it when there is no clone (RadHarmony is an
  installed package). It can be newer or older than the installed version.

When a page does not answer, read the code: the signature, the docstring, `LABEL_COLS`. In a clone, the code is
in `radharmony/`. Outside a clone, use `inspect` (step 2). The code wins over the docs. Do not answer from
memory.

| The user wants to | Read first (`<page>`) |
|---|---|
| Install, or pick an extra | `get-started/install` |
| Do a first run from start to end | `get-started/first-benchmark` |
| Know the parts (harmonizer, harmonized table, dataset, evaluator) | `get-started/concepts` |
| Download a dataset | `how-to/get-the-data` |
| Find a dataset (registry key, task, outputs, size, license) | `reference/datasets/index`, then the dataset page (the Page link of its row, `reference/datasets/<group>/<name>`) |
| Load a dataset and pick the outputs (`cls`, `mask`, `bbox`, `report`, `reg`) | `how-to/load-a-dataset`, `reference/api-datasets` |
| Save, load or filter a harmonized table | `how-to/save-and-reuse` |
| Split a dataset (official split, train and val, k folds), augment | `how-to/split-data`, `reference/transforms` |
| Cache the samples | `how-to/cache` |
| Run a linear, kNN, SVM or prototype probe, or zero-shot | `how-to/run-a-probe`, `reference/evaluators/index`, `reference/evaluators/zero_shot` |
| Fine-tune | `reference/evaluators/finetune` |
| Run segmentation | `how-to/run-segmentation` |
| Run report generation | `how-to/run-report-generation` |
| Pick or compare backbones | `reference/backbones/index`, `how-to/compare-backbones` |
| Run my own encoder (an `nn.Module` or a function, no recipe) | `reference/evaluators/index` (section "Encoder contract"), `how-to/add-a-backbone` (steps 1 to 3), `how-to/compare-backbones` |
| Read the result table, report a CI, plot the results | `reference/evaluators/index` (section "Result table"), `references/results.md` |
| Use the app (look at samples, try transforms). It needs a clone: `app.py` is not in the package | `how-to/use-the-app` |
| Know why (label policy, DICOM load, design decisions) | `explanation/label-policy`, `explanation/dicom-loading`, `design` |

`references/results.md` is in this skill folder. It has the CI rules and 2 plot snippets.

## Step 1. Find the names

Get the names from the installed package, not from memory. The catalog changes. Use the Python of the venv
that has RadHarmony (in a clone, the dev venv: `.venv/bin/python`). Replace `python` in the commands with that
path. A snippet with a backbone recipe needs the packages of its extra. If the dev venv does not have them,
use the model venv: `$RADHARMONY_MODEL_PYTHON`, the Python of a venv with the backbone packages, for example
`/path/to/model/.venv/bin/python`.

```bash
python -c "from radharmony.registry import list_datasets; print(list_datasets())"
python -c "from radharmony.evaluator import list_evaluators; print(list_evaluators())"
python -c "import radharmony.evaluator.backbones as b; print(b.__all__)"
python -c "from radharmony.registry import list_datasets, resolve_dataset; print([k for k in list_datasets() if 'mask' in resolve_dataset(k).SUPPORTED_OUTPUTS])"
```

The last command lists the datasets with an output, here `mask`. Use `bbox`, `report`, `cls` or `reg` for the
others.

`resolve_dataset(key)` (in `radharmony.registry`) and `resolve_evaluator(key)` (in `radharmony.evaluator`) give
the class of a key.

## Step 2. Read the page and the signature

Read the page from the table. Take the snippet of the page as the start. Check each argument that you add:

```bash
python -c "import inspect; from radharmony.evaluator import LinearProbeEvaluator as C; print(inspect.signature(C.__init__))"
```

An evaluator passes `**base_kwargs` to its base class. So also read the signature of the base class, or the
"Arguments" table of its page.

## Step 3. Write the snippet

- Build the dataset with the `transform` of the backbone recipe (D-16). Example: `transform, encoder =
  make_raddino(device=device)`, then `MontgomeryCXRDataset(..., transform=transform)`.
  - No backbone (only load the data): `transform=None` gives the default image steps.
  - The user's own encoder: give a transform that matches the encoder. Example:
    `RadiologyTransform2D(img_size=224, channels=3, output_keys={"img", "cls"}).get_transform()`, or
    `EncoderPreprocessTransform.from_custom` with `RadiologyEncoderTransform` (`how-to/add-a-backbone`, step
    2). An own transform lists all the keys of the flags in `output_keys`, or the dataset warns (D-11).
- Pick the device: `device = "cuda" if torch.cuda.is_available() else "cpu"`. Give the same `device` to the
  recipe and to the evaluator.
- Use `/data/...` for the data folder, or the folder that the user gives. Ask the user when you do not know
  it. When no user can answer (a sub-agent, a headless run), write the question and the answer that you chose
  in the report, and go on. Never write a path of another person or another machine.
- Use `cache_dir=None` for a short try. The default `"./cache"` writes files (`how-to/cache`).
- Turn on only the `output_*` flags that the task needs.

Traps that users often hit. Say the one that applies:

- `ds.get_datasets(n_splits=5)` gives one train and val split (1/5 of the patients in val), not 5 folds.
  For folds, use `ds.get_folds(n_splits=5)`, or `n_folds=5` of the evaluator (`how-to/split-data`).
- K-fold mode (`dataset=`, `n_folds=`) ignores `n_seeds` and `n_bootstrap`, and warns (D-19). For bootstrap
  CIs, use `train_dataset=` and `test_dataset=`.
- Zero-shot gives NaN for F1 and the other threshold metrics, unless you set
  `threshold_strategy="fixed:<value>"`. AUROC and AUPRC need no threshold (`how-to/run-a-probe`, step 5).
- Zero-shot has 1 run, so its CI is NaN unless you set `n_bootstrap` (for example 100).
- The segmentation evaluators take 2-D images and masks only. A dataset with 3-D masks (RSNA 2022 Cervical
  Spine) does not fit them (`how-to/run-segmentation`).
- The probes L2-normalize the features (D-17). Zero-shot always uses cosine similarity.
- 2 datasets can name the same finding in different ways. Example: ChestX-ray14 has `effusion`, CheXpert has
  `pleural_effusion`. Compare `ds.LABEL_COLS` before you mix datasets. An evaluator takes one RadHarmony dataset,
  because it needs the patient ids of its table. `torch.utils.data.ConcatDataset` is only for your own
  training loop.
- The 2 report generators need their own venv (`get-started/install`, section "More than one venv").

## Step 4. Check the snippet

Save the snippet as a `.py` file in a temporary folder. Run `scripts/check_snippet.py` of this skill folder
(the folder of this SKILL.md) with the Python of the venv:

```bash
python <skill folder>/scripts/check_snippet.py /tmp/snippet.py
```

- In a clone, the script uses the functions of the block runner of the docs (`scripts/evalkit/run_blocks.py`).
  It finds the clone from the skill folder, or else from the current folder (`git rev-parse`). It writes no
  file in the repo. Outside a clone, it does the same static checks itself.
- The static checks: the snippet compiles, each radharmony import exists, and each keyword argument is in the
  signature. The script must print `PASS`. On `FAIL`, fix the snippet and check again.
- `--run` also runs the snippet in a temporary folder, and prints the last 20 lines of its output. Use these
  lines for the output comments of step 5. The limit is 300 s for each block. Give your shell command a longer
  time limit. In a clone, set `RADHARMONY_MODEL_PYTHON`: the script runs the snippet again with that Python
  when a package is missing.
- `BLOCKED` means that data, a package or a GPU is missing. A `/data/...` path always gives `BLOCKED`. To run
  on real data, run a copy of the snippet with the real folder, and give the user the `/data/...` version.
  The real folder is the one that the user gives. In a clone, it can also come from the data paths file
  (`get-started/install`, "Data paths for the checks"). `paths("<key>")` of `scripts/dataset_paths.py` reads
  `configs/data_paths.json`, or the file that `RADHARMONY_DATA_PATHS` names. With no entry, it raises
  `MissingPathsError` with a hint. Then ask the user for the folder, or tell the user which part you did not
  run.
- When the user asks for a result, run the snippet on the user's data and give the real numbers. A small run
  is fine as a check: on Montgomery (138 images, 1 GPU), a linear probe took 44 s and zero-shot took 29 s. Do
  not start a long training run or a large weight download only to check a snippet.

## Step 5. Answer

1. The idea, in 1 to 3 sentences.
2. The snippet, with the output as a comment when you know it from the page or a run.
3. The `<page>` to read next, as a repo path in a clone, or as a URL.
4. The check result (`PASS`, or what you could not run).

Give each result with its CI, as `mean [ci_lo, ci_hi]` (`references/results.md`).

## Datathon 2026

When the user asks about Datathon 2026 (`radharmony.datathon26`), use the `use-datathon26` skill. It is in
`.claude/skills/use-datathon26/` of the repo. The page is `reference/datathon26`.

## Do not

- Do not extend the package here. A new recipe file or a new dataset class is the job of add-backbone or
  add-dataset. To wrap the user's own encoder for a run is in scope.
- Do not give a kwarg or a registry key that you did not check.
- Do not call `plt.show()`. Save each figure to a file.
- Do not commit data with a data use agreement (for example MIMIC report text or images), or an API key.
