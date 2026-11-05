---
name: add-backbone
description: >
  Add a new image encoder to RadHarmony as a backbone recipe. It writes the make_<name>() factory in
  radharmony/evaluator/backbones/, the exports, the install extra, the backbone page and its table rows. It
  checks the input size, the crop, the mean, the std and the pooled token against the authors' code. Use when
  the user says "add a backbone", "wire <model> into the evaluator", "make a recipe for <model>", or gives a
  notebook in notebooks/backbones/ to turn into a recipe. To run an existing backbone, use use-radharmony. To
  add a dataset, use add-dataset. To review the branch, use evaluate.
---

# Add a backbone

A backbone recipe is a `make_<name>()` function. It returns a dataset transform and an encoder, and for a
text-and-image model also a text encoder and a tokenizer. The person's guide is
`docs/wiki/how-to/add-a-backbone.md`. This skill follows the same rules and adds the steps for the repo.

This skill changes the repo. Run it in a RadHarmony clone, not in a project that installs RadHarmony as a
package. It uses 2 venvs:

- The dev venv: `.venv/bin/python`.
- The model venv: `$RADHARMONY_MODEL_PYTHON`, the Python of a venv with the backbone packages, for example
  `/path/to/model/.venv/bin/python`. Make it with `uv pip install -e ".[<name>]"`
  (`docs/wiki/get-started/install.md`). If the variable is not set, ask the user for the path, and set it.

Real data. Steps 6 and 7 need 4 chest X-rays of Montgomery County CXR (key `montgomery_cxr`). The data
folders come from the data paths file: `configs/data_paths.json`, or the file that `RADHARMONY_DATA_PATHS`
names (`docs/wiki/get-started/install.md`, "Data paths for the checks"). Read them with `paths(key)` of
`scripts/dataset_paths.py`. If the file or the key is missing, `paths()` raises `MissingPathsError` with a hint.
Then ask the user for the folder, and offer one of 3 choices:

1. Add the entry to their data paths file. Copy `configs/data_paths.example.json` first if the file is missing.
2. Give the folder for this run only. Montgomery is public and small: `docs/wiki/how-to/get-the-data.md`
   shows the download.
3. Skip the real-data check, and say so in the report.

A sub-agent cannot ask the user: it skips, and it reports what it skipped.

`<name>` is the short name of the new recipe, in snake case, for example `my_model`. It names the file
`radharmony/evaluator/backbones/<name>.py`, the factory `make_<name>()`, the install extra `<name>` and the page
`docs/wiki/reference/backbones/<name>.md`.

The references are in `.claude/skills/add-backbone/references/`:

- `authors-check.md`: the fact table, the compare script and the test (J-05).
- `recipe-rules.md`: the rules for the recipe file, and more `model_call` functions.
- `backbone-page.md`: the backbone page template and the other doc changes.

## Step 0. Read first

1. `CLAUDE.md`, section "Design decisions and rules". Read D-16 (backbones match their authors) and D-17
   (L2-normalize features) in `docs/wiki/design.md`.
2. `CLAUDE.md`, section "Base code needs explicit approval". A recipe does not need base code. If you think
   that it does, stop and follow that section.
3. `docs/wiki/how-to/add-a-backbone.md` and `docs/wiki/reference/backbones/index.md`.
4. Ask the user for what you do not know: the model card or the authors' repo, the variant, `<name>`, and
   where the weights are. Ask one question at a time. When no user can answer (a sub-agent, a headless run),
   write the question and the answer that you chose in the report, and go on.

## Step 1. Find the authors' facts

Fill the fact table of `references/authors-check.md`, part 1, before you write code. It has 4 facts: the input
size, the resize and crop, the mean and std, and the pooled token. Give a source for each value. Also find:

- The feature size `D`: `model.config.hidden_size`, or one forward pass on a zero tensor.
- The patch size and the register tokens `R` (ViT only), for the segmentation grid.
- The ecosystem: Hugging Face, OpenCLIP, timm, a pip package, or a clone of the authors' repo.

Show the table to the user. When 2 sources of the authors give different values and D-16 does not say which
one to use, ask the user with `.claude/skills/evaluate/references/decisions.md`.

## Step 2. Write the recipe

1. Copy `radharmony/evaluator/backbones/_template.py` to `radharmony/evaluator/backbones/<name>.py`. Rename
   `make_my_model`, `_MY_MODEL_HUB`, `MyModel` and the `my_model` extra to the names of `<name>`.
2. Fill the sections in order. Follow `references/recipe-rules.md`.
3. Pick `model_call` from the table in the how-to page (step 1) or in `references/recipe-rules.md`.
4. Image only: return `(transform, encoder)`. Text and image: return
   `(transform, image_encoder, text_encoder, tokenizer)`.

## Step 3. Register the recipe

In `radharmony/evaluator/backbones/__init__.py`:

- Add an entry to the module docstring, next to the others. Give the function, one line on the model, the
  feature size, the input size, the return value and the install command.
- Add `from .<name> import make_<name>`.
- Add `"make_<name>"` to `__all__`.

## Step 4. Add the install extra

1. In `pyproject.toml`, add `<name> = [...]` under `[project.optional-dependencies]`, next to the other
   backbone extras. Name each package that the recipe imports. When the authors' model code is a clone or
   Hugging Face remote code, also name each package that it imports (L-019). A pip package of the authors
   (for example `rad-dino`) brings its own dependencies, so name only that package. A Hugging Face recipe often
   needs only `"transformers>=5.6.0"`. Add `sentencepiece` and `protobuf` when the tokenizer needs them.
2. Add pairs to the `[tool.uv]` `conflicts` list. A pair is one line:
   `[{ extra = "chexagent_gen" }, { extra = "<name>" }],`. If the extra needs `transformers>=5.6.0`, add a
   pair with `chexagent_gen` and a pair with `maira2_gen`. If it needs only `torchvision>=0.26.0`, add the
   pair with `chexagent_gen`.
3. If the recipe imports `torchvision`, add `"<module>": "<extra>"` to `BACKBONE_EXTRA` in
   `tests/test_packaging.py`, for example `"eva_x": "eva_x"`.
4. If the recipe imports the authors' model code (a clone or Hugging Face remote code), add a case to
   `_L019_CASES` in `tests/test_lessons.py`.

## Step 5. Smoke test

Without the extra, the import must work, because the model imports are lazy. Block each top-level package
that the recipe imports (here `transformers`), so the check works also in a venv that has the extra:

```bash
.venv/bin/python -c "import sys; sys.modules['transformers'] = None; import radharmony.evaluator.backbones as b; print('make_<name>' in b.__all__)"
```

Then use the model venv and one free GPU (here GPU 0). Check both modes:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD $RADHARMONY_MODEL_PYTHON - <<'EOF'
import torch
from radharmony.evaluator.backbones import make_<name>
x = torch.zeros(1, 3, 224, 224, device="cuda")  # use the input size of the recipe
_, encoder = make_<name>(device="cuda", dtype=torch.float32)[:2]
print(encoder(x).shape)  # [1, D]
_, encoder = make_<name>(device="cuda", output_keys={"img", "mask"}, dtype=torch.float32)[:2]
print(encoder(x).shape)  # [1, D, G, G], G = input size / patch size
EOF
```

## Step 6. Check against the authors

A right shape does not prove a right recipe. Do parts 2 and 3 of `references/authors-check.md`:

1. Write the authors' pipeline in a file in a temporary folder, for example `/tmp/`. Run
   `.claude/skills/add-backbone/scripts/compare_with_authors.py` with the model venv on 4 real images
   ("Real data" above). It must print `PASS`. If the recipe has a segmentation mode, run it again with
   `--seg`. It must print `PASS` again.
2. Add a test to `tests/test_backbones.py` that compares the input and the feature with the authors' code.
   Run it with the model venv. It must pass.

If the script fails, fix the recipe, not the authors file. Then run the script again.

## Step 7. Write the docs

1. Run the Quick use and Segmentation mode blocks of the template (`references/backbone-page.md`) one time by
   hand. Use the model venv, and the Montgomery folder ("Real data" above) as `base_image_dir`.
2. Write `docs/wiki/reference/backbones/<name>.md` from the template. Put the printed output of item 1 in the
   comments of the blocks. Keep `/data/...` in the page.
3. Do each item of "Other doc changes" in `references/backbone-page.md`.

Check the new page. The block runner runs a page again with `$RADHARMONY_MODEL_PYTHON` when a package is
missing:

```bash
.venv/bin/python scripts/evalkit/wiki_audit.py --page reference/backbones/<name>.md
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evalkit/run_blocks.py --level full \
    --page docs/wiki/reference/backbones/<name>.md
```

The audit must give 0 errors. The block runner must give no `failed`. It gives `blocked needs data` for a
`/data/...` path, because it does not map that path to the real folder. That is why item 1 runs by hand.

## Step 8. Proof

1. A fast first check: run the STE linter on each changed `.md` and `.py` file,
   `.venv/bin/python scripts/evalkit/ste_lint.py <files>`. It must print nothing.
2. Run the fast checks: `.venv/bin/python scripts/evalkit/run_all.py --group lint,docs`, and the tests of
   step 4 and step 6 with the model venv:
   `PYTHONPATH=$PWD $RADHARMONY_MODEL_PYTHON -m pytest tests/test_backbones.py tests/test_packaging.py -q`
   (about 30 s). Add `tests/test_lessons.py -k l019` when you added an L-019 case.
3. Run the automatic checks once: `.venv/bin/python scripts/evalkit/run_all.py`. Run it in the background and
   wait for the notice: it can take more than 1 hour. Add `--full` only where the data is:
   `.venv/bin/python -m pytest tests/test_data_paths.py -k paths_exist_on_nas -q` passes (it skips when the
   data is not mounted). The checks must pass.
4. Run `/evaluate` on the work. Uncommitted work: use `--base <current branch>`. Committed work: use
   `--base <the commit before the first change>`. Do not run the full checks a second time: `/evaluate` step 3
   runs them.
5. The work is done only when the automatic checks pass and the report has no open high finding. If
   `/evaluate` reports an open high finding, fix it, or tell the user why not.
6. If you found a new kind of mistake, add a rule to `eval/RULES.md` as a test or a lint. Follow its section
   "How to add a rule (lessons)". Do not add the rule to this skill as text.

## Report to the user

1. The recipe file, and the `model_call` that you chose, in one sentence.
2. The filled fact table, with a source for each fact.
3. The output of `compare_with_authors.py`, and the name of the new test.
4. The other changed files, one line each.
5. The checks and their results. Name each check that you could not run (no GPU, no weights, no extra), so the
   user can run it.
