---
name: add-backbone
description: >
  Scaffold a new image encoder backbone into the RadHarmony evaluator subsystem.
  Creates the make_<name>() factory in radharmony/evaluator/backbones/, wires it
  into the package exports, adds an optional-dependency extra in pyproject.toml,
  and updates both backbone-recipe tables (README extras + evaluator wiki). Use
  when the user says "add a backbone", "wire <model> into the
  evaluator", "make a recipe for <model>", or hands you a notebook in
  notebooks/backbones/ and asks to turn it into a recipe. Covers both vision-only
  and vision-language (zero-shot capable) models.
---

# Add Backbone Skill

Wire a new image encoder into RadHarmony's evaluator subsystem as a drop-in
`make_<name>()` factory. The factory returns a ready-to-use MONAI transform
plus an `ImageEncoderWrapper`-wrapped encoder, so the user can pair it with
any RadHarmony dataset and any evaluator (`LinearProbeEvaluator`,
`ZeroShotEvaluator`, etc.) in three lines.

## When This Skill Triggers

- User asks to "add a backbone", "wire <model> into the evaluator", "make a
  recipe for <model>", "scaffold a backbone".
- User hands you a notebook in `notebooks/backbones/` (e.g.
  `backbone_<model>.ipynb`) and asks to turn it into a callable recipe.
- User asks how `make_raddino` / `make_biomed_clip` / etc. were built and
  wants to replicate the pattern for a new model.

## Mental model

RadHarmony's evaluator contract is `Callable[[Tensor], Tensor[B, D]]` — every
encoder must satisfy it. Two pieces solve the gap from raw weights to that
contract, and they live on opposite sides:

| Piece | Lives on | Responsibility |
|---|---|---|
| **Preprocessing transform** | Dataset side (CPU, DataLoader workers) | Read image file → resize → normalize → return `Tensor[C, H, W]` per sample |
| **Encoder wrapper** | Model side (GPU) | Route `forward(x)` to the backbone, extract a pooled `Tensor[B, D]` |

Each backbone has a unique pretraining recipe (ImageNet norm for HF ViTs,
`(0.5, 0.5, 0.5)` for BiomedCLIP, raw `[-1, 1]` for some CXR-SSL models).
Mixing readers or normalization between recipes silently perturbs pixel
statistics. **Each `make_<name>()` factory bundles both pieces — preprocessing
on the transform side, forward routing on the wrapper side — so the caller
gets a guaranteed-correct pair.**

## Mandatory design rules

These hold for **every** recipe. The template (`_template.py`) follows them;
existing recipes (`raddino.py`, `chexagent.py`, `medsiglip.py`, `dinov3.py`)
illustrate them.

1. **All backbone imports are lazy** (inside the factory body). This lets
   `from radharmony.evaluator import backbones` succeed even when the user
   hasn't installed the optional dep.
2. **A single model instance** is created and shared between the transform's
   preprocess closure and the encoder wrapper — never load weights twice.
3. **Preprocessing lives entirely on the transform side**, never inside
   `ImageEncoderWrapper`. If you find yourself resizing or normalizing inside
   `_model_call`, move it back to `_preprocess`.
4. **Read variant-specific config dynamically** when the model exposes it.
   E.g. for DINOv2/v3 read `model.config.num_register_tokens` rather than
   hardcoding `4` — this makes the recipe survive variant swaps.
5. **Two-mode encoder**: classification (default, returns `[B, D]`) vs
   segmentation (`"mask"` in `output_keys`, returns `[B, D, H, W]`). Gate
   the difference on `_seg_mode = output_keys is not None and "mask" in
   output_keys`. For ViTs use `patches_to_spatial` (from `_utils.py`).
6. **Mask preprocess closure** mirrors the image one but with
   `do_normalize=False, do_rescale=False, resample=0` (nearest-neighbour)
   so class indices survive.

## File touched by every recipe

| File | What you do |
|---|---|
| **New** `radharmony/evaluator/backbones/<name>.py` | The recipe itself — copy `_template.py` and fill in. |
| **New** `docs/wiki/backbones/<name>.md` | Per-backbone wiki page (embed dim, input size, install, override knobs, usage, segmentation mode). |
| **Modified** `radharmony/evaluator/backbones/__init__.py` | `from .<name> import make_<name>`, add to `__all__`, add docstring entry. |
| **Modified** `pyproject.toml` | One row under `[project.optional-dependencies]`. |
| **Modified** `docs/wiki/backbones/index.md` | One row in the recipes table (linking to `<name>.md`); for manual-setup backbones also add a row in the override-paths table. |
| **Modified** `mkdocs.yml` | One `nav:` entry under `Backbones:` pointing at the new page. |
| **Modified** `README.md` | One row in the extras table. |

Skipping any of these breaks discovery, install, or docs. The verification
step below catches a missing export; doc drift is caught by the
`update-docs` skill.

## Workflow

### 1. Confirm the model's ecosystem

Different ecosystems use different call signatures — figure out which one
applies before writing the recipe.

| Ecosystem | `_model_call` pattern | Notes |
|---|---|---|
| HuggingFace ViT (DINOv2/v3, RAD-DINO HF version, BEiT) | `lambda m, x: m(pixel_values=x).last_hidden_state[:, 0]` | CLS token |
| HuggingFace SigLIP / CLIP-family | `lambda m, x: m(pixel_values=x).pooler_output` | learned pooler projection |
| HuggingFace SigLIP `.vision_model` | `lambda m, x: m.vision_model(pixel_values=x).pooler_output` | when `AutoModel` returns the joint model |
| OpenCLIP / BiomedCLIP | `lambda m, x: m.encode_image(x)` | image tower call |
| timm (`num_classes=0`) | `None` (model already returns `(B, D)`) | bare wrap with `ImageEncoderWrapper.from_custom(model)` |
| RAD-DINO (`rad-dino` package) | `cls, _ = m.encode(BatchFeature({"pixel_values": x})); return cls` | custom return tuple |
| ViT with register tokens (DINOv3) | Skip CLS + register tokens for segmentation: `m(...).last_hidden_state[:, 1+R:]` | read `R` from `model.config.num_register_tokens` |
| SSL backbone (MAE, SimCLR) | `lambda m, x: m.backbone(x)` or whichever attribute holds the extractor | check the package's docs |

If you have a working notebook (`notebooks/backbones/backbone_<name>.ipynb`),
read its forward-pass cells — the pattern is usually visible in two or three
lines. Convert that pattern verbatim; don't second-guess.

### 2. Confirm input shape, embedding dim, register tokens

You need three numbers before writing the recipe:

- **Input size** (e.g. 224, 518). Usually set by the model's processor; check
  `processor.crop_size` or `processor.size` for HF, or `preprocess_val` for
  OpenCLIP.
- **Embedding dim `D`** (e.g. 768, 1024). Run a forward pass on a dummy
  tensor and print `.shape[-1]`, or check `model.config.hidden_size`.
- **Register tokens (`R`, ViT only)**. Check `getattr(model.config,
  "num_register_tokens", 0)`. Non-zero R changes the segmentation-mode
  slice: skip `1 + R` tokens (CLS + registers) before reshaping.

These numbers go into the doc-table row, the docstring's "Returns" section,
and the segmentation-mode slice.

### 3. Scaffold from `_template.py`

```bash
cp radharmony/evaluator/backbones/_template.py radharmony/evaluator/backbones/<name>.py
```

The template has marked blocks for every concern: lazy imports, model load,
helper transforms, loader compose, preprocess closure, model-call closure,
mask preprocess closure, assemble. **Fill the blocks in order — don't
delete the section headers; they make future recipe authors' jobs easier.**

For a typical HuggingFace recipe, the body shrinks to ~40 lines plus
docstring. `dinov3.py` is a good size reference; `chexagent.py` is the
canonical vision-language reference (4-tuple return with text encoder).

### 4. Pick a hub identifier

Make `hub` a kwarg if the model family has multiple variants (ViT-B / ViT-L
/ ViT-H), so users can swap variants without editing the recipe. Use the
default that matches the project's typical compute budget (often ViT-B/16
for ~85M-param ViT models). Set a module-level `_<NAME>_HUB` constant for
the default.

### 5. Vision-only vs vision-language

- **Vision-only** (RAD-DINO, DINOv3, CheXFound): return
  `(transform, encoder)`.
- **Vision-language** (BiomedCLIP, CheXagent, MedSigLIP, MedImageInsights):
  return `(transform, image_encoder, text_encoder, tokenizer_or_processor)`.
  The text encoder closure typically wraps `model.text_model` or
  `model.encode_text` and moves outputs to CPU as float32 (so the evaluator
  can manipulate them with NumPy).

For vision-language, also update the doc table column "Returns" with the
full 4-tuple form.

### 6. Wire up exports

In [radharmony/evaluator/backbones/__init__.py](radharmony/evaluator/backbones/__init__.py):

- Add a docstring entry near the others (8-line block: function ref +
  one-line summary + extra install command).
- Add `from .<name> import make_<name>`.
- Append `"make_<name>"` to `__all__`.

### 7. Add the extras row

In [pyproject.toml](pyproject.toml), under `[project.optional-dependencies]`:

```toml
<name> = ["<pip-package>", ...]
```

If the model is HF-distributed, this is typically just
`["transformers>=5.6.0"]`. Add extras for tokenizer packages (e.g.
`sentencepiece`, `protobuf`) when the processor needs them.

Order: place the new extra near related extras (other backbones), keep the
file's existing grouping intact.

### 8. Write the per-backbone wiki page + update doc tables

New file: [docs/wiki/backbones/<name>.md](docs/wiki/backbones/). Match the
structure of the existing pages (raddino, biomed_clip, etc. for hub-fetch
recipes; ark_plus, medical_mae, eva_x, chexfound for manual-setup recipes).
Required sections:

- One-paragraph model intro (family, intent, hub identifier).
- Defaults table: embed dim, input size, returns-tuple shape, extra name.
- Install section: extras command + manual setup steps if any.
- Override default paths (only for manual-setup recipes): table of kwargs +
  defaults + purpose, plus a code example showing a non-default redirect.
- Usage code block: `make_<name>(...)` + a dataset call.
- Segmentation mode block: `output_keys={"img", "mask"}` + explicit
  `forward(x) -> Tensor[B, D, H, W]` shape (compute `H = W = input/patch`).

Then update the same row in two tables — keep the columns consistent:

- [docs/wiki/backbones/index.md](docs/wiki/backbones/index.md) — recipes
  table (Factory / Embed dim / Input size / Returns / Extra). Link the
  Factory cell to the new `<name>.md`. For manual-setup recipes also add a
  row to the "Override default paths" table.
- [README.md](README.md) — extras table (Extra / Adds / Install).

Append the new row under [mkdocs.yml](mkdocs.yml) `nav:` → `Backbones:`.

Append below the existing entries; don't alphabetize unless the existing
order is alphabetical.

### 9. Verify

```bash
# Smoke import — confirms exports are wired
python -c "from radharmony.evaluator.backbones import make_<name>; print(make_<name>)"
```

If the user has the extra installed and can spare a few seconds:

```bash
python <<'EOF'
import torch
from radharmony.evaluator.backbones import make_<name>
transform, encoder = make_<name>(device="cpu")
# Match the input size you documented:
out = encoder(torch.zeros(1, 3, 224, 224))
print("Embed shape:", out.shape)   # should match the doc-table embed dim
EOF
```

If the doc table claims `D=768` and the forward pass returns `D=1024`,
something in `_model_call` or the model load is wrong — fix before
considering the work done.

### 10. (Optional) Segmentation smoke test

Only run this if segmentation mode is in scope.

```bash
python <<'EOF'
import torch
from radharmony.evaluator.backbones import make_<name>
transform, encoder = make_<name>(device="cpu", output_keys={"img", "mask"})
out = encoder(torch.zeros(1, 3, 224, 224))
print("Spatial shape:", out.shape)   # should be [1, D, H, W] with H = W = input/patch
EOF
```

For a 224-px input with 16-px patches and 0 register tokens, you should see
`[1, D, 14, 14]`. For DINOv3 with 4 register tokens, the patches grid is
still 14×14 — the recipe skips the register-token positions, it doesn't
shrink the grid.

## What NOT to do

- **Do not import the heavy dependency at module top.** It will break
  `from radharmony.evaluator import backbones` for any user who hasn't
  installed the extra. Always import inside the factory body.
- **Do not load weights twice.** Some authors instinctively create one
  model for the transform and another for the encoder. Use a single
  `_model` and capture it in both closures.
- **Do not preprocess images inside `ImageEncoderWrapper`.** Channel
  expansion, resize, and normalization belong in `_preprocess`. The
  wrapper is a routing adapter, not a pipeline.
- **Do not hardcode register-token counts** when `model.config` exposes
  them. Variants change; reading the config makes the recipe robust.
- **Do not skip the docstring's "Returns" section.** Users discover the
  embed dim and segmentation shape from there — leaving it generic forces
  them to inspect the source.
- **Do not invent recipe parameters** for things the model doesn't expose.
  `_template.py`'s parameter list (`device`, `hub`, `output_keys`) is what
  every backbone needs. Add more only when the model has a real knob
  (e.g. CheXFound's `checkpoint_path=` because it needs a manual download).
- **Do not commit a recipe whose smoke-import test fails.** A broken
  import in `backbones/__init__.py` blocks every evaluator workflow.

## Style rules

- Module docstring: model name, input format, key constraints (input size,
  bit depth, etc.), and the install line as an indented code block.
- Section comments: keep the template's `# ── Section ──` dividers so
  future readers see the same structure across recipes.
- Type annotations: match `_template.py` — `mt.Compose, ImageEncoderWrapper`
  for vision-only, plus `Callable[[torch.Tensor], torch.Tensor], Any` for
  the text-encoder and tokenizer in vision-language.
- Hub constants: module-level `_<NAME>_HUB = "..."` (private, single
  source of truth, overridable via the `hub=` kwarg).
- Inline comments: explain the *why* of non-obvious slices
  (`# CLS + register tokens`), never the *what*.

## Editing this skill

The copy in the repo, `.claude/skills/add-backbone/SKILL.md`, is the one to edit and commit.
There is no `docs/` mirror anymore.

## Output Format

For each artifact:
1. Show the new recipe file path + a one-sentence summary of the
   `_model_call` pattern chosen.
2. Show the modified files (init, pyproject, three doc tables) — one
   line each.
3. Report the smoke-import result.
4. Flag anything that couldn't be tested without the heavy extra
   installed (e.g. a forward pass) so the user can run it manually.
