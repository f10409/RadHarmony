# MAIRA-2 (Generative)

Chest-X-ray report generator from Microsoft Health Futures ([`microsoft/maira-2`](https://huggingface.co/microsoft/maira-2)). A RAD-DINO vision tower paired with a Vicuna text decoder, purpose-built for free-text FINDINGS + IMPRESSION generation with optional phrase-level grounding.

| Base model | Input | Returns | Extra |
|------------|-------|---------|-------|
| microsoft/maira-2 | image file path | `(transform, report_generator)` | `maira2_gen` (isolated venv, `transformers>=4.46,<4.47`) |

## Contract

Same shape as [`make_chexagent_generator`](chexagent_gen.md):

- **`transform`**: a callable `sample_dict -> sample_dict`. It loads `sample["img"]` (any MONAI-readable format including DICOM), normalises to uint8 RGB, writes it to a temporary PNG, and stores the PNG **path string** back under `img`. `report` passes through unchanged.
- **`report_generator`**: a callable `list[str] -> list[str]`. Given a list of PNG paths, returns generated report text.

Pair with [`ReportGenerationEvaluator`](../evaluator/index.md).

## Install

MAIRA-2 pins `transformers>=4.46,<4.47`, incompatible with [`chexagent_gen`](chexagent_gen.md) (pins `4.40.0`) and RadEval (pins `>=5.0`). Install in its own venv:

```bash
uv venv .venv-maira2-gen
source .venv-maira2-gen/bin/activate
uv pip install -e ".[maira2_gen]"
```

Weights are auto-downloaded on first call. The HuggingFace repo is gated (accept the license on the model card first).

## Usage

```python
from radharmony.evaluator.backbones import make_maira2_generator
from radharmony.evaluator import ReportGenerationEvaluator
from radharmony.dataset import MIMICCXRDataset

transform, report_generator = make_maira2_generator(device="cuda")

ds = MIMICCXRDataset(
    base_image_dir="/data/mimic-cxr/2.0.0/files",
    report_csv_path="/data/mimic-cxr/reports.csv",
    transform=transform,
    output_report=True,
    cache_dir=None,
)

ev = ReportGenerationEvaluator(
    report_generator,
    dataset=ds,
    ref_section="findings",
    max_samples=50,
    output_dir="outputs/maira2_gen",
)
df = ev.evaluate()
ev.save_results(df)
```

## Generation recipe

`report_generator` follows the official MAIRA-2 single-image reporting path. For each PNG:

1. Open the PNG (PIL, RGB).
2. Call `processor.format_and_preprocess_reporting_input(current_frontal=img, current_lateral=None, prior_frontal=None, indication=None, technique=None, comparison=None, prior_report=None, ...)`. RadHarmony datasets currently surface only the image and reference report, so all other inputs are `None`.
3. Generate greedily (`do_sample=False, num_beams=1, use_cache=True, max_new_tokens=300`).
4. Decode the new tokens with `skip_special_tokens=True`.

The returned string is the raw model output (typically a FINDINGS+IMPRESSION narrative). Pass `get_grounding=True` to include the model's phrase-grounding markup.

## Recipe arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `device` | `str` | `"cuda"` | Passed to `.to(device)` after model load |
| `hub` | `str` | `"microsoft/maira-2"` | HuggingFace model identifier |
| `max_new_tokens` | `int` | `300` | Cap per generation call |
| `dtype` | `torch.dtype` | `torch.bfloat16` | Model precision |
| `tmp_dir` | `str` | auto (`tempfile.mkdtemp(prefix="maira2_png_")`) | Directory for the intermediate PNGs the transform writes |
| `get_grounding` | `bool` | `False` | Forward `get_grounding` to MAIRA's reporting-input processor (emit phrase-grounding markup in the output text) |

## Two-stage workflow

Same pattern as [`chexagent_gen`](chexagent_gen.md#two-stage-workflow): call `ev.generate_only("outputs/maira2_gen/pairs.parquet")` to produce a `(sample_id, reference, hypothesis)` parquet without importing RadEval, then score the parquet in a separate `radeval` venv.

## Notes

- The HuggingFace `microsoft/maira-2` repo is a **custom-code model**; `trust_remote_code=True` is required and set by the recipe.
- MAIRA-2 supports prior-study conditioning (`prior_frontal`, `prior_report`). RadHarmony's dataset abstractions surface only the current image and reference report, so those parameters are pinned to `None`. If you need the prior-study path, wrap the processor call directly.
- Segmentation mode is not supported (this is a generative decoder, not an encoder).
