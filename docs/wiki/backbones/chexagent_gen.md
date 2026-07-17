# CheXagent (Generative)

Generative chest-X-ray report writer from Stanford AIMI ([`StanfordAIMI/CheXagent-2-3b`](https://huggingface.co/StanfordAIMI/CheXagent-2-3b)). A 3B-param vision-language decoder that consumes chest-X-ray images and emits FINDINGS + IMPRESSION narrative text. Distinct from [`make_chexagent`](chexagent.md) (a contrastive image tower with no text-generation head).

| Base model | Input | Returns | Extra |
|------------|-------|---------|-------|
| StanfordAIMI/CheXagent-2-3b | image file path | `(transform, report_generator)` | `chexagent_gen` (isolated venv, `transformers==4.40.0`) |

## Contract

Unlike the encoder recipes, `make_chexagent_generator` returns a pair with a different contract:

- **`transform`**: a callable `sample_dict -> sample_dict` (not a MONAI `Compose`). It loads `sample["img"]` (any MONAI-readable format including DICOM), normalises to uint8 RGB, writes it to a temporary PNG, and stores the PNG **path string** back under `img`. `report` passes through unchanged. `img` stays a string, not a tensor: CheXagent-2 consumes file paths.
- **`report_generator`**: a callable `list[str] -> list[str]`. Given a list of PNG paths, it returns a list of generated report strings of the same length.

Pair the returned tuple with [`ReportGenerationEvaluator`](../evaluator/index.md), which orchestrates dataset iteration, generation, and RadEval scoring.

## Install

CheXagent-2 pins `transformers==4.40.0`, which is incompatible with several other RadHarmony extras (RadEval, MAIRA-2, most classification backbones). Install in its own venv:

```bash
uv venv .venv-chexagent-gen
source .venv-chexagent-gen/bin/activate
uv pip install -e ".[chexagent_gen]"
```

`chexagent_gen` also pins `torch==2.7.1` per the model card. Weights are auto-downloaded from HuggingFace on first call.

## Usage

```python
from radharmony.evaluator.backbones import make_chexagent_generator
from radharmony.evaluator import ReportGenerationEvaluator
from radharmony.dataset import MIMICCXRDataset

transform, report_generator = make_chexagent_generator(device="cuda")

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
    output_dir="outputs/chexagent_gen",
)
df = ev.evaluate()
ev.save_results(df)
```

## Generation recipe

By default (`prompt=None`) `report_generator` runs the official Stanford AIMI multi-step "ABCDE" recipe from [`demos/app_demo.py`](https://github.com/Stanford-AIMI/CheXagent) (commit `e4f31e6e`):

1. **Findings.** One generation call per anatomical region (Airway, Breathing, Cardiac, Diaphragm, Everything-else). Outputs are concatenated with spaces.
2. **Impression.** A text-only call over the concatenated findings, prompted with "Write the Impression section for the following Findings: ...".

The returned string is formatted as `"FINDINGS: ...\n\nIMPRESSION: ..."`, matching the reference-report style [`ReportGenerationEvaluator`](../evaluator/index.md) parses out of MIMIC-CXR notes.

### Choosing / overriding the prompt

- **Single-shot** — pass a `prompt` string to generate the whole report in one image call instead of the ABCDE recipe:

  ```python
  transform, report_generator = make_chexagent_generator(
      prompt="Write the FINDINGS and IMPRESSION for this chest X-ray.",
  )
  ```

- **Customize the ABCDE path** — keep `prompt=None` and override the per-anatomy prompts and/or the impression template:

  ```python
  transform, report_generator = make_chexagent_generator(
      findings_prompts=["Describe the lungs.", "Describe the heart and mediastinum."],
      impression_prompt="Summarize the impression from these findings: {findings}",
  )
  ```

When the evaluator runs with `use_indication=True`, the parsed indication is prepended to the prompt(s) as clinical context in either mode.

## Recipe arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `device` | `str` | `"cuda"` | Passed to `.to(device)` after model load |
| `hub` | `str` | `"StanfordAIMI/CheXagent-2-3b"` | HuggingFace model identifier |
| `prompt` | `str \| None` | `None` | `None` → multi-step ABCDE recipe; a string → single-shot generation with that prompt |
| `findings_prompts` | `list[str] \| None` | `None` (demo's 5 anatomy prompts) | Override the per-anatomy findings prompts (ABCDE path only) |
| `impression_prompt` | `str \| None` | `None` (demo's template) | Override the impression template (must contain `{findings}`; ABCDE path only) |
| `max_new_tokens` | `int` | `512` | Cap per generation call |
| `dtype` | `torch.dtype` | `torch.bfloat16` | Model precision |
| `tmp_dir` | `str` | auto (`tempfile.mkdtemp(prefix="chexagent_png_")`) | Directory for the intermediate PNGs the transform writes |

## Two-stage workflow

`ReportGenerationEvaluator` has a `generate_only(out_parquet)` method that runs the generator over the dataset and writes `(sample_id, reference, hypothesis)` to a parquet file **without importing RadEval**. This lets you run generation in the `chexagent_gen` venv, then score the parquet in a separate `radeval` venv (which pins incompatible `transformers`).

```python
# In the chexagent_gen venv:
ev.generate_only("outputs/chexagent_gen/pairs.parquet")

# In the radeval venv, load the parquet and call RadEval directly on the
# stored (reference, hypothesis) columns.
```

## Notes

- The transform writes PNGs to `tmp_dir` (or an auto-created temp directory). Point at a fast local scratch when running over large datasets.
- `chexagent_gen` and [`maira2_gen`](maira2_gen.md) cannot share a venv; see the `[tool.uv].conflicts` block in [pyproject.toml](../../../pyproject.toml).
- Segmentation mode is not supported (this is a generative decoder, not an encoder).
