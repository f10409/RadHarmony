# ReportGenerationEvaluator

Registry key: `"report_generation"`.

Runs an arbitrary vision-language *generator* over a RadHarmony dataset and scores the generated hypotheses against reference reports using the [RadEval](https://github.com/jbdel/RadEval) metric suite (BLEU, ROUGE, BERTScore, RadGraph, RadCliQ, and more).

Report generation has no training phase, so the k-fold / fixed-split machinery of the classification evaluators does not apply here. This evaluator takes a single `dataset=`, iterates it once, and returns a `pandas.DataFrame` of metric scores. Variance (if requested) comes from bootstrapping the scored `(reference, hypothesis)` pairs.

## Encoder contract

The evaluator never inspects model internals. Wrap any chest-X-ray VLM (CheXagent-2, MAIRA-2, MedGemma, LLaVA-Rad, ...) into one minimal callable:

```text
report_generator(imgs, indications=None) -> list[str]   # len(imgs) reports
```

`indications` is an optional per-image list of clinical-indication strings
(`None` entries → no indication); see [Indication-conditioned generation](#indication-conditioned-generation). A legacy single-argument `report_generator(imgs)` also works — the evaluator detects the accepted arity.

For CheXagent-2 and MAIRA-2, use the [`make_chexagent_generator`](../backbones/chexagent_gen.md) / [`make_maira2_generator`](../backbones/maira2_gen.md) recipes. They return `(transform, report_generator)` where `transform` handles image loading (any MONAI-readable format including DICOM) and stores the PNG **path string** under `img`. For those recipes, `imgs` in the callable above is a list of file-path strings, one per image; the batch loader collates them as `list[str]`.

## Usage

```python
from radharmony.evaluator import ReportGenerationEvaluator
from radharmony.evaluator.backbones import make_chexagent_generator
from radharmony.dataset import MIMICCXRDataset

transform, report_generator = make_chexagent_generator(device="cuda")

ds = MIMICCXRDataset(
    base_image_dir="/data/mimic-cxr/2.0.0/files",
    report_csv_path="/data/mimic-cxr/reports.csv",
    transform=transform,
    output_report=True,           # required (evaluator reads sample["report"])
    cache_dir=None,
)

ev = ReportGenerationEvaluator(
    report_generator,
    dataset=ds,
    ref_section="findings",
    n_bootstrap=100,
    output_dir="outputs/chexagent_gen",
)
df = ev.evaluate()
ev.save_results(df)
```

`save_results` writes three files:

- `results.csv`: one row per bootstrap resample (plus the point estimate)
- `results_summary.csv`: mean / std / 95% CI per metric
- `generations.csv`: the raw `(sample_id, reference, hypothesis)` triples that the metric panel was computed over

## Constructor arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `report_generator` | callable | — | `(imgs) -> list[str]` of length `len(imgs)` |
| `dataset` | dataset | — | RadHarmony dataset constructed with `output_report=True` |
| `metrics` | `tuple[str, ...]` | `FULL_METRICS` (16 metrics) | RadEval metric names. See [Metric suite](#metric-suite) below |
| `ref_section` | `str` | `"findings"` | One of `"findings"`, `"impression"`, `"both"`, or `"full"` |
| `use_indication` | `bool` | `False` | Parse the indication section from each report and pass it to the generator as per-sample context (indication-conditioned reporting) |
| `indication_section` | `str` | `"indication"` | Section name handed to the parser for the indication text |
| `device` | `str` | `"cuda"` | Currently unused (the generator owns its device); kept for symmetry with other evaluators |
| `batch_size` | `int` | `8` | Inference DataLoader batch size |
| `num_workers` | `int` | `4` | Inference DataLoader workers |
| `max_samples` | `int` | `None` | Optional cap on scored pairs (useful for smoke tests) |
| `n_bootstrap` | `int` | `0` | Bootstrap resamples of the `(ref, hyp)` list for metric CIs |
| `bootstrap_seed` | `int` | `0` | RNG seed for the bootstrap resampler |
| `output_dir` | `str` | `None` | Directory for CSV output |

## Reference-section parsing

RadHarmony datasets surface `sample["report"]` as a **file path** (see [Dataset API](../api.md)). The evaluator reads that file and extracts the requested section using `parse_report_section`:

| `ref_section` | Meaning |
|---------------|---------|
| `"findings"` (default) | FINDINGS section, or IMPRESSION if FINDINGS is missing |
| `"impression"` | IMPRESSION / CONCLUSION section, or FINDINGS if missing |
| `"both"` | Findings text + impression text, space-joined |
| `"full"` | The entire report, whitespace-normalized |

The parser is permissive by design: when no recognisable section header is present, it falls back to the whole de-headered body rather than raising. Pairs with an empty reference (or an empty generation) are dropped before scoring.

## Metric suite

Two named metric bundles ship with the language-metrics module:

| Constant | Metrics | Notes |
|----------|---------|-------|
| `FULL_METRICS` (default) | `bleu`, `rouge`, `bertscore`, `radeval_bertscore`, `f1chexbert`, `f1radbert_ct`, `radgraph`, `ratescore`, `radgraph_radcliq`, `radcliq`, `srrbert`, `temporal`, `green`, `mammo_green`, `crimson`, `radfact_ct` | Includes local-LLM (`green`, `mammo_green`, ~7B) and API-key metrics (`crimson`, `radfact_ct`) |
| `LIGHT_METRICS` | `bleu`, `rouge`, `bertscore`, `radgraph`, `f1chexbert` | Lexical + clinical subset; no LLM, no API keys |

Pick a subset explicitly to skip heavy metrics:

```python
from radharmony.evaluator.metrics import LIGHT_METRICS
ev = ReportGenerationEvaluator(report_generator, dataset=ds, metrics=LIGHT_METRICS)
```

Unknown / unavailable metrics surface RadEval's own per-metric error (missing model weights, missing API key, incompatible dependency) rather than being silently dropped.

## Output schema

`evaluate()` returns a `pandas.DataFrame` with these columns:

| Column | Meaning |
|--------|---------|
| `label` | Always `"report"` (no per-label axis for generation) |
| `fold` | Always `-1` |
| `seed` | Always `-1` |
| `bootstrap` | `-1` = point estimate; `0..n_bootstrap-1` = resample |
| `<metric columns>` | One column per metric leaf (nested RadEval outputs are flattened, e.g. `radgraph.f1`, `radgraph.precision`) |

## Indication-conditioned generation

The **indication** (the clinical reason for the exam) is legitimate generation
*input* — it is what the radiologist has before dictating — and **not** a
scoring target, so conditioning on it is fair (the reference remains the
FINDINGS/IMPRESSION). MedGemma (`"<INDICATION> findings:"`) and MAIRA-2 (its
native `indication` reporting-input slot) both use it.

Set `use_indication=True` and the evaluator parses the indication section out
of each report and passes it to the generator as the per-sample context:

```python
ev = ReportGenerationEvaluator(
    report_generator, dataset=ds, ref_section="findings", use_indication=True,
)
```

`make_maira2_generator` wires the string into MAIRA-2's `indication=` slot;
`make_chexagent_generator` prepends it to the anatomy prompts; and
`make_medgemma_generator` prepends it as `"Indication: <ind>. "` before its
`prompt`. The indication parser returns `""` when no indication header is
found (so absence reads as "no indication" — it never falls back to the report
body).

## Two-stage workflow

The full 16-metric suite pulls in RadEval (which pins incompatible `transformers` vs the `chexagent_gen` / `maira2_gen` extras), so generation and scoring often run in separate venvs. `ReportGenerationEvaluator` exposes `generate_only(out_parquet)`:

```python
# In the chexagent_gen or maira2_gen venv (no RadEval import):
ev = ReportGenerationEvaluator(report_generator, dataset=ds, output_dir="outputs/run1")
ev.generate_only("outputs/run1/pairs.parquet")
```

The parquet has columns `sample_id`, `reference`, `hypothesis`. Load it in a separate `radeval` venv and score the columns directly with RadEval.

Two worked notebooks walk through the full two-stage flow end to end:
[`notebooks/evaluator/report_generation_example.ipynb`](https://github.com/f10409/RadHarmony/blob/main/notebooks/evaluator/report_generation_example.ipynb)
(Stage A — generate) and
[`notebooks/evaluator/score_reports_radeval.ipynb`](https://github.com/f10409/RadHarmony/blob/main/notebooks/evaluator/score_reports_radeval.ipynb)
(Stage B — score).

## Notes

- The dataset must be built with `output_report=True`. For MIMIC-CXR (DICOM) additionally pass `report_csv_path=`; MIMIC-CXR-JPG has no reports.
- Bootstrap CIs use resampling *of the scored pair list*, not per-sample-metric variance: each resample re-runs every metric on a bootstrap draw of the pairs. With the full suite this is expensive; keep `n_bootstrap` small (10–50 is typical).
- No fold iteration happens. Passing `train_dataset` / `test_dataset` is not supported (only `dataset=`).
