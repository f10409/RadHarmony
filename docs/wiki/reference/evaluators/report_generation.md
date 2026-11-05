# ReportGenerationEvaluator

| | |
|---|---|
| Registry key | `report_generation` |
| Task | Report generation, scores generated reports |
| Dataset outputs it needs | `report` (`output_report=True`) |
| Split modes | test only (`dataset=`) |
| Encoder it needs | a report generator, `(imgs, indications=None) -> list[str]`, one report for each image ([Encoder contract](index.md#encoder-contract)) |
| Output | The result table, one row for the point estimate and one for each bootstrap sample. See [Output](#output) |

The report generation evaluator runs a vision-language model over a dataset and scores each generated
report against the real report. The scores come from the [RadEval](https://github.com/jbdel/RadEval) metrics
(BLEU, ROUGE, BERTScore, RadGraph, F1-CheXbert and more). Use it to compare report generators on the same
images. It does not look inside the model, so any generator works.

## Quick use

The generator recipes [`make_chexagent_generator`](../backbones/chexagent_gen.md) and
[`make_maira2_generator`](../backbones/maira2_gen.md) need their own venv (`chexagent_gen` or
`maira2_gen` extra). RadEval needs other package versions, so you generate and score in 2 steps. In the
generator venv, `generate_only()` writes the generated reports to a parquet file. This block makes 4
reports on Montgomery (138 images) with CheXagent-2, in batches of 2:

<!-- block: run=skip reason="needs the chexagent_gen venv (transformers 4.40.0), the data and a GPU" -->
```python
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator import ReportGenerationEvaluator
from radharmony.evaluator.backbones import make_chexagent_generator

transform, report_generator = make_chexagent_generator(device="cuda")
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_report=True,
    cache_dir=None,
)

ev = ReportGenerationEvaluator(
    report_generator,
    dataset=ds,
    batch_size=2,
    max_samples=4,
    output_dir="outputs/report_generation",
)
ev.generate_only("outputs/report_generation/pairs.parquet")
# pairs.parquet columns: sample_id, reference, hypothesis, patient_id
```

Then score the parquet file in the RadEval venv (`radeval` extra). `--light` uses `LIGHT_METRICS` (see
[Metrics](#metrics)). Without it, the script uses all 16 metrics:

<!-- block: run=skip reason="needs the radeval venv and the parquet file of step 1" -->
```bash
python notebooks/evaluator/score_reports_radeval.py \
    --pairs-parquet outputs/report_generation/pairs.parquet \
    --output-dir outputs/report_generation/scores \
    --light --n-bootstrap 10
```

The script writes `results.csv` with one row of scores and one row for each bootstrap sample. It also
writes `results_summary.json` with the number of pairs and the scores of the first row. Run the script from
the repo, because it loads `radharmony/evaluator/language/hypothesis.py` by its file path. Montgomery reports
are short (for example "normal"). So the scores mean little, and the example only shows the steps.

## How it works

- Generation: the evaluator runs `report_generator` on each batch of `img`. It needs one report for each
  image. With the CheXagent and MAIRA-2 recipes, `img` is the path of a PNG file, so the batch is a list
  of path strings.
- Reference: the evaluator takes the section `ref_section` of `sample["report"]` (D-38). See
  [Reference sections](#reference-sections).
- Dropped pairs: the evaluator drops a pair when the reference or the generated report is empty.
- Cleaning: `evaluate()` cleans each generated report with `normalize_hypothesis` before it scores it.
  The scoring script of step 2 does the same, so both ways give the same score for the same generated
  reports (D-50).
- Bootstrap: each bootstrap sample draws patients with replacement and keeps all their pairs (D-32). With
  no `patient_id`, the bootstrap draws pairs.
- IDs: `sample_id` is the `image_path` with no file ending (D-46). For example, `MCUCXR_0001_0.png` gives
  `MCUCXR_0001_0`.
- Splits: the evaluator has no folds and no training ([Split modes](index.md#split-modes)).

## Indication

The indication is the reason for the exam, for example "cough". The radiologist has it before they write
the report, so it is a fair input. The evaluator never scores it. With `use_indication=True`, the evaluator takes
the section `indication_section` of `sample["report"]` and gives it to the generator:

```text
report_generator(imgs, indications) -> list[str]
```

`indications` has one string for each image. An entry is `None` when that report has no indication
section. With `use_indication=False`, the generator gets only `imgs`. With `use_indication=True`, a
generator that takes only `imgs` raises `TypeError` at once (D-52).

The recipes use the indication in different ways:

- [`make_maira2_generator`](../backbones/maira2_gen.md) puts it in the `indication` input of MAIRA-2.
- [`make_chexagent_generator`](../backbones/chexagent_gen.md) puts it before each prompt.
- `make_medgemma_generator` puts `"Indication: <text>. "` before its prompt.

## Reference sections

`ref_section` picks the part of the real report that is the reference (D-38):

| `ref_section` | Reference |
|---|---|
| `"findings"` | The FINDINGS section. With no FINDINGS section, the IMPRESSION section |
| `"impression"` | The IMPRESSION or CONCLUSION section. With neither, the FINDINGS section |
| `"both"` | The FINDINGS text and then the IMPRESSION text, joined with a space |
| `"full"` | The whole report, with single spaces |

A report with neither section gives the whole report, with single spaces. A header is a name and a colon,
for example `FINDINGS:`. At the start of a line, it can be in upper or lower case. Inside a line, it must be
in capitals. So "No new findings: none." in a sentence is not a header.

The parser and the text cleaning need no data:

<!-- block: run=ci -->
```python
from radharmony.evaluator.language import normalize_hypothesis
from radharmony.evaluator.language._report_parser import parse_report_section

report = "INDICATION: cough.\nFINDINGS: Heart is normal. No effusion.\nIMPRESSION: No acute disease."
print(parse_report_section(report, "findings"))  # Heart is normal. No effusion.
print(parse_report_section(report, "both"))  # Heart is normal. No effusion. No acute disease.
print(parse_report_section("IMPRESSION: Stable.", "findings"))  # Stable.
print(normalize_hypothesis("[Lungs: Pleura] **small effusion**."))  # small effusion.
```

`normalize_hypothesis` removes the CheXagent grounding tags (`[Region: Subregion]`), the boxes
(`<|box|> ... <|/box|>`), the ref marks (`<|ref|>`, `<|/ref|>`) and the bold marks (`**`). Then it makes
all white space single spaces.

## Metrics

`radharmony.evaluator.metrics` has 2 metric lists:

| Name | Metrics | Notes |
|---|---|---|
| `FULL_METRICS` (default) | `bleu`, `rouge`, `bertscore`, `radeval_bertscore`, `f1chexbert`, `f1radbert_ct`, `radgraph`, `ratescore`, `radgraph_radcliq`, `radcliq`, `srrbert`, `temporal`, `green`, `mammo_green`, `crimson`, `radfact_ct` | 16 metrics. `green` and `mammo_green` run a local LLM of about 7B weights. `crimson` and `radfact_ct` need an API key |
| `LIGHT_METRICS` | `bleu`, `rouge`, `bertscore`, `radgraph`, `f1chexbert` | No LLM and no API key |

A metric name that RadEval does not know raises `ValueError`. A metric with a missing model file, API key
or package raises the error from RadEval. The evaluator does not skip that metric.

This block scores 2 given texts in the RadEval venv, with no generator:

<!-- block: run=gpu timeout=600 -->
```python
from radharmony.evaluator.metrics import LIGHT_METRICS, compute_report_metrics

refs = ["Heart size is normal. No pleural effusion.", "Small left pleural effusion."]
hyps = ["The heart is normal. No effusion.", "Left pleural effusion."]
scores = compute_report_metrics(refs, hyps, LIGHT_METRICS)
print(sorted(scores))
# ['bertscore', 'bleu', 'f1chexbert_5_macro_f1', ..., 'radgraph_simple', 'rouge1', 'rouge2', 'rougeL']
```

## Score in one call

`evaluate()` generates and scores in one call. It needs a venv with the generator and RadEval. The
`chexagent_gen` and `maira2_gen` venvs do not have RadEval, so use the 2 steps of
[Quick use](#quick-use) for them. This block uses `ReportGenerationEvaluator`, `report_generator` and `ds`
of [Quick use](#quick-use), with a generator that runs in the RadEval venv:

<!-- block: run=skip reason="needs a report generator that runs in the RadEval venv, the data and a GPU" -->
```python
from radharmony.evaluator.metrics import LIGHT_METRICS

ev = ReportGenerationEvaluator(
    report_generator,
    dataset=ds,
    metrics=LIGHT_METRICS,
    n_bootstrap=10,
    output_dir="outputs/report_generation",
)
df = ev.evaluate()
ev.save_results(df)  # writes results.csv, results_summary.csv and generations.csv
```

## Arguments

<!-- args: radharmony.evaluator.ReportGenerationEvaluator -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `report_generator` | callable | required | A function `(imgs, indications=None) -> list[str]`. It returns one report for each image of the batch ([Encoder contract](index.md#encoder-contract)) |
| `metrics` | `tuple[str, ...]` or `list[str]` | `('bleu', 'rouge', 'bertscore', 'radeval_bertscore', 'f1chexbert', 'f1radbert_ct', 'radgraph', 'ratescore', 'radgraph_radcliq', 'radcliq', 'srrbert', 'temporal', 'green', 'mammo_green', 'crimson', 'radfact_ct')` | The RadEval metric names. The default is `FULL_METRICS`, all 16 metrics. See [Metrics](#metrics) |
| `ref_section` | `str` | `"findings"` | The report section to compare with: `"findings"`, `"impression"`, `"both"` or `"full"` |
| `use_indication` | `bool` | `False` | Give the indication section of each report to the generator. See [Indication](#indication) |
| `indication_section` | `str` | `"indication"` | The report section that gives the indication |
| `normalize_hyp` | `bool` | `True` | Clean each generated report with `normalize_hypothesis` before scoring (D-50). `False` scores the raw text |

### Shared arguments (inherited from BaseEvaluator)

These arguments have the same names as in the other evaluators. `ReportGenerationEvaluator` sets them in
its own signature. It gives only `dataset` and `output_dir` to `BaseEvaluator`.

| Argument | Type | Default | Description |
|---|---|---|---|
| `dataset` | dataset | required | Test only: the one dataset, made with `output_report=True` ([Split modes](index.md#split-modes)) |
| `device` | `str` | `"cuda"` | Not used. The generator sets its own device. A value other than `"cuda"` warns (D-52) |
| `batch_size` | `int` | `8` | The batch size of the generator |
| `num_workers` | `int` | `4` | The number of DataLoader workers |
| `max_samples` | `int` | `None` | The largest number of pairs to score. Use it for a short test run |
| `n_bootstrap` | `int` | `0` | The number of bootstrap samples of the patients (of the pairs when the table has no `patient_id`) |
| `bootstrap_seed` | `int` | `0` | The seed of the bootstrap samples |
| `output_dir` | `str` | `None` | The folder for `save_results()` |

## Output

`evaluate()` returns the [result table](index.md#result-table), one row for the point estimate and one
row for each bootstrap sample. A real run with `LIGHT_METRICS` gave these columns:

| Column | Meaning |
|---|---|
| `label` | Always `"report"` |
| `fold` | Always `-1` |
| `seed` | Always `-1` |
| `bootstrap` | `-1` for the point estimate. `0` to `n_bootstrap - 1` for a bootstrap sample |
| `bleu`, `bertscore` | One score each |
| `rouge1`, `rouge2`, `rougeL` | The scores of `rouge` |
| `radgraph_simple`, `radgraph_partial`, `radgraph_complete` | The scores of `radgraph` |
| `f1chexbert_5_micro_f1`, `f1chexbert_all_macro_f1`, ... | The 6 scores of `f1chexbert` (5 or all labels, and micro, macro or weighted F1) |

A metric that gives more than one score gets one column for each score. The column name joins the names
with `_`. A score that is not a number is not in the table.

`save_results(df)` writes 3 files to `output_dir`:

- `results.csv`: the result table.
- `results_summary.csv`: one row. For each metric, it gives `<metric>_mean` (the point estimate),
  `<metric>_n`, and `<metric>_std`, `<metric>_ci_lo` and `<metric>_ci_hi` over the bootstrap samples
  (D-19). With `n_bootstrap` below 2, the std and the CI are NaN.
- `generations.csv`: the columns `sample_id`, `reference` and `hypothesis`. The hypothesis is the text
  that the metrics scored, so it is the cleaned text when `normalize_hyp=True`.

`generate_only(path)` writes a parquet file with `sample_id`, `reference` and `hypothesis`, and
`patient_id` when the table has it. The hypothesis is the raw text. The scoring script of step 2 cleans it,
unless you pass `--no-normalize-hyp`.

## Notes

- `output_report=True` drops the rows with no report (D-33). For MIMIC-CXR, also pass `report_csv_path`.
  MIMIC-CXR-JPG has no reports.
- Each bootstrap sample runs every metric again. With `FULL_METRICS` this is slow, so keep `n_bootstrap`
  small, for example 10 to 50.
- For a full example, see
  [`notebooks/evaluator/report_generation_example.py`](https://github.com/f10409/RadHarmony/blob/main/notebooks/evaluator/report_generation_example.py).
- For a hands-on notebook with MedGemma and VQA, see
  [`datathon26/6_radharmony_hands_on_vqa_report_generation.ipynb`](https://github.com/f10409/RadHarmony/blob/main/datathon26/6_radharmony_hands_on_vqa_report_generation.ipynb).
