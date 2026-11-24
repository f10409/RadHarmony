# Run report generation

`ReportGenerationEvaluator` runs a report generator on a dataset and scores each generated report against
the real report with [RadEval](https://github.com/jbdel/RadEval). The generator recipes and RadEval need
different package versions, so you usually generate in one venv and score in another. This page does
both steps on [Montgomery County CXR](../reference/datasets/cxr-boxes-masks/montgomery_cxr.md) with
CheXagent-2.

Before you start, [get the Montgomery data](get-the-data.md#montgomery-county-cxr). Step 2 needs a GPU.

## 1. Make the 2 venvs

The `chexagent_gen` extra (and the `maira2_gen` extra) conflicts with the `radeval` extra
(the `conflicts` list in `pyproject.toml`, and
[Report generators](../reference/backbones/index.md#report-generators)). So install each one in its own
venv. RadEval needs Python 3.11 or newer.

<!-- block: run=skip reason="installs packages" -->
```bash
uv venv .venv-chexagent
uv pip install --python .venv-chexagent/bin/python -e ".[chexagent_gen]"
uv venv --python 3.12 .venv-radeval
uv pip install --python .venv-radeval/bin/python -e ".[radeval]"
```

For MAIRA-2, make `.venv-maira2` with the `maira2_gen` extra in the same way.

## 2. Generate the reports

Run this step in `.venv-chexagent`, for example with `.venv-chexagent/bin/python`. `generate_only()` writes
the generated reports and the real reports to a parquet file. It does not score. `output_report=True` keeps
only the rows with a report. `max_samples=4` makes a short test run.

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
ev = ReportGenerationEvaluator(report_generator, dataset=ds, batch_size=2, max_samples=4)
ev.generate_only("outputs/report_generation/pairs.parquet")
```

The parquet file has the columns `sample_id`, `reference`, `hypothesis` and `patient_id`. The
`hypothesis` is the raw generated text. In a real run, the 4 reports took 35 seconds. The first row:

<!-- block: run=skip reason="sample output of step 2" -->
```text
sample_id   MCUCXR_0001_0
reference   normal
hypothesis  FINDINGS: The hilar contours are normal. The lungs are clear. ... IMPRESSION: No acute
            cardiopulmonary process.
```

## 3. Score the reports

In the RadEval venv, run the scoring script from its place in the repo, not from a copy. It finds
`radharmony/evaluator/language/hypothesis.py` from its own path, so it cleans the text in the same way as
`evaluate()` (D-50). `--light` uses the 5 metrics of `LIGHT_METRICS`. Without it, the script uses all 16
metrics. One metric can give more than one score column, for example `radgraph` gives 3.

<!-- block: run=skip reason="needs the radeval venv and the parquet file of step 2" -->
```bash
.venv-radeval/bin/python notebooks/evaluator/score_reports_radeval.py \
    --pairs-parquet outputs/report_generation/pairs.parquet \
    --output-dir outputs/report_generation/scores \
    --light --n-bootstrap 10
# n=4  results → outputs/report_generation/scores/
```

The script writes `results.csv` to `--output-dir`, with one row of scores (`bootstrap` is `-1`) and one row
for each bootstrap sample. It also writes `results_summary.json`, with the number of pairs and the scores
of the first row.

## 4. Or: generate and score in one venv

When the generator and RadEval run in the same venv, `evaluate()` generates and scores in one call. Use it in
place of steps 2 and 3. This block needs a `report_generator` and a `ds` that you build as in step 2, in that
venv.

<!-- block: run=skip reason="needs a report generator that runs in the RadEval venv, the data and a GPU" -->
```python
from radharmony.evaluator import ReportGenerationEvaluator
from radharmony.evaluator.metrics import LIGHT_METRICS

ev = ReportGenerationEvaluator(
    report_generator,
    dataset=ds,
    metrics=LIGHT_METRICS,
    max_samples=4,
    n_bootstrap=10,
    output_dir="outputs/report_generation",
)
df = ev.evaluate()
ev.save_results(df)  # writes results.csv, results_summary.csv and generations.csv
```

The report generator is any function from the `img` batch to a `list[str]` with one report for each image.
With the 2 generator recipes, the `img` batch is a list of image paths. So you can also score your own model
in this way.

## Check

- `outputs/report_generation/pairs.parquet` has up to `max_samples` rows.
- `outputs/report_generation/scores/results.csv` has 11 rows: the scores (`bootstrap` is `-1`) and 10
  bootstrap samples. Its columns are `label`, `bootstrap`, `bleu`, `rouge1`, `rouge2`, `rougeL`,
  `bertscore`, the 3 `radgraph_*` scores and the 6 `f1chexbert_*` scores.
- Montgomery reports are short, for example "normal". So the scores mean little here. Use a dataset with
  full reports, for example [MIMIC-CXR](../reference/datasets/cxr-labels/mimic_cxr.md), for a real
  comparison.

## See also

- [Report generation](../reference/evaluators/report_generation.md): the reference sections
  (`ref_section`), the metric lists, the arguments and the output files.
- [CheXagent-2 (generator)](../reference/backbones/chexagent_gen.md) and
  [MAIRA-2 (generator)](../reference/backbones/maira2_gen.md): the 2 generator recipes.
