# VQAEvaluator

Registry key: `"vqa"`.

Runs a **question-conditioned** vision-language model (VLM) over a RadHarmony VQA dataset and scores the generated answers against the reference answers, reproducing the **MedGemma technical report** VQA protocol (report §3.4, Table 9).

VQA has no training phase, so the k-fold / fixed-split machinery of the classification evaluators does not apply. This evaluator takes a single `dataset=`, iterates it once, and returns a `pandas.DataFrame` of metric scores. Variance (if requested) comes from bootstrapping the scored `(reference, hypothesis)` pairs.

## Answerer contract

The evaluator never inspects model internals. Unlike `ReportGenerationEvaluator` (whose generator takes only images), the VQA answerer also receives the per-sample **questions**:

```text
vqa_answerer(imgs, questions: list[str]) -> list[str]     # len(imgs) answers
```

For CheXagent-2 and MedGemma use the [`make_medgemma_vqa`](../backbones/index.md) / `make_chexagent_vqa` recipes. They return `(transform, vqa_answerer)` where `transform` handles image loading (any MONAI-readable format including DICOM) and stores the PNG **path string** under `img`; for those recipes `imgs` is a list of file-path strings.

## Metrics (MedGemma protocol)

Scores are reported in three buckets, one row per bucket (matching Table 9):

| `label` | Metric(s) | Over which QAs |
|---------|-----------|----------------|
| `overall` | `token_f1` | **all** QAs |
| `closed` | `accuracy`, `token_f1` | the **yes/no** subset (reference answer is exactly `yes`/`no`) |
| `open` | `token_f1`, `token_recall` | the remaining (non-yes/no) QAs |

- **Tokenized F1** is the standard VQA-RAD / SLAKE token-overlap F1: normalize (lowercase, strip punctuation + articles, collapse whitespace), then multiset token-overlap F1 between prediction and reference.
- **Closed accuracy** extracts the predicted `yes`/`no` (first such token, so a fuller sentence still grades) and compares to the reference.
- **Open token recall** is the fraction of reference tokens recovered — the SLAKE open-ended column.

The metric helpers are public: `from radharmony.evaluator.metrics import tokenized_f1, token_recall, extract_yes_no, is_yes_no_answer`.

## Replicating the MedGemma paper numbers

Use `make_medgemma_vqa` (loads `google/medgemma-4b-it`) — the numbers in Table 9 are MedGemma's own, so a different backbone (e.g. `make_chexagent_vqa`) produces a *different model's* score, not a replication. The recipe encodes the paper's faithfulness details: greedy decoding (temperature 0), the VQA-RAD prompt from Appendix Table A7, and **no** persona system message for MedGemma (per the Table A7 caption).

```python
from radharmony.evaluator import VQAEvaluator
from radharmony.evaluator.backbones import make_medgemma_vqa
from radharmony.dataset import VQARadDataset

transform, answerer = make_medgemma_vqa(device="cuda")   # gated HF model

ds = VQARadDataset(
    base_image_dir="/path/to/VQA-RAD/VQA_RAD Image Folder",
    transform=transform,          # yields the PNG path the answerer expects
    cache_dir=None,
)

ev = VQAEvaluator(answerer, dataset=ds, n_bootstrap=0, output_dir="outputs/vqa")
df = ev.evaluate()
ev.save_results(df)
```

!!! note "Split comparability"
    Table 9's VQA-RAD numbers use the contamination-free split of Yang et al. (2024) / Xu et al. (2023). That specific split is **not** shipped here, so by default the evaluator runs over the **entire** VQA-RAD set and the numbers are not directly comparable. `VQARadHarmonizer` does emit a `split` column marking the dataset's *original* (contamination-flagged) 451-question test set; to evaluate on it, filter the harmonized DataFrame and rebuild the dataset with `harmonized_df=`.

## Constructor arguments

Specific to `VQAEvaluator`:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `vqa_answerer` | callable | — | `(imgs, questions) -> list[str]` |
| `is_closed` | callable | `is_yes_no_answer` | `(reference) -> bool` marking the closed (yes/no) subset |

### Shared arguments (inherited from `GenerativeEvaluator`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `dataset` | dataset | — | A VQA dataset (`VQARadDataset`, `MIMICExtCXRQBADataset`) |
| `device` | `str` | `"cuda"` | Kept for symmetry; the answerer owns its device |
| `batch_size` | `int` | `8` | Inference DataLoader batch size |
| `num_workers` | `int` | `4` | Inference DataLoader workers |
| `max_samples` | `int` | `None` | Optional cap on scored pairs (smoke tests) |
| `n_bootstrap` | `int` | `0` | Bootstrap resamples of the `(ref, hyp)` list for CIs |
| `bootstrap_seed` | `int` | `0` | RNG seed for the bootstrap resampler |
| `output_dir` | `str` | `None` | Directory for CSV output |

## Output schema

`evaluate()` returns a `pandas.DataFrame`:

| Column | Meaning |
|--------|---------|
| `label` | `"overall"` / `"closed"` / `"open"` |
| `fold`, `seed` | Always `-1` (no folds/seeds for VQA) |
| `bootstrap` | `-1` = point estimate; `0..n_bootstrap-1` = resample |
| `token_f1`, `accuracy`, `token_recall` | Bucket metrics; `NaN` where a metric does not apply to that bucket |

`save_results(df, output_dir)` writes `results.csv`, `results_summary.csv`, and `generations.csv` (the raw `question, reference, hypothesis, closed` rows behind the scores).

## Two-stage workflow

`generate_only(out_parquet)` runs the answerer and writes a `(question, reference, hypothesis, closed)` parquet without scoring — useful when generation and scoring run in separate environments.
