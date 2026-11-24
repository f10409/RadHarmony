# VQAEvaluator

| | |
|---|---|
| Registry key | `vqa` |
| Task | Visual question answering, scores generated answers |
| Dataset outputs it needs | `question` and `answer` (every VQA dataset gives them) |
| Split modes | test only (`dataset=`) |
| Encoder it needs | a VQA answerer, `(imgs, questions) -> list[str]`, one answer for each image ([Encoder contract](index.md#encoder-contract)) |
| Output | The result table, one row for each answer group, for the point estimate and for each bootstrap sample. See [Output](#output) |

The VQA evaluator runs a vision-language model over a VQA dataset. The model gets each image and its
question, and writes an answer. The evaluator scores each model answer against the reference, with the
metrics of the MedGemma technical report (section 3.4, Table 9). The reference is the real answer of the
dataset, `sample["answer"]`. A pair is one question with its reference and its model answer.

Use the evaluator to compare VQA models on the same questions. It does not look inside the model, so any answerer
works.

## Quick use

The recipe [`make_medgemma_vqa`](../backbones/index.md) needs its own venv (`medgemma_gen` extra).
`google/medgemma-4b-it` is a gated Hugging Face model. Accept its license on the model page and run
`huggingface-cli login` first. This block runs MedGemma on the 451 test questions of
[VQA-RAD](../datasets/vqa-text/vqa_rad.md), and scores the model answers:

<!-- block: run=skip reason="needs the medgemma_gen venv, the gated MedGemma model, the data and a GPU" -->
```python
from radharmony.dataset import VQARadTestDataset
from radharmony.evaluator import VQAEvaluator
from radharmony.evaluator.backbones import make_medgemma_vqa

transform, vqa_answerer = make_medgemma_vqa(device="cuda")
ds = VQARadTestDataset(
    base_image_dir="/data/VQA-RAD/VQA_RAD Image Folder/",
    transform=transform,
    cache_dir=None,
)

ev = VQAEvaluator(vqa_answerer, dataset=ds, n_bootstrap=100, output_dir="outputs/vqa")
df = ev.evaluate()
ev.save_results(df)  # writes results.csv, results_summary.csv and generations.csv
print(df[df["bootstrap"] == -1][["label", "token_f1", "accuracy", "token_recall"]])
```

## How it works

- Generation: the evaluator runs `vqa_answerer` on each batch of `img` and `question`. It needs one answer
  for each image. With the MedGemma and CheXagent recipes, `img` is the path of a PNG file, so the batch is
  a list of path strings. An answerer can also take only `imgs`. Then the evaluator does not give it the
  questions.
- Reference: the evaluator uses `sample["answer"]` as the reference.
- Dropped pairs: the evaluator drops a pair when the reference or the model answer is empty.
- Cleaning: the evaluator removes the white space at the start and the end of each model answer.
- Groups: a pair is closed when its reference is `yes` or `no` (`is_closed`). The other pairs are open. See
  [Metrics](#metrics).
- Bootstrap: each bootstrap sample draws pairs with replacement. It does not group them by patient. A
  bootstrap sample with no closed pair has no `closed` row. A bootstrap sample with no open pair has no
  `open` row.
- Splits: the evaluator has no folds and no training ([Split modes](index.md#split-modes)).

## Metrics

The result table has one row for each group of pairs:

| `label` | Metrics | Answers |
|---|---|---|
| `overall` | `token_f1` | All pairs |
| `closed` | `accuracy`, `token_f1` | The pairs with the reference `yes` or `no` |
| `open` | `token_f1`, `token_recall` | The other pairs |

- `token_f1`: the token F1 that the MedGemma report uses for VQA-RAD and SLAKE. The metric first changes each text to lower case. It
  changes each character that is not a letter, a digit or a space to a space. It removes the words `a`,
  `an` and `the`. Then it counts the tokens that the model answer and the reference share. The precision
  is this count divided by the number of model answer tokens. The recall is this count divided by the
  number of reference tokens. `token_f1` is the F1 of this precision and this recall.
- `accuracy`: the part of the closed pairs with a correct model answer. A model answer is correct when its
  first `yes` or `no` token is the reference. So the model answer "Yes, the heart is large." is correct
  for the reference `yes`.
- `token_recall`: the part of the reference tokens that the model answer has. The MedGemma report gives
  it for the open SLAKE questions.

`radharmony.evaluator.metrics` exports the metric functions. They need no data:

<!-- block: run=ci -->
```python
from radharmony.evaluator.metrics import extract_yes_no, is_yes_no_answer, token_recall, tokenized_f1

print(tokenized_f1("The left lung.", "left lung"))  # 1.0
print(tokenized_f1("left pleural effusion", "pleural effusion"))  # 0.8
print(round(token_recall("pleural effusion", "left pleural effusion"), 3))  # 0.667
print(extract_yes_no("No, there is no effusion."))  # no
print(is_yes_no_answer(" Yes "), is_yes_no_answer("yes, small"))  # True False
```

## Compare with the MedGemma report

Table 9 of the MedGemma report gives the scores of MedGemma. To compare with Table 9, use
`make_medgemma_vqa`. It follows the report in 3 ways:

- It uses greedy decoding. At each step, it picks the most likely token.
- Its prompt is the VQA-RAD prompt of Table A7 of the report. The question comes after the prompt.
- It gives no system message by default (`system_message=None`). The caption of Table A7 says that
  MedGemma got no system message.

The VQA-RAD scores of Table 9 use a test split with no overlap with the train data (Yang et al. 2024, Xu et
al. 2023). RadHarmony does not have this split. `VQARadTestDataset` gives the 451 test questions of the
VQA-RAD paper.

202 of the 203 images of these questions also have train questions. So your scores can differ from
Table 9, because the test questions are not the same.

## Arguments

<!-- args: radharmony.evaluator.VQAEvaluator -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `vqa_answerer` | callable | required | A function `(imgs, questions) -> list[str]`. It returns one answer for each image of the batch ([Encoder contract](index.md#encoder-contract)) |
| `is_closed` | callable | `is_yes_no_answer` | A function `(reference) -> bool`. `True` puts the answer in the closed group. The default gives `True` when the reference is `yes` or `no`, in any case |

### Shared arguments (inherited from GenerativeEvaluator)

These arguments have the same names as in the other evaluators. `VQAEvaluator` sets them in its own
signature and gives them to `GenerativeEvaluator`.

| Argument | Type | Default | Description |
|---|---|---|---|
| `dataset` | dataset | required | Test only: the one VQA dataset ([Split modes](index.md#split-modes)) |
| `device` | `str` | `"cuda"` | Not used. The answerer sets its own device |
| `batch_size` | `int` | `8` | The batch size of the answerer |
| `num_workers` | `int` | `4` | The number of DataLoader workers |
| `max_samples` | `int` | `None` | The largest number of pairs to score. Use it for a short test run |
| `n_bootstrap` | `int` | `0` | The number of bootstrap samples of the pairs |
| `bootstrap_seed` | `int` | `0` | The seed of the bootstrap samples |
| `output_dir` | `str` | `None` | The folder for `save_results()` |

## Output

`evaluate()` returns the [result table](index.md#result-table), one row for each group, for the point
estimate and for each bootstrap sample. With `n_bootstrap=0`, `evaluate()` gives 3 rows: `overall`,
`closed` and `open`.

| Column | Meaning |
|---|---|
| `fold` | Always `-1` |
| `seed` | Always `-1` |
| `bootstrap` | `-1` for the point estimate. `0` to `n_bootstrap - 1` for a bootstrap sample |
| `label` | `overall`, `closed` or `open` |
| `token_f1` | The mean token F1 of the group |
| `accuracy` | The accuracy of the closed pairs. NaN in the `overall` and `open` rows |
| `token_recall` | The mean token recall. NaN in the `overall` and `closed` rows |

A dataset with no closed pair gives no `closed` row and no `accuracy` column. A dataset with no open pair
gives no `open` row and no `token_recall` column.

`save_results(df)` writes 3 files to `output_dir`:

- `results.csv`: the result table.
- `results_summary.csv`: one row for each group. For each metric, it gives `<metric>_mean` (the point
  estimate), `<metric>_n`, and `<metric>_std`, `<metric>_ci_lo` and `<metric>_ci_hi` over the bootstrap
  samples (D-19). With `n_bootstrap` below 2, the std and the CI are NaN. They are also NaN for a
  group that is in fewer than 2 bootstrap samples.
- `generations.csv`: the columns `question`, `reference`, `hypothesis` and `closed`. `hypothesis` is the
  model answer.

`generate_only(path)` runs the answerer with no scoring. It writes a parquet file with the same 4 columns.
You can score the file later with the metric functions of [Metrics](#metrics).

## Notes

- A transform must keep the `question` and `answer` keys. The default transform of a VQA dataset and the
  VQA recipes keep them. A missing key raises `KeyError`.
- `save_results()` warns that some metrics are NaN in some rows. These are the metrics that do not apply
  to a group, for example `accuracy` in the `overall` row. The summary is correct.
- The CheXagent recipe `make_chexagent_vqa` needs the `chexagent_gen` venv. It puts a short instruction
  before each question.
- For a hands-on notebook with MedGemma, see
  [`datathon26/6_radharmony_hands_on_vqa_report_generation.ipynb`](https://github.com/f10409/RadHarmony/blob/main/datathon26/6_radharmony_hands_on_vqa_report_generation.ipynb).
