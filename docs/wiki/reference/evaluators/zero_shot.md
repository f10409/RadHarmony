# ZeroShotEvaluator

| | |
|---|---|
| Registry key | `zero_shot` |
| Task | Classification with text prompts, no training |
| Dataset outputs it needs | `cls` (`output_cls=True`) |
| Split modes | test only (`test_dataset=` or `dataset=`) |
| Encoder it needs | an image encoder and a text encoder with one shared feature space |
| Output | The result table, one row for each label, for the point estimate and each bootstrap sample. See [Output](#output) |

The zero-shot evaluator scores each test image against text prompts for each label, with the cosine of
the image feature and the prompt feature. It trains nothing and uses only test data. Use it for a
vision-language model, for example a CLIP or SigLIP model.

## Quick use

This block scores [Montgomery](../datasets/cxr-boxes-masks/montgomery_cxr.md) (138 images) with BiomedCLIP,
1 positive and 1 negative prompt, and 100 bootstrap samples.

<!-- block: run=gpu timeout=600 -->
```python
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator import ZeroShotEvaluator
from radharmony.evaluator.backbones import make_biomed_clip

transform, encoder, text_encoder, tokenizer = make_biomed_clip(device="cuda")
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_cls=True,
    cache_dir=None,
)

ev = ZeroShotEvaluator(
    encoder,
    text_encoder,
    prompts={"tuberculosis": ["a chest X-ray with tuberculosis"]},
    negative_prompts={"tuberculosis": ["a normal chest X-ray"]},
    tokenizer=tokenizer,
    test_dataset=ds,
    n_bootstrap=100,
    threshold_strategy="fixed:0",  # without it, F1 and the other threshold metrics are NaN
    output_dir="outputs/zero_shot",
)
df = ev.evaluate()
ev.save_results(df)  # writes results.csv and results_summary.csv
print(df.columns.tolist())
# ['n_train', 'fold', 'seed', 'bootstrap', 'label', 'auroc', 'auprc', 'f1', ...]
```

## How it works

- Image features: the evaluator runs the image encoder one time over the test data. It always
  L2-normalizes the features, as CLIP does (D-17).
- Prompt features: the text encoder encodes each prompt. The evaluator L2-normalizes each prompt feature,
  takes the mean over the prompts of a label, and normalizes the mean again.
- Score: `cos(image, positive prompts)`. When the label has negative prompts, the score is
  `cos(image, positive prompts) - cos(image, negative prompts)`.
- Labels: each evaluated label needs positive prompts, or the run raises `KeyError`. The prompt lookup
  ignores case. To evaluate only some labels, pass `labels=`. A label with no negative prompts uses only
  its positive prompts.
- Threshold: there is no train data, so there is no validation part. The threshold metrics are NaN,
  except with `threshold_strategy="fixed:<float>"` (D-29). The evaluator warns about this. When every
  label has negative prompts, `"fixed:0"` means "closer to the positive prompts".
- Bootstrap: `n_bootstrap` works with `dataset=` too (D-32).

## Arguments

<!-- args: radharmony.evaluator.ZeroShotEvaluator -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `text_encoder` | callable | required | It gets the output of `tokenizer`, or the `list[str]` of prompts with no `tokenizer`. It returns `Tensor[T, D]` |
| `prompts` | `dict[str, list[str]]` | required | The positive prompts of each label |
| `negative_prompts` | `dict[str, list[str]]` | `None` | The negative prompts of each label. A label can have none |
| `tokenizer` | callable | `None` | It turns the prompts into the input of `text_encoder`. `None` gives the strings to `text_encoder` |

### Shared arguments (inherited from BaseClsEvaluator)

| Argument | Type | Default | Description |
|---|---|---|---|
| `image_encoder` | `nn.Module` or callable | required | The image encoder. It returns `Tensor[B, D]` ([Encoder contract](index.md#encoder-contract)) |
| `dataset` | dataset | `None` | The test data. It is the same as `test_dataset`. Pass only one of the 2 |
| `train_dataset` | dataset | `None` | Not used. With `test_dataset`, the class drops it with no warning. With `dataset=`, or alone, it raises `ValueError` |
| `test_dataset` | dataset | `None` | The test data |
| `labels` | `list[str]` | `None` | The labels to evaluate. `None` gives all labels. The match ignores case. An unknown name raises `KeyError` |
| `device` | `str` | `"cuda"` | `"cuda"`, `"cuda:N"` or `"cpu"` |
| `batch_size` | `int` | `64` | The batch size of the feature extraction |
| `num_workers` | `int` | `4` | The number of DataLoader workers |
| `autocast_dtype` | `torch.dtype` | `torch.bfloat16` | The autocast type of the image encoder. `None` gives no autocast |
| `embedding_cache` | `str` | `None` | The path of the feature file. The file is `<path>.test.pkl` ([Embedding cache](index.md#embedding-cache)) |
| `l2_normalize` | `bool` | `False` | No effect: the score is always a cosine. This class sets `True`, and `False` warns (D-17) |
| `output_dir` | `str` | `None` | The folder for `save_results()` |
| `n_seeds` | `int` | `1` | Not used. A value other than 1 warns |
| `base_seed` | `int` | `0` | Not used |
| `n_bootstrap` | `int` | `0` | The number of bootstrap samples of the test set |
| `bootstrap_seed` | `int` | `0` | The seed of the bootstrap samples |
| `threshold_strategy` | `str` | `"youden"` | Use `"fixed:<float>"` for the threshold metrics ([How it works](#how-it-works)) |

## Output

`evaluate()` returns the [result table](index.md#result-table) with these columns: `n_train`, `fold`,
`seed`, `bootstrap`, `label`, `auroc`, `auprc`, `f1`, `accuracy`, `balanced_accuracy`, `tpr`, `tnr`,
`ppv`, `npv`, `mcc` and `threshold`. `n_train`, `fold` and `seed` are always `-1`. Each label has
`1 + n_bootstrap` rows: the point estimate (`bootstrap = -1`) and one row for each bootstrap sample.
`save_results(df)` writes `results.csv` and `results_summary.csv` to `output_dir`.

## Notes

- A Hugging Face processor (an object with an `image_processor`) gets the prompts as `text=`. A SigLIP
  processor pads to its `model_max_length`, because SigLIP trains with a fixed text length.
  Another processor pads to the longest prompt.
- A longer prompt than the `model_max_length` of a SigLIP processor is cut to that length, as the Hugging Face
  zero-shot pipeline does. The class warns with the number of these prompts (D-52).
- Any callables work in place of a backbone recipe, if they follow the
  [encoder contract](index.md#encoder-contract). See [Add a backbone](../../how-to/add-a-backbone.md).
