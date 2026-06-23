# ZeroShotEvaluator

Registry key: `"zero_shot"`.

Scores test images against per-label text prompts using cosine similarity.
No training loop — only the test set is used. Requires a vision-language
model that exposes both an image encoder and a text encoder.

Score per label:

- **Positive prompts only**: `score = cos(img_emb, mean(pos_prompt_embs))`
- **With negative prompts**: `score = cos(img_emb, mean(pos)) − cos(img_emb, mean(neg))`

---

## Usage

```python
from radharmony.evaluator import ZeroShotEvaluator
from radharmony.evaluator.backbones import make_biomed_clip

transform, encoder, text_encoder, tokenizer = make_biomed_clip(device="cuda")

ev = ZeroShotEvaluator(
    image_encoder=encoder,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    test_dataset=test_ds,
    prompts={
        "pneumothorax": ["a chest X-ray showing pneumothorax", "pneumothorax present"],
        "cardiomegaly": ["a chest X-ray showing cardiomegaly"],
    },
    negative_prompts={
        "pneumothorax": ["a normal chest X-ray without pneumothorax"],
    },
    n_bootstrap=100,
)
df = ev.evaluate()
```

Without a backbone recipe — pass callables directly:

```python
import open_clip

model, _, _ = open_clip.create_model_and_transforms("ViT-B-16-SigLIP")
tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")

ev = ZeroShotEvaluator(
    image_encoder=lambda x: model.encode_image(x),
    text_encoder=lambda toks: model.encode_text(toks),
    tokenizer=tokenizer,
    test_dataset=test_ds,
    prompts={"pneumothorax": ["a chest X-ray showing pneumothorax"]},
)
```

---

## Constructor arguments

`ZeroShotEvaluator` takes the same shared arguments as other evaluators (see
[Common constructor arguments](index.md#common-constructor-arguments)) plus:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `text_encoder` | callable | — | `(tokens: Tensor) -> Tensor[T, D]`; receives tokenized text |
| `prompts` | `dict[str, list[str]]` | — | Maps each label name to one or more positive prompt strings |
| `negative_prompts` | `dict[str, list[str]]` \| `None` | `None` | Optional negative prompts per label |
| `tokenizer` | callable \| `None` | `None` | Tokenizer forwarded to `text_encoder` when provided |

---

## Notes

- Only the test set is used. `train_dataset` is ignored when paired with
  `test_dataset`. `n_seeds` is ignored (no train subsampling).
- `n_bootstrap` still applies — test-row resamples generate confidence intervals
  on the zero-shot metric panel.
- Prompts not listed in `prompts` are silently skipped, so you can pass a
  partial `prompts` dict to evaluate only a label subset.
- Multiple prompts per label are averaged (ensemble) before the cosine
  similarity is computed.
- Labels missing from `negative_prompts` are scored with positive prompts only.
