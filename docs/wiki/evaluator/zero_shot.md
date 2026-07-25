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
from radharmony.dataset import VinDrCXRTestDataset

transform, encoder, text_encoder, tokenizer = make_biomed_clip(device="cuda")

# VinDr-CXR test split (carries both pneumothorax and cardiomegaly labels).
# Pass the recipe's transform so preprocessing matches the encoder.
ds = VinDrCXRTestDataset(
    base_image_dir="/mnt/NAS4/datasets/external/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/test/",
    transform=transform,
    output_cls=True,
    cache_dir=None,
)

ev = ZeroShotEvaluator(
    image_encoder=encoder,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    dataset=ds,
    prompts={
        "pneumothorax": ["a chest X-ray showing pneumothorax", "pneumothorax present"],
        "cardiomegaly": ["a chest X-ray showing cardiomegaly"],
    },
    negative_prompts={
        "pneumothorax": ["a normal chest X-ray without pneumothorax"],
        # cardiomegaly is left out on purpose: labels absent here fall back to
        # positive-prompt-only scoring (see Notes).
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
    dataset=ds,  # the VinDrCXRTestDataset built above
    prompts={"pneumothorax": ["a chest X-ray showing pneumothorax"]},
)
```

---

## Constructor arguments

Specific to `ZeroShotEvaluator`:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `text_encoder` | callable | — | `(tokens: Tensor) -> Tensor[T, D]`; receives tokenized text |
| `prompts` | `dict[str, list[str]]` | — | Maps each label name to one or more positive prompt strings |
| `negative_prompts` | `dict[str, list[str]]` \| `None` | `None` | Optional negative prompts per label |
| `tokenizer` | callable \| `None` | `None` | Tokenizer forwarded to `text_encoder` when provided |

### Shared arguments (inherited from `BaseClsEvaluator`)

Zero-shot uses only a test set, so `train_dataset`, `n_seeds`, and the
train-subsample knobs are ignored (see Notes). The remaining shared arguments
apply:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `image_encoder` | `nn.Module` / callable | — | `forward(imgs) → Tensor[B, D]` |
| `dataset` | dataset | `None` | Test set in k-fold-style slot; `test_dataset=` is routed here |
| `test_dataset` | dataset | `None` | Fixed-split mode test set (train side is dropped) |
| `device` | `str` | `"cuda"` | `"cuda"`, `"cuda:N"`, or `"cpu"` |
| `batch_size` | `int` | `64` | Inference DataLoader batch size |
| `num_workers` | `int` | `4` | Inference DataLoader workers |
| `autocast_dtype` | `torch.dtype` | `torch.bfloat16` | Inference autocast dtype; `None` disables |
| `embedding_cache` | `str` | `None` | Path prefix to pickle cached embeddings (caches store **raw** features; normalization is reapplied after load) |
| `l2_normalize` | `bool` | `False` | L2-normalize features at the cache boundary |
| `output_dir` | `str` | `None` | Directory for CSV output |
| `n_bootstrap` | `int` | `0` | Test-row bootstrap resamples for confidence intervals |
| `bootstrap_seed` | `int` | `0` | RNG seed base for test-row resampling |
| `threshold_strategy` | `str` | `"youden"` | `"youden"`, `"f1"`, or `"fixed:<float>"` |

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
