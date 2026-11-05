# Intent: backbones

Each `make_<name>()` factory in `radharmony/evaluator/backbones/` gives the model as its authors released
it, so that the benchmark compares fair features.

**Design decisions that apply** (`docs/wiki/design.md`). Test each one, and read its known gaps:
- D-16: input size, crop, mean, std and the feature token
- D-17: L2 norm in the probes
- Any other entry in the design.md section "Backbones and evaluators" that applies to this part.

**Other test steps:**

1. **Preprocessing.** Also check the resize method, the channel count, and the [0, 1] or [0, 255] range.
   Compare each value with the Hugging Face `preprocessor_config.json` (cached in
   `~/.cache/huggingface/hub`), or with the official code that the file copies or cites.
2. **Features.** The shape matches `feature_dim` or `embed_dim`. Patch tokens have the right grid for
   segmentation.
3. **Weights.** The model is in eval mode with no dropout. The right checkpoint loads, and a weights
   mismatch is not hidden (`strict=False` with missing keys).
4. **Zero-shot text.** For CLIP-like models, the text encoder, tokenizer, context length and logit scale
   match the official code. The prompt templates are as documented.
5. **Generative models.** `make_chexagent_generator` and `make_maira2_generator` make text with the documented prompt format.
6. **Determinism.** The same input gives the same features two times.

**How to test:** load each model on the GPU from your task. Run 2 to 4 real CXR images, for example from
`paths("chexpert_valid")` (see `references/parts.md`, "Data and environments", for a key with no entry). Compare with a reference pipeline that you build from the model card, for
example HF `AutoImageProcessor` + `AutoModel`. Report the max abs difference and the cosine similarity. If
the weights of a model are not cached locally, skip it and mark it BLOCKED.
