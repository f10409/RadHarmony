# Recipe rules

The how-to page `docs/wiki/how-to/add-a-backbone.md` has the 3 main rules (step 4), the common `model_call`
functions (step 1) and the loader functions of `_utils.py` (step 4). This file adds what an agent also needs.
Example recipes:

- `raddino.py`: a pip package. `dinov3.py`: a short Hugging Face recipe.
- `chexagent.py`: text and image.
- `ark_plus.py` and `eva_x.py`: a clone or a checkpoint of the authors.

## More `model_call` functions

The how-to table has the common cases. These cases are not in it:

| Model family | `model_call` | Note |
|---|---|---|
| Hugging Face SigLIP, when `AutoModel` gives the joint model | `lambda m, x: m.vision_model(pixel_values=x).pooler_output` | The image tower only |
| ViT with register tokens (DINOv3), segmentation mode | `m(pixel_values=x).last_hidden_state[:, 1 + R:]`, then `patches_to_spatial` | Skip the CLS token and the `R` register tokens |
| SSL model (MAE, SimCLR) | `lambda m, x: m.backbone(x)`, or the attribute that holds the encoder | Read the package docs |

A notebook in `notebooks/backbones/` often shows the call in 2 or 3 lines. Copy it, then check it with
`references/authors-check.md`. A notebook is not the authors' code.

## Rules for the recipe file

Follow the 3 rules of `docs/wiki/how-to/add-a-backbone.md`, step 4 (lazy imports, one model, all image steps
in the transform). Also:

1. Start from `radharmony/evaluator/backbones/_template.py`. It is a working Hugging Face recipe,
   `make_my_model()`. Keep its `# ── Section ──` lines.
2. Keep the parts that the template already has:
   - the image loader that starts with `load_image()`, and the mask loader with `mt.LoadImage` (L-012),
   - the number of register tokens from the model config, not a number in the code,
   - the 2 modes (`_seg_mode`): classification gives `[B, D]`, segmentation gives `[B, D, G, G]`
     (`patches_to_spatial`),
   - the mask preprocess with the same resize and crop as the image, nearest neighbor, and no normalization,
   - the `dtype` argument, passed to `RadiologyEncoderTransform`,
   - `processor=` in `EncoderPreprocessTransform.from_custom`, so the box warning can read the crop and pad
     (D-28).
3. Register tokens do not make the grid smaller. Example: DINOv3 (4 register tokens) with a 224 input and patch
   16 gives a 14 × 14 grid.
4. For an OpenCLIP or torchvision model, the mask preprocess takes the resize and the crop of the image steps
   with `InterpolationMode.NEAREST`. See `biomed_clip.py` or `eva_x.py`.
5. A text-and-image model returns `(transform, image_encoder, text_encoder, tokenizer)`. The text encoder
   gives a float32 tensor on the CPU. The end of the template shows it.
6. Add a parameter only for a real setting of the model. Examples: `hub=` for the variants of one family,
   `checkpoint_path=` for a manual download. Put the default hub name in a module constant `_<NAME>_HUB`.

## Docstrings and comments

- Module docstring: the model name, the input format, the input size, and the install line as an indented
  code block.
- Factory docstring: give a "Returns" section with the feature size in classification mode and the shape in
  segmentation mode. Users read the feature size there.
- Comments say why, not what (J-10). Example: `# CLS + register tokens`.
- Type hints as in `_template.py`: `tuple[mt.Compose, ImageEncoderWrapper]`. For text and image, add
  `Callable[[torch.Tensor], torch.Tensor], Any`.

## Do not

- Do not trust a shape check alone. Run `references/authors-check.md`.
- Do not leave a recipe whose import fails. A broken import in `backbones/__init__.py` stops every evaluator.
