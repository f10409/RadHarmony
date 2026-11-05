# The backbone page

Write `docs/wiki/reference/backbones/<name>.md` from the template below.
`scripts/evalkit/wiki_audit.py` (`check_template`) checks the At a glance rows and the `##` sections. Copy the
style of `docs/wiki/reference/backbones/raddino.md` (image only), `biomed_clip.md` (text and image) or
`ark_plus.md` (a checkpoint of the authors).

## The template

- At a glance: a 2-column table right under the `#` title, with these 7 rows in this order: `Recipe`,
  `Model card`, `Input size`, `Normalization`, `Feature`, `Install extra`, `Segmentation features`.
- One paragraph: the model, its authors and paper, its training data, and what to use it for.
- The `##` sections start with `Quick use`, `How the input is prepared`, `Segmentation mode`. They end with
  `Arguments`, `Notes`. Optional sections go between them, for example `Text encoder`, `Feature` or
  `Weights and code`.

The skeleton (replace each `<...>`):

~~~markdown
# <Model name>

| | |
|---|---|
| Recipe | `make_<name>()`, returns `(transform, encoder)` |
| Model card | [<org/model>](https://huggingface.co/<org/model>) |
| Input size | <S> × <S> (<resize and crop, in words>) |
| Normalization | <mean and std>, from <source> |
| Feature | <token>, <D> values |
| Install extra | `<extra>` |
| Segmentation features | `Tensor[B, <D>, <G>, <G>]` |

<Model> is a <architecture> from <authors> (<paper, year>). The authors trained it on <data>. Use it for <task>.

## Quick use

This block encodes one [Montgomery](../datasets/cxr-boxes-masks/montgomery_cxr.md) image.

<!-- block: run=gpu timeout=600 -->
```python
import torch
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator.backbones import make_<name>

transform, encoder = make_<name>(device="cuda")  # text and image: transform, encoder, *_ = ...
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_cls=True,
    cache_dir=None,
)
img = ds.get_datasets()[0]["img"]  # Tensor[3, <S>, <S>], torch.bfloat16
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    feature = encoder(img[None].to("cuda"))
print(feature.shape)  # torch.Size([1, <D>])
```

## How the input is prepared

1. Load: the shared loader gives a 3-channel `uint8` image in 0 to 255
   ([How a recipe loads an image](index.md#how-a-recipe-loads-an-image)).
2. Resize: <...> (<source, for example preprocessor_config.json>).
3. Crop: <...>.
4. Scale: <...>.
5. Mean and std: <...>.
6. Dtype: the transform casts the image to `dtype`.

## Segmentation mode

<!-- block: run=gpu timeout=600 -->
```python
transform, encoder = make_<name>(device="cuda", output_keys={"img", "mask"})
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    patches = encoder(img[None].to("cuda"))  # img from Quick use
print(patches.shape)  # torch.Size([1, <D>, <G>, <G>])
```

<Which tokens, and the grid: <S> / <patch> = <G>.> See [Segmentation mode](index.md#segmentation-mode).

## Arguments

<!-- args: radharmony.evaluator.backbones.make_<name> -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `device` | `str` | `"cuda"` | The device of the encoder, for example `"cuda:1"` or `"cpu"` |
| `output_keys` | `set[str]` | `None` | The keys of each sample. `None` gives `{"img", "cls"}`. With `"mask"`, the encoder gives patch features ([Segmentation mode](index.md#segmentation-mode)) |
| `dtype` | `torch.dtype` | `torch.bfloat16` | The dtype of the image tensor ([Output dtype](index.md#output-dtype)) |

## Notes

- <The source of the feature and of each input step, and each D-16 choice.>
~~~

## Block notes

- A block that loads the model gets `<!-- block: run=gpu timeout=600 -->`. The block runner runs it only at
  `--level full` on a machine with a GPU. Without a GPU, the data or the package, the result is "blocked",
  not "failed".
- A block that needs another venv (for example a report generator) or installs packages gets
  `run=skip reason="..."`. A skip needs a reason.
- Write the real output as a comment, for example `# torch.Size([1, 768])`. Take it from a real run.
- Use `/data/...` paths. Never a NAS path (L-002).
- The `<!-- args: ... -->` line names the factory. Then `wiki_audit.py` compares each row and each default
  with the signature.

## Other doc changes

- `docs/wiki/reference/backbones/index.md`: a row in the "Recipes" table, and the recipe count in the first
  paragraph ("This page lists the 13 recipes"). The row has 7 columns: Recipe, Model, Feature size, Input
  size, Zero-shot (`yes` for a text-and-image recipe), Install extra, Page. For a clone or a checkpoint of the authors, also a row in the
  "Weights and code locations" table.
- `docs/wiki/get-started/install.md`: a row in the "Extras" table. If the extra conflicts with
  `chexagent_gen` or `maira2_gen`, update "More than one venv" too.
- `mkdocs.yml`: a line under `Backbones:` in `nav`, before the 2 report generators.
- `README.md` has no backbone table and no extras table. It links to the Install and Backbones pages. Do not
  add a table there.
