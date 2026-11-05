# Check the recipe against the authors (J-05, D-16)

A recipe must give the same features as the authors' own pipeline. A shape check is not enough. A wrong crop or
a wrong mean still gives the right shape. This file has 3 parts: the fact table, the run of
`.claude/skills/add-backbone/scripts/compare_with_authors.py`, and the test for `tests/test_backbones.py`.

## Two past mistakes (the evidence for J-05)

- V-B-1, EVA-X: the recipe took the CLS token. The authors' fine-tune scripts use `--use_mean_pooling`: the
  mean of the patch tokens, then the final norm `fc_norm`. Same shape, different feature.
- V-B-2, Ark+: the recipe resized to 768 with no crop. The authors' fine-tune code resizes to 896 × 896 and then
  takes a center crop of 768 (`--img_size 896 --input_size 768`). Same shape, different input.

## 1. Fill the fact table before you write code

Find each fact in the authors' sources. Write the source next to each value: a URL, or a file and a line.

| Fact | Authors' value | Source | Recipe value |
|---|---|---|---|
| Input size | for example 518 × 518 | | |
| Resize and crop | for example short side to 518 (bicubic), then center crop 518 | | |
| Mean and std | for example 0.5307 and 0.2583 on each channel | | |
| Pooled token | for example the CLS token after the final norm, 768 values | | |

Where to look, in this order:

1. The model card. Its code example is the first source. With no network, the model card text can also be in
   the `METADATA` file of the authors' pip package.
2. `preprocessor_config.json` of a Hugging Face model: `size`, `crop_size`, `resample`, `image_mean`,
   `image_std`, `do_center_crop`.
3. The authors' eval code: the linear-probe script, the fine-tune data loader in `"valid"` or `"test"` mode,
   and its command-line arguments in the README.

When the authors' sources do not agree, D-16 decides:

- Two input steps (for example pretraining and fine-tune): use the downstream (fine-tune) input steps.
- Two features (for example fine-tune and linear probe): the classification feature follows the linear-probe
  code. Example: Ark+ gives the projector output (1376 values), not `forward_features` (1536 values).

If D-16 does not cover a difference, ask the user with `.claude/skills/evaluate/references/decisions.md`.
Write each fact in the backbone page: the At a glance rows, "How the input is prepared", and "Notes" for the
source and any choice.

## 2. Run the compare script

Write the authors' pipeline in a small file in a temporary folder, for example `/tmp/`. The file has one
function, `load(device)`. It returns `prepare(path) -> Tensor[C, H, W]` and `feature(x) -> Tensor[B, D]`. For
`--seg`, it also returns `patches(x) -> Tensor[B, D, G, G]`. Copy the authors' code. Do not use RadHarmony code
in this file.

Example for RAD-DINO. Its authors' code is the `rad-dino` package (class `RadDino`), and the model card runs it:

```python
"""RAD-DINO as its authors run it: the rad-dino package that the microsoft/rad-dino model card uses."""
from PIL import Image
from rad_dino import RadDino
from transformers.feature_extraction_utils import BatchFeature


def load(device):
    rad_dino = RadDino().to(device)

    def prepare(path):
        return rad_dino.preprocess(Image.open(path).convert("RGB"))["pixel_values"][0]

    def feature(x):
        cls_token, _ = rad_dino.encode(BatchFeature({"pixel_values": x}))
        return cls_token  # the CLS token after the final norm

    def patches(x):
        _, patch_tokens = rad_dino.encode(BatchFeature({"pixel_values": x}))
        return rad_dino.reshape_patch_tokens(patch_tokens)  # [B, 768, 37, 37]

    return prepare, feature, patches
```

When the authors' code is a plain `transformers` call, write the transformers equivalent. Example:
`AutoImageProcessor.from_pretrained(hub)` for `prepare`, and `AutoModel.from_pretrained(hub)(pixel_values=x)`
with the authors' token for `feature`. Use the processor settings of the authors' code. Example: `RadDino` loads
the processor with `use_fast=False` (the PIL resize). The default processor of `transformers` 5.6 uses another
resize. On one Montgomery image, the 2 inputs differ by up to 0.015.

Run the script with the model venv (`$RADHARMONY_MODEL_PYTHON`) on one free GPU (here GPU 0). Give 4 real
images in one of 2 ways:

- `--dataset montgomery_cxr`: the script reads the folder from the data paths file with
  `scripts/dataset_paths.py`. With no file or no entry, it stops with the hint ("Real data" in SKILL.md).
- `--images`: 4 image files that the user gives.

```bash
CUDA_VISIBLE_DEVICES=0 $RADHARMONY_MODEL_PYTHON .claude/skills/add-backbone/scripts/compare_with_authors.py \
    --recipe make_raddino --authors /tmp/authors_raddino.py --dataset montgomery_cxr --n 4
```

Run it a second time with `--seg` when the recipe has a segmentation mode. Then the script builds the recipe
with `output_keys={"img", "mask"}`, and compares its patch features with `patches`. It does not check the mask
steps.

Output for RAD-DINO:

```text
image                            recipe input     authors input       input    token      end
MCUCXR_0001_0.png                (3, 518, 518)    (3, 518, 518)     1.00000  1.00000  1.00000
...
feature size: recipe (768,), authors (768,)
PASS
```

With `--seg`, the last line before `PASS` is `feature size: recipe (768, 37, 37), authors (768, 37, 37)`.

- `input` compares the 2 input tensors. A low value means a wrong size, crop, mean or std.
- `token` runs the recipe encoder on the authors' input. A low value means a wrong pooled token.
- `end` compares the whole recipe with the whole authors' pipeline.

Each value must be 0.999 or more. Example: we gave the script a wrong authors file for RAD-DINO. It had a
resize to 518 × 518 with no crop, and the mean of the patch tokens. The script gave input 0.80, token 0.18
and FAIL. For a recipe with a manual checkpoint, pass its arguments as JSON: `--recipe-kwargs '{"checkpoint_path": "/data/models/x.pth"}'`.

## 3. Add a test

Add a test to `tests/test_backbones.py`, so that the check runs again after each change. Use the helpers
`_gray` and `_cos` of that file. Skip the test when the package or the weights are not on the machine. Name
the source in the docstring. Example for RAD-DINO (it passed with the model venv):

```python
def test_raddino_input_and_feature_match_the_rad_dino_package(tmp_path):
    """D-16: the rad-dino package of the microsoft/rad-dino model card: RadDino.preprocess, then RadDino.encode."""
    rad_dino = pytest.importorskip("rad_dino")
    hub = pytest.importorskip("huggingface_hub")
    if not isinstance(hub.try_to_load_from_cache("microsoft/rad-dino", "config.json"), str):
        pytest.skip("RAD-DINO weights not cached")
    from PIL import Image
    from transformers.feature_extraction_utils import BatchFeature

    from radharmony.evaluator.backbones import make_raddino

    path = tmp_path / "gray.png"
    Image.fromarray(_gray(300, 400)).save(path)
    transform, encoder = make_raddino(device="cpu", dtype=torch.float32)
    authors = rad_dino.RadDino()

    ours = transform({"img": str(path), "cls": torch.zeros(1)})["img"]
    theirs = authors.preprocess(Image.open(path).convert("RGB"))["pixel_values"][0]
    assert ours.shape == theirs.shape == (3, 518, 518)  # input size and crop
    assert _cos(ours, theirs) > 0.999  # mean and std
    with torch.no_grad():
        got = encoder(theirs[None])
        expected, _ = authors.encode(BatchFeature({"pixel_values": theirs[None]}))
    assert got.shape == expected.shape == (1, 768)
    assert _cos(got, expected) > 0.999  # pooled token
```

When the authors' input steps are torchvision code, compare with that code, as in
`test_ark_plus_image_geometry_matches_the_authors`. When the feature needs the weights, compare with the
authors' pooling on the same model, as in `tests/test_eva_x.py`. Run the test with the model venv:

```bash
PYTHONPATH=$PWD $RADHARMONY_MODEL_PYTHON -m pytest tests/test_backbones.py -v -k <name>
```
