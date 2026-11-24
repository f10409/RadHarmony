# Backbones

A backbone recipe is a `make_<name>()` function in `radharmony.evaluator.backbones`. It loads one foundation
model and returns a dataset transform and an encoder. Give the transform to the dataset and the encoder to an
[evaluator](../evaluators/index.md). Each recipe follows the input steps and the feature of the authors' own
code (D-16). This page lists the 16 recipes and the facts that they share.

A recipe of an image-only model returns `(transform, encoder)`. A recipe of a text-and-image model also
returns a text encoder and its tokenizer or processor, for [zero-shot](../evaluators/zero_shot.md). For
`make_medimageinsights`, the 4th value is `None`. A recipe of a report model returns
`(transform, report_generator)`. A recipe of a visual question answering (VQA) model returns
`(transform, vqa_answerer)`, for [`VQAEvaluator`](../evaluators/vqa.md).

## Recipes

| Recipe | Model | Feature size | Input size | Zero-shot | Install extra | Page |
|---|---|---|---|---|---|---|
| `make_raddino` | RAD-DINO | 768 | 518 × 518 | no | `raddino` | [RAD-DINO](raddino.md) |
| `make_dinov3` | DINOv3 ViT-B/16 | 768 | 224 × 224 | no | `dinov3` | [DINOv3](dinov3.md) |
| `make_eva_x` | EVA-X | 768 (base), 384 (small), 192 (tiny) | 224 × 224 | no | `eva_x` | [EVA-X](eva_x.md) |
| `make_medical_mae` | Medical MAE | 768 (base), 384 (small) | 224 × 224 | no | `medical_mae` | [Medical MAE](medical_mae.md) |
| `make_ark_plus` | Ark+ | 1376 | 768 × 768 | no | `ark_plus` | [Ark+](ark_plus.md) |
| `make_chexfound` | CheXFound | 4096 | 512 × 512 | no | `chexfound` | [CheXFound](chexfound.md) |
| `make_biomed_clip` | BiomedCLIP | 512 | 224 × 224 | yes | `biomed` | [BiomedCLIP](biomed_clip.md) |
| `make_chexagent` | XraySigLIP | 1024 | 512 × 512 | yes | `chexagent` | [CheXagent (XraySigLIP)](chexagent.md) |
| `make_medsiglip` | MedSigLIP | 1152 | 448 × 448 | yes | `medsiglip` | [MedSigLIP](medsiglip.md) |
| `make_medimageinsights` | MedImageInsight | 1024 | 480 × 480 | yes | `medimageinsights` | [MedImageInsight](medimageinsights.md) |
| `make_siglip2` | SigLIP 2 so400m | 1152 | 384 × 384 | yes | `siglip2` | [SigLIP 2](siglip2.md) |
| `make_chexagent_generator` | CheXagent-2-3b | a report (text) | 512 × 512 (in the model code) | no | `chexagent_gen` | [CheXagent-2 (generator)](chexagent_gen.md) |
| `make_maira2_generator` | MAIRA-2 | a report (text) | set by the MAIRA-2 processor | no | `maira2_gen` | [MAIRA-2 (generator)](maira2_gen.md) |
| `make_medgemma_generator` | MedGemma 4B (`google/medgemma-4b-it`, gated) | a report (text) | set by the MedGemma processor | no | `medgemma_gen` | no page yet |
| `make_medgemma_vqa` | MedGemma 4B (`google/medgemma-4b-it`, gated) | an answer (text) | set by the MedGemma processor | no | `medgemma_gen` | no page yet. See [VQA](../evaluators/vqa.md) |
| `make_chexagent_vqa` | CheXagent-2-3b | an answer (text) | 512 × 512 (in the model code) | no | `chexagent_gen` | no page yet. See [VQA](../evaluators/vqa.md) |

Feature size is the length of the feature in classification mode. Each page gives the shape of the patch
features in [segmentation mode](#segmentation-mode). Install the extra of a recipe before you use it:

<!-- block: run=skip reason="installs packages" -->
```bash
uv pip install -e ".[raddino]"
```

## How a recipe loads an image

All recipes start with the same loader (`load_image()` and `display_255()` in
`radharmony/evaluator/backbones/_utils.py`):

1. MONAI `LoadImage` reads the file. PNG, JPEG, DICOM and NIfTI files work.
2. The image is transposed to `(C, H, W)`.
3. The values go to 0 to 255, in the same way as in the dataset load (D-13, D-27). A DICOM file gets its VOI
   window. An 8-bit image keeps its values. Any other image is scaled with its minimum and maximum.
4. A 1-channel image is repeated to 3 channels.
5. The values are rounded to `uint8`. BiomedCLIP and MedImageInsight make a PIL RGB image instead. The
   transform of a report or VQA recipe saves a PNG file.

The model's own steps (resize, crop, mean and std) come after this. Each page lists them. The transform runs
on the CPU, in the DataLoader workers. The encoder and the text encoder run on `device`. The text encoder
gives its output on the CPU.

## Segmentation mode

Pass `output_keys={"img", "mask"}` (or `{"img", "cls", "mask"}`) to an encoder recipe. Then the encoder
returns patch features, `Tensor[B, D, H', W']`, for the segmentation evaluators
([linear](../evaluators/linear_probe_seg.md), [conv](../evaluators/conv_probe_seg.md) and
[UPerNet](../evaluators/upernet_seg.md)). Without `"mask"`, the encoder returns the feature, `Tensor[B, D]`.
One encoder gives the feature or the patch features, not both. To get the other output, call the recipe again.

- The patch grid follows from the input size: `H' = W' =` input size / patch size. For a hierarchical model
  (the Swin of Ark+), divide by the stride of the last stage.
- The transform gives the mask the same resize and crop as the image, with nearest-neighbor interpolation and
  no scaling.
- The mask keeps the values of the file, for example 0 and 255. The segmentation probes take each value above
  0 as foreground in a binary mask (D-14).
- The report and VQA recipes do not take `output_keys`.

## Output dtype

`dtype` (default `torch.bfloat16`) is the dtype of the image tensor that the transform gives. The labels
(`cls`, `reg`, `mask`) are always float32. The evaluators run the encoder under autocast (`autocast_dtype`,
default `torch.bfloat16`), so the float32 weights take the bfloat16 image.

When you call an encoder yourself, use `torch.autocast`, as on each page in Quick use. Or pass
`dtype=torch.float32` to the recipe. The report and VQA recipes use `dtype` for the model weights instead.

## Weights and code locations

Four recipes need files that do not come from Hugging Face. They need a clone of the authors' repo, a
checkpoint that you download by hand, or both. By default, the recipe looks in `third_party_models/` at the
repo root. The arguments in this table point to other folders, for example shared storage.

| Recipe | Clone | Checkpoint | Arguments |
|---|---|---|---|
| [`make_eva_x`](eva_x.md) | `third_party_models/EVA-X/` | from Hugging Face | `repo_path` |
| [`make_medical_mae`](medical_mae.md) | `third_party_models/medical_mae/` | `<mae_dir>/<variant file>.pth` | `mae_dir`, `checkpoint_path`, `legacy_timm_dir` |
| [`make_ark_plus`](ark_plus.md) | not needed | `third_party_models/Ark/Ark_Plus/Ark6_swinLarge768_ep50.pth.tar` | `checkpoint_path`, `legacy_timm_dir` |
| [`make_chexfound`](chexfound.md) | `third_party_models/CheXFound/` | `<chexfound_dir>/weights/teacher_checkpoint.pth` | `chexfound_dir`, `checkpoint_path` |

`make_ark_plus` and `make_medical_mae` need an old `timm` to build the model. On the first call, they install
it into `legacy_timm_dir` with `uv pip install --target --no-deps`. This step needs `uv` and the network. The
old `timm` is on `sys.path` only while the recipe builds the model. The rest of the session keeps the `timm`
of the venv.

Point `legacy_timm_dir` to a fast local folder that all users can read. Then the install runs one time, and it
stays after a cleanup of `third_party_models/`.

<!-- block: run=skip reason="needs the Ark+ checkpoint at a /data/ path" -->
```python
from radharmony.evaluator.backbones import make_ark_plus

transform, encoder = make_ark_plus(
    checkpoint_path="/data/models/Ark_Plus/Ark6_swinLarge768_ep50.pth.tar",
    legacy_timm_dir="/data/cache/ark-timm-054",
    device="cuda",
)
```

## Report generators

`make_chexagent_generator` and `make_maira2_generator` wrap a model that writes a report. They do not give a
feature.

- `transform`: a function from a sample dict to a sample dict. It writes the image of the sample to a
  temporary PNG file and puts the path (a `str`) in `img`. The report stays as it is.
- `report_generator`: a function from a `list[str]` of image paths to a `list[str]` of reports. Give it to
  [`ReportGenerationEvaluator`](../evaluators/report_generation.md).

The `chexagent_gen` extra pins `transformers==4.40.0` (from the model card). The `maira2_gen` extra pins
`transformers>=4.46.0,<4.47`, and the MAIRA-2 model card asks for `transformers>=4.48.0,<4.52`. So
`pyproject.toml` declares both extras as conflicting with each other and with most other extras. Install each
one in its own venv.

## Authoring a new recipe

See [Add a backbone](../../how-to/add-a-backbone.md).
