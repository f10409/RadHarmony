# MAIRA-2 (generator)

| | |
|---|---|
| Recipe | `make_maira2_generator()`, returns `(transform, report_generator)` |
| Model card | [microsoft/maira-2](https://huggingface.co/microsoft/maira-2) |
| Input size | The MAIRA-2 processor (`format_and_preprocess_reporting_input`) resizes and crops the PNG. See the model card |
| Normalization | The MAIRA-2 processor applies it |
| Feature | None: it generates a report |
| Install extra | `maira2_gen`, in its own venv |
| Segmentation features | None: it generates a report |

<!-- claim: id=maira2_gen.card.params source="Microsoft, MAIRA-2 model card 2024" expect="7B" -->
MAIRA-2 is a chest X-ray report generator from Microsoft Health Futures (Bannur et al., arXiv 2024). It
has 7B parameters: a RAD-DINO-MAIRA-2 image encoder and a Vicuna-7B v1.5 language model. Microsoft
trained it on MIMIC-CXR, PadChest and a private US dataset. Use it with
[`ReportGenerationEvaluator`](../evaluators/report_generation.md) to compare report generators.

## Quick use

This block writes one report for one [Montgomery](../datasets/cxr-boxes-masks/montgomery_cxr.md) image.

<!-- block: run=skip reason="needs the maira2_gen venv" -->
```python
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator.backbones import make_maira2_generator

transform, report_generator = make_maira2_generator(device="cuda")
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_report=True,
    cache_dir=None,
)
sample = ds.get_datasets()[0]  # sample["img"] is the path of a temporary PNG
reports = report_generator([sample["img"]])
print(reports[0])  # the findings text of one report
```

To score many reports, see [Report generation](../evaluators/report_generation.md).

## How the input is prepared

The recipe works with image files, not tensors, the same as
[CheXagent-2 (generator)](chexagent_gen.md).

1. Load: the shared loader gives a 3-channel `uint8` image in 0 to 255
   ([How a recipe loads an image](index.md#how-a-recipe-loads-an-image)).
2. PNG: the transform writes an RGB PNG to `tmp_dir`, and puts its path in `sample["img"]`, as a string.
   The PNG name has a hash of the source path, so 2 images with the same file name get 2 PNG files. When
   the PNG is already there, the transform uses it again. The transform does not change `report`.
3. Model input: `report_generator` opens the PNG as RGB. The MAIRA-2 processor resizes, crops and
   normalizes it.

The transform is a plain function `sample -> sample`, not a MONAI `Compose`.

## Segmentation mode

None: MAIRA-2 generates a report and gives no patch features.

## Generation

`report_generator` takes a `list[str]` of PNG paths and gives a `list[str]` of reports, one for each path.
It follows the single-image report steps of the model card:

1. Open the PNG as RGB.
2. Call `processor.format_and_preprocess_reporting_input()` with the image as `current_frontal`. The other
   inputs are `None`: `current_lateral`, `prior_frontal`, `indication`, `technique`, `comparison` and
   `prior_report`. RadHarmony datasets give only the image and the real report.
3. Generate with greedy decoding (`do_sample=False`, `num_beams=1`, `use_cache=True`), with at most
   `max_new_tokens` new tokens.
4. Decode the new tokens with `skip_special_tokens=True`, and strip the spaces at the start and the end.

The recipe does not change the text of the model. With `get_grounding=True`, the text also has the
grounding marks of the model. [`ReportGenerationEvaluator`](../evaluators/report_generation.md) does not
remove these marks: its cleaning step knows only the CheXagent-2 marks (D-50). So use
`get_grounding=False` for a report that you score.

## Install

MAIRA-2 needs an older `transformers` than the other extras and RadEval. So install the `maira2_gen`
extra in its own venv:

<!-- block: run=skip reason="installs packages" -->
```bash
uv venv .venv-maira2
source .venv-maira2/bin/activate
uv pip install -e ".[maira2_gen]"
```

Generate the reports in this venv with `ReportGenerationEvaluator.generate_only()`, and score them in the
`radeval` venv. See [Report generation](../evaluators/report_generation.md).

`generate_only(path)` writes a parquet file with the columns `sample_id`, `reference` and `hypothesis`,
and `patient_id` when the dataset has patient IDs. It does not import RadEval.

## Arguments

<!-- args: radharmony.evaluator.backbones.make_maira2_generator -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `device` | `str` | `"cuda"` | The device of the model. The recipe moves the whole model to it with `.to(device)` |
| `hub` | `str` | `"microsoft/maira-2"` | The Hugging Face name of the model |
| `max_new_tokens` | `int` | `300` | The maximum number of new tokens of each report |
| `dtype` | `torch.dtype` | `torch.bfloat16` | The dtype of the model weights |
| `tmp_dir` | `str` | `None` | The folder for the PNG files. `None` makes a new folder under the system temp folder, and deletes it when Python exits. A folder that you give stays (D-16) |
| `get_grounding` | `bool` | `False` | `True` asks for a grounded report, with a box for each finding |

## Notes

- The first call downloads the weights from Hugging Face.
- The Hugging Face model is gated. Accept its license on the model card, then run `hf auth login` before the
  first call.
- The `maira2_gen` extra pins `transformers>=4.46.0,<4.47`, because `transformers` 5 breaks the model
  code. The current model card asks for `transformers>=4.48.0,<4.52`.
- For a grounded report, the model card uses `max_new_tokens=450`. Pass `max_new_tokens=450` together
  with `get_grounding=True`.
- The model card turns the decoded text into plain text with
  `processor.convert_output_to_plaintext_or_grounded_sequence()`. The recipe does not call it.
- The Hugging Face model has its own code, so the recipe loads it with `trust_remote_code=True`.
- MAIRA-2 can also use a prior image and a prior report. The recipe always gives `None` for them. To use
  them, call the processor yourself.
- `maira2_gen` and [`chexagent_gen`](chexagent_gen.md) cannot share a venv. See the `[tool.uv]`
  `conflicts` list in `pyproject.toml`.
