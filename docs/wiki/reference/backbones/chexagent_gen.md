# CheXagent-2 (generator)

| | |
|---|---|
| Recipe | `make_chexagent_generator()`, returns `(transform, report_generator)` |
| Model card | [StanfordAIMI/CheXagent-2-3b](https://huggingface.co/StanfordAIMI/CheXagent-2-3b) |
| Input size | 512 × 512 (bicubic resize of the whole image, no crop, in the model code) |
| Normalization | mean 0.5 and std 0.5 on each channel, from the XraySigLIP processor that the model code reads |
| Feature | None: it generates a report |
| Install extra | `chexagent_gen`, in its own venv |
| Segmentation features | None: it generates a report |

<!-- claim: id=chexagent_gen.card.params source="Stanford AIMI, CheXagent-2-3b model card 2024" expect="3B" -->
CheXagent-2 is a chest X-ray report generator from Stanford AIMI (Chen et al., arXiv 2024). It is a
vision-language model with 3B parameters. Its image encoder is
[XraySigLIP](chexagent.md), and a language model writes the text. Use it with
[`ReportGenerationEvaluator`](../evaluators/report_generation.md) to compare report generators.

## Quick use

This block writes one report for one [Montgomery](../datasets/cxr-boxes-masks/montgomery_cxr.md) image.

<!-- block: run=skip reason="needs the chexagent_gen venv" -->
```python
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator.backbones import make_chexagent_generator

transform, report_generator = make_chexagent_generator(device="cuda")
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_report=True,
    cache_dir=None,
)
sample = ds.get_datasets()[0]  # sample["img"] is the path of a temporary PNG
reports = report_generator([sample["img"]])
print(reports[0])
# FINDINGS: The hilar contours are normal. The lungs are clear. ...
#
# IMPRESSION: No acute cardiopulmonary process.
```

The output above is an example of one run. To score many reports, see
[Report generation](../evaluators/report_generation.md).

## How the input is prepared

The recipe works with image files, not tensors, because the CheXagent-2 code reads image paths.

1. Load: the shared loader gives a 3-channel `uint8` image in 0 to 255
   ([How a recipe loads an image](index.md#how-a-recipe-loads-an-image)).
2. PNG: the transform writes an RGB PNG to `tmp_dir`, and puts its path in `sample["img"]`, as a string.
   The PNG name has a hash of the source path, so 2 images with the same file name get 2 PNG files. When
   the PNG is already there, the transform uses it again. The transform does not change `report`.
3. Resize: the model code opens the PNG as RGB, and the whole image goes to 512 × 512 with bicubic
   interpolation. There is no crop.
4. Scale: the values are divided by 255.
5. Mean and std: 0.5 and 0.5 on each channel, so the values go to -1 to 1.

Steps 3 to 5 are the image transform of the model code (`modeling_visual.py` of the model).

The transform is a plain function `sample -> sample`, not a MONAI `Compose`.

## Segmentation mode

None: CheXagent-2 generates a report and gives no patch features.

## Generation

`report_generator(img_paths, indications=None)` takes a `list[str]` of PNG paths and gives a `list[str]`
of reports, one for each path.
Each call is greedy (`do_sample=False`, `num_beams=1`), with a system message `"You are a helpful
assistant."`, as on the model card.

With `prompt=None` (the default), it runs the multi-step "ABCDE" steps of the
[CheXagent demo](https://github.com/Stanford-AIMI/CheXagent) (`demos/app_demo.py`, commit `e4f31e6e`):

1. Findings: one call with the image for each of 5 regions. The regions are Airway, Breathing, Cardiac,
   Diaphragm, and "Everything else" (mediastinal contours, bones, soft tissues, tubes, valves and
   pacemakers). The prompt is `Please provide a detailed description of "<region>" in the chest X-ray`.
   The recipe joins the answers that are not empty with spaces.
2. Impression: one call with text only: `Write the Impression section for the following Findings:
   <findings>`.

The demo first asks for the view of the image and does not use the answer. The recipe skips this call.

The report is `"FINDINGS: <findings>\n\nIMPRESSION: <impression>"`.
[`ReportGenerationEvaluator`](../evaluators/report_generation.md) scores this whole text against the
`ref_section` of the real report. Before it scores, it removes the bold marks (`**`) that CheXagent-2 can
write (D-50).

With a `prompt` string, the generator makes one call for each image with that prompt. The recipe does not
change the answer.

To change the ABCDE steps, keep `prompt=None`. Then pass `findings_prompts` (one prompt for each region),
or `impression_prompt` (a template with `{findings}`):

<!-- block: run=skip reason="needs the chexagent_gen venv" -->
```python
transform, report_generator = make_chexagent_generator(
    findings_prompts=["Describe the lungs.", "Describe the heart and mediastinum."],
    impression_prompt="Summarize the impression from these findings: {findings}",
)
```

With `use_indication=True` in [`ReportGenerationEvaluator`](../evaluators/report_generation.md#indication),
the generator gets the indication of each image. It puts `"Clinical indication: <text>. "` before the
`prompt` string, or before each findings prompt. The impression call does not get it.

## Install

The model card asks for `transformers==4.40.0`, `torch==2.7.1` and `torchvision==0.22.1`. These versions
do not work with the other extras, so install the `chexagent_gen` extra in its own venv:

<!-- block: run=skip reason="installs packages" -->
```bash
uv venv .venv-chexagent
source .venv-chexagent/bin/activate
uv pip install -e ".[chexagent_gen]"
```

RadEval needs other versions too. Generate the reports in this venv with
`ReportGenerationEvaluator.generate_only()`, and score them in the `radeval` venv. See
[Report generation](../evaluators/report_generation.md).

`generate_only(path)` writes a parquet file with the columns `sample_id`, `reference` and `hypothesis`,
and `patient_id` when the dataset has patient IDs. It does not import RadEval.

## Arguments

<!-- args: radharmony.evaluator.backbones.make_chexagent_generator -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `device` | `str` | `"cuda"` | The device of the model. The recipe loads with `device_map=device`, so the whole model is on this device |
| `hub` | `str` | `"StanfordAIMI/CheXagent-2-3b"` | The Hugging Face name of the model |
| `prompt` | `str` | `None` | `None` runs the ABCDE steps. A string makes one call for each image with that prompt |
| `findings_prompts` | `list[str]` | `None` | The findings prompts of the ABCDE steps, one call for each. `None` uses the 5 region prompts of the demo |
| `impression_prompt` | `str` | `None` | The template of the impression call. It must have `{findings}`. `None` uses the template of the demo |
| `max_new_tokens` | `int` | `512` | The maximum number of new tokens of each call |
| `dtype` | `torch.dtype` | `torch.bfloat16` | The dtype of the model weights |
| `tmp_dir` | `str` | `None` | The folder for the PNG files. `None` makes a new folder under the system temp folder, and deletes it when Python exits. A folder that you give stays (D-16) |

## Notes

- The first call downloads the weights from Hugging Face.
- The model card loads with `device_map="auto"`, which can spread the model over all GPUs. The device
  does not change the output (D-16).
- The Hugging Face model has its own code, so the recipe loads it with `trust_remote_code=True`.
- With the default `prompt=None`, each image needs 6 calls of the model. Use `max_samples` of the
  evaluator for a short test.
- Put `tmp_dir` on a fast local disk for a large dataset.
- `chexagent_gen` and [`maira2_gen`](maira2_gen.md) cannot share a venv. See the `[tool.uv]`
  `conflicts` list in `pyproject.toml`.
- For the image and text encoder of CheXagent (zero-shot and probes), see
  [CheXagent (XraySigLIP)](chexagent.md).
