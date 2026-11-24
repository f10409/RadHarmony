# Install

RadHarmony needs Python 3.10 or newer. The base install gives the harmonizers, the dataset classes, the
evaluators and PyTorch. An extra adds the packages of one backbone, the app or the notebooks.

## Install with uv

We recommend [uv](https://docs.astral.sh/uv/). These commands make a venv and install RadHarmony from
the source files.

<!-- block: run=skip reason="installs packages" -->
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # only if you do not have uv
git clone https://github.com/f10409/RadHarmony.git
cd RadHarmony
uv venv
source .venv/bin/activate
uv pip install -e .
```

To add an extra, put its name in brackets. You can give more than one extra.

<!-- block: run=skip reason="installs packages" -->
```bash
uv pip install -e ".[raddino]"
uv pip install -e ".[model,app]"
```

## Install with pip

<!-- block: run=skip reason="installs packages" -->
```bash
git clone https://github.com/f10409/RadHarmony.git
cd RadHarmony
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Add an extra in the same way, for example `pip install -e ".[raddino]"`.

## Extras

These are all the extras in `pyproject.toml`.

| Extra | What it adds | Use it for |
|---|---|---|
| `model` | `torchvision`, `transformers` 5.6 or newer, and the NVIDIA CUDA libraries (cuDNN, cuSPARSELt, NCCL, NVSHMEM) | Models in your own code |
| `notebook` | `ipykernel`, `ipywidgets` | The notebooks in `notebooks/` |
| `app` | `gradio` | The app ([Use the app](../how-to/use-the-app.md)) |
| `all` | `model`, `notebook` and `app` | All 3 extras above |
| `raddino` | `rad-dino` | [RAD-DINO](../reference/backbones/raddino.md) |
| `biomed` | `open-clip-torch`, `transformers`, `torchvision` | [BiomedCLIP](../reference/backbones/biomed_clip.md) |
| `chexagent` | `transformers`, `sentencepiece`, `protobuf` | [CheXagent](../reference/backbones/chexagent.md) (the vision encoder) |
| `medsiglip` | `transformers`, `sentencepiece`, `protobuf` | [MedSigLIP](../reference/backbones/medsiglip.md) |
| `siglip2` | `transformers`, `sentencepiece`, `protobuf` | [SigLIP 2](../reference/backbones/siglip2.md) |
| `dinov3` | `transformers` | [DINOv3](../reference/backbones/dinov3.md) |
| `medimageinsights` | `mup`, `fvcore`, `timm`, `transformers` and 8 more | [MedImageInsights](../reference/backbones/medimageinsights.md) |
| `chexfound` | `omegaconf`, `torchvision` | [CheXFound](../reference/backbones/chexfound.md) |
| `eva_x` | `timm`, `huggingface_hub`, `torchvision` | [EVA-X](../reference/backbones/eva_x.md) |
| `ark_plus` | `torchvision` | [Ark+](../reference/backbones/ark_plus.md) |
| `medical_mae` | `torchvision` | [Medical MAE](../reference/backbones/medical_mae.md) |
| `chexagent_gen` | `transformers` 4.40.0, `torch` 2.7.1, `torchvision` 0.22.1 and more | [CheXagent report generator](../reference/backbones/chexagent_gen.md), in its own venv |
| `maira2_gen` | `transformers` 4.46, `huggingface-hub` older than 1.0, and more | [MAIRA-2 report generator](../reference/backbones/maira2_gen.md), in its own venv |
| `medgemma_gen` | `transformers` 5.6 or newer, `torch` 2.7.1, `torchvision` 0.22.1 and more | MedGemma report generator and [VQA](../reference/evaluators/vqa.md) recipes, in its own venv. Accept the MedGemma license on Hugging Face and log in first |
| `radeval` | RadEval from GitHub (`radeval[api]`) | The report metrics ([Run report generation](../how-to/run-report-generation.md)). It needs Python 3.11 or newer |

Some backbones also need a clone of the authors' code or a checkpoint. The page of each backbone says what
to get ([Backbones](../reference/backbones/index.md)).

## More than one venv

The 3 generator extras need versions of `transformers` and `torch` that differ from the other extras. So
these extras cannot share a venv with the other extras:

- `chexagent_gen`: its own venv. It can share a venv only with `notebook`, `app` and `raddino`.
- `maira2_gen`: its own venv. It cannot share a venv with `chexagent_gen`, `medgemma_gen`, `radeval`, `model`,
  `all`, `biomed`, `chexagent`, `medsiglip`, `dinov3`, `siglip2` or `medimageinsights`.
- `medgemma_gen`: its own venv. It can share a venv only with `notebook`, `app`, `raddino` and `radeval`.
- `radeval`: it can go in your main venv when that venv has Python 3.11 or newer. On Python 3.10, the extra
  installs nothing and gives no error. It cannot share a venv with `chexagent_gen` or `maira2_gen`.

The `[tool.uv]` `conflicts` list in `pyproject.toml` gives each pair that cannot share a venv. A report
generation run uses 2 venvs: the generator venv makes the reports, and the `radeval` venv scores them.
[Run report generation](../how-to/run-report-generation.md#1-make-the-2-venvs) makes the 2 venvs.

## Data paths for the checks

You give the data folders to each dataset class as arguments, so RadHarmony needs no data paths file.
The integrity check (`scripts/check_datasets.py`) and the tests on real data read the folders from
`configs/data_paths.json`. To run them for some datasets, make this file once:

<!-- block: run=skip reason="copies a file in a clone" -->
```bash
cp configs/data_paths.example.json configs/data_paths.json
```

Then keep the entries of the datasets that you want to check, and give their real folders. Your copy
stays out of git. To keep the file in another place, set `RADHARMONY_DATA_PATHS` to its path. A dataset
with no entry is skipped, and a check of it gives a message that names this file.

## Claude Code skills

The repo has skills for Claude Code in `.claude/skills/`: `use-radharmony` (use the package),
`add-dataset`, `add-backbone`, `evaluate` (review the code and docs) and `fix`. In a clone of the repo,
Claude Code loads them with no step. To use a skill in another project, copy its folder to your skills
folder:

<!-- block: run=skip reason="copies a folder" -->
```bash
cp -r .claude/skills/use-radharmony ~/.claude/skills/
```

`add-dataset`, `add-backbone`, `evaluate` and `fix` change the repo, so use them in a clone.

## Check

<!-- block: run=ci -->
```python
from radharmony.registry import list_datasets

print("montgomery_cxr" in list_datasets())  # True
```

## Next steps

- [Your first benchmark in 10 minutes](first-benchmark.md): run your first benchmark, step by step.
- [Concepts](concepts.md): the parts of RadHarmony and how they work together.
