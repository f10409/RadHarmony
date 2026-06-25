# RadHarmony

[![Tests](https://github.com/f10409/RadHarmony/actions/workflows/tests.yml/badge.svg)](https://github.com/f10409/RadHarmony/actions/workflows/tests.yml)

**RadHarmony** is a Python library for loading and harmonizing radiological datasets with a unified API. It wraps [MONAI](https://monai.io/) to deliver ready-to-use **PyTorch `Dataset`** objects — drop them straight into a `DataLoader` for training and evaluation with minimal configuration. Chest X-ray is the primary, fully-supported modality; CT, MRI, and non-chest radiographs are available as <sup>beta</sup> and still under testing.

**Harmonize datasets** — load and unify multiple sources into one format:

![Dataset demo](docs/figs/demo_dataset.gif)

**Evaluate models** — probe foundation-model backbones across the supported datasets:

![Evaluator demo](docs/figs/demo_evaluator.gif)

## Documentation

Full docs: **[f10409.github.io/RadHarmony](https://f10409.github.io/RadHarmony)** — quickstart, per-dataset pages, transforms, architecture, the [Evaluator API](https://f10409.github.io/RadHarmony/evaluator/index.html) (linear / k-NN / SVM / prototype probes, zero-shot, fine-tune, three segmentation heads, backbone recipes), and the [App Guide](https://f10409.github.io/RadHarmony/app.html) (including remote access over an SSH tunnel).

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
git clone <repo-url>
cd RadHarmony
uv venv
uv pip install -e .
```

Python ≥ 3.10 required.

### Optional extras

The base install gives you the dataset API. Install extras for models, notebooks, or the Gradio app:

| Extra | Adds | Install |
|---|---|---|
| `model` | PyTorch, torchvision, transformers, CUDA runtime libs | `uv pip install -e ".[model]"` |
| `notebook` | ipykernel, ipywidgets | `uv pip install -e ".[notebook]"` |
| `app` | Gradio (for the visualizer app) | `uv pip install -e ".[app]"` |
| `all` | Everything above | `uv pip install -e ".[all]"` |
| `raddino` | RAD-DINO backbone (`rad-dino`) | `uv pip install -e ".[raddino]"` |
| `biomed` | BiomedCLIP backbone (`open-clip-torch`, `transformers`) | `uv pip install -e ".[biomed]"` |
| `chexagent` | XraySigLIP backbone (`transformers`) | `uv pip install -e ".[chexagent]"` |
| `medsiglip` | MedSigLIP backbone (`transformers`) | `uv pip install -e ".[medsiglip]"` |
| `medimageinsights` | MedImageInsights backbone (`huggingface_hub`, `mup`, …) | `uv pip install -e ".[medimageinsights]"` |
| `chexfound` | CheXFound backbone (`omegaconf`) + manual checkpoint | `uv pip install -e ".[chexfound]"` |
| `dinov3` | DINOv3 backbone (`transformers`) — Meta ViT, register tokens | `uv pip install -e ".[dinov3]"` |
| `eva_x` | EVA-X CXR ViT (`timm`, `huggingface_hub`, `torchvision`) — needs `third_party_models/EVA-X` submodule | `uv pip install -e ".[eva_x]"` |
| `ark_plus` | Ark+ Swin-L (`torchvision`) — manual checkpoint + side-installs `timm==0.5.4` into `third_party_models/Ark/timm-054` on first call | `uv pip install -e ".[ark_plus]"` |
| `medical_mae` | Medical MAE CXR ViT (`torchvision`) — cloned repo + manual checkpoint + side-installs `timm==0.4.12` into `third_party_models/medical_mae/timm-0412` on first call | `uv pip install -e ".[medical_mae]"` |
| `siglip2` | SigLIP 2 backbone (`transformers`) — general-domain vision-language baseline, 1152-d pooled embeddings, 384×384 input | `uv pip install -e ".[siglip2]"` |

Combine extras as needed, e.g. `uv pip install -e ".[model,app]"`.

## Quick Start

```python
import torch
from radharmony.dataset import CheXpertTrainDataset

ds = CheXpertTrainDataset(
    base_image_dir="/data/CheXpert-v1.0/train/",
    output_cls=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=10, num_cores=4)

sample = train_ds[0]
# sample["img"]  → torch.Tensor, shape (1, 224, 224), float in [-1, 1]
# sample["cls"]  → torch.Tensor, shape (14,), binary labels
```

For augmentations, k-fold splits, dataset registry, harmonizer save/reuse, and the Gradio visualizer — see the [wiki Quickstart](https://f10409.github.io/RadHarmony/quickstart.html) and per-dataset pages.

## Evaluators

`radharmony.evaluator` pairs a supported dataset with an image encoder and produces
a standardized results DataFrame (AUROC, AUPRC, F1, and eight other per-label
metrics). Six classification evaluators are available — linear probe, k-NN
probe, SVM probe, prototype probe, zero-shot, and fine-tune — plus three
segmentation evaluators (`LinearProbeSegEvaluator` 1×1 Conv2d head,
`ConvProbeSegEvaluator` conv-block head, and `UPerNetSegEvaluator` simple
feature pyramid + `UperNetHead`), all over frozen dense features.

Use a built-in backbone recipe for the fastest setup:

```python
from radharmony.evaluator.backbones import make_raddino
from radharmony.evaluator import LinearProbeEvaluator
from radharmony.dataset import VinDrCXRTrainDataset, VinDrCXRTestDataset

transform, encoder = make_raddino(device="cuda")

train_ds = VinDrCXRTrainDataset(base_image_dir="/data/VinDr-CXR/", transform=transform, output_cls=True, cache_dir=None)
test_ds  = VinDrCXRTestDataset (base_image_dir="/data/VinDr-CXR/", transform=transform, output_cls=True, cache_dir=None)

ev = LinearProbeEvaluator(
    encoder,
    train_dataset=train_ds,
    test_dataset=test_ds,
    n_seeds=5,
    n_train_samples=[500, 1500],
    n_bootstrap=100,
    output_dir="outputs/linprobe",
)
df = ev.evaluate()
ev.save_results(df)
# → outputs/linprobe/results.csv, results_summary.csv
```

Other evaluators follow the same interface:

```python
from radharmony.evaluator import KNNProbeEvaluator, SVMProbeEvaluator
from radharmony.evaluator import PrototypeProbeEvaluator, ZeroShotEvaluator
from radharmony.evaluator.backbones import make_biomed_clip

# k-NN probe (k-fold mode)
ev = KNNProbeEvaluator(encoder, dataset=train_ds, n_folds=5, k=20)

# SVM probe
ev = SVMProbeEvaluator(encoder, dataset=train_ds, n_folds=5)

# Prototype probe (parameter-free baseline)
ev = PrototypeProbeEvaluator(encoder, dataset=train_ds, n_folds=5)

# Zero-shot (vision-language models only)
_, vl_encoder, text_encoder, tokenizer = make_biomed_clip(device="cuda")
ev = ZeroShotEvaluator(
    image_encoder=vl_encoder,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    test_dataset=test_ds,
    prompts={"pneumothorax": ["a chest X-ray showing pneumothorax"]},
)
```

See the [evaluator wiki](https://f10409.github.io/RadHarmony/evaluator/index.html) for the full API and all constructor arguments.

## Supported Datasets

A broad collection of public radiological datasets covering 2D CXR, 3D CT, and 3D MRI — including CheXpert, CheXpert-Plus, MIMIC-CXR (DICOM/JPG), ChestX-ray14, PadChest, ReXGradient-160K, VinDr-CXR, SIIM-ACR PTX, SIIM COVID-19, RSNA Pneumonia, RSNA PE Detection <sup>beta</sup>, RSNA Bone Age <sup>beta</sup>, RSNA 2022 Cervical Spine <sup>beta</sup>, RSNA 2023 Abdominal Trauma <sup>beta</sup>, RSNA 2024 Lumbar Spine <sup>beta</sup>, CT-RATE <sup>beta</sup>, RAD-ChestCT <sup>beta</sup>, TAIX-Ray, BRAX, RANZCR CLiP, OpenI IU CXR, Montgomery County CXR, and Shenzhen Hospital CXR.

See the [Datasets inventory](https://f10409.github.io/RadHarmony/datasets.html) for the full table with registry keys, modalities, and label counts.

## Notebooks

Worked examples in [`notebooks/`](notebooks/):

| Notebook | Contents |
|---|---|
| [`tutorials/dataset_api_tour.ipynb`](notebooks/tutorials/dataset_api_tour.ipynb) | Full dataset-API tour: harmonizers, preprocessors, datasets, transforms, registry |
| [`tutorials/custom_dataset_tutorial.ipynb`](notebooks/tutorials/custom_dataset_tutorial.ipynb) | Template for integrating a new dataset |
| [`tutorials/evaluator_tutorial.ipynb`](notebooks/tutorials/evaluator_tutorial.ipynb) | End-to-end evaluator walkthrough: linear probe on VinDr-CXR with RAD-DINO |
| [`evaluator/fm_comparison.ipynb`](notebooks/evaluator/fm_comparison.ipynb) | FM benchmark — 11 foundation-model image encoders × 6 probing strategies on VinDr-CXR / TAIX-Ray |
| [`evaluator/fm_comparison_seg.ipynb`](notebooks/evaluator/fm_comparison_seg.ipynb) | FM benchmark — 11 encoders × three segmentation heads (linear / conv / UPerNet) on SIIM-ACR PTX and Montgomery-CXR |
