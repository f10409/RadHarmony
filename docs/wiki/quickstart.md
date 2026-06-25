# Quickstart

## Install

Requires **Python ≥ 3.10**.

### With [uv](https://docs.astral.sh/uv/) (recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/f10409/RadHarmony.git

# Create and activate a virtual environment
uv venv /path/to/my/env
source /path/to/my/env/bin/activate  # Windows: /path/to/my/env/Scripts/activate

uv pip install -e /path/to/RadHarmony
```

### Optional extras

The base install gives you the dataset API. Install extras for models, notebooks, or the Gradio app:

| Extra | Adds | Install |
|---|---|---|
| `model` | PyTorch, torchvision, transformers, CUDA runtime libs | `uv pip install -e "/path/to/RadHarmony[model]"` |
| `notebook` | ipykernel, ipywidgets | `uv pip install -e "/path/to/RadHarmony[notebook]"` |
| `app` | Gradio (for the visualizer app) | `uv pip install -e "/path/to/RadHarmony[app]"` |
| `all` | Everything above | `uv pip install -e "/path/to/RadHarmony[all]"` |

Backbone extras for the evaluator (install only the ones you need):
`raddino`, `biomed`, `chexagent`, `medsiglip`, `medimageinsights`, `chexfound`,
`dinov3`, `eva_x`, `ark_plus`, `medical_mae`, `siglip2`.
E.g. `uv pip install -e "/path/to/RadHarmony[raddino]"`. See the
[Evaluator API](evaluator/index.md) for the backbone recipes table.

Combine extras as needed, e.g. `uv pip install -e "/path/to/RadHarmony[model,app]"`.

### With pip

```bash
git clone https://github.com/f10409/RadHarmony.git

# Create and activate a virtual environment
python -m venv /path/to/my/env
source /path/to/my/env/bin/activate  # Windows: /path/to/my/env/Scripts/activate

pip install -e /path/to/RadHarmony
```

## Launch the app

```bash
python app.py
# or with uv:
uv run python app.py
```

Open http://localhost:7860 in your browser. The app lists all supported datasets grouped by modality (CXR, CT, MRI, Radiograph). Point it at your local data directory and click **Load**.

For remote access (server + local laptop), see [App Guide → Remote access](app.md#remote-access-ssh-tunnel).

## Load your first dataset in code

```python
import torch
from radharmony.dataset import CheXpertTrainDataset

ds = CheXpertTrainDataset(
    base_image_dir="/data/CheXpert-v1.0/train/",
    output_cls=True,
    dtype=torch.float32,   # float32 works on CPU and all GPUs
)
train_ds, val_ds = ds.get_datasets(n_splits=5)
sample = train_ds[0]
# sample["img"]  → torch.Tensor of shape (1, 224, 224)  # 224 is the default img_size
# sample["cls"]  → torch.Tensor of shape (14,)  — 14 CheXpert labels
```

The CSV (`train.csv`) is auto-discovered relative to `base_image_dir`. Pass `csv_path=` explicitly if auto-discovery fails. Use `CheXpertValidDataset` for the valid split.

## dtype: float32 vs bfloat16

| dtype | When to use |
|-------|-------------|
| `torch.float32` | CPU, any GPU — universally supported |
| `torch.bfloat16` | Ampere+ NVIDIA GPUs (A100, RTX 30xx+) — faster training |

Default is `torch.bfloat16`. Pass `dtype=torch.float32` if you are on CPU or an older GPU.

## Next steps

- [Datasets](datasets.md) — full list of supported datasets
- [Dataset API](api.md) — splits, folds, data dict keys, caching
- [Transforms](transforms.md) — augmentations and custom pipelines
