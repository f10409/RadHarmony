# Compare backbones

To compare 2 or more encoders, run the same probe on the same dataset with each backbone recipe. Then put
the results in one table. This page compares RAD-DINO, DINOv3 and BiomedCLIP with the linear probe on
[Montgomery County CXR](../reference/datasets/cxr-boxes-masks/montgomery_cxr.md) (138 images).

Before you start, install the extras of the recipes, and
[get the Montgomery data](get-the-data.md#montgomery-county-cxr).

<!-- block: run=skip reason="installs packages" -->
```bash
uv pip install -e ".[raddino,dinov3,biomed]"
```

## 1. Run the probe with each recipe

Each recipe has its own transform, so each recipe gets its own dataset object. Each recipe also gets its own
`embedding_cache` path, named after the recipe, for example `montgomery_raddino`. A cache file from a
different encoder raises `ValueError`. Keep the dataset, `n_folds` and `base_seed` the same for all recipes.
Then all recipes get the same folds and the same validation part.

<!-- block: run=gpu timeout=900 -->
```python
import gc

import torch

from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator import LinearProbeEvaluator
from radharmony.evaluator.backbones import make_biomed_clip, make_dinov3, make_raddino

recipes = {"raddino": make_raddino, "dinov3": make_dinov3, "biomed_clip": make_biomed_clip}
for name, make in recipes.items():
    transform, encoder = make(device="cuda")[:2]  # a text-and-image recipe also gives a text encoder
    ds = MontgomeryCXRDataset(
        base_image_dir="/data/MontgomerySet/CXR_png/",
        transform=transform,
        output_cls=True,
        cache_dir=None,
    )
    ev = LinearProbeEvaluator(
        encoder,
        dataset=ds,
        n_folds=5,
        embedding_cache=f"outputs/cache/montgomery_{name}",
        output_dir=f"outputs/compare/{name}",
    )
    ev.save_results(ev.evaluate())  # writes results.csv and results_summary.csv
    del transform, encoder, ds, ev  # the transform also holds the model: free the GPU memory
    gc.collect()
    torch.cuda.empty_cache()
```

In a real run on a shared GPU, the 3 recipes took about 2 minutes. For DINOv3, `save_results()` warns that
`ppv` (the positive predictive value) is NaN in 1 of the 5 folds (D-19).

## 2. Make one table

Read the `results_summary.csv` of each backbone, and keep the `macro_average` row. The summary gives the
mean, the std and the 95% CI over the 5 folds ([Result table](../reference/evaluators/index.md#result-table)).

<!-- block: run=gpu timeout=600 -->
```python
import pandas as pd

summary = pd.concat(
    pd.read_csv(f"outputs/compare/{name}/results_summary.csv").assign(backbone=name) for name in recipes
)
table = summary[summary["label"] == "macro_average"].set_index("backbone")
print(table[["auroc_mean", "auroc_std", "auroc_ci_lo", "auroc_ci_hi", "f1_mean"]].round(3))
#              auroc_mean  auroc_std  auroc_ci_lo  auroc_ci_hi  f1_mean
# backbone
# raddino           0.971      0.043        0.895        1.000    0.880
# dinov3            0.869      0.082        0.747        0.964    0.575
# biomed_clip       0.932      0.084        0.786        0.997    0.701
table.to_csv("outputs/compare/table.csv")
```

With 138 images and 5 folds, the std is large. A difference between 2 backbones that is smaller than the
std can come from the split only. For a real comparison, use a larger dataset, or the fixed-split mode with
`n_bootstrap` ([Split modes](../reference/evaluators/index.md#split-modes)).

A second run with the same `embedding_cache` path reads the features from the file. It still builds the
recipe, but it does not run the encoder on the images. So you can add a probe, for example
`KNNProbeEvaluator`, at a low cost. [Run a probe](run-a-probe.md) writes the same `montgomery_raddino` file,
so a run of both pages extracts the RAD-DINO features one time. To add a recipe, add it to `recipes` and run
step 1 again.

## Check

- `outputs/compare/` has one folder for each backbone, each with `results.csv` and `results_summary.csv`.
- `outputs/cache/` has one `.pkl` file for each backbone.
- `outputs/compare/table.csv` has one row for each backbone.

## See also

- [Run a probe](run-a-probe.md): one probe, and how to read the result table.
- [Backbones](../reference/backbones/index.md): all recipes, their feature sizes and install extras.
- [Embedding cache](../reference/evaluators/index.md#embedding-cache): the cache files and their checks.
- [`notebooks/evaluator/fm_comparison.ipynb`](https://github.com/f10409/RadHarmony/blob/main/notebooks/evaluator/fm_comparison.ipynb):
  a full comparison of 11 recipes and 6 probes on VinDr-CXR and TAIX-Ray.
