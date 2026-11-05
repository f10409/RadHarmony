# Results: confidence intervals and plots

The home of the CI facts is the page `reference/evaluators/index` (section "Result table") and D-19 in the
page `design`. In a clone, read `docs/wiki/<page>.md`. Else read
`https://f10409.github.io/RadHarmony/<page>.html`. This file gives the rule for the answer and 2 plot snippets.

## Report a result with its CI

Give each metric as `mean [ci_lo, ci_hi]`, not as a bare mean. Take the 3 values from `results_summary.csv`
(`<metric>_mean`, `<metric>_ci_lo`, `<metric>_ci_hi`). Example: `AUROC 0.97 [0.90, 1.00]`.

The CI is the 2.5 and 97.5 percentiles of the bootstrap rows. With no bootstrap rows, it comes from the folds
or the seeds (D-19). With 5 folds, it is close to the lowest and the highest fold. Say which one gave the CI.
For a CI from many test samples, use an official test split and `n_bootstrap`. K-fold mode ignores
`n_bootstrap`.

## Plot the results

Save each figure to a file. Do not call `plt.show()`: an agent has no screen.

The mean and the CI of one metric, for each label:

```python
import matplotlib.pyplot as plt
import pandas as pd

summary = pd.read_csv("outputs/first_benchmark/results_summary.csv")
m = "auroc"
err = [summary[f"{m}_mean"] - summary[f"{m}_ci_lo"], summary[f"{m}_ci_hi"] - summary[f"{m}_mean"]]
fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(summary)), 4))
ax.bar(summary["label"], summary[f"{m}_mean"], yerr=err, capsize=4)
ax.set_ylabel(m)
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=45)
fig.savefig("outputs/first_benchmark/results.png", dpi=150, bbox_inches="tight")
```

With more than one `n_train`, plot one figure for each `n_train`, or filter the rows first.

Segmentation masks on the image. `preds` comes from `predict(ds, indices=..., return_images=True)` of a
segmentation probe (page `reference/evaluators/linear_probe_seg`, section "Save and reuse"):

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, len(preds), figsize=(4 * len(preds), 4), squeeze=False)
for ax, p in zip(axes[0], preds):
    ax.imshow(p["image"], cmap="gray")
    ax.imshow(np.ma.masked_where(p["y_pred"] == 0, p["y_pred"]), cmap="autumn", alpha=0.5)
    ax.contour(p["y_true"] > 0, levels=[0.5], colors="cyan", linewidths=1)
    ax.set_title(p["image_id"])
    ax.axis("off")
fig.savefig("outputs/linear_probe_seg/samples.png", dpi=150, bbox_inches="tight")
```

The predicted mask is orange. The true mask is the cyan line. Use the rows of `inner_val_indices()` or of a
test split. The rows that trained the head look better than new images.
