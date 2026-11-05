# Your first benchmark in 10 minutes
<!-- blocks: run=gpu timeout=600 -->

In this lesson, you test how good the RAD-DINO features are at finding tuberculosis (TB) on chest X-rays.
You download a small dataset, load it, make the encoder, run a linear probe with 5 folds, and read the
result table.

You need a computer with an NVIDIA GPU and about 1.5 GB of free disk space. On our server, all steps took
about 3 minutes: the download took 81 s and the Python steps took 70 s.

## 1. Install

Follow [Install](install.md) to clone RadHarmony and make a venv. Then add the `raddino` extra.

<!-- block: run=skip reason="installs packages" -->
```bash
uv pip install -e ".[raddino]"
python -c "import rad_dino; print('ok')"  # ok
```

## 2. Download Montgomery County CXR

Montgomery County CXR has 138 chest X-rays, 58 with TB and 80 normal. It needs no account. These
commands download it (about 600 MB) to `MontgomerySet/` in the current folder.

<!-- block: run=skip reason="downloads data" -->
```bash
BASE="https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set"
wget -q -r -np -nH --cut-dirs=3 -e robots=off --reject "index.html*" "${BASE}/MontgomerySet/index.html"
ls MontgomerySet/CXR_png | wc -l  # 138
# set base_image_dir to MontgomerySet/CXR_png/
```

The images are in `MontgomerySet/CXR_png/`. In the next steps, replace `/data/MontgomerySet/CXR_png/` with
the full path of this folder. For other datasets, see [Get the data](../how-to/get-the-data.md).

## 3. Load the dataset

The dataset class reads the source files and makes the harmonized table: one row for each image.

```python
from radharmony.dataset import MontgomeryCXRDataset

ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    output_cls=True,
    cache_dir=None,
)
df = ds.get_harmonized_df()
print(len(df), df["patient_id"].nunique())  # 138 135
print(ds.LABEL_COLS)  # ['tuberculosis']
print(int(df["tuberculosis"].sum()), "images with TB")  # 58 images with TB

sample = ds.get_datasets()[0]
print(sample["img"].shape, sample["cls"])  # torch.Size([1, 224, 224]) tensor([0.])
```

`cache_dir=None` turns off the image cache, so the class writes no files. The table has 138 images of 135
patients. The label `tuberculosis` is 1 for TB and 0 for normal. A sample is a dict:
`img` is the image and `cls` is the label vector.

## 4. Make the encoder

`make_raddino()` returns a transform and an encoder. The transform prepares each image as RAD-DINO
needs it. The encoder turns a batch of images into one feature for each image. The first call
downloads the RAD-DINO weights (331 MB) from Hugging Face. Then make the dataset again with this
transform.

```python
from radharmony.evaluator.backbones import make_raddino

transform, encoder = make_raddino(device="cuda")
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    transform=transform,
    output_cls=True,
    cache_dir=None,
)
print(ds.get_datasets()[0]["img"].shape)  # torch.Size([3, 518, 518])
```

Now each image has the size and scale that RAD-DINO needs: 3 channels of 518 × 518.

## 5. Run the linear probe

The linear probe extracts the features of all images one time. Then it fits and tests a probe 5 times, one
time for each fold. Each time, it fits a logistic regression on the images of the other 4 folds, and tests it
on this fold. It keeps 10% of the train patients apart (the validation part) to pick the threshold of F1
(D-29). The folds group the images by patient, so all images of one patient are in the same fold.

```python
from radharmony.evaluator import LinearProbeEvaluator

ev = LinearProbeEvaluator(encoder, dataset=ds, n_folds=5, output_dir="outputs/first_benchmark")
results = ev.evaluate()
ev.save_results(results)
print(results[["fold", "label", "auroc", "auprc", "f1"]].round(3).to_string(index=False))
#  fold         label  auroc  auprc    f1
#     0  tuberculosis  1.000  1.000 1.000
#     1  tuberculosis  0.974  0.975 0.909
#     2  tuberculosis  0.995  0.994 0.957
#     3  tuberculosis  1.000  1.000 1.000
#     4  tuberculosis  0.886  0.856 0.533
#     0 macro_average  1.000  1.000 1.000
#     1 macro_average  0.974  0.975 0.909
#     2 macro_average  0.995  0.994 0.957
#     3 macro_average  1.000  1.000 1.000
#     4 macro_average  0.886  0.856 0.533
```

The result table has one row for each fold and label. `macro_average` is the mean over the labels. With
only 1 label, it is the same as `tuberculosis`.

## 6. Read the summary

`save_results()` wrote 2 files to `outputs/first_benchmark/`. `results.csv` is the table above.
`results_summary.csv` gives, for each metric, the mean over the folds. It also gives the standard deviation
(std) and the 95% confidence interval (CI).

```python
import pandas as pd

summary = pd.read_csv("outputs/first_benchmark/results_summary.csv")
cols = ["label", "auroc_mean", "auroc_std", "auroc_ci_lo", "auroc_ci_hi"]
print(summary[cols].round(3).to_string(index=False))
#         label  auroc_mean  auroc_std  auroc_ci_lo  auroc_ci_hi
# macro_average       0.971      0.043        0.895          1.0
#  tuberculosis       0.971      0.043        0.895          1.0
```

RAD-DINO features find TB with a mean AUROC of 0.971 over the 5 folds.

The AUROC is noisy, because the dataset has only 138 images. Each test fold has about 28 images, so one wrong
image changes the AUROC a lot. Here, the AUROC of one fold is 0.886, and the AUROC of 2 other folds is 1.000.
Use this dataset to learn the steps. Do not use it to compare encoders. For a real benchmark, use a larger
dataset with an official test split.

## Next steps

- [Concepts](concepts.md): the parts of RadHarmony and how they work together.
- [Run a probe](../how-to/run-a-probe.md): fixed splits, other probes and the embedding cache.
- [Compare backbones](../how-to/compare-backbones.md): run more encoders on the same data.
- [Linear probe](../reference/evaluators/linear_probe.md): all the arguments of the probe.
