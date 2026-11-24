# RadHarmony

RadHarmony gives many public radiology datasets (chest X-ray, CT and MRI) and many foundation models one
interface.

A harmonizer reads the source files of a dataset and makes one harmonized table. A dataset class
gives samples with the same keys for every dataset. An evaluator (probe, zero-shot, fine-tune, segmentation or
report generation) takes any dataset and the encoder of a backbone recipe. It gives the same result table for
all of them. Report generation takes a report generator in place of an encoder. So you can test one encoder on
many datasets, or many encoders on one dataset, with no code for each pair.

## Example

This block runs a linear probe with RAD-DINO features on Montgomery County CXR (138 chest X-rays, a TB label),
with 5 folds. [Your first benchmark in 10 minutes](get-started/first-benchmark.md) runs it step by step.

<!-- block: run=gpu timeout=600 -->
```python
from radharmony.dataset import MontgomeryCXRDataset
from radharmony.evaluator import LinearProbeEvaluator
from radharmony.evaluator.backbones import make_raddino

transform, encoder = make_raddino(device="cuda")
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/", transform=transform, output_cls=True, cache_dir=None
)
ev = LinearProbeEvaluator(encoder, dataset=ds, n_folds=5, output_dir="outputs/first_benchmark")
results = ev.evaluate()
ev.save_results(results)  # writes results.csv and results_summary.csv
print(results.groupby("label")["auroc"].mean().round(3).to_dict())
# {'macro_average': 0.971, 'tuberculosis': 0.971}
```

## The 4 parts of this wiki

- Get started: lessons for a new user.
    - [Install](get-started/install.md)
    - [Your first benchmark in 10 minutes](get-started/first-benchmark.md)
    - [Concepts](get-started/concepts.md)
- How-to guides: the steps of one task.
    - [Get the data](how-to/get-the-data.md)
    - [Load a dataset and pick outputs](how-to/load-a-dataset.md)
    - [Save and reuse a harmonized table](how-to/save-and-reuse.md)
    - [Split data](how-to/split-data.md)
    - [Cache the samples](how-to/cache.md)
    - [Run a probe](how-to/run-a-probe.md)
    - [Run segmentation](how-to/run-segmentation.md)
    - [Run report generation](how-to/run-report-generation.md)
    - [Compare backbones](how-to/compare-backbones.md)
    - [Add a dataset](how-to/add-a-dataset.md)
    - [Add a backbone](how-to/add-a-backbone.md)
    - [Use the app](how-to/use-the-app.md)
- Reference: facts to look up, such as arguments, keys and labels.
    - [Dataset catalog](reference/datasets/index.md)
    - [Evaluators](reference/evaluators/index.md)
    - [Backbones](reference/backbones/index.md)
    - [Dataset API](reference/api-datasets.md)
    - [Transforms](reference/transforms.md)
    - [Datathon26 module](reference/datathon26.md)
- Explanation: why the project does a thing.
    - [Design decisions](design.md)
    - [Label policy](explanation/label-policy.md)
    - [MIMIC-CXR label extraction](explanation/phase-grounding.md)
    - [DICOM loading](explanation/dicom-loading.md)
