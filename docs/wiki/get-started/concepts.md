# Concepts

RadHarmony has 3 main parts: the harmonizer, the dataset class and the evaluator. Each part passes one
thing to the next part.

<!-- block: run=skip reason="diagram, not code" -->
```text
source files (CSV, JSON, DICOM, PNG, NIfTI, RLE masks, ...)
   │  harmonizer: harmonize()
   ▼
harmonized table (pandas DataFrame, one row for each image)
   │  dataset class + transform
   ▼
samples {"img": Tensor, "cls": Tensor, "mask": ..., "bbox": ..., "report": ...}
   │  evaluator + backbone recipe (transform, encoder)
   ▼
result table (pandas DataFrame) → results.csv, results_summary.csv
```

## Harmonizer

A harmonizer reads the source files of one dataset and returns the harmonized table. Each dataset has its
own harmonizer class, for example `MontgomeryCXRHarmonizer`. All of them inherit from `BaseHarmonizer`.
`harmonize()` runs the same steps for every dataset (`_build_patient_id()`, `_build_image_path()`,
`_build_labels()`, `_build_bbox()`, ...). A harmonizer changes these steps, not `harmonize()` (D-30).

`h.save(path)` saves the table. `MontgomeryCXRHarmonizer.load_from_saved(path)` gives back a harmonizer with
the same `harmonized_df` (D-08). So you
harmonize a large dataset only one time ([Save and reuse a harmonized table](../how-to/save-and-reuse.md)).

## Harmonized table

The harmonized table is the only input of the rest of the code (D-01). After
`harmonize()`, no class reads the source files again.

- Required columns: `patient_id`, `study_id` and `image_path`.
- Optional columns: the label columns, `bbox`, `bbox_labels`, `mask_path`, `report`, `view_position`,
  `age_years`, `sex` and more (D-01).
- One row is one image (D-03). Some datasets use one row for each study, series or volume. Their page says
  so.
- `image_path` is relative to `base_image_dir`, so the same table works on any computer (D-04).
- Label columns have snake case names, in alphabetical order. A label is 1, 0, -1 (uncertain) or NaN
  ([Label policy](../explanation/label-policy.md)).

## Dataset class

A dataset class wraps the harmonized table in a MONAI `PersistentDataset` (a plain MONAI `Dataset` when
`cache_dir=None`). Give it `base_image_dir`, and it
looks for each source file only at 3 places: the path that you give, `base_image_dir/<name>` and
`<parent of base_image_dir>/<name>`. It does not search other folders (D-35). Each official split has its own
class, for example `VinDrCXRTrainDataset` and `VinDrCXRTestDataset` (D-09). Each class also has a registry
key, for example `vindr_cxr_train`. `resolve_dataset("vindr_cxr_train")` in `radharmony.registry` gives the
class ([Dataset catalog](../reference/datasets/index.md)).

A sample, `ds.get_datasets()[i]`, is a dict. It always has `img`. Each `output_*` flag adds its own key:
`output_cls` adds `cls`, `output_mask` adds `mask`, `output_bbox` adds `bbox` and `bbox_labels`,
`output_report` adds `report` and `output_reg` adds `reg`. `output_bbox` also adds `bbox_findings` when the
class has it. A flag that the dataset does not support gives a warning and no key (D-11).

The transform loads and prepares each image. The default is `RadiologyTransform2D` or `RadiologyTransform3D`
([Transforms](../reference/transforms.md)). A 2-D image is scaled to [-1, 1] (D-14). `cache_dir` stores the
prepared images before the first random step (D-15). So a random step, for example a random flip, runs again
each time you read a sample. See [Dataset API](../reference/api-datasets.md) for all keys and arguments.

## Evaluator and backbone recipe

A backbone recipe, for example `make_raddino()`, returns a transform and an encoder. The transform prepares
the image as the model authors did (D-16). The encoder turns a batch of images into features
([Backbones](../reference/backbones/index.md)).

An evaluator takes an encoder and a dataset made with the recipe's transform. It runs the task and returns
the result table: one row for each label and run. `save_results()` writes `results.csv` and
`results_summary.csv` ([Evaluators](../reference/evaluators/index.md)). The frozen-feature probes
L2-normalize the features (D-17).

## Box order

A box is `[dim0_min, dim0_max, dim1_min, dim1_max]`, row first. For a 2-D image, this is
`[y_min, y_max, x_min, x_max]`, not `[x, y, w, h]`. A 3-D box adds `dim2_min, dim2_max`. The values are
fractions in [0, 1] of the image size (D-07). Example: on a 1000 × 2000 (H × W) image, a box from row 100
to row 300 and from column 500 to column 1000 is `[0.1, 0.3, 0.25, 0.5]`. Why rows first: the first axis
of a numpy array is the row, so `img[y0:y1, x0:x1]` uses the box in its own order.

- In a sample, the box matches `sample["img"]`. Each resize, pad, crop, flip and warp of the default
  transforms moves the box with the image (D-28).
- `bbox` is a float32 tensor of shape `(N, 4)` (2-D) or `(N, 6)` (3-D). Use
  `radharmony.dataset.bbox_collate` to batch samples with different box counts.
- A harmonizer clips each box to the image, and drops a box with no area (D-07).
- The model processor of a backbone recipe can crop or pad the image. The box does not move with these
  steps. The recipe transform warns when `bbox` is in `output_keys` and the processor does more than
  resize.
- A 3-D volume is reoriented to `axcodes` (default `IPL`) and then transposed to `(D, H, W)`. The box goes
  through the same 2 steps (D-12).

## Splits

There are 2 kinds of split:

- Official splits: when the dataset authors publish a split, each split has its own class and registry
  key, for example `mimic_cxr_jpg_train` and `mimic_cxr_jpg_test` (D-09). Images with no labels go in a
  separate test class (D-25). An official split stays as published, also when 2 splits share patients
  (D-34).
- Your own splits: `get_datasets(n_splits=5)` gives a train and a val split, and `get_folds()` gives
  k folds. Both split by `patient_id`, so all images of a patient are in one split (D-10). They split only
  the rows of that dataset class, so they never mix 2 official splits ([Split data](../how-to/split-data.md)).

An evaluator takes one dataset for k folds (`dataset=`), or a train and a test dataset for a fixed split
(`train_dataset=`, `test_dataset=`). Its folds and its bootstrap also group by patient (D-32).

## Next steps

- [Load a dataset and pick outputs](../how-to/load-a-dataset.md): load a dataset and choose its outputs.
- [Run a probe](../how-to/run-a-probe.md): test an encoder with a probe.
- [Dataset catalog](../reference/datasets/index.md): all datasets and their registry keys.
