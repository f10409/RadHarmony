# Architecture

## Layer diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Raw dataset                          │
│   CSVs / DICOMs / JPEGs / NIfTI / RLE masks / JSON bbox    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                      Harmonizer layer                       │
│  XxxHarmonizer  →  standardised pd.DataFrame                │
│  columns: patient_id, study_id, image_path,                 │
│           <label_cols>, bbox, bbox_labels,                  │
│           mask_path, report, view_position                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                      Dataset layer                          │
│  BaseRadiologicalDataset → patient-level split →            │
│  MONAI PersistentDataset (cached preprocessed tensors)      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                     Transform layer                         │
│  RadiologyTransform2D / RadiologyTransform3D                │
│  MONAI Compose pipeline: load → normalise → augment →       │
│  to tensor                                                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data dict (sample)                       │
│  {"img": Tensor, "cls": Tensor, "bbox": [...], ...}         │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                       App / trainer                         │
│  Gradio app  or  your PyTorch DataLoader                    │
└─────────────────────────────────────────────────────────────┘
```

---

## What each layer does

**Harmonizer** reads raw dataset files (CSVs, annotation JSONs, RLE files) and produces a standardised `pd.DataFrame`. Every harmonizer inherits from `BaseHarmonizer` and implements `_build_patient_id()`, `_build_study_id()`, `_build_image_path()`, and optionally `_build_labels()`, `_build_bbox()`, `_build_mask_path()`, `_build_report()`. The output DataFrame is the contract between the harmonizer and the dataset layer.

**Dataset** wraps the harmonized DataFrame in a PyTorch-compatible dataset. It handles patient-level train/val splitting (`get_datasets`), k-fold cross-validation (`get_folds`), image path resolution, and optional image existence verification (`verify_images`). It delegates preprocessing and augmentation entirely to the transform layer. Under the hood it uses MONAI's `PersistentDataset` to cache preprocessed (but not augmented) tensors on first access.

**Transform** defines the full preprocessing and augmentation pipeline as a MONAI `Compose`. The `RadiologyTransform2D` and `RadiologyTransform3D` classes provide sensible defaults (load, channel-first, intensity normalisation, resize, to-tensor) and a fluent builder API for adding augmentations. The transform is stateless; augmentation randomness comes from MONAI's per-worker RNG.

**App** registers each dataset in `DATASET_REGISTRY` as a `DatasetConfig` that maps a UI name to a build function, modality, and input field labels. The Gradio UI groups datasets by modality tab and calls the build function with user-supplied paths and flags.

---

## Registry system

Datasets are registered in `app.py`:

```python
DATASET_REGISTRY: dict[str, DatasetConfig] = {
    "CheXpert": DatasetConfig(
        is_3d=False,
        extra_field_label="",
        build=_build_chexpert,
        modality="CXR",
        base_dir_placeholder="e.g. /data/CheXpert-v1.0/train/",
        csv_placeholder="auto: train.csv",
    ),
    # ...
}
```

Adding a new dataset requires: a harmonizer, a dataset class, and a `DatasetConfig` entry in `DATASET_REGISTRY`.

---

## `infer_path()` auto-discovery

When `csv_path` is omitted, `infer_path()` searches for the CSV automatically. Resolution order:

1. `user_path` — if explicitly provided and exists, use it
2. `base_image_dir/candidate` — look directly inside the image dir
3. `parent/candidate` then glob siblings of `base_image_dir` — searches adjacent directories, never descends into the image tree itself
4. Glob children of grandparent — falls back to uncle directories

This means that if your CSV lives in the same parent directory as your image folder (a common layout), it is found automatically.

---

## Pipeline invariants

### 3D orientation + axis transpose

The 3D pipeline performs two coupled steps before intensity normalisation:

1. **Orientation normalisation** — `_SITKOrientD(axcodes=...)` reorients every volume to a canonical axis code (default `"IPL"`) using SimpleITK's `DICOMOrientImageFilter`, so that studies load with consistent axis semantics regardless of scanner orientation. When `output_bbox=True`, `_ReorientBbox` runs immediately before it and pre-permutes bbox coord pairs so they survive the reorient.
2. **Axis transpose** — `TransposeD([0, 3, 2, 1])` then reorders the axes from ITK's `(W, H, D)` load order to the row-first `(D, H, W)` convention that RadHarmony uses for bounding box coordinates (`[dim0_min, dim0_max, dim1_min, dim1_max]`).

**Both steps are load-bearing and must stay paired.** They are gated by `base_transpose=True`. If you remove either step (or change `axcodes` between the image and bbox reorient) while using `output_bbox=True`, bbox coordinates will be misaligned with the image.

### Bounding box normalisation

Pixel-space bounding boxes are normalised to fractional `[0, 1]` coordinates by dividing by the image dimensions. The order is always `[dim0_min, dim0_max, dim1_min, dim1_max]` (row-first), not `[x, y, w, h]`.

### Patient-level splitting

`get_datasets` and `get_folds` split on `patient_id`, not on row index. A patient's studies appear in only one of train or val. This prevents data leakage when a patient has multiple scans.
