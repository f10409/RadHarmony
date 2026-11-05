# Review — How RadHarmony exposes train / val / test splits

**Status:** For team discussion (no decision yet)
**Authors:** Judy (raised question), drafted with Claude
**Date:** 2026-04-26
**Pertains to:** every dataset class under `radharmony/dataset/`, the registry entries in `app.py`, and the on-disk layout under `/path/to/datasets/external/`.

## TL;DR

The repo handles train / val / test split exposure **inconsistently** across datasets. Today there are at least four different patterns in active use, and one large class of datasets (Kaggle competitions) intentionally exposes only `train` — which surprised at least one user. This doc inventories what's actually shipping, explains why, and lists the open questions for the team to resolve.

## How the question came up

While testing in Gradio, the user noticed that **RSNA 2024 Lumbar Spine** has both `train_images/` and `test_images/` on disk under `/path/to/datasets/external/rsna-2024-lumbar-spine-degenerative-classification/`, but the `RSNA2024LumbarSpineDataset` class only auto-discovers the `train_*` files. By contrast, **VinDr-CXR** ships as two separate dataset classes (`VinDrCXRTrainDataset` / `VinDrCXRTestDataset`) and **MIMIC-CXR** ships as one class with a `split` column inside the metadata. The natural question: *what's the convention, and how should a user ask for the test split?*

---

## Audit 1 — on-disk inventory under `/path/to/datasets/external/`

Datasets that ship a separate `train` and `test` (or `valid`) on disk:

| Dataset directory | Train labels? | Test labels? | Notes |
|---|---|---|---|
| `MIMIC-CXR-V2` | ✅ in metadata CSV | n/a — `test/` is just an empty / minor variant | split is a per-row column, not a dir |
| `NIH-CXR` | ✅ via `train_val_list.txt` | ✅ via `test_list.txt` | text-file split; CSV labels apply to both |
| `rsna-2022-cervical-spine-fracture-detection` | ✅ `train.csv` + bboxes | ❌ Kaggle holdout | test images present, no labels |
| `rsna-2024-lumbar-spine-degenerative-classification` | ✅ `train.csv` + `train_label_coordinates.csv` | ❌ Kaggle holdout | only `test_series_descriptions.csv` (IDs + series type) and `sample_submission.csv` |
| `rsna-boneage` | ✅ `train.csv` | (val ships in a separate dir) | val has labels (Bone Age Validation Set/) |
| `rsna-mammo` | ✅ `train.csv` | ❌ Kaggle holdout | |
| `rsna-miccai-brain-tumor-radiogenomic-classification` | ✅ `train_labels.csv` | ❌ Kaggle holdout | |
| `rsna_pe_dataset` | ✅ `train.csv` | ❌ Kaggle holdout | `test.csv` is just `StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID` |
| `rsna-pneumonia-detection-challenge` | ✅ `stage_2_train_labels.csv` | ❌ Kaggle holdout | (also has a separate format-mismatch issue — see `proposals/rsna_pneumonia_data_format.md`) |

**Six of the nine "two-folder" datasets are Kaggle competitions where the test labels are held back for leaderboard scoring.** The test images on disk are useful only for inference + submission; you cannot train, validate, or browse them with labels.

---

## Audit 2 — patterns in code

Five patterns are currently in use across the dataset classes. Listed in rough order of how much the user has to know:

### Pattern A — train-only ("test set is intentionally not wired")

**Datasets:** `RSNA2024LumbarSpineDataset`, `RSNAPEDetectionDataset`, `RSNAPneumoniaDataset` (and de facto every Kaggle competition above).

The class auto-discovers `train.csv` (etc.) and the corresponding image folder. Test images on disk are simply ignored.

**Why it's like this:** these are Kaggle competitions; the test labels are not in the Kaggle download. There is nothing useful for the labeled-sample browser to do with the test images.

**Cost:** new users see `train_images/` AND `test_images/` on disk and assume both are accessible — confusion. The convention is undocumented.

### Pattern B — one class, swap the CSV file ("user names the split via `csv_path`")

**Datasets:** `CheXpertDataset` (`train.csv` ↔ `valid.csv`), `CTRATEDataset` (`train_predicted_labels.csv` ↔ `validation_predicted_labels.csv`), `RadChestCTDataset` (`imgtrain/imgvalid/imgtest_..._Labels.csv`).

The class auto-discovers any of several known filenames. To switch splits, the user passes a different `csv_path=`.

**Why it's like this:** train and val/test have identical schemas; a single class works; the user just points at the file they want.

**Cost:** the user has to know the exact filename. In Gradio that means typing it into the CSV path textbox.

### Pattern C — one class, explicit `split=` constructor + Gradio dropdown

**Datasets:** `RSNABoneAgeDataset(split="train" | "val")`.

The class takes a string `split` arg; in Gradio it's exposed as a dropdown via `extra_dropdown_kwarg="split"`.

**Why it's like this:** train and val live in differently-named files / folders that aren't simple variants ("Bone Age Validation Set/" vs `train.csv`); the class hides the file/folder mapping behind a clean enum.

**Cost:** every consumer (notebook, app, CI) has to remember the `split` arg.

### Pattern D — one class, internal `split` column + `get_datasets_predefined()`

**Datasets:** `TAIXRayDataset` / `TAIXRay512Dataset` (new in this branch).

All splits live in one `annotation.csv` with a `split` column (`train`/`val`/`test`). The standard `get_datasets()` does a random patient-level split as usual; an additional `get_datasets_predefined()` honours the in-CSV split. The Gradio app currently ignores the predefined helper and uses the standard random-split path.

**Why it's like this:** TAIX-Ray ships its splits inline; reproducing the paper's evaluation requires honouring them.

**Cost:** Gradio doesn't surface `get_datasets_predefined()` — only programmatic users see it. (Could be wired into the dropdown like Pattern C if useful.)

### Pattern E — two completely separate dataset classes

**Datasets:** `VinDrCXRTrainDataset` / `VinDrCXRTestDataset`.

Two sibling classes share a prefix; each has its own harmonizer.

**Why it's like this:** VinDr-CXR train uses majority-vote across 3 radiologists and the column `"Other diseases"`; VinDr-CXR test uses 5-radiologist consensus and the column `"Other disease"` (singular). Same conceptual schema, different aggregation logic + a column-name fix-up — enough divergence that a single class would need branching everywhere.

**Cost:** the most code; two registry entries; users have to know both classes exist.

---

## Why Kaggle competitions don't expose test

It's worth saying explicitly because it's the source of the original confusion:

> **Kaggle's competition format intentionally ships an unlabeled test set.** The label CSV for the test rows is held on Kaggle's servers and only consulted when scoring a submission. The test directory on disk is therefore useful only for *inference + submission to Kaggle*, never for training, validation, or labeled browsing.

RadHarmony's harmonizers all assume labels are present (most produce a `cls` tensor by reading them). Wiring a Kaggle test set into a dataset class would require either:

1. Returning rows with `cls = NaN` everywhere (and the Gradio app would either crash or display "No finding" for every sample — neither is useful), or
2. Building a parallel "inference-only" dataset class with no `cls`, used only for generating submission.csv files.

Neither is currently in scope.

---

## Open questions for the team

1. **Should we document Pattern A (Kaggle = train only) prominently?** A one-paragraph note in `dataset_guide.md`'s per-dataset sections — and maybe a generic "About Kaggle datasets" callout — would prevent the same confusion in the future.

2. **Should we converge on one pattern when train/val/test are all labeled?** Today CheXpert, CT-RATE, RAD-ChestCT use the CSV-swap pattern; Bone Age uses a `split=` dropdown; TAIX-Ray uses an internal column + helper. All three work, but for a new contributor "which one do I pick?" is unclear.

3. **Should the Gradio app surface `get_datasets_predefined()` for TAIX-Ray (and similar future datasets)?** Currently the app always uses the random-split path (`get_datasets(n_splits=...)`), even when a predefined split is available. Adding a "Use predefined split" checkbox or a dropdown choice would make the paper's evaluation reproducible from the UI.

4. **Should we ever support inference-only flows?** Wiring Kaggle test sets for "load model → run → write submission.csv" is a meaningfully new feature category (no `cls`, different transforms, write outputs). Worth considering whether the project wants to go there or stay strictly a labeled-sample / training pipeline.

5. **Should VinDr-CXR's two-class split survive long term?** Now that we have Pattern D (internal `split` column), VinDr could plausibly merge into one class with a column-rename pre-step. Cost-vs-benefit unclear; leaving it alone is also fine.

---

## Appendix — verification commands

To reproduce the audit:

```bash
# Which external datasets ship train/test/val/valid as siblings on disk
for d in /path/to/datasets/external/*/; do
    n=$(basename "$d")
    subs=$(ls "$d" 2>/dev/null | grep -iE "^(train|test|val|valid)" | tr '\n' ' ')
    [ -n "$subs" ] && printf "%-50s → %s\n" "$n" "$subs"
done

# Which dataset classes mention 'split' / 'train' / 'test' in their constructors
grep -nE "split|train|test|valid" radharmony/dataset/*.py | head -50
```
