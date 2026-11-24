# Proposal — RSNA Pneumonia data-format mismatch

**Status:** Open
**Authors:** Judy (incident report), drafted with Claude
**Date:** 2026-04-26
**Pertains to:** `radharmony/harmonizer/rsna_pneumonia.py`, `radharmony/dataset/rsna_pneumonia.py`, `notebooks/download_kaggle.ipynb`, `app.py:_build_rsna_pneumonia`.

## Symptom

Loading **RSNA Pneumonia** in the Gradio app appears to hang for several minutes when pointed at the locally-downloaded data, then eventually errors with `Adjudicated JSON annotation file is required.`

## Root cause

The repo ships **two incompatible distributions** of the same competition:

| Distribution | What's in it | Where the repo gets it |
|---|---|---|
| **Kaggle competition** (CSV format) | `stage_2_train_labels.csv`, `stage_2_detailed_class_info.csv`, `stage_2_train_images/<SOPInstanceUID>.dcm` (flat tree) | `notebooks/download_kaggle.ipynb`, lands at `/path/to/datasets/external/rsna-pneumonia-detection-challenge/` |
| **Adjudicated JSON export** | `pneumonia-challenge-annotations-adjudicated-kaggle_2018.json`, `<StudyUID>/<SeriesUID>/<SOPUID>.dcm` (3-level nested tree) | not downloaded by any notebook in this repo |

`RSNAPneumoniaHarmonizer` reads the **adjudicated JSON** and expects the **nested DICOM tree**. The Kaggle download brings format A; the harmonizer needs format B; nothing tells the user.

The "hang" is a side effect: when the JSON isn't found in the obvious places, `infer_path()` falls through to its uncle-recursive-glob fallback and walks every sibling under `/path/to/datasets/external/` — 22 multi-TB dataset directories — searching for the missing JSON. That's the multi-minute pause. (See companion proposal `infer_path_recursion_bound.md` for the uncle-glob fix.)

## Options

### Option A — provide the adjudicated JSON (data-only fix)

Drop `pneumonia-challenge-annotations-adjudicated-kaggle_2018.json` and the nested DICOM tree under `/path/to/datasets/external/rsna-pneumonia-detection-challenge/`. The existing harmonizer "just works" with no code change.

**Pros:** zero code change.
**Cons:** the adjudicated JSON export isn't trivially downloadable; requires manual access to wherever the JSON lived originally. Re-organising ~30k flat DICOMs into the nested tree is a one-off scripting task.

### Option B — add a Kaggle-format harmonizer (code fix)

Either a new class `RSNAPneumoniaKaggleHarmonizer` + `RSNAPneumoniaKaggleDataset` (mirrors the MIMIC `_DICOM` vs `_JPG` split) **or** auto-detect format inside the existing `RSNAPneumoniaHarmonizer`:

- if `pneumonia-challenge-annotations-adjudicated-kaggle_2018.json` exists → adjudicated JSON path (today's behaviour),
- elif `stage_2_train_labels.csv` exists → Kaggle path (new),
- else → clear error.

The Kaggle path needs to:
- read `stage_2_train_labels.csv` (one row per bbox; multiple rows per `patientId` mean multiple opacities),
- read `stage_2_detailed_class_info.csv` to get the 3-class label (`Normal` / `No Lung Opacity / Not Normal` / `Lung Opacity`),
- group by `patientId` to merge multi-bbox rows + collapse to one record per image,
- build `image_path = "stage_2_train_images/" + patientId + ".dcm"`.

**Pros:** the Kaggle download notebook becomes self-sufficient — no separate manual data fetch.
**Cons:** the two format paths can drift; need tests to keep label semantics aligned.

### Option C — both

Ship the Kaggle path so the existing notebook works end-to-end (covers the most common case) **and** keep the adjudicated-JSON path for users who want the original adjudicated annotations (richer bboxes per opacity).

## Recommendation

**Option B (auto-detect inside the existing harmonizer)**, leaning toward **Option C** if the adjudicated-JSON path needs to stay supported. Either way:
- Update `notebooks/download_kaggle.ipynb`'s RSNA Pneumonia cell to clarify *which* format it brings.
- Update `docs/dataset_guide.md`'s RSNA Pneumonia section to document both formats and which file lives where.
- Update the observed-paths table to reference `/path/to/datasets/external/rsna-pneumonia-detection-challenge/`.

## Acceptance

- Loading "RSNA Pneumonia" in Gradio against the Kaggle-downloaded path returns a working dataset (no hang, no missing-file error).
- 3-class label distribution matches the paper / known counts: ~9,790 Normal, ~12,788 No Lung Opacity / Not Normal, ~11,390 Lung Opacity.
- Bounding boxes are populated when `output_bbox=True` and the row's class is `Lung Opacity`.
