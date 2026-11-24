# Audit — Annotation coverage across all integrated datasets

**Status:** ⏳ **awaiting team discussion** — fact-finding done, no code changes proposed yet.
**Authors:** Judy + Claude
**Date:** 2026-04-26
**Triggered by:** the RSNA 2023 Abdominal Trauma integration shipped today (labels-only, with segmentations + per-slice point annotations explicitly deferred).  Rather than carry that decision through dataset-by-dataset, audit *every* integrated dataset for annotation types that exist in the source release but are not currently wired through to a harmonized output, and decide as a team which to land vs. leave deferred.

---

## What's wired vs. what exists in the source

Legend: ✓ wired and exposed via the standard output flag · ⏳ exists in source release but not yet wired · — not present in source release · ✗ not yet wired AND would need investigation to confirm presence

| Dataset | Labels (`cls`) | View pos | Masks (`mask`) | BBoxes (`bbox`) | Reports (`report`) | Reg target (`reg`) | Notes |
|---|---|---|---|---|---|---|---|
| **ChestXray14** | ✓ 14 findings | ✓ "View Position" | — | — (BBox in separate variant) | — | — | NIH release — bbox subset is a separate dataset class below |
| **ChestXray14 BBox** | ✓ 15 findings | ✓ | — | ✓ | — | — | The ~880-image bbox subset from NIH |
| **CheXpert** | ✓ 14 findings | ✓ AP/PA | — | — | — | — | Reports not in original release |
| **CheXpert-Plus** | ✓ 14 findings | ✓ ap_pa | — | — | ✓ inline | — | DICOM source, reports preserved as a column |
| **CT-RATE** | ✓ 18 findings | ✓ ImageOrientationPatient | — | — | ⏳ exists in source release but `REPORT_PATH_COL` is not set | — | The CT-RATE release includes free-text reports — needs wiring |
| **MIMIC-CXR (DICOM)** | ✓ 14 CheXpert | ✓ DICOM headers | — | — | ✓ via `REPORT_JOIN_COLS` + `REPORT_PATH_COL` | — | Fully wired |
| **MIMIC-CXR-JPG** | ✓ 14 CheXpert | ✓ ViewPosition | — | — | ⏳ not wired in JPG harmonizer (the paired DICOM release has them) | — | If trainees use only JPG, they lose reports — worth wiring |
| **RAD-ChestCT** | ✓ 84 findings | — | — | ✓ via `BBOX_SOURCE_COL = "bbox"` | — | — | |
| **RSNA 2024 Lumbar Spine** | ✓ 25 level-specific | ✓ series_description | ⏳ `segmentations.zip` exists with vertebra/spinal-canal masks (sub-set of series) | ✓ via custom `_build_bbox` reading `train_label_coordinates.csv` (commit `d1d0535`) | — | — | Mask wiring is the main gap |
| **RSNA Abdominal Trauma (2023)** | ✓ 14 patient-level | — | ⏳ `segmentations/` directory has organ masks for a subset of series | ⏳ `image_level_labels_2024.csv` has 15,632 Active_Extravasation point annotations — could be 3-D bbox dots like lumbar spine | — | — | **Just integrated today (labels only)**; explicit deferrals in dataset docstring |
| **RSNA Bone Age** | — | — | — | — | — | ✓ `age_months` (`REG_COLS`) | Sex carried as `EXTRA_OUTPUT_COLS` |
| **RSNA PE Detection** | ✓ 13 binary | — | — | — | — | — | No bboxes / segmentations in source release |
| **RSNA Pneumonia** | ✓ pneumonia + opacity | — | — | ✓ via custom `_build_bbox` reading MD.AI JSON | — | — | Lung-opacity local annotations preserved |
| **SIIM ACR PTX** | ✓ pneumothorax | — | ✓ RLE → mask via `MASK_SOURCE_COL = "EncodedPixels"` | — | — | — | |
| **SIIM COVID-19** | ✓ study + image | — | — | ✓ image-level bboxes via custom `_build_bbox` | — | — | |
| **TAIX-Ray** | ✓ findings | — | — | — | — | — | Adds `sex / age / study_date / fold / split / physician_id` as `EXTRA_OUTPUT_COLS`.  Confirm with the upstream release whether any masks/bboxes exist. |
| **VinDr-CXR (Train)** | ✓ 28 findings | — | — | ✓ via custom `_build_bbox` reading `annotations_train.csv` | — | — | |
| **VinDr-CXR (Test)** | ✓ 28 findings | — | — | ✓ via custom `_build_bbox` reading `annotations_test.csv` | — | — | |

---

## Open items, summarized

These are the rows in the table above with a ⏳, regrouped by what kind of work each needs:

### Segmentation masks (3-D)

1. **RSNA 2024 Lumbar Spine** — `segmentations.zip` contains per-series vertebra + spinal-canal segmentation masks (subset of series).  Adding requires:
   - Unzip on disk during dataset prep, OR read from the zip on the fly
   - A custom `_build_mask_path` that joins on `series_id` and emits per-row mask paths
   - `output_mask=True` support in the dataset class

2. **RSNA Abdominal Trauma (2023)** — `segmentations/` directory has per-series organ masks for a subset of patients.  Same shape of work as lumbar spine.

### Image-level / point annotations (could become BBoxes or "key slice" labels)

3. **RSNA Abdominal Trauma (2023)** — `image_level_labels_2024.csv` has 15,632 rows of `(patient, series, instance, "Active_Extravasation")`.  Same template as the lumbar-spine bbox-dots integration in `d1d0535` — point annotations rendered as tiny cube bboxes.  Needs a custom `_build_bbox` plus extending `bbox_labels` with the `injury_name`.

### Reports

4. **CT-RATE** — reports exist in the release but `REPORT_PATH_COL` / `REPORT_JOIN_COLS` not set in our harmonizer.  Needs a small wiring change.

5. **MIMIC-CXR-JPG** — paired DICOM release has reports; JPG harmonizer doesn't wire them.  Either point users at the DICOM version when they need reports, or duplicate the wiring (slightly redundant but useful for "JPG-only" workflows).

### Other

6. **TAIX-Ray** — confirm with the upstream HuggingFace release whether bboxes, masks, or other per-image annotations exist that we're not exposing.  Just shipped today by the parallel session; might be 100% wired already.

---

## Decision framework — when to wire vs. defer

Suggested heuristic for the team to apply per-row above:

| Signal | Implication |
|---|---|
| Annotation is widely used in the published baseline / leaderboard | → wire it |
| Annotation type already wired for a similar dataset (template exists) | → cheap to add — wire it |
| Annotation requires a non-trivial new file format reader (DICOM-RT, NIfTI segmentation, etc.) | → propose first, discuss before wiring |
| Annotation covers a small subset of samples (segmentations often do) | → wire it but document the partial coverage in the dataset's docstring + the integrity check |
| Annotation requires upstream contact (license, format question) | → defer until clarified |

For the trauma dataset specifically: organ segmentations are widely used in the abdominal trauma literature (the 2023 RSNA challenge specifically called them out as a core resource), so my recommendation would be to **prioritize wiring them next** — even ahead of the Active_Extravasation point annotations.

---

## Out of scope

- Augmentation behavior on the various annotation types — see `docs/proposals/bbox_augmentation_lost_in_pipeline.md` for the in-flight investigation of `bbox_mask` round-trip issues.  Solving the augmentation issue first might affect how worth-it segmentation wiring becomes.
- Changing the `output_mask` / `output_bbox` / `output_report` flag API itself — keep flags as is; just wire the missing harmonizer side.
- Re-auditing harmonizers that are already complete (most of the table — left in place for completeness).

---

## Open questions for the team

1. **Priority order.**  Of the 6 ⏳ items above, which 1–2 are worth doing first?  My (Claude's) lean: lumbar-spine masks → trauma masks → CT-RATE reports.  Lumbar-spine because the template (custom `_build_mask_path` reading a zip/dir) would also unblock trauma.
2. **Mask file storage.**  For both lumbar-spine and trauma masks the raw segmentations are in a zip/subdir alongside the imaging tree.  Should the harmonizer assume they're already extracted to the filesystem, or read them from the zip on demand?  PE Detection's "DICOM series → ITKReader directory" pattern assumes pre-extracted; mask wiring could follow the same.
3. **TAIX-Ray gap.**  Frank or whoever shipped TAIX-Ray today: is the dataset really labels-only upstream, or did we miss any bbox / segmentation annotations in the HuggingFace release?
4. **MIMIC-CXR-JPG reports.**  Worth wiring (duplicating the DICOM harmonizer's report logic) or leave as "use the DICOM harmonizer if you need reports"?  Slight redundancy vs. a friendlier JPG-only experience.
