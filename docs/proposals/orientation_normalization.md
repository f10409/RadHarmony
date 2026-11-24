# Proposal — Canonical Orientation in 3-D Load Pipeline

**Status:** ✅ **implemented** — `_ReorientBbox` pre-transforms bbox coordinate pairs to match the post-`OrientationD`+`TransposeD` axis order before `OrientationD` runs, resolving the bbox alignment conflict.  `_SITKOrientD` and `_ReorientBbox` are both wired into `_base_load_3d` (gated on `base_transpose`).  `_SITKOrientD` replaces `mn.transforms.OrientationD` entirely: it converts each MetaTensor to a SimpleITK image, applies `DICOMOrientImageFilter`, and converts back — no nibabel dependency.  `RadiologyTransform3D` exposes an `axcodes` parameter (default `"IPL"`) that controls both transforms.  Source orientation code is read from the MetaTensor affine via SimpleITK's `GetOrientationFromDirectionCosines` — no nibabel dependency in `_ReorientBbox` either.
**Authors:** Judy + Claude
**Date:** 2026-04-26
**For:** Frank, who wrote the 3-D load pipeline (`radharmony/dataset/transforms.py:_base_load_3d`).
**Pertains to:** `radharmony/dataset/transforms.py` — the 3-D pipeline shared by every CT/MRI dataset.

---

## What surfaced this

Loading random samples from RSNA PE Detection in the Gradio app, Judy noticed that the rendered axial / coronal / sagittal mid-slices were sometimes upright and sometimes inverted along z.  Same dataset, same pipeline — the "upside-down-ness" varied across samples.

This isn't a quirk of the dataset.  The DICOM metadata is correct in every case; the pipeline isn't using it.

---

## What's actually in the data

Sampled 50 random studies from `rsna_pe_dataset/train/` and inspected the first DICOM of each series:

| Variant | Count | What it means |
|---|---|---|
| `FFS, z↑` (Feet-First Supine, slices stored superior→inferior) | ~24 | DICOM `ImagePositionPatient[2]` decreases with slice number |
| `FFS, z↓` (Feet-First Supine, slices stored inferior→superior) | ~24 | `ImagePositionPatient[2]` increases with slice number |
| `HFS, z↓` (Head-First Supine, slices stored inferior→superior) | ~2 | Different `PatientPosition` entirely |

Even within FFS — the dominant position — half the studies have `z↑` and half have `z↓`.  All have identical `ImageOrientationPatient = [1, 0, 0, 0, 1, 0]` (no in-plane rotation), so the in-plane axes agree across studies.  The variance is purely **slice-stack direction** plus the occasional HFS outlier.

DICOM captures every one of these correctly.  ITKReader populates the resulting MetaTensor's `affine` from the metadata.  The bug is downstream.

---

## Where the pipeline drops the affine

`radharmony/dataset/transforms.py:_base_load_3d` (pre-fix):

```python
t = [
    mn.transforms.LoadImageD(keys=img_keys, reader="ITKReader"),
    mn.transforms.EnsureChannelFirstD(keys=img_keys),
]
if pixdim is not None:
    t.append(mn.transforms.SpacingD(keys=img_keys, pixdim=pixdim, ...))
if base_transpose:
    t.append(mn.transforms.Transposed(keys=img_keys, indices=[0, 3, 2, 1]))
```

`SpacingD` respects the affine, but it's gated on `pixdim is not None`.  The unconditional `Transposed([0, 3, 2, 1])` is a **fixed axis permutation** that assumes one specific input layout — it has no awareness of which physical axis is z.  Volumes whose ITK-reported axis order disagrees with that assumption come out flipped along z.  That's exactly what Judy was seeing.

---

## Proposed fix (implemented on this branch)

Insert `mn.transforms.OrientationD(keys=img_keys, axcodes="RAS")` immediately after `EnsureChannelFirstD` and before any spatial step.  After it, every volume has axes in canonical Right-Anterior-Superior order regardless of the source scanner's convention.

```python
t = [
    mn.transforms.LoadImageD(keys=img_keys, reader="ITKReader"),
    mn.transforms.EnsureChannelFirstD(keys=img_keys),
]
if base_transpose:
    t.append(mn.transforms.OrientationD(keys=img_keys, axcodes="RAS"))   # NEW
if pixdim is not None:
    t.append(mn.transforms.SpacingD(...))
if base_transpose:
    t.append(mn.transforms.Transposed(keys=img_keys, indices=[0, 3, 2, 1]))
```

`OrientationD` is gated on `base_transpose` (same gate as the existing `Transposed`) because both rely on the affine, which is missing for the raw-numpy-array path (RAD-ChestCT).  That keeps the new behaviour scoped to the same datasets the existing `Transposed` already touches.

---

## Iteration history (all reverted)

Five iterations on this branch, every one of them reverted in the final state.  Recording the dead-ends so the next attempt doesn't repeat them.

**Iteration 1.** Kept the existing `Transposed`, added `OrientationD(RAS)` before it.  Made every PE Detection sample render consistently (the canonicalization worked) but **upside-down** on coronal/sagittal — the `Transposed` was now double-correcting volumes that `OrientationD` had already canonicalized.

**Iteration 2.** Dropped `Transposed`, kept `OrientationD(RAS)` only.  Panels now in the **wrong order** (sagittal | coronal | axial instead of axial | coronal | sagittal).  RAS gave axes in `(R, A, S)` but the display code (`_sample_to_rgb`) expects `(D, H, W)` = `(z, y, x)` order.

**Iteration 3.** Switched to `axcodes="IPL"` (still no `Transposed`).  Axes now in `(I, P, L)` order matching the display's expectation.  PE Detection rendered correctly across many random samples — anatomically right and consistent.  But **lumbar-spine bboxes landed on wrong vertebrae**.

**Iteration 4.** Re-added `Transposed` BEFORE `_BboxToMask`, kept `OrientationD(IPL)` AFTER.  Commit `d1d0535` (the original bbox-dots wiring) deliberately authored bbox coords in the post-`Transposed` `(D, H, W)` order.  Without the `Transposed`, `_BboxToMask` painted `bbox[0:2]` into ITK's W axis (horizontal) instead of the slice direction, so labels appeared at wrong vertebra levels.  Re-adding the `Transposed` got the bbox coords back into a usable axis system, but **panel order was again wrong** — `OrientationD(IPL)` on top of post-`Transposed` axes produced an axis assignment where the densely-sampled L↔R direction (for sagittal MRI series) ended up at axis 1 (the "coronal" panel slot), so the good sagittal view appeared in the middle panel rather than the right.

**Iteration 5 (this commit).** Reverted the whole orientation work.  Pipeline is back to the original `Transposed`-only state — same as `integration` branch.  PE Detection orientation is again **inconsistent across samples** (the original problem), but lumbar-spine bboxes work and panel order is back to what users expect.

## The deeper problem

The bbox harmonizer (`d1d0535`) authors bbox coordinates in **image-relative axes**, not patient-canonical axes.  For a sagittal MRI series, axis 1 is "the head-feet axis of the displayed image"; for an axial CT series, axis 1 might be A↔P.  The same bbox coord vector means different anatomical things depending on the source acquisition geometry.

`OrientationD(IPL)` (or any canonical reorientation) by definition **moves volumes into patient-canonical axes** — which guarantees the bbox coord vector now means something *different* than what the harmonizer encoded.  Wrapping the bbox-mask as a MetaTensor with `img.affine` and including it in `OrientationD`'s keys list should let `OrientationD` reorient it the same way as the image, but in practice the resulting bbox positions still ended up wrong (likely because the bbox coords were already in image-relative space when `_BboxToMask` painted them, and the affine's axis-direction information doesn't reverse that mapping).

A proper fix needs to either:

1. **Re-author the bbox harmonizer** to emit coords in patient-canonical axes (e.g., always `(I, P, L)` order).  Then `OrientationD` doesn't need to touch them, and `_BboxToMask` paints them into a canonical-orientation image directly.  This is the cleanest fix but requires going back to the source DICOM annotations (`train_label_coordinates.csv` for lumbar spine) and reapplying them with axis-direction awareness.
2. **Apply the same affine-derived axis transformation to the bbox coord vector** before `_BboxToMask` runs.  Compute the permutation+sign-flip that `OrientationD(IPL)` would apply to a `(D, H, W)`-encoded coord vector and apply it to `bbox` first.  Math-heavy but local to one helper.
3. **Accept per-dataset orientation inconsistency** as the lesser evil and skip canonicalization entirely.  The current state.

---

## Open questions for the next attempt

1. Is there a known canonical convention for bbox coords across all 3-D datasets?  If not, would it be worth defining one (e.g., always `(I, P, L)` post-`OrientationD`) and migrating existing harmonizers?
2. The PE Detection orientation inconsistency is a real problem — is it a blocker for downstream training, or is it tolerable for visualization-only use?  That affects priority.
3. For the per-volume direction variation (`FFS-z↑` vs `FFS-z↓` vs `HFS-z↓`), is there an alternative that doesn't require a full reorientation pass — e.g., a "z-axis-flip on load" hook driven by the `PatientPosition` DICOM tag?

---

## Risk and rollback

Branch is feature-isolated.  If anything goes wrong:

```bash
git checkout integration             # back to known-good
```

If `OrientationD` interacts badly with a specific dataset (e.g. a NumPy-array dataset where `base_transpose=False` already skips the gate, but some other 3-D dataset that I haven't tested), the right move is probably to keep `OrientationD` and selectively disable the existing `Transposed` for that dataset.

---

## Out of scope

- Per-dataset display conventions (some institutions display CTPA flipped on purpose for radiologist convention).  We're just normalizing to RAS; downstream display code can still flip for cosmetic reasons.
- 2-D pipelines.  Radiographs don't have z to invert, and `_base_load_2d` doesn't apply to this conversation.
- Whether RAS is the right canonical (vs LAS or LPS).  RAS is the standard MONAI choice and matches `nii.gz` conventions; sticking with it unless there's a reason otherwise.
