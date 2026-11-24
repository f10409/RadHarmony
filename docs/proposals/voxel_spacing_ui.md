# Proposal — Voxel Spacing UI (`pixdim`) UX

**Status:** ✅ all items resolved — see "Resolution status" below
**Authors:** Judy (UX feedback), drafted with Claude
**Date:** 2026-04-25
**For:** Frank, who shipped the original `iso_spacing` + `pixdim` controls (commit `50cbe1d`, "mri spacing").
**Pertains to:** `app.py` 3-D dataset tabs (currently CT and MRI).

## Resolution status

| Item | State | Notes |
|---|---|---|
| **1.** Replace `gr.Number` with bounded `gr.Slider` (0.25–5.0 mm, step 0.05) | ✅ **done** | Implemented in commit `edc2d24` (2026-04-26).  Step refined from 0.1 → 0.05 so the range can land on Judy's empirical sweet spot value of 0.25 along z. |
| **2.** Re-run on `.release()` instead of waiting for click-Run | ✅ **done** | Implemented in commit `edc2d24` (2026-04-26).  Wires each pixdim slider's `.release()` AND the Resample checkbox's `.change()` to re-run via `run_same_fn` (re-renders the SAME volume rather than yanking a fresh random sample, so the spacing change is the only variable).  Tradeoff stands: ~1–2 s pause per release for lumbar spine MRI.  If this becomes annoying in practice, gate behind a "Live preview" checkbox. |
| **3.** Show native spacing readout + preset buttons ("1 mm iso", "Match native", "Half native") | ✅ **done** | Implemented in commit `edc2d24` (2026-04-26).  Native spacing read at load time via pydicom (DICOM series dirs) or MONAI ITKReader (single files); rendered as `_Native: 0.59 × 0.59 × 3.30 mm_` caption below the sliders.  "Match native" reads from state; "Half native" halves it.  Preset clicks set sliders only; user clicks Run (or drags then releases for live re-run) to render. |
| **4(a).** "Voxel spacing (mm)" section header + "Resample" checkbox + per-axis x/y/z labels | ✅ **done** | Implemented in commit `aaa0d42` (2026-04-25), pushed to `origin/integration`. |
| **4(b).** Single iso slider + "Per-axis spacing" sub-toggle | ❌ **rejected** | Frank picked 4(a) instead. |

---

## Context — what surfaced this

While testing the new MRI-spacing controls on the **RSNA 2024 Lumbar Spine** dataset, the workflow was opaque:

- The controls are `gr.Number` widgets with no min/max — easy to type "11" and end up with nonsense.
- No visible relationship between the value entered and the actual rendered output until **Run** is clicked.
- The label is **"Isotropic spacing (mm)"** but exposes **three independent x/y/z fields** — semantically mismatched ("isotropic" means same in all axes).
- After much trial and error, `(x=1, y=1, z=0.25)` mm produced a coronal/sagittal view that "looked proportional." There was no way to know that ahead of time.

**The empirical answer turned out to depend on data we never showed the user**: lumbar spine MRI is natively *highly anisotropic* (sub-millimeter in-plane × 3–5 mm slice thickness), so the resampling target you pick has to respect that geometry. A user who isn't familiar with MRI acquisition physics has no way to guess this from the current UI.

---

## Background — what the literature says

### Typical voxel geometries

| Modality | Native (typical clinical) | Common DL resampling target |
|---|---|---|
| **CT** | 0.39–0.98 mm in-plane × 1.25–3.75 mm slice | **1×1×1 mm isotropic** (overwhelming default in DL); 2×2×2 or 3×3×3 for memory-bound 3D models; 1×1×2 to preserve some axial detail |
| **MRI (general)** | Wide variance — sub-mm in-plane + 3–5 mm slice for 2D; ~1 mm isotropic for 3D acquisitions | 1×1×1 mm if 3D-isotropic acquisition; otherwise leave anisotropy or pick a target that respects the lower-resolution axis |
| **MRI (lumbar spine T1/T2)** | **3.30 × 0.59 × 0.59 mm** (sagittal T1/T2 — highly anisotropic, thick slices); 0.90 × 0.47 × 0.47 mm (T2 SPACE 3D-iso); axial slices commonly 4 mm thick | No fixed standard — depends on whether you want to "fix" the slice direction or preserve it |

### nnU-Net's approach (the reference standard)

nnU-Net does **not** prescribe a fixed target spacing. Instead it analyses each dataset's "fingerprint" (per-volume native spacings) and adapts the target to the data. Notably, nnU-Net **defaults to anisotropic targets, not isotropic** — confirming that for inherently anisotropic modalities (most clinical MRI), forcing isotropy can hurt segmentation performance and visual fidelity. This is the formal version of what Judy discovered empirically with her `(1, 1, 0.25)` setting.

### Implication for our app

The current `(1, 1, 1)` default is the right choice for **CT** but actively unhelpful for **anisotropic MRI** sequences. There is no single sane default that works across both modalities.

---

## Proposed plan

Four changes, ordered easiest → most opinionated. We can land any subset.

### 1. Replace `gr.Number` with bounded `gr.Slider`  *(small, defensible)*

```python
pixdim_x = gr.Slider(0.25, 5.0, value=1.0, step=0.1, label="pixdim x (mm)")
pixdim_y = gr.Slider(0.25, 5.0, value=1.0, step=0.1, label="pixdim y (mm)")
pixdim_z = gr.Slider(0.25, 5.0, value=1.0, step=0.1, label="pixdim z (mm)")
```

**Why these bounds:** 0.25 mm covers high-res 3D MRI / fine CT (and Judy's empirical sweet-spot value); 5 mm covers thick-slice clinical MR. Anything outside this range is not a real medical-imaging spacing and risks running the system out of memory.

**Result:** drag handle is visible, "go to 11" is impossible, and the user can see where their value sits within the realistic range.

### 2. Re-run on `release()`, not on `change()`  *(small, defensible)*

Wire `slider.release(...)` to the same handler `Run` uses. Releasing the handle re-renders; dragging doesn't trigger every tick (a 3-D pipeline is too heavy for that).

```python
for s in (pixdim_x, pixdim_y, pixdim_z):
    s.release(fn=run_fn, inputs=[...same as run_btn...], outputs=[...])
```

**Tradeoff:** each release fires a transform pipeline run. For lumbar spine MRI, that's a ~1–2 s pause per release — interactive, not instantaneous. Acceptable for exploration, not so good for fine-tuning. Could be gated behind a "Live preview" checkbox if some users find it disruptive.

### 3. Show native spacing + add presets  *(small UX, big payoff)*

After the dataset loads, read the first sample's `MetaTensor.affine` (which `ITKReader` populates from DICOM `PixelSpacing` + `SliceThickness`) and surface it as a small caption above the sliders:

```
Native: 0.59 × 0.59 × 3.30 mm  (axes: x, y, z post-transpose)
```

Plus three preset buttons above the sliders:

| Button | Sets to | Rationale |
|---|---|---|
| **1 mm isotropic** | (1.0, 1.0, 1.0) | Most-common DL default; reasonable for CT |
| **Match native** | reads from the loaded volume | Disables actual resampling (or sets target = native) — useful for "show me what's there" inspection |
| **Half native** | (native_x/2, native_y/2, native_z/2) | Quick high-res experiment without typing |

This turns "fiddle until it looks right" into "look at native, pick a preset, fine-tune." Eliminates the discoverability problem for new users without removing the power-user knobs.

### 4. Disambiguate the iso/anisotropic story  *(more opinionated — needs Frank's nod)*

Current state: **"Isotropic spacing (mm)" checkbox + 3 independent x/y/z fields** — the words and the controls disagree.

Two clean alternatives:

- **(a) Keep three sliders, drop "Isotropic" from the label.** Just call the section "Voxel spacing (mm)" with a checkbox "Resample" to toggle it on/off. Honest about what the controls do.
- **(b) Make it actually iso when checked.** Default to one slider (locked across axes); a sub-toggle "Per-axis spacing" exposes the three sliders. Cleaner for the common case.

(b) is more work but a meaningfully better default UX. Worth discussing whether it's worth the change before doing it.

---

## Open questions for Frank

1. **Axis ordering after transpose.** Judy's `(1, 1, 0.25)` worked because *some* axis got heavy upsampling. Which one ended up being "z" after MONAI's transpose in `_base_load_3d`? If we add a "Native: ..." readout, the labels need to match what the user types.
2. **Per-modality defaults.** Worth having different defaults per modality tab? CT could default to `(1, 1, 1)` iso; MRI to `(0.5, 0.5, 3)` aniso (rough lumbar-spine-shaped). Or stay with one global default + good presets.
3. **Real-time preview cost.** Is the ~1–2 s per release acceptable for the lumbar spine pipeline, or do we need a faster preview path (e.g., re-apply transform to the cached last_sample without re-loading)?
4. **Item 4 above** — keep as-is, simplify the labels (a), or restructure to a single-slider iso default with an anisotropic escape hatch (b)?

---

## Out of scope for this proposal

- Changing the underlying `RadiologyTransform3D.with_spacing` API. Frank's transforms work; this is purely a UI/UX layer.
- Changing the harmonizer or dataset class to expose native spacing in some richer way. Reading `MetaTensor.affine` at sample-load time is enough.
- Multi-stage cached preview pipelines. Treat as a future optimization if the on-release re-run is too slow.

---

## References

- [Effect of resampling of voxel size on CT Radiomic features extraction (J Nucl Med, 2022)](https://jnm.snmjournals.org/content/63/supplement_2/3241)
- [Intrinsic dependencies of CT radiomic features on voxel size and number of gray levels (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5462462/)
- [Introduction to 3D medical imaging for machine learning (AI Summer)](https://theaisummer.com/medical-image-processing/)
- [The Role of ITK in the Era of Deep Learning in Medical Imaging (Kitware)](https://www.kitware.com/the-role-of-itk-in-the-era-of-deep-learning-in-medical-imaging/)
- [Considerations on Image Preprocessing Techniques Required by Deep Learning Models — Knee MRI (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11565144/)
- [A Structured Review and Quantitative Profiling of Public Brain MRI Datasets (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12733697/)
- [nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation (arXiv)](https://arxiv.org/html/2404.09556v1)
- [nnUNet repository (MIC-DKFZ)](https://github.com/MIC-DKFZ/nnUNet/tree/master)
- [Lumbar spine segmentation in MR images: a dataset and a public benchmark (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10908819/)
- [Voxel size — Radiopaedia reference article](https://radiopaedia.org/articles/voxel-size-1)
- [MRI Exam-Specific Parameters: Spine Module (ACR Accreditation)](https://accreditationsupport.acr.org/support/solutions/articles/11000061020-mri-exam-specific-parameters-spine-module-revised-5-10-2022-)
