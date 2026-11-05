# Review — Mixing 2-D and 3-D datasets within one Gradio modality tab

**Status:** For team discussion (no decision yet)
**Authors:** Judy (raised question), drafted with Claude
**Date:** 2026-04-26
**Pertains to:** `app.py:build_dataset_tab` (around line 2129), `app.py:DATASET_REGISTRY`, and the `DatasetConfig.is_3d` flag.

## TL;DR

Adding the new `RSNA2022CervicalSpineBboxDataset` to the Gradio app forced the question: where does a **2-D-per-slice** view of a CT dataset belong in the modality tabs? The user expects it under "CT" (the parent dataset is CT). The framework's current invariant says "no — every dataset in a modality tab must share the same `is_3d` flag." We currently park it under `"Other"`, which works but reads weird. This doc lays out three resolutions for the team to pick.

## How the question came up

While integrating the RSNA 2022 Cervical Spine dataset, we made the call to ship a **second** dataset class for slice-level bounding-box work — the bboxes in `train_bounding_boxes.csv` are per-slice 2-D, and the natural way to expose them is one row per `(study, slice)` rather than per-volume.

The 3-D class went under `modality="CT"` next to RSNA PE Detection and CT-RATE. The 2-D bbox class triggered:

```
ValueError: modality 'CT' mixes is_3d values ({False, True}) across
['CT-RATE', 'RAD-ChestCT', 'RSNA 2022 Cervical Spine',
 'RSNA 2022 Cervical Spine (BBox)', 'RSNA PE Detection'].
Split into separate modalities or add per-dataset adapter logic
before enabling this mix.
```

The guard was added in commit `80f65d0` ("safeguards against pipeline-axis regressions"). Its docstring is explicit:

> All datasets within a modality must currently share the same `is_3d` dimensionality. Mixed-dim modalities aren't supported yet: they'd need per-dataset control adaptation (3-D controls would appear/disappear as the dropdown changes) which is a bigger UI change than we want to take on here.

## Why the guard exists

Each modality tab in `build_dataset_tab` builds a fixed set of UI controls based on the **first** dataset's `is_3d` value. For 3-D tabs that means HU-window sliders, slice-axis selector, isotropic-spacing controls, voxel-spacing readout. For 2-D tabs those controls aren't built. If a user switches the dropdown from a 3-D entry to a 2-D entry within the same tab, the 3-D controls would still be present but inert (or worse, send junk into the 2-D builder). The guard prevents that misconfiguration at app startup.

So the constraint is real and intentional, but it's UI-shape, not data-shape. Nothing about the underlying dataset class objects requires the separation.

## Three resolutions

### Option 1 — Status quo: keep it under `"Other"`

Where we are today. The 2-D class lives under a single-dataset "Other" tab.

**Pros:** zero code change. No risk of broken UX.
**Cons:** discoverability — users looking for "CT-related datasets" won't find it under "CT". Long-term we'll accumulate other 2-D-from-3-D variants and "Other" becomes a junk drawer.

### Option 2 — Drop the guard and accept stale controls

Delete the `is_3d_flags` check in `build_dataset_tab` and let `"CT"` contain both. The 2-D class would be selectable from the CT dropdown, but users would see HU window / slice axis / spacing controls that do nothing for it (and might silently send wrong values into the 2-D builder if anything reads them).

**Pros:** trivial change (~3 lines deleted), satisfies the discoverability ask.
**Cons:** broken UX. The unused controls aren't just visually noisy — the existing build flow reads them unconditionally. Likely needs at least a defensive ignore in the 2-D build function. Can also produce confusing behaviour if a user changes a control expecting it to do something.

### Option 3 — Conditional UI: hide/show controls based on the selected dataset's `is_3d`

The right long-term fix per the docstring: wire the dropdown's `.change()` event to toggle visibility of the 3-D-only controls. When the user picks the 2-D entry, the HU window / slice axis / spacing widgets become `gr.update(visible=False)`; switching back to a 3-D entry re-shows them.

**Scope estimate (rough):**
- Add `gr.update(visible=cfg.is_3d)` to the dropdown change handler for each 3-D-specific widget.
- Audit `_run_dataset` (or wherever the controls are read) to gracefully handle the "control exists but isn't visible" case — Gradio still passes its current value, so the build function may need to treat them as defaults when `is_3d=False`.
- Update `build_dataset_tab` to build the union of 2-D and 3-D controls when the modality contains both, rather than picking one or the other based on the first dataset.

**Pros:** correct long-term architecture. Removes the guard cleanly. Future "2-D slice from 3-D volume" datasets (PE detection axials, CT-RATE slice subsets) can reuse the pattern.
**Cons:** real piece of work — likely a dozen widget toggles plus the `_run_dataset` audit. Higher chance of regressions in adjacent flows. Probably warrants its own branch/PR.

## Recommendation

**Defer to the team.** Option 1 (current) is fine if no one minds the "Other" location. Option 3 is the principled fix and would unblock similar future cases. Option 2 should be ruled out unless we add explicit defensive handling in every 3-D-control-reader — otherwise it just trades an explicit error for a silent bug.

## Open questions

1. **Is `"Other"` an acceptable parking spot for 2-D-from-3-D variants, or is the discoverability cost too high?**
2. **If we go with Option 3, who scopes the conditional-UI work?** It's adjacent to the orientation-normalization work in `80f65d0` (same author area).
3. **Are there other "2-D slice-from-3-D volume" dataset variants we expect to add?** If yes (e.g. axial-slice CT-RATE, axial-slice PE detection), Option 3 amortizes; if not, Option 1 is fine.
4. **Should the `DatasetConfig` taxonomy itself evolve?** Today `is_3d` is a single boolean. We could replace it with `dimensionality: Literal["2D", "3D", "2D_from_3D"]` and let `build_dataset_tab` do something useful with the third case.

## References

- Guard introduction: commit `80f65d0` "Add safeguards against pipeline-axis regressions + contributor docs"
- Companion proposals: `docs/proposals/orientation_normalization.md`, `docs/proposals/bbox_augmentation_lost_in_pipeline.md`
- Related review: `docs/reviews/dataset_split_handling.md` (separate but adjacent — both surface design pressure on the dataset/registry/UI seams)
