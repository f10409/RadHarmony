# Proposal — Per-dataset opt-in for canonical 3-D orientation

**Status:** Open — extends `orientation_normalization.md` (deferred)
**Authors:** Judy (re-surfaced via cervical spine), drafted with Claude
**Date:** 2026-04-26
**Pertains to:** `radharmony/dataset/transforms.py:_base_load_3d`, every 3-D dataset class.

## Symptom

Loading random samples from `RSNA2022CervicalSpineDataset` (3-D CT) in the Gradio app, the user noticed the same upside-down rendering that prompted `docs/proposals/orientation_normalization.md` for RSNA PE Detection earlier this week — the head appears at the bottom and the neck at the top across some samples but not others. Same dataset, same pipeline, varying flip.

This is the exact failure mode `orientation_normalization.md` documents and proposes a fix for. That proposal is currently **deferred** because adding `OrientationD(axcodes="RAS")` globally to `_base_load_3d` breaks bbox alignment on RSNA 2024 Lumbar Spine — the lumbar bbox harmonizer (commit `d1d0535`) authored coords in image-relative axes that don't survive a reorientation.

So we have:
- One known-broken-rendering symptom across **every** 3-D CT/MRI dataset.
- One known-broken-bbox-alignment symptom that appears **only** for the bbox harmonizer that hard-codes image-axes coords.

A global on/off switch can't satisfy both. But the impacted set is tiny and asymmetric.

## Datasets affected, by category

| Dataset class | Has bboxes that depend on image-axes coords? | Wants `OrientationD(RAS)`? |
|---|---|---|
| `CTRATEDataset` | no | yes |
| `RadChestCTDataset` | yes (from `Extrema_35747.csv`) — **needs verification** | unclear |
| `RSNAPEDetectionDataset` | no | yes (the original motivating case) |
| `RSNA2022CervicalSpineDataset` | no | yes (today's case) |
| `RSNA2024LumbarSpineDataset` | yes — **the one that breaks under global `OrientationD`** | no (until bbox coords are reauthored) |

So the conflict is really between the lumbar-spine bbox harmonizer and everyone else. A per-dataset opt-in resolves it without re-litigating the global question.

## Proposed change

Add an opt-in `orientation` argument to `_base_load_3d` (default `None` = no canonicalization, preserves today's behaviour):

```python
def _base_load_3d(
    *,
    img_keys: list[str],
    pixdim: tuple | None,
    base_transpose: bool,
    hu_window: tuple | None,
    orientation: str | None = None,    # NEW — e.g. "RAS"
) -> list:
    t = [
        mn.transforms.LoadImageD(keys=img_keys, reader="ITKReader"),
        mn.transforms.EnsureChannelFirstD(keys=img_keys),
    ]
    if base_transpose and orientation is not None:
        t.append(mn.transforms.OrientationD(keys=img_keys, axcodes=orientation))
    if pixdim is not None:
        t.append(mn.transforms.SpacingD(keys=img_keys, pixdim=pixdim, mode=spacing_modes))
    if base_transpose:
        t.append(mn.transforms.Transposed(keys=img_keys, indices=[0, 3, 2, 1]))
    ...
```

Then `RadiologyTransform3D.__init__` forwards an `orientation` kwarg, and dataset classes that opt in pass `orientation="RAS"` when constructing their default transform. Today only the lumbar bbox harmonizer has a coord-axis assumption that conflicts; everyone else can opt in safely.

Three opt-in dataset classes to land alongside the framework change:
- `RSNAPEDetectionDataset`
- `RSNA2022CervicalSpineDataset`
- `CTRATEDataset`

`RadChestCTDataset` needs the bbox-coord audit before it opts in. `RSNA2024LumbarSpineDataset` stays opt-out until the bbox coords get reauthored against canonical axes.

## Why this is safer than the deferred global fix

The original `orientation_normalization.md` proposal was reverted because making the change global broke a dataset that depends on the current (broken) orientation. The opt-in pattern flips the responsibility: each dataset author decides whether their bbox/mask coord conventions survive `OrientationD`. The blast radius shrinks from "every 3-D dataset" to "the dataset I'm enabling."

It also creates a clean migration path for the lumbar-spine fix: when the bbox coords get reauthored against canonical axes, that dataset opts in and the special case disappears.

## What this does NOT solve

- The cosmetic flip in datasets that haven't opted in (e.g. RAD-ChestCT) — those stay where they are until someone audits their bbox/mask coord conventions.
- The bbox-coord drift problem itself for lumbar spine — still needs the reauthoring work `orientation_normalization.md` documents in its iteration history.
- 2-D datasets are unaffected (no analogous problem).

## Acceptance

- `RSNA2022CervicalSpineDataset` rendered in Gradio shows a consistent superior-anterior-right orientation across samples (no more random flips).
- `RSNAPEDetectionDataset` shows the same consistency that the original orientation work delivered, but without breaking lumbar-spine bboxes.
- `tests/test_bbox_axis_alignment.py` (added in commit `80f65d0`) still passes for lumbar spine because it stays opt-out.

## References

- Existing deferred proposal: `docs/proposals/orientation_normalization.md`
- Bbox-axis safeguards added with the deferral: `tests/test_bbox_axis_alignment.py`
- Related contributor doc (introduced in `80f65d0`): `docs/PIPELINE_INVARIANTS.md`
