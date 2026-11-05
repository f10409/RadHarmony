# Investigation — Bboxes vanish during the Resize step

**Status:** Superseded by D-28 in `docs/wiki/design.md` (2026-10-28). The box now moves with exact math, and only the affine and elastic warps use a mask round trip, on the final image grid.

**Old status:** ⏳ **awaiting decision** — root cause identified, fix options drafted, no code change made.
**Authors:** Judy + Claude
**Date:** 2026-04-26
**Pertains to:** `radharmony/dataset/transforms.py` — the `_BboxToMask` / `_MaskToBbox` round-trip and the `ResizeD(mode="nearest")` step on `bbox_mask`.

---

## What surfaced this

Loading **RSNA 2024 Lumbar Spine** in the Gradio app with `output_bbox=True` enabled, then enabling **Flip** in the augmentations panel.  Expectation: image flips, bbox mirrors with it.  Observed: image flips, **bbox stays in the same place**.

Chasing it revealed a deeper issue — the bbox doesn't survive the pipeline cleanly even *without* augmentation.  The Flip-not-tracking symptom is a downstream consequence.

---

## Root cause

The lumbar-spine bbox harmonizer (commit `d1d0535`) emits **tiny "dot" cube bboxes** — each annotated point becomes a ~1-voxel-cube with `bbox_epsilon` half-widths.  In the source coordinate system that's a cube of about `0.005 × 0.005 × 0.030` in normalized coords (or roughly `3 × 3 × 1` voxels in the native `640 × 640 × 15` volume).

The pipeline does:

1. `_BboxToMask` paints each bbox cube into a binary mask channel at the **native** image size (e.g. `(5 channels, 15, 640, 640)`).  Per-channel non-zero voxel counts: ~380–400.
2. `ResizeD(spatial_size=112, size_mode="longest", mode="nearest")` shrinks the mask.  Scale factor = `112 / 640 = 0.175`.  Each ~400-voxel cube scales by `0.175³ ≈ 0.0054`, giving a target of ~2 voxels — but with **nearest-neighbor** resampling at this scale, the dots **mostly fall between sample points and disappear entirely**.
3. `_MaskToBbox` then walks each channel, finds no non-zero voxels in most of them, and drops them.  The fallback for "all empty" is a single `[0, 0, 0, 0, 0, 0]` placeholder.

Empirically reproducible — standalone test on the first lumbar spine sample:

```
Raw bbox: list of 5 rows
After Load+ECF+Transpose: img (1, 15, 640, 640), bbox still 5 rows
After _BboxToMask:        bbox_mask (5, 15, 640, 640), non-zero per channel = [380, 380, 380, 400, 400]
After full pipeline:      bbox shape (1, 6) — all zeros
```

Five real bboxes in, one zeroed-out placeholder out.  When the user enables Flip, the bbox_mask that gets flipped is *already empty*; flipping zeros gives zeros.  The bbox *visible in the UI* is rendered from a separately-cached path or by chance from one channel that survives — that's why a label like `left_subarticular_stenosis` shows up but never moves.

---

## Why this didn't show up earlier

The lumbar-spine dataset is the only currently-integrated 3-D dataset that emits bboxes, and the `output_bbox=True` flag isn't ticked by default in the Gradio UI.  Most testing flows (and the existing integrity check) don't enable bboxes on lumbar spine.  The bbox round-trip was correct on smaller-scale 2-D datasets where the resize ratio is gentler.

---

## Fix options

Three real paths forward, with trade-offs:

### A. Inflate the bbox cube size in the harmonizer

Change `bbox_epsilon` in the lumbar-spine harmonizer (commit `d1d0535` defaulted it to a small value) from "1 voxel" to e.g. "5 mm in physical space."  Each dot annotation becomes a ~`10 × 10 × 2`-voxel cube at native scale, which survives a `0.175×` downsample as a ~`2 × 2` dot.

**Pro:** smallest code change; localized to the harmonizer.
**Con:** the bbox no longer represents a "point" annotation — it's a small region.  May affect downstream training if anyone is using these as exact-position labels.  Also: any other dataset that emits sub-voxel bboxes has the same problem.

### B. Resize the bbox mask with `mode="bilinear"` then re-binarize

Change the second `ResizeD` in `_build_preprocessing` (the one for `bbox_mask`) from `mode="nearest"` to `mode="bilinear"`, then add a `Lambdad(keys=["bbox_mask"], func=lambda x: (x > 0).to(x.dtype))` to re-binarize.  Trilinear interpolation preserves a fractional value at downsampled positions even when the source is sub-voxel; the threshold step recovers a binary mask.

**Pro:** generic — works for any bbox size, any dataset.  No harmonizer changes.
**Con:** slight semantic change — bboxes that overlap two coarse voxels at threshold may grow by one voxel.  Acceptable for visualization; needs care for training labels.

### C. Skip `_BboxToMask` entirely for sub-voxel bboxes

Convert bbox to mask at the *target* (post-resize) size instead of the source size.  Move `_BboxToMask` to *after* `ResizeD` for the image, and have it use the bbox coords (already normalized) directly against the smaller volume.

**Pro:** geometrically exact — no resize-then-resize loss.
**Con:** breaks the existing pipeline ordering convention (mask is created at source size so that augmentations can apply uniformly to img + mask).  Augmentations that are spatial-only (Flip, Affine, Elastic) would still work, but anything that needs a "before resize" mask would not.

### My (Claude's) recommendation

**B** as the immediate fix — it's the most general, doesn't change harmonizer semantics, and doesn't reorder the pipeline.  The slight bbox dilation it introduces is acceptable for a visualizer; if it becomes a problem for training, we can revisit.

**A** as a per-dataset quality-of-life follow-up — inflating the lumbar-spine `bbox_epsilon` to ~3 mm equivalent makes the bboxes legible in the UI even at 112³ resolution, regardless of which fix we pick.

---

## Reproduction

```python
import warnings; warnings.filterwarnings("ignore")
from radharmony.dataset.transforms import RadiologyTransform3D
from radharmony.dataset import RSNA2024LumbarSpineDataset

ds = RSNA2024LumbarSpineDataset(
    base_image_dir="/path/to/datasets/external/rsna-2024-lumbar-spine-degenerative-classification/train_images/",
    cache_dir=None, output_cls=True, output_bbox=True,
)
data_dict = ds.get_datasets().data[0]
print(f"raw bbox count: {len(data_dict['bbox'])}")

b = RadiologyTransform3D(
    img_size=112, output_keys={"img", "cls", "bbox"},
    pad=True, hu_window=None,
).with_bbox_as_mask()
out = b.get_transform()(dict(data_dict))
print(f"post-pipeline bbox: shape={out['bbox'].shape}, values={out['bbox']}")
```

Expected: 5 raw bboxes → 1 all-zero output bbox.

---

## Out of scope

- The `_MaskToBbox` "1 placeholder when all empty" fallback (line in `_MaskToBbox.__call__`) is reasonable behavior; we don't need to change it.
- Affine / Flip augmentations themselves are wired correctly (they include `bbox_mask` in their keys list).  This isn't an augmentation-wiring bug.
- Test coverage: a regression test for "bboxes survive the pipeline non-empty" is a natural follow-up once we pick a fix.

---

## Open questions

1. Is option **B** acceptable for downstream training use, or are these bboxes used as exact-position labels somewhere?
2. Should the lumbar-spine `bbox_epsilon` default change regardless of which fix we pick, just for legibility in the visualizer?
3. Are there other datasets (CT-RATE, RAD-ChestCT, etc.) that emit bboxes I haven't checked?  If so, fix B's generality matters more than fix A's locality.
