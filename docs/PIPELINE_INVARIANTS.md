# Pipeline Invariants — Things That Look Removable But Aren't

A growing list of load-bearing assumptions in the data-loading and transform pipeline that are easy to break "while cleaning up" but expensive to debug after the fact.  Each entry has a one-line description, the file it lives in, the regression test that protects it (if any), and the commit that made it load-bearing.

**Before refactoring `radharmony/dataset/transforms.py`, `radharmony/dataset/base.py`, or any harmonizer's `_build_*` method, scan this list.**

---

## 1. `Transposed([0, 3, 2, 1])` in `_base_load_3d`

**File:** `radharmony/dataset/transforms.py` — inside `_base_load_3d`, gated on `base_transpose=True`.
**Test:** `tests/test_bbox_axis_alignment.py`
**Made load-bearing by:** commit `d1d0535` (2026-04-22, "Wire train_label_coordinates.csv point annotations as 3-D bbox dots")

The Transposed step looks like a simple ITK→display axis correction (and the inline comment says exactly that).  But the lumbar spine bbox harmonizer authors `bbox` coordinates **in the post-Transposed `(D, H, W)` axis order**.  If the Transposed is removed, `_BboxToMask` paints `bbox[0:2]` into ITK's W axis instead of the slice direction — labels appear at wrong vertebra levels, image still looks "fine", and there's no error.

April 2026 we discovered this the hard way: an attempt to add `OrientationD` for cross-scanner orientation canonicalization removed the Transposed (thinking `OrientationD` subsumed it).  PE Detection looked great (no bboxes); lumbar spine bboxes silently moved.  Five iterations later we reverted everything.  The full story is in `docs/proposals/orientation_normalization.md`.

If you need to change the load-time axis order, the right approach is to change the bbox harmonizer's coord convention at the same time, then run `pytest tests/test_bbox_axis_alignment.py` before declaring victory.

## 2. _(template for future entries)_

**File:** `radharmony/...`
**Test:** `tests/...`
**Made load-bearing by:** commit `...`

Description...

---

## How to add an entry

When you fix a non-obvious bug whose root cause was "X in shared code looked redundant and got removed":

1. Add an entry here describing the invariant.
2. Add a regression test under `tests/` that fails fast if the invariant is broken.
3. Add an inline comment at the call site pointing at this doc and the test, e.g.:

   ```python
   # ⚠ DO NOT REMOVE without reading docs/PIPELINE_INVARIANTS.md
   # AND running tests/<your_regression_test>.py
   ```

The comment, the test, and this index are belt-and-suspenders.  Each one alone is fragile.  Together they make the invariant hard to break by accident.
