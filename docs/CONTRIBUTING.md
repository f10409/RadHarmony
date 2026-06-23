# Contributing to RadHarmony

A short guide for humans and AI agents working on this codebase.  Most of it is about not breaking existing things — the load/transform pipeline in particular is full of subtle invariants that look removable but aren't.  Read the **"Before you change shared infrastructure"** section before refactoring anything in `radharmony/dataset/transforms.py`, `radharmony/dataset/base.py`, or any harmonizer's `_build_*` method.

---

## Where things live

- `radharmony/dataset/` — `BaseRadiologicalDataset` + per-dataset subclasses + the `RadiologyTransform2D/3D` builders.
- `radharmony/harmonizer/` — `BaseHarmonizer` + per-dataset subclasses that read raw CSVs and produce a standardized DataFrame with `image_path`, `cls`, `bbox`, etc.
- `app.py` — the Gradio visualizer.  Datasets are registered in `DATASET_REGISTRY` near the top.
- `notebooks/` — runnable examples + the `dataset_integrity_check.ipynb` that exercises every dataset end-to-end.
- `docs/` — user-facing guides (`dataset_guide.md`, `app_guide.md`) + `proposals/` for in-flight design discussions + `PIPELINE_INVARIANTS.md` for "things that look removable but aren't."
- `tests/` — pytest regression tests.  Run with `.venv/bin/python -m pytest tests/`.

## How to add a new dataset

The fastest path is to copy an existing harmonizer + dataset class that has the closest properties:
- 2-D classification with bboxes → `chestxray14_bbox.py` template
- 3-D CT volume per study → `rsna_pe_detection.py` template
- 3-D MRI with multiple series per study → `rsna_2024_lumbar_spine.py` template

Pattern:
1. Write the harmonizer (`radharmony/harmonizer/<name>.py`) — produces a DataFrame with the standard columns.
2. Write the dataset wrapper (`radharmony/dataset/<name>.py`) — wraps the harmonizer + provides MONAI dataset semantics.
3. Export both from the respective `__init__.py`.
4. Add a `_build_<name>` function and a `DATASET_REGISTRY` entry in `app.py`.
5. Add a cell to `notebooks/dataset_integrity_check.ipynb`.

The Gradio app loader updates automatically once the registry entry exists.

---

## Before you change shared infrastructure

Anything in `radharmony/dataset/transforms.py`, `radharmony/dataset/base.py`, or `radharmony/harmonizer/base.py` is reached by every dataset.  A subtle change here can silently break datasets you didn't think to test.

**Required steps before touching shared infra:**

1. **Read [`docs/PIPELINE_INVARIANTS.md`](PIPELINE_INVARIANTS.md).** It's the index of "things that look removable but actually aren't."  If your change touches anything listed there, you need to update the corresponding test, not delete it.

2. **Run the existing tests.**  At minimum:

   ```bash
   .venv/bin/python -m pytest tests/
   ```

   These tests are intentionally fast (~30 s total).  If they fail before you've made any changes, fix that first.

3. **Read recent commit history** for the file you're changing:

   ```bash
   git log --follow --oneline -20 -- radharmony/dataset/transforms.py
   git blame -L <start>,<end> -- radharmony/dataset/transforms.py
   ```

   Commit messages often carry rationale that the inline comments don't.  In particular, comments may say things like *"ITK axis correction"* without mentioning *"this is also load-bearing for bbox alignment in dataset X."*  The commit that originally added the line is often the only place that connection is documented.

4. **Find downstream callers.**  For example:

   ```bash
   grep -rn "_base_load_3d\|_BboxToMask" .
   ```

   If a function is consumed by 14 build functions, your change will affect all 14.

5. **Add a test if you discover a non-obvious invariant.**  The pattern is:
   - Add a regression test under `tests/`.
   - Add an entry to `docs/PIPELINE_INVARIANTS.md`.
   - Add an inline `⚠ DO NOT REMOVE` comment at the call site pointing at both.

   See the `Transposed([0, 3, 2, 1])` line in `_base_load_3d` for the canonical example of all three.

---

## Pull request hygiene

- **One concern per PR.**  Don't bundle a refactor with a feature with a docs change.  Each is easier to review and revert independently.
- **Don't refactor opportunistically while you're "in there."**  If a function looks ugly but works, leave it.  The 5-iteration orientation episode in April 2026 happened because a "while we're here, this Transposed looks redundant" cleanup got bundled with an unrelated orientation feature.  The Transposed wasn't redundant.
- **Use `docs/proposals/` for design discussions** before landing changes that touch shared infra or change user-visible behavior.  Mark the status explicitly (`⏳ awaiting review`, `✅ done in commit X`, `⛔ blocked on Y`).

## Branches and commits

- Cut feature branches from `integration`, not `main`.
- Small fixes can go directly to `integration`; larger features go through PR review first.
- Squash-merge feature branches whose history is iterative or contains reverts (most of them).  Fast-forward when the history is genuinely linear and informative.
- Commit messages should explain the *why*, not just the *what*.  Future-you will read these; make them count.

---

## If you are an AI agent working on this codebase

In addition to all of the above:

- **Treat shared infrastructure as load-bearing until proven otherwise.**  If you're tempted to remove a line of code that "looks unnecessary," check the commit that added it — often the rationale is in the message rather than the inline comment.
- **Never refactor multiple concerns in the same change.**  If you're adding feature X, do exactly that.  Cleanup that "looks adjacent" needs to wait for a separate change so the user can review each independently.
- **Run the test suite before declaring a fix complete.**  `pytest tests/` is fast; there's no excuse to skip it.
- **When you're not sure why a line exists, ask the user before removing it.**  The April 2026 orientation episode took five iterations of breaking-and-fixing because a `Transposed([0, 3, 2, 1])` line was removed without first checking that it was load-bearing for bbox coord alignment.  Asking would have saved hours.
- **Add a regression test the moment you discover a non-obvious invariant.**  This is your debt to future-you.
