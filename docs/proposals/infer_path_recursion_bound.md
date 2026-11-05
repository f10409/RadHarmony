# Proposal — Bound `infer_path`'s uncle-recursive glob

**Status:** Open
**Authors:** Judy (surfaced via RSNA Pneumonia incident), drafted with Claude
**Date:** 2026-04-26
**Pertains to:** `radharmony/utils/infer.py:infer_path`.

## Symptom

When a dataset's expected CSV/JSON is *missing* from the user-supplied `base_image_dir` (and its siblings), `infer_path` silently spends **minutes** walking unrelated dataset trees before returning `""`. From the user's perspective the Gradio app "hangs."

Concrete example surfaced today:
- User pointed RSNA Pneumonia at `/path/to/datasets/external/rsna-pneumonia-detection-challenge/` (Kaggle distribution; the adjudicated JSON the harmonizer needs is not present).
- `infer_path` fell into step 4 — *"recursive glob inside each uncle of `base_dir`"* — i.e. every sibling under `/path/to/datasets/external/`.
- That parent contains **22 multi-TB dataset directories** (CheXpert-Plus, MIMIC-CXR, NIH-CXR, PadChest, RAD-ChestCT, …).
- Recursive glob over all of them across NAS = several minutes of stat traffic before the function gives up and returns the (legitimate) "file not found."

## Why the current behaviour exists

From `infer.py`'s docstring:

> 4. Recursive glob inside each *uncle* of `base_dir` (children of grandparent, excluding parent) — only reached if step 3 finds nothing.

This is convenient when datasets are co-located in shallow trees (e.g. `~/data/<dataset>/{images,csvs}`) and the user happens to pass the wrong subdir. But on real NAS layouts where all datasets sit as siblings under a single root, step 4 escalates a "file not found" into a multi-minute wait.

## Options

### Option A — bound the search depth

Add a `max_glob_depth: int = 2` (or similar) parameter to `infer_path` and pass it through to the `_glob.glob(..., recursive=True)` call. Step 4's recursion stops at depth N, capping the worst-case work regardless of how cluttered the parent is.

**Pros:** smallest change; preserves step-4 behaviour for shallow trees.
**Cons:** picking N is heuristic; might miss legitimately-deep CSV placements.

### Option B — short-circuit when uncles are obviously huge

Skip a sibling directory in step 4 when it contains > N (say 10) sub-directories. Heuristic: a "datasets root" with many siblings is the user's NAS, not a co-located dataset bundle.

**Pros:** no API change.
**Cons:** still heuristic; could be confusing when it works for one tree and skips another.

### Option C — surface progress + abort

Emit a `tqdm` progress bar (or log line per sibling visited) so the user can see what's happening, and short-circuit on Ctrl+C or a configurable deadline.

**Pros:** the symptom ("appears to hang") goes away even if the wall-clock cost stays.
**Cons:** doesn't solve the underlying excess work; tqdm in a Gradio worker doesn't show in the UI.

### Option D — drop step 4 entirely

Make step 4 opt-in (off by default). When users put their data in a flat layout, they almost always know the exact path; auto-discovery beyond the parent directory is rarely worth a multi-minute scan.

**Pros:** eliminates the failure mode entirely.
**Cons:** breaks the (small number of) workflows that relied on it; users would need to pass `csv_path=` explicitly more often.

## Recommendation

**Option A (depth bound) + Option C (progress logging)** together. A `max_glob_depth=3` default keeps step 4 useful for genuinely-nested layouts while capping pathological cases at seconds rather than minutes; a per-sibling log line lets the user see *something* moving rather than wondering if the app froze.

## Acceptance

- Pointing a dataset at a `base_image_dir` whose parent contains many large unrelated trees returns a clear "file not found" within **seconds**, not minutes.
- Existing per-dataset workflows (CheXpert, MIMIC, etc.) still resolve their CSVs correctly via auto-discovery.
