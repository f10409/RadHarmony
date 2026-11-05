# RadHarmony v0.2 to-do list

Updated: 2026-10-29. Branch: `v0.2-dev`.

The order is the one that Frank Li set: finish the pipeline, then fix the findings, then build D-31. The
other phases of the plan come after that. "Needs your yes" means that the agent asks before it acts.

## 1. Finish the pipeline

- [x] **Commit the pipeline work** to `v0.2-dev`. Done on 2026-10-28 in 4 commits, the first is
  48bbd15. No base code changed.
- [x] **Turn on the git hooks in this clone.** Done on 2026-10-28: `core.hooksPath` is `.githooks`. The old
  value pointed to the hooks folder of an old clone, which does not exist.
- [x] **Put the 4 tool-cost lessons into the agent briefs.** Done on 2026-10-28:
  - `hunter.md`: asks only for `findings.json`. Nothing read `report.md`, and 4 bug hunters could not write it.
  - `checker.md`: run in the foreground, and write `checked.json` before the final message.
  - `rules_reviewer.md` and step 4 of `SKILL.md`: the rules reviewer gets the old findings files.
  - `intents/datasets.md`: a filter for each large VQA set. Their `harmonize()` has no `study_ids=`, so the
    report's advice ("use `study_ids=`") did not work.
- [x] **Decide the draft rules L-008 to L-011.** All 4 approved on 2026-10-28. The answers are in the
  lessons of `reports/eval/2026-10-27-full/report.md`. All 4 are built (2026-10-28 and 2026-10-29). See
  `eval/RULES.md` for each rule, its test and its proof.
- [x] **Decide what to do with the 22 failing doc blocks until they are fixed.** Done on 2026-10-28 (Frank
  Li). They are on the known-problems list `eval/known_problems/blocks_run.json`. `run_all.py --full` now
  passes, and only a new broken block fails. The fix phase makes the list shorter.
- [x] **Update the intent files** in `.claude/skills/evaluate/references/intents/`, and make them point to
  `docs/wiki/design.md`. Example: `harmonizers.md` says that `image_path` is absolute, but D-04 says that
  it is relative to `base_image_dir`.
- [x] **Make the pipeline files concise** (review of 2026-10-28). Each item was decided one at a time:
  1. Done (Frank Li, 2026-10-28): keep `PIPELINE.md` and `summary.html`, and keep both up to date. If they
     disagree, `PIPELINE.md` is correct. `SUMMARY.md` (652 lines) was removed.
  2. Done (Frank Li, 2026-10-28): each intent file keeps only test steps that `design.md`
     does not have. A repeated rule becomes a pointer, for example "Boxes: see D-07 and D-28". It fixed 2
     conflicts: `image_path` is absolute (D-04: relative), and the bootstrap resamples rows (D-32: patients).
  3. Done (Frank Li, 2026-10-28): each known gap in `design.md` is one item, with the place and the finding
     ID, or "no ID". The gap text went from 98 to 81 lines, and `design.md` from 645 to 637 lines. Some
     details are in section 2 of this file. `/fix` now removes a gap from `design.md` when it fixes it.
  4. Done (Frank Li, 2026-10-28): plan 6d now says that each skill points to `CLAUDE.md` for the
     base-code rule, and does not repeat the steps. The 3 briefs stay as they are, because their proof
     keys differ on purpose.
  5. Done (Frank Li, 2026-10-28): new open questions get no O- ID. The 10 old O- IDs stay only in the "Was"
     column of the index. A question that is left open becomes a
     D- entry with the status "Open", and it keeps its D- ID when it is decided.
- [x] **Point to the rules from `CLAUDE.md` and `CONTRIBUTING.md`.** Done on 2026-10-28. `CLAUDE.md` has
  the new section "Design decisions and rules" (approved by Frank Li). Its 2 rules: read `design.md` before
  you change code, and ask about a behavior that no entry covers. `CONTRIBUTING.md`, `radharmony_internal/README.md` and
  `scripts/SYNCING.md` no longer send real paths to `configs/local/`. They point to `configs/data_paths.json`.
- [x] **Make the finding IDs point to one place.** Done on 2026-10-28 (Frank Li chose an ID index).
  `merge.py` writes every finding ID of every run to `reports/eval/index.json`. A new finding gets the next
  free number if an earlier run used its ID. `eval/RULES.md` and `docs/wiki/design.md` point to the index.
- [x] **Test a branch review** (plan, phase 7). Done on 2026-10-28, result PASS. A scratch branch had 3
  seeds: a Montgomery label regex bug, swapped Shenzhen counts and a base edit (no label sort). Both bug
  hunters found all 3 with proof, and 3 review parts flagged the base edit. No automatic check found any of
  them. Report: `reports/eval/2026-10-28-scratch-branch-review-test/report.md`. The branch is deleted. Frank Li
  chose 2 changes after it. L-008 also checks the label order. The checker keeps a real bug outside the
  diff in the list `outside_diff`.
- [x] **Add 4 items to plan 6d** (skills upgrade). Done on 2026-10-28:
  - The first 2 (read `docs/wiki/design.md` before you write code, and use `decisions.md` when the user must
    choose) are now rules in `CLAUDE.md`. Plan 6d says that each skill points to them.
  - The skill test of `update-docs` is now a test of `/fix`.
  - The `add-backbone` check also compares the crop and the pooled token (rule J-05, V-B-1, V-B-2).

## 2. Fix the findings

- [x] **Base-code changes to ask Frank Li in one round, at the end of this fix stage.** Done on 2026-11-02. Frank Li, 2026-11-01,
  for this fix stage only. In this stage, the agent decides each question that does not touch base code.
  It takes its recommended option and marks it "Decided by the agent, for Frank Li to confirm" in
  `docs/wiki/design.md`. After the stage, each question goes to Frank Li one at a time again.
  - `radharmony/dataset/transforms.py` (`_base_load_3d` and `with_spacing`). Done on 2026-11-02 (Frank Li).
    The docstrings said that ITKReader takes the z spacing from SliceThickness. ITK takes it from the slice
    positions (c9b947c). Docstrings only, no behavior change.
  - `radharmony/dataset/base.py` (`output_report` docstring). Done on 2026-11-02 (Frank Li). The docstring
    said that `output_report` gives the "report path" and needs a `report_path` column. The sample has the
    report text, from the `report` column (V-E-26). Docstring only, no behavior change.
  - `radharmony/dataset/base.py` (`_sync_label_cols_from_harmonizer`). Done on 2026-11-02 (Frank Li).
    After `harmonizer_path=` or `harmonizer=`, the dataset class took every column that is not in
    `_HARMONIZED_NON_LABEL_COLS` as a label, also the extra output columns. A run on 2026-11-01 (L-014)
    loaded the 41 real image keys with `expected_stats` this way. 16 keys got wrong label columns, and the
    6 TAIX-Ray keys raised `ValueError`, because `physician_id` is text. Examples: CheXpert Plus got `age`,
    `ethnicity` and `race`, and VinDr-CXR got `bbox_labels`. The fix takes the label columns from
    `h.get_label_cols()` (snake case) that are in the table. The 22 strict known failures are gone, and
    `tests/test_save_load.py::test_dataset_takes_only_the_declared_labels` covers it in CI.
  - `radharmony/dataset/base.py` (`_infer_label_cols_from_df`). Done on 2026-11-02 (Frank Li). A dataset
    class with empty `LABEL_COLS` took every column that is not a standard column as a label after
    `harmonized_df=` or `set_harmonized_df()`. `verify_images(drop_missing=True)` calls
    `set_harmonized_df()`, so the wrong labels also came back after `harmonizer_path=`. Example: Chest
    ImaGenome gold got `bbox_findings`, `bbox_labels`, `image_height` and `image_width`. Found by the
    critic of the label fix, 2026-11-02. It affected 11 keys: the 5 Chest ImaGenome keys, `ms_cxr_t`,
    `medseg_covid_ct_val` and the 4 `rsna_bone_age` keys. The fix: a class with `_HARMONIZER_CLS` takes
    the declared labels of that harmonizer that are in the table.
    `tests/test_save_load.py::test_harmonized_df_takes_no_extra_columns_as_labels` covers it in CI.
  - `radharmony/dataset/transforms.py` (`_base_load_2d` and `_append_warp`). Done on 2026-11-02 (Frank Li).
    The `Lambdad` steps used lambdas, so a DataLoader worker with the "spawn" or "forkserver" start method
    could not pickle the transform. Every 2-D dataset and every warp failed there (critic of d8d69c2). The
    venvs use Python 3.12.9 ("fork"), so the runs before were not affected. The fix: 3 module-level
    functions (`_drop_dicom_axis`, `_shift_up`, `_shift_down`) with the same code. The sample tensors of 6
    transforms are equal to the old code. `tests/test_transforms.py::test_transform_pickles_for_spawn_workers`
    covers it in CI.
  - The base-code group is in `reports/eval/2026-10-31-full/report.md`, section 9. It has V-D-27, V-D-28,
    V-D-30, V-D-36, V-D-37, V-E-25, V-E-31, V-H1-31, V-H2-28 and R-code-18. It also has the gaps of Q-5,
    Q-6, Q-8, Q-13 and Q-20. Each one is written here with its planned change when the agent reaches it.
    V-D-30 and V-H1-31 are fixed by d45bb39 (the label rule after `harmonizer_path=`). Their proof commands
    give the expected output on 2026-11-02.
    V-D-27 is fixed on 2026-11-02 (Frank Li): one channel count for each dataset (D-42). The fix is the step
    `_SetChannelsD` in `_base_load_2d` and `IMG_CHANNELS` in `base_vqa.py`. Its proof command gives the
    expected output.
    V-D-28 is fixed on 2026-11-02 (Frank Li): `_ReorientBbox` and `_SITKOrientD` are MONAI transforms, so
    the 3-D cache holds the preprocessed volume (D-15). One CT-RATE cache file went from 1,053 MB to 3 MB.
    V-D-36 is fixed on 2026-11-02 (Frank Li): `visualize_samples()` shows only positive labels, so a NaN
    (unknown) or -1 (uncertain) label no longer raises `ValueError` (D-05).
    V-D-37 is fixed on 2026-11-02 (Frank Li). The cache key has a hash of each function in a cached step,
    and of the `__call__` of each RadHarmony step (D-15).
    V-E-25 is fixed on 2026-11-02 (Frank Li). The embedding cache keeps a fingerprint of all weights, the
    encoder code and settings, and the transform (D-41).
    V-E-31 is fixed on 2026-11-02 (Frank Li). With fewer than 2 values, the summary std and CI are NaN (D-19).
    The Q-8 gap is fixed on 2026-11-02 (Frank Li). The summary has `<metric>_n` and warns per label (D-19).
    V-H2-28 is fixed on 2026-11-02 (Frank Li). The VQA `verify_images()` raises for relative paths with no
    folder (D-04).
    R-code-18 is fixed on 2026-11-02 (Frank Li). The 3-D box step has no silent `except` (D-28, J-03).
    The Q-20 gap is fixed on 2026-11-02 (Frank Li). A resample keeps the image centered (D-43).
    The Q-13 gap is fixed on 2026-11-02 (Frank Li). 2-D `view_position` uses PA, AP and LATERAL. Left (not
    base): AP or PA in place of FRONTAL for CheXlocalize val, SIIM-ACR PTX and OpenI.
  - `radharmony/dataset/base.py` (`_describe`). Done on 2026-11-02 (Frank Li). Some changed steps kept their
    cache key. Examples: a `functools.partial` with other arguments, and other default arguments. Also a
    builtin such as `torch.abs` in place of `torch.neg`, a bound method whose object changed, and a
    `staticmethod` object. Found by the critic of V-D-37 (D-15). Now each of them gives a new key.

- [x] **Pick the findings to fix.** Done on 2026-10-28 (Frank Li): fix all open findings, batch by batch,
  in this order. Each fix starts with a failing test. The agent asks only for base code, or when
  `docs/wiki/design.md` does not choose the behavior. Low findings go with their batch.
  1. B1: the 6 high findings of `reports/eval/audit-2026-10-27/report.md` (they change benchmark numbers),
     and V-H1-23 (same code as V-H1-2). The first fix builds the stats test (`tests/test_dataset_stats.py`).
  2. B2: test images in training (D-09, D-25).
  3. B3: labels, save and load (D-05, D-08, D-24, L-008). It ran as 2 batches: B3a (labels, save and load,
     output flags) and B3b (harmonizer data, sex and age).
  4. B4: evaluators (D-18, D-29, D-32, L-009).
  5. B5: boxes (D-28, L-011).
  6. B6: preprocessing and readers (D-27).
  7. B7: docs, with the 22 failing doc blocks.
  8. B8: the app, scripts, other backbones and packaging.
- [ ] **Add `expected_stats` to existing datasets** (decided 2026-10-28). Start with the datasets that have
  known gaps. The stats test is the failing test that `/fix` writes first.
  - Done in B1: ChestX-ray14, CheXpert, RSNA Pneumonia (Kaggle), SIIM-ACR and SIIM-COVID19.
  - Done in B2: MIMIC-CXR, MIMIC-CXR-JPG, RANZCR CLiP, CheXinstruct VQA, TAIX-Ray, TAIX-Ray 512, MS-CXR,
    CheXpert Plus, VQA-RAD, CT-RATE, RAD-ChestCT and CheXlocalize.
  - Done in B3b and B5: VinDr-CXR train and test, Shenzhen and Chest ImaGenome gold. 11 files that already
    had stats got more stats (sex, age or boxes), for example ChestX-ray14 and RAD-ChestCT.
  - Left: 16 of the 37 files in `configs/datasets/`, for example PadChest, BRAX and NODE21. Some datasets
    have no file in `configs/datasets/` yet.
- [x] **D-09 official splits (B2).** Done on 2026-10-28: 47 new split keys, for example
  `mimic_cxr_jpg_train` and `ranzcr_clip_test`. Each old key that mixes splits works as before, and its
  docs say that it mixes splits.
- [x] **B4 (evaluators).** Done on 2026-10-28.
  - The 8 evaluators that use the bootstrap now draw patients (D-32). These are the 6 classification
    evaluators, the segmentation linear probe and report generation.
  - The 5 probes that train (linear, SVM, kNN, prototype, fine-tune) drop NaN train labels. The new lesson
    check L-009 in `eval/RULES.md` tests this.
  - These 5 probes fit on 90% of the training patients and pick the threshold on the other 10% (D-29).
  - The evaluators set the torch seed.
  - Example of a changed result: MedSigLIP features, a tuberculosis task with 138 train images and 3,000
    test images. Linear probe AUROC 0.856 to 0.858, balanced accuracy 0.796 to 0.538. The train set has
    42% positives and the test set 5.5%.
- [x] **B8 (app, scripts, integrity check, backbones, packaging).** Done on 2026-10-28. Ark+ now uses the
  input of the authors' fine-tune code (resize 896, crop 768). SigLIP2 now lowercases the text. Results
  made before 2026-10-28 with these 2 backbones must run again (see the last items).
- [x] **B3a (labels, save and load, output flags).** Done on 2026-10-29.
  - D-05: VinDr-CXR, VinDr-PCXR, Open-I and MR-RATE sort their label columns.
  - D-08: 12 harmonizer families keep their arguments after a load (27 classes). With `harmonizer_path=`,
    each registry key now loads a file of its own harmonizer. Before, 34 keys failed.
  - D-11: every image class takes `output_cls`, `output_mask`, `output_bbox` and `output_report`, and
    warns for a flag that it does not support (19 classes raised `TypeError` before).
  - New lesson checks: L-008 (harmonizer contract) and L-010 (output flags).
- [x] **B3b (harmonizer data, sex and age).** Done on 2026-10-29.
  - D-26: one `sex` column ("M", "F" or NaN) from a base step, in 21 harmonizers.
  - D-06: `age_years` and `sex` from the DICOM tags for RSNA Pneumonia Kaggle, SIIM-ACR, SIIM-COVID19
    (sex only) and VinDr-CXR. `age_years` from the source for ChestX-ray14, CT-RATE, CheXpert Plus and
    CheXlocalize.
  - 11 more commits fix other data problems, or warn about them (12 finding IDs). For example, the
    ChestX-ray14 box label `infiltrate` is now `infiltration`, as the label column. MS-CXR-T `image_path` has no `files/` prefix now, so make the saved MS-CXR-T tables
    again.
- [x] **B5 (boxes).** Done on 2026-10-29.
  - D-28: the box moves with every resize, pad, flip, transpose and warp. A sample box is an `(N, 4)` or
    `(N, 6)` float32 tensor. New lesson check L-011.
  - 7 harmonizer box fixes, for example SIIM-COVID19 boxes clipped to the image and RSNA 2024 lumbar boxes
    on the annotated slice (not the mirrored one).
  - `radharmony.dataset.bbox_collate` makes a batch of samples with different box counts.
  - Old cache files are not read since 2026-10-29 (D-15 transform key). Delete them to free the disk.
- [x] **B6 (preprocessing and image readers).** Done on 2026-10-29.
  - `CXRPreprocessor`: MONOCHROME1 is inverted once, a DICOM CXR gives a 2-D image, and the VOI LUT works
    for any DICOM file name.
  - The encoder readers scale 12-bit and 16-bit images to 0 to 255 and do not clip (D-27).
  - Shenzhen gives 1 channel, not 3. MS-CXR `harmonize()` takes `preprocess_output_dir`.
  - The other findings of B6 were base-code changes. They are done (item "Base-code proposals from the
    batches" below).
- [x] **B7 (docs).** Done on 2026-10-29.
  - 17 of the 22 failing doc blocks do not fail now. Most run or have `run=skip` with a reason. 3 report
    "blocked", because they need data or a checkpoint (`api.md:53`, `shenzhen_cxr.md:156` and
    `backbones/index.md:95`). The other 5 are in the item "Fix the 22 failing doc blocks" below.
  - `docs/wiki/reference/datasets/index.md` lists all 127 registry keys, with a claim note for the count.
  - The home page, quickstart, API, architecture and dataset pages match the code. They cover the `sex`
    column, the box tensor, `bbox_collate`, the split keys, D-29 and D-32. They say that an unsupported
    `output_*` flag gives a warning.
  - The app has 11 new split entries (ROCO, ROCOv2, VQA-RAD and CheXinstruct VQA). `check_datasets.py`
    has 33 new specs. `check_datasets_internal.py` has 4 (EmoryCXR DICOM and the 3 MS-CXR splits).
  - Embedding sizes and the RAD-DINO and SigLIP2 inputs in the backbone docs match the model cards.
- [x] **D-24 (V-H1-23).** The ChestX-ray14 train and test harmonizers kept all 111,240 rows in
  `harmonized_df`. Fixed on 2026-10-28, with V-H1-2. `harmonized_df` now has only the rows of the split
  (86,524 train, 25,596 test).
- [x] **Decide the open parts of 3 decisions** before their fixes. Done on 2026-10-28 (Frank Li):
  - D-26: a base step with `SEX_SOURCE_COL`, like age. The base-code change to `base.py` is approved.
  - D-28: decided on 2026-10-28, math on the box corners, and the mask round trip only for affine and
    elastic. The base-code change to `transforms.py` is approved.
  - D-29: decided on 2026-10-28, 10% of the training patients (`val_fraction=0.1`).
- [ ] **The other known gaps in `docs/wiki/design.md`.** The base-code gaps are done (next item). The open
  parts wait for the questions below.
- [x] **Base-code proposals from the batches.** Done on 2026-10-29. Frank Li approved each change below, one
  at a time. Each fix has a test in `tests/`. The proposed diffs are in
  `reports/eval/2026-10-29-base-proposals/` (local, not in git).
  - `harmonizer/base.py`:
    - `load_from_saved` sorts `LABEL_COLS` like `harmonize()`. Before, `["Calcification", "COPD"]` became
      `["COPD", "Calcification"]` after a load (D-08).
    - Save and load keep an empty `LABEL_COLS` empty (D-08, V-H1-15, V-H2-3). Before, Chest ImaGenome gold got
      4 box and size columns as labels after a load.
    - Remove `_HARMONIZED_CORE_COLS`, which no code used after f72315d.
    - The base merges raise a `ValueError` on a repeated key (V-H2-17, D-03). No dataset changed. SIIM-ACR
      joins its mask lines in its own code.
    - A `_read_source()` step, so a harmonizer does not copy `harmonize()` (D-30). The 45 copied
      `harmonize()` methods moved to the base steps on 2026-11-03 (R-code-8, plan 6b, merges refactor-r1 to r4).
    - `preprocess_masks` skips an image that nobody annotated (D-33, V-H2-4). Before, 20,988 of 30,083 RANZCR
      train images got an all-zero mask. The RANZCR CLiP and CheXlocalize segmentation results must run again.
    - The base `harmonize()` makes the masks before the image copies (V-H2-21). Before, a call with both
      `preprocess_output_dir` and `mask_output_dir` crashed on SIIM-ACR. The dataset classes never pass both,
      so no result changes.
    - With an absolute `image_path`, the mask goes into `mask_output_dir` (V-H2-21). Before, for a PNG image,
      `mask_path` became the X-ray itself.
  - Not needed: the `study_ids=` filter in `harmonizer/base_vqa.py`. D-02 (2026-10-30) puts it only in
    `MIMICExtCXRQBAHarmonizer`, which is not base code.
  - `dataset/base.py`:
    - `visualize_samples` draws each box of an `(N, 4)` tensor (V-D-22). Before, it drew no box at all.
    - A transform key in the cache folder name (V-D-8, D-15). Before, a run with a new `img_size` and the same
      `cache_dir` got the old tensors.
      Follow-up after the critic review: the key leaves out the train mode of a model and has the reader. A
      list of steps and any tensor work.
    - The same transform key for `dataset/base_vqa.py` (V-D-8).
    - A new random seed for each worker and each epoch (V-D-7). Before, with a torch `DataLoader`, the workers
      repeated the same augmentation in every epoch. The same step for `dataset/base_vqa.py` was approved too.
    - `_apply_voi_lut` and `_fix_missing_highbit` handle a signed MONOCHROME1 image (D-13), with the shared
      helper `radharmony/utils/dicom.py`.
    - `_apply_voi_lut` and `_fix_missing_highbit` find a DICOM file by a tag, not by its file name (D-13). The
      tag is `0028|0004` in the image information.
      Follow-up after the critic review: an `.mha`, `.nrrd` or NIfTI copy keeps the tag, so these endings are
      not DICOM (`is_dicom` in `radharmony/utils/dicom.py`). `CXRPreprocessor` uses the same helper.
  - `preprocessor/base.py`:
    - `run_batch` raises an error when images fail (V-P-3, V-H2-19). Before, `image_path` pointed to copies
      that were never written.
    - Each image gets its own output file, below its own folders (V-P-4, V-H2-18). Before, the 138,358
      CheXpert train images went into 6 files.
      Follow-up after the critic review: `.dicom` is a known ending, so a VinDr copy is `abc.nii`, not
      `abc.dicom.nii`.
    - `_crop_3d` crops the right region when an axis is flipped (V-P-5). Before, a flipped y axis cut away the
      whole body.
    - `_resize_isotropic` finds the long side by its size in mm, not by its voxel count (V-P-11).
    - Remove `_fix_monochrome`, which had no caller.
  - `evaluator/base.py`:
    - The summary mean uses only the point-estimate rows (`bootstrap = -1`). The std and the CI use the
      bootstrap rows (V-E-15).
    - Remove `_bootstrap_metrics`, which no code called (R-code-11).
  - `evaluator/classification/base.py`: the embedding cache stores the image paths, the encoder name and all
    label columns (V-E-2). On other images or another encoder, it raises an error (Frank Li chose this over
    building the cache again). Follow-up after the critic review: the rows come from `dataset_rows(ds)`.
    The encoder name comes from the model inside `ImageEncoderWrapper`. A cache file from before
    2026-10-29 raises. `extract_embeddings` takes its patient ids from `dataset_rows(ds)` too (V-E-9).
  - Docstrings and comments from B7 (`b7_docstring_proposals.md` in the same folder):
    - `harmonizer/base.py`: the docstrings say that `image_path` is relative (D-04, V-H1-22, V-H2-23).
      `verify_images()` takes `self.base_image_dir` when it exists. Otherwise it raises for relative paths.
      Before, it dropped 138 of 138 Montgomery rows.
    - `dataset/base.py`: remove 2 commented-out debug lines and 1 commented-out argument (R-code-12), and the
      unused `_convert_monochrome1_to_2`.
    - `evaluator/classification/base.py` and `evaluator/segmentation/base.py`: rewrite the bfloat16 autocast
      docstring in STE style (R-code-14).
    - `evaluator/base.py`: rewrite the module docstring in STE style (R-code-15).
- [x] **The 6 high findings of the audit (B1).** Done on 2026-10-28, each with a failing test first:
  - V-H1-1: CheXpert keeps the lateral images.
  - V-H1-2 and V-H1-23: the ChestX-ray14 test split has all 25,596 images.
  - V-H3-1 and V-H3-2: RSNA Pneumonia and SIIM-ACR give one row per image.
  - V-H3-3: SIIM-COVID19 takes the patients from the DICOM `PatientID`.
  - V-B-1: EVA-X gives the mean of the patch tokens, then `fc_norm`. The small probe AUROC went from 0.758
    to 0.844.
- [x] **Fix the 22 failing doc blocks.** B7 fixed 17 on 2026-10-29. The D-35 fix of 2026-10-30 fixed the
  other 5 (`roco.md`, `rsna_2024_lumbar_spine.md`, `siim_covid19.md`, `chexphoto.md` and
  `rsna_pneumonia.md`). With no data, each now reports "blocked, needs data".
- [x] **`docs/wiki/reference/datasets/cxr-boxes-masks/ms_cxr.md`.** It said that the split has only train and test. Fixed by d5e3dd9:
  the page lists `ms_cxr_val` (155 images, 212 boxes).
- [x] **`docs/wiki/reference/datasets/cxr-boxes-masks/shenzhen_cxr.md:195`.** It said that `df.shape` is `(662, 27)`. Fixed in B3b:
  it is `(662, 29)` with `sex`.
- [x] **Found while fixing B1 (no ID, critic review of 2026-10-28).**
  `ChestXray14TestDataset(harmonizer_path=p)` with no `base_image_dir` failed, and the train class too.
  Fixed in B3a, with V-H3-17.
- [ ] **Found while fixing B2 and B8 (no ID).**
  - Done in B7: `scripts/check_datasets.py` has a spec for each registry key with data, for example
    `siim_acr_ptx_test`, `taix_ray_train` and `rsna_bone_age_test`. The MS-CXR splits are in
    `check_datasets_internal.py`.
  - Done in B7: the app has split entries for ROCO, ROCOv2, VQA-RAD and CheXinstruct VQA. MR-RATE has no
    split entry, because it has no registry key.
  - Done in B7: `docs/wiki/reference/datasets/index.md` said "52 dataset configurations". It now says "127 registry keys".
  - `ChestXDet10TrainDataset(csv_path=".../test.json")` reads the test JSON as train data. A test image
    with a box raises `FileNotFoundError`, because the class looks for it in `train-old/`. The JSON has no
    split field, so the class cannot tell train from test.
  - Some old dataset classes name their harmonizer class directly. So their split classes copy
    `__init__` or `_get_harmonized_df`.
  - Dataset modules import the private `_FixedSplitArgMixin`.
  - The CheXinstruct split classes are in the harmonizer module, not in a `*_split.py` file.
- [ ] **Found during the design questions (no ID).**
  - The segmentation probes run the frozen backbone again in every epoch. They have no feature cache like
    `embedding_cache`. On Montgomery (138 images) with RAD-DINO, an epoch took about 60 s without
    `cache_dir` and 1.2 s with it.
  - BRAX: `docs/wiki/reference/datasets/cxr-labels/brax.md` and `configs/datasets/brax.json` say 19,351 patients and about 24,959
    studies (the paper). `master_spreadsheet_update.csv` has 40,967 images, as the paper says, but 18,442
    `PatientID` values and 23,276 `AccessionNumber` values. Add `expected_stats` for BRAX and find the reason.
    Fixed on 2026-11-03: the wiki page, the config and the harmonizer docstring give the file counts and the
    paper counts (V-H1-10). The `expected_stats` and the reason are still to do.
  - The report parser takes the first IMPRESSION header. 6 of 2,216 MIMIC-CXR test reports (117 of 171,373
    train reports) have 2 IMPRESSION headers. The first is often a preliminary read (a "wet read" or a
    "provisional findings impression") before the FINAL REPORT. Should the last one win?
  - Done on 2026-10-31: Q-9 (D-38). The IMPRESSION stop list gets FINDINGS (7 of 20,000 random MIMIC-CXR
    reports have IMPRESSION first). Fixed in `_report_parser.py` on 2026-11-01.
- [ ] **Found while fixing B7 (no ID).**
  - Fixed on 2026-10-30: `ROCOHarmonizer` warns for each missing split folder, and raises
    `FileNotFoundError` when it finds none (D-35).
  - Fixed on 2026-10-31: 7 docstring lines in 4 dataset files said that an unsupported flag is "silently
    ignored". They now say "A warning, and no key (D-11)", as the code does.
  - Done on 2026-10-30 (D-25): keep the RSNA 2023 abdominal and RSNA 2022 cervical test classes. Fixed on
    2026-10-30: their docstrings and wiki pages say so.
  - Done on 2026-10-30 (D-21): L-002 also checks `.claude/skills/`. Fixed on 2026-10-30.
- [ ] **Questions for Frank Li after the batches.** Ask them one at a time.
  - Done on 2026-10-29: D-34, option (a), keep the published splits. Fixed on 2026-10-30:
    `mimic_cxr_jpg_validate` keeps the 3 patients of the labeled test set.
  - Done on 2026-10-29: D-35, option (a), `infer_path` looks only in exact places. Fixed on 2026-10-30.
  - Done on 2026-10-29: D-36, option (c), group the boxes by IoU > 0.5. Fixed on 2026-10-30 (V-H3-4):
    VinDr-CXR train has 8,328 boxes on 4,167 images.
  - Done on 2026-10-30: the open part of D-02, option (c). Fixed on 2026-10-30: `MIMICExtCXRQBAHarmonizer`
    takes `study_ids=` and `patient_ids=`.
  - Done on 2026-10-30: the 4 open parts of D-06. Fixed on 2026-10-30 for BRAX, VinDr-CXR and PadChest
    (V-H1-8). MR-RATE stays a known gap (restricted data).
  - Done on 2026-10-30: the open part of D-09. A split class raises on an old file with rows of another
    split, as today. No code change.
  - Done on 2026-10-30: the open part of D-11. Every image key takes `output_reg`. Fixed on 2026-10-30.
  - Done on 2026-10-30: the open part of D-14. A 2-D mask in `ds[i]` holds 0 and 1. Fixed on 2026-10-30
    with one step in `RadiologyTransform2D` (base code, approved by Frank Li on 2026-10-30, V-D-10).
  - Done on 2026-10-30: the open part of D-26. CheXpert `sex` comes from the CSV. Fixed on 2026-10-30.
  - Done on 2026-10-30: V-D-9 (D-33). With `output_report=True`, the dataset class drops a row with no
    report and warns with the count. The dropped mask rows get the same warning. Fixed on 2026-10-30.
  - Done on 2026-10-30: V-P-16, new entry D-37. `RadiologyEncoderTransform` fills the warp border with the
    darkest value of each channel of the image. Fixed on 2026-10-30 with an override of `_append_warp` in
    `radharmony/evaluator/transforms.py` (no base code).
  - Done on 2026-10-30: V-H1-9 (D-01). BRAX keeps the `AccessionNumber` as `study_id`, with the warning and
    the wiki note. No code change.
  - Done on 2026-10-30: V-H1-12 (D-03). Fixed on 2026-10-30: 8 label and 17 report conflicts are left.
  - Done on 2026-10-30: V-H1-27 (D-01). MedPMC with no tag table keeps the warning. No code change.
  - Done on 2026-10-30: V-H1-28 (D-01). CT-RATE `view_position` is the nearest plane name, and the vector
    is in the new extra column `image_orientation`. Fixed on 2026-10-30.
  - Done on 2026-10-30: D-07 has both B5 rules as general rules. No code change.
  - Done on 2026-10-30: D-29 part 1. The 90% / 10% way is confirmed. No code change.
  - Done on 2026-10-30: D-29 part 2. Zero-shot threshold metrics are NaN unless `threshold_strategy` is
    `fixed:<value>`, with one warning that names `fixed:0` for negative prompts. Fixed on 2026-10-30
    (R-code-2).
  - Done on 2026-10-30: D-17 zero-shot. It warns for `l2_normalize=False`. Fixed on 2026-10-30.
  - Done on 2026-10-30: D-17 segmentation heads. They keep the raw features. No code change.
  - Done on 2026-10-30: V-E-7, new entry D-38. A report section keeps the fallback to the other section.
    No code change. The wiki table already says it.
  - Done on 2026-10-30: V-E-6 (D-14). A mask with more than 2 classes holds the class indices 0 to K-1. No
    code change.
  - Done on 2026-10-30: V-E-8 (D-38). The B4 header rule is confirmed. No code change.
  - Done on 2026-10-30: MIMIC-CXR-JPG test (D-09). Fixed on 2026-10-30: the key
    `mimic_cxr_jpg_test_chexpert` (3,556 images, 2,216 studies, 287 patients, as `mimic_cxr_test`).
  - Done on 2026-10-30: V-A-13 (D-11). The app "BBox" checkbox follows the build function default of the
    dataset. Fixed on 2026-10-30.
  - Done on 2026-10-30: V-A-5, new entry D-39. A bad box or a NaN or Inf pixel stays ERROR. No code change.
  - Done on 2026-10-30: V-B-2 (D-16). The Ark+ fine-tune input (bilinear 896, crop 768) is confirmed. No
    code change.
  - Done on 2026-10-30: A-6 (D-39). Fixed on 2026-10-30: the 6 specs are in `check_datasets.py`.
  - Done on 2026-10-30: missing data (D-35). Fixed on 2026-10-30: `infer_path(..., required=True)` in 14
    dataset classes, and the runner rule for the pandas text "Invalid file path ..." is gone. Fixed on
    2026-10-31 in 15 more classes.
  - Done on 2026-10-30: `check_datasets.py` (D-39). A spec for restricted or not_on_nas data runs only with
    `--datasets`. The B7 rule is confirmed. No code change.
  - Done on 2026-10-31: D-12 RAD-ChestCT. Its arrays are already `IPL`, so `base_transpose=False` follows
    the rule. No code change.
  - Done on 2026-10-31: D-12. The `_base_load_3d` docstring names `_SITKOrientD` and `IPL`. Fixed on
    2026-10-31 (base code, docstring only, approved by Frank Li on 2026-10-31).
  - Done on 2026-10-31: D-13. GDCM inverts MONOCHROME1, and
    `test_dataset_load_of_signed_monochrome1_dicom_with_window` fails if GDCM stops the inversion. No code
    change.
  - Done on 2026-10-31: D-28. `RadiologyEncoderTransform` warns when `bbox` is in `output_keys` and the
    processor crops or pads, or is a custom `preprocess` with no processor. The warning is in the code since
    2026-10-31, and the gap stays.
  - Done on 2026-10-31: D-30. The 45 `harmonize()` overrides and the 10 `_KeepSplitMixin` split classes
    get their fix in the code cleanup (plan 6b), with golden outputs. No code change now.
  - Done on 2026-10-31: D-35, file for one output. A file that a class reads for one output is required when
    the user asks for that output. Fixed on 2026-10-31 for MIMIC-CXR, RSNA Pneumonia Kaggle and CheXpert Plus.
  - Done on 2026-10-31: D-35, empty image folder. `montgomery_cxr`, `siim_acr_ptx_test` and `siim_covid19_test`
    raise `FileNotFoundError` with the folder and the file type. Fixed on 2026-10-31.
  - Done on 2026-10-31: Q-7 of the 2026-10-31 review, new entry D-40. A new argument
    `n_train_unit="image" | "patient"` (default `"image"`) says what `n_train` counts. To build.
  - Done on 2026-10-31: Q-10 (D-33). CheXlocalize images with all 10 labels negative are labeled negatives
    and get an all-zero mask (test 667 of 668, val 234 of 234). Fixed on 2026-11-01. Then the CheXlocalize
    segmentation results run again (see the rerun list).
  - Done on 2026-10-31: Q-15 (D-05). PadChest rows with an empty label list get NaN in every label (777
    rows), so the probes skip them. Fixed on 2026-11-01, with the wiki line.
  - Done on 2026-10-31: Q-4 (D-05). In `cls`, -1 means uncertain and NaN means not mentioned (the
    CheXpert-style codes). Fixed on 2026-11-01: CheXpert, CheXpert Plus, MIMIC-CXR and MIMIC-CXR-JPG keep -1,
    and `drop_uncertain` drops the rows with -1. A blank is 0 by default, or NaN with `not_mentioned="nan"`
    (Frank Li, 2026-11-01). V-H1-34 is then not a bug for BRAX, Emory CXR and the
    MIMIC-CXR-JPG labeled test.
  - Done on 2026-10-31: Q-6, new entry D-41. The embedding cache holds features only, and each run reads
    the labels from the dataset. Fixed on 2026-11-02 (Frank Li, `_dataset_labels`).
  - Done on 2026-10-31: Q-17 (D-05). A blank RSNA 2024 lumbar severity gives NaN in its 3 grade columns
    (572 cells). Fixed on 2026-11-01, with the wiki line.
  - Done on 2026-10-31: Q-8 (D-19). The summary gives `<metric>_n` (the runs with a value), and the probe
    warns for a label with NaN runs. Fixed on 2026-11-02 (`_summarize`, option A).
  - Done on 2026-10-31: Q-5 and V-D-27, new entry D-42. One channel count for each 2-D dataset: 1 for
    gray modalities, 3 for color figures. Fixed on 2026-11-02 (`_SetChannelsD`, `IMG_CHANNELS`).
  - Done on 2026-10-31: Q-19 (D-06). TAIX-Ray age 0 days stays 0.0 (real newborns). No code change. The
    wiki page says so.
  - Done on 2026-10-31: Q-13 (D-01). 2-D `view_position` uses one name list: PA, AP, LATERAL, NaN for
    unknown, other values raw in upper case. To fix with one base step (needs approval at the edit).
  - Done on 2026-10-31: Q-14 (D-01). Emory CXR: when `Frontal` and `ViewPosition` disagree, `Frontal` wins
    (Frontal 0 gives Lateral, the source spelling until Q-13, and Frontal 1 with Lateral gives NaN). To fix in the Emory harmonizer. A sample of
    24 of the 2,202 Frontal=0 / AP images was mostly frontal or decubitus, so this group gets NaN (Frank Li,
    2026-11-01). Fixed in the Emory harmonizer on 2026-11-01.
  - Done on 2026-10-31: Q-18 (D-01). RSNA 2024 lumbar `view_position` becomes the nearest plane, and the
    description goes to `series_description`. Frank Li wants to revisit this later, so wait for him
    before the fix.
  - Done on 2026-10-31: Q-12 (D-01). MIMIC IDs are digits in every harmonizer. Fixed in MIMIC-Ext-CXR-QBA on
    2026-11-01 (its id filters accept both forms) and in GEMeX VQA.
  - Done on 2026-10-31: Q-11 (D-04). CheXinstruct VQA uses relative paths (one source folder today, a
    `data_source` column for several sources later). Fixed on 2026-11-01 in its harmonizer and dataset class.
  - Done on 2026-10-31: Q-16 (D-33). A placeholder report counts as no report. Fixed in Montgomery (1 row)
    on 2026-11-01, after the patient link.
  - Done on 2026-10-31: Q-24 (D-11). The 5 `output_*` flags are for image classes only. The VQA classes keep
    `TypeError`. To fix: `docs/wiki/reference/datasets/index.md:123` says "each dataset" (A-18 of the 2026-10-31 review).
  - Done on 2026-10-31: Q-1 (D-39). `check_dataset` gives ERROR for a box with min not less than max. Fixed on
    2026-11-01.
  - Done on 2026-10-31: Q-2 (D-39). `check_dataset` warns when `output_bbox=True` and no sample has
    `bbox_labels`. Fixed on 2026-11-01, with V-D-29 and V-H3-23.
  - Done on 2026-10-31: Q-21 (D-10). `split_data` raises `ValueError` for a split with no patient. Fixed on
    2026-11-01 in `radharmony/utils/data_utils.py`, with `kfold_splits`.
  - Done on 2026-10-31: Q-20, new entry D-43. A resample keeps the image centered. To fix in
    `_resize_isotropic` (base code, needs approval at the edit).
  - Done on 2026-10-31: Q-22, new entry D-44. "Stable X" means X is present (score 2 with no severity word).
    To fix: the reason in `docs/wiki/explanation/phase-grounding.md:177` (R-docs-18).
  - Done on 2026-10-31: Q-23 (D-21). The local `.ipynb_checkpoints` copies under `docs/` stay. Every check
    skips them. No change.
  - Done on 2026-10-31: Q-25 (D-09). The app lists the split keys only, and every dataset. To fix in the
    `app.py` entries, with A-17. Emory CXR DICOM stays out of the app (Frank Li, 2026-11-01). Fixed on
    2026-11-02: 17 old mixed entries left the app, and `mimic_cxr_jpg_test_chexpert` got an entry.
  - Done on 2026-10-31: Q-3 (D-11). An app VQA row shows only its own question and answer. Fixed in the
    GEMeX VQA app builder on 2026-11-01.
  - Done on 2026-11-01: V-A-38, new entry D-45. Any new entry in a known-problems list fails
    `--check-shrink`. A new rule is named with `--new-rule`. Fixed on 2026-11-01.
  - Done on 2026-11-01 (D-45): in CI, the PR description names each new rule (`New rule: L-013`). The
    shrink step passes the description with `--pr-body`. Fixed on 2026-11-01.
- [ ] **Run the results again that B1, B4, B5 and B8 changed** (when Frank Li asks).
  - EVA-X results made before 2026-10-28 used the CLS token.
  - Backbone recipe results made before 2026-11-01 stretched every image with its minimum and maximum, also
    an 8-bit PNG, and skipped the DICOM window (V-B-12). On 40 VinDr-CXR DICOMs, 8 changed by more than 2
    gray levels. The RAD-DINO feature cosine was 0.98 to 1.0. Delete the old `embedding_cache` files first.
  - CheXFound classification results made before 2026-11-01 used the last CLS token (1024 values). The
    authors join the CLS tokens of the last 4 blocks (4096 values). On 234 CheXpert valid images, the
    macro AUROC went from 0.774 to 0.803 (V-B-13). Delete the old `embedding_cache` files first.
  - SIIM-ACR results counted 624 images 2 to 10 times.
  - CheXpert results had no lateral images.
  - `results_summary.csv` files made before 2026-10-29 mixed the point estimates and the bootstrap rows in
    the mean (V-E-15). Make them again from `results.csv` with `save_results()`. No new run is needed.
  - RANZCR CLiP and CheXlocalize segmentation results counted the images that nobody annotated as "no
    finding" (before 2026-10-29).
  - Training runs with a random augmentation and `num_workers > 0` (fine-tune, segmentation heads) made
    before 2026-10-29 repeated the same augmentation in every worker and epoch (V-D-7).
  - Probe results made before 2026-10-28 fit on all of the train sample and took the threshold from the
    test labels. Their bootstrap drew images, not patients.
  - Ark+ used another input, and SigLIP2 zero-shot used text that was not lowercased.
  - Box results: before 2026-10-29, a sample box did not move with the pad. At 224 pixels, it was up to 22
    pixels off (VinDr-CXR test, 949 images with boxes). RSNA 2024 lumbar boxes were on the mirrored slice in about half of
    the series. VinDr-CXR train boxes before 2026-10-30 were the mean of all boxes of a class (D-36).
  - Zero-shot F1, balanced accuracy and the other threshold metrics before 2026-10-30 used a threshold from
    the test labels (D-29). They are NaN now unless `threshold_strategy="fixed:<value>"`.
  - CheXlocalize segmentation results before 2026-11-01 had no labeled negatives: 499 of 668 test images,
    not 667 (Q-10).
  - PadChest results before 2026-11-01 counted the 777 rows with an empty label list as negative for all
    193 labels (Q-15). RSNA 2024 lumbar results counted 572 blank severity cells as 0 (Q-17).
  - Backbone recipe and `reader="monai"` results on Shenzhen before 2026-10-31. These readers read 334 of
    the 662 PNGs as their raw palette index, so these images were inverted (V-B-11, V-P-19). The RAD-DINO
    probe AUROC on 300 images was 0.923, against 0.972 with the right reader. Any other dataset with palette
    PNGs is affected too. RGBA PNGs (a few in ChestX-ray14) crashed 9 of the 11 encoder recipes. Delete the
    old `embedding_cache` files of these runs first. Since 2026-11-02, a file from before raises
    `ValueError` (D-41).
- [x] **Run `/evaluate --full` again** after the fixes. Done on 2026-11-02:
  `reports/eval/2026-11-02-full/report.md`. 41 old findings are fixed, 71 are still open, and 38 are new
  (1 high).
- [x] **Fix the findings of the 2026-11-02 review.** Done on 2026-11-03. Frank Li asked the agent to decide
  the non-base choices in this round. He approved each of the 17 base-code changes on 2026-11-03.
  - Fixed: about 105 items, by the lead and by 8 fix groups and 4 base groups (branches fix-a1 to fix-g and
    base-b, base-e, base-h, base-t). Rules L-016 to L-021 are built and proved.
  - Skipped with a reason: a decision already covers some items, for example D-05, D-09, D-34, and D-02
    for the VQA `study_ids=` filter. The failure of some items did not show. R-code-23 waits for plan 6b.
  - The agent decisions of this round say "Decided by the agent, for Frank Li to confirm" in design.md.
- [ ] **Run the results again** that this round changed (when Frank Li asks): CheXpert Plus (D-48 labels),
  Ark+ (D-16 projector feature) and CheXFound (V-B-14 input). Delete the old embedding caches first: since
  2026-11-03 a cache file from before raises `ValueError` (V-E-34, D-41).
- [x] **Base-code proposals of the fix agents.** Frank Li decided each one on 2026-11-03.
  - Done: `harmonize()` warns about `sample_n` alone. `preprocess()` and `preprocess_masks()` take
    `self.base_image_dir` when no folder is given (D-04, D-30).
  - Not done (Frank Li): a warning for image-copy arguments given by position (D-30, L-022 covers it).
  - Done: `_fix_missing_highbit` skips a color DICOM (D-13).
  - Done: `BaseVQADataset.get_datasets` and `get_folds` warn about random steps in the val split (D-49).
  - Done: the 2 error texts of `radharmony/evaluator/segmentation/base.py` have no semicolon and no em dash.
- [x] **Outside base code (done on 2026-11-03):** TAIX-Ray and RANZCR `get_datasets_predefined` warn about
  random steps in the val and test splits (D-49). The 4 stub preprocessors of the tests are now one,
  `tests/fixtures/stub_preprocessor.py`.
- [ ] **For Frank Li:** the owner of the MedSeg slice folders writes the new ones (command in the session of 2026-11-03).
  The shared `.venv` has no `cv2` (`uv sync`, needs Frank Li's yes).
- [ ] **Phase-grounding labels against D-44** (found 2026-11-04). The owner of the script: in `extract_mimic_labels.py`, a
  "stable X" sentence must give X the default grade 2, not 0. Then make the labels parquet again. Then remove
  the D-44 "Known gaps (script)" line and the note in `docs/wiki/explanation/phase-grounding.md`.
- [ ] **Data loader messages in a second probe** (D-41 known gap, found 2026-11-04). Reproduce with
  `notebooks/tutorials/evaluator_tutorial.ipynb` (cell 15), then propose a base fix in `extract_embeddings`
  (for example, delete the loader and run gc after the extraction). Base code: ask first.
- [ ] **MAIRA-2 against its model card** (D-16 known gap, found 2026-11-04). Fix and test these 3 points with
  the real model, when someone has the gated weights:
  - the `maira2_gen` pin `transformers>=4.46.0,<4.47` against the card's `>=4.48.0,<4.52`.
  - the card's `processor.convert_output_to_plaintext_or_grounded_sequence()` and `lstrip()` against our `strip()`.
  - `normalize_hypothesis` does not remove the MAIRA-2 box marks of `get_grounding=True`.

## 3. Build D-31

- [ ] **Study and patient tables from the image table** (`to_level()`), with `study_date`, `study_time` and
  `visit_number`. The draft is `docs/proposals/harmonized_df_levels_and_visits.md` (2026-10-04). It
  touches base code, so it needs your yes.

## 4. Later phases of the plan

- [ ] **Code cleanup (plan 6b).** One comment style, and no dead code. Save golden outputs before each
  batch and compare after it. `vulture` is approved and installed. Base files get comment
  changes only, each file with your yes.
  - [x] D-30 (decided on 2026-10-31): move the 45 copied `harmonize()` methods to the base steps. Give the
    10 `_KeepSplitMixin` split classes a `_read_source()` that removes the other splits first. Done on
    2026-11-03 (merges refactor-r1 to r4). Golden: 0 differences on 122 specs.
- [ ] **Rewrite `scripts/evalkit/wiki_audit.py`.** Part 1 is done (2026-11-03). The import and argument
  checks of the code blocks moved to the static level of `run_blocks.py` (L-005). The signature helpers moved
  to `scripts/evalkit/signatures.py`. The audit has 666 lines and gives the same 200 findings. Part 2 is in
  6c (Frank Li, 2026-11-03). One `<!-- args: <class> -->` line above each argument table replaces the guess
  of the class. The new page template replaces the section-title and shared-subsection checks.
- [ ] **Claims (plan, phase 6).** Add claim notes for the counts in the docs, for example "4,555" images in
  PadChest-GR. The docs have 1 claim note (the 127 registry keys in `docs/wiki/reference/datasets/index.md`, B7).
- [ ] **Docs restructure and rewrite (plan 6c).** You approve the new outline first. `mkdocs-redirects` is
  a new package, so it needs your yes. At the end, the known-problems lists are empty.
  - **A conversion page for each dataset, in a new docs tab** (Frank Li, 2026-11-01). Users and developers
    must see how the raw data becomes the RadHarmony format. Each page says:
    - the source files that the harmonizer reads, and how it joins and drops rows
    - how the images are processed (reader, inversion, window, resize, value range)
    - how the targets are made (label codes and uncertain labels, boxes, masks, reports, ids, splits)
    - the `docs/wiki/design.md` entry that sets each step
  - The name of the tab is for the outline. "Recipe" already names a backbone recipe (`make_<name>`), so
    another word, for example "Data conversion", keeps one word for one meaning. The `add-dataset` skill
    (plan 6d) writes this page for each new dataset.
- [ ] **Skills upgrade (plan 6d).** It includes the 3 decisions of 2026-10-28 for `add-dataset`: no change
  before the upgrade, a config review, and `expected_stats`. Still open: new config fields for the official
  splits, age and sex. B7 left these skill findings for it: R-docs-1, R-docs-9, R-docs-15, and the skill
  part of R-docs-4 (`add-dataset/SKILL.md:756-757`: the `cls` order is `sorted(ds.LABEL_COLS)`). Found in the
  6c rewrite (2026-11-04): the skill still lists the old dataset page sections (`add-dataset/SKILL.md:818-821`
  and `:899-903`). Use the 6c dataset page template (`reference/datasets/cxr-labels/chexpert.md`).
- [ ] **Full sync of the public repo with the dev repo** (Frank Li, 2026-11-04, after 6d). Progress on 2026-11-05:
  public/main merged into dev (b73ea6e). `scripts/sync_public.py` makes the snapshot, and
  `docs/internal/syncing.md` has the steps (it replaces `scripts/SYNCING.md`). The snapshot passes all checks as
  a public clone. Left for Frank Li: the archived copy of the public repo, the snapshot commit and the push. The public repo
  gets everything in the dev repo, except a short exclude list: `configs/data_paths.json`, `docs/internal/`,
  the internal Emory checks (`scripts/check_datasets_internal.py`,
  `notebooks/datasets/dataset_integrity_check_internal.ipynb`) and `radharmony_internal/`. Before the sync,
  replace the NAS paths and the user names in `model/`, `notebooks/`, `eval/TODO.md`, `docs/proposals/` and
  `docs/reviews/` with placeholders or `paths(key)` reads. The public `.gitignore` gets
  `configs/data_paths.json`. Update `scripts/SYNCING.md` and CONTRIBUTING for the new model. Frank Li pushes.
  Method (Frank Li, 2026-11-04): first fork the public repo as it is today (v0.1 with its history) to an archived
  repo, for example `f10409/RadHarmony-v0.1`. Then one snapshot commit of the cleaned files on top of the public
  history. The dev history stays private, because its old commits hold the NAS paths.
  This replaces the item below: ruff, the STE linter, the static doc checks, the known-problems lists and CI
  now go with everything else. Never copy `configs/data_paths.json`. Also copy the
  skills of `.claude/skills/` (add-dataset, add-backbone, use-radharmony, evaluate, fix) with
  `scripts/evalkit/` and `eval/RULES.md`, which evaluate and fix need (Frank Li, 2026-11-04). The skills must
  work in a public clone: a step that needs a dev-only file is conditional.
- [x] **Tell users how to install the skills** (Frank Li, 2026-11-04). Done in c53df70: a short section on
  docs/wiki/get-started/install.md and one line in README.md. In a clone, Claude Code loads `.claude/skills/`
  with no step. In another project, copy a skill to `~/.claude/skills/`.
- [ ] **Merge `v0.2-dev` into `main` through a PR.** If base code changed, the PR description starts with
  the base-code section.
