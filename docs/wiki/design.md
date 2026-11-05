# Design decisions

This page records the design choices of RadHarmony and the reason for each choice.
Use it to answer the question "is this a bug, or is it how we want it?".

- **Contributors:** read it before you add a dataset, a backbone or an evaluator.
- **Review agents:** compare the code with these decisions. If no decision covers a behavior and you
  cannot tell if it is a bug, write an open question (see the review briefs). Do not guess the intent.

Related pages: [Concepts](get-started/concepts.md) shows how the layers work. The file `eval/RULES.md` in the
repo has the review rules (J-xx) and the lesson checks (L-xxx). The folder `docs/proposals/` has the
design drafts.

## How to read an entry

Each decision has an ID (D-xx) and these lines:

- **Status:**
  - **Decided** means that a maintainer chose it. The entry names the person and date, or the file that
    records it.
  - **Decided, planned** means that a maintainer chose it, but the feature is not built yet. Do not report
    the missing feature as a bug.
  - **Decided by the agent, for Frank Li to confirm** means that an agent chose its recommended option.
    This was in the fix stage that started on 2026-11-01, and for Q-9 to Q-20 and the draft rules of the
    2026-11-02 review. Frank Li asked for it both times. Treat it as decided until Frank Li confirms or
    changes it.
  - **From the code** means that the code does it today, but no maintainer has confirmed it yet. Treat it
    as the intent until a maintainer confirms or changes it.
  - **Open** means that a maintainer has not decided yet. The entry gives the question, what the code
    does today, the options and the evidence (a command and its output, or a file and line). It has no
    "Known gaps" line. Do not report the behavior as a bug. When a maintainer decides, the status changes,
    and the ID stays the same.
- **Checked by:** the review rule, lesson or test that checks it, or "none".
- **Known gaps:** places where the code breaks the decision today, one line each, with the finding ID. A
  gap is a bug to fix, not a reason to change the decision. An ID (for example V-H1-23) always names the same
  finding, in every run. `reports/eval/index.json` gives its run folder. The index and the folder are local
  only (not in git). "No ID" means
  that a maintainer found the gap by reading the code, and no report has it yet. Some gaps have more
  details in `eval/TODO.md`, section 2. `/fix` removes a gap when it fixes it. A gap whose fix needs a
  base-code change names its strict xfail test, if one exists.
- **Open part:** a detail of a decided entry that is not decided yet. It gives the question, an example
  and the options.
- **Notes:** facts that help, but that are not gaps.
- **Related:** other entries about the same topic.

An ID never changes. A new decision gets the next free D- number, and it goes in the section where it
belongs. So the numbers in a section are not in order. Use the [index](#index) to find an ID.

Words on this page:

- **Table:** the `harmonized_df` that a harmonizer makes.
- **Harmonizer:** a class that reads the source files and makes the table.
- **Dataset class:** a class that reads the table and gives samples (`ds[i]`).
- **Official split:** a train, validation or test set that the dataset authors provide.
- **Bootstrap sample:** one draw, with replacement, from the test set.
- **Open question:** a behavior that no decision covers. A review agent writes it in its report (Q- ID). A
  maintainer decides it, or it becomes an entry with the status "Open".
- **Open part:** a detail of a decided entry that is not decided yet. For example, D-29 had one until
  2026-10-30: which threshold zero-shot uses.

## Index

The "Was" column gives the old O- ID of the 10 questions that were decided on 2026-10-28. A new open
question gets no O- ID. It gets a D- ID with the status "Open".

| ID | Title | Status | Was |
|---|---|---|---|
| D-01 | One table is the contract | From the code | |
| D-02 | Harmonize first, then filter | From the code | |
| D-03 | One row per image | From the code | |
| D-04 | Image paths are relative to `base_image_dir` | From the code | |
| D-05 | Labels | From the code | |
| D-06 | Age in years | Decided | |
| D-07 | Box format | From the code | |
| D-08 | Save and load give the same table | From the code | |
| D-09 | Official splits are separate classes | Decided | O-2 |
| D-10 | Our own splits group by patient | From the code | |
| D-11 | Output flags | From the code | |
| D-12 | 3-D orientation | Decided | |
| D-13 | CXR pixel values | From the code | |
| D-14 | Pixel value range | From the code | |
| D-15 | Cache before augmentation | From the code | |
| D-16 | Backbones match their authors | From the code | |
| D-17 | Frozen-feature probes L2-normalize features | Decided | |
| D-18 | Test data stays out of training | From the code | |
| D-19 | Variance of a result | From the code | |
| D-20 | Base code needs approval | Decided | |
| D-21 | One file for data paths | Decided | |
| D-22 | Docs do not hide code bugs | Decided | |
| D-23 | Tests get their expected values from the source | Decided | |
| D-24 | `harmonized_df` equals the last `harmonize()` result | Decided | O-1 |
| D-25 | Images with no labels go in a separate test class | Decided | O-2 |
| D-26 | One `sex` column | Decided | O-3 |
| D-27 | A reader never clips pixel values | Decided | O-4 |
| D-28 | The box in a sample matches the image in that sample | Decided | O-5 |
| D-29 | The threshold comes from validation data | Decided | O-6 |
| D-30 | A harmonizer changes steps, not `harmonize()` | Decided | O-7 |
| D-31 | Study and patient tables come from the image table | Decided, planned | O-8 |
| D-32 | The bootstrap draws patients | Decided | O-9 |
| D-33 | An empty `mask_path` means "not annotated" | Decided | O-10 |
| D-34 | Official splits stay as published, also when they share patients | Decided | |
| D-35 | `infer_path` looks only in exact places, and missing data raises `FileNotFoundError` | Decided | |
| D-36 | Boxes of several readers are grouped by overlap | Decided | |
| D-37 | A warp fills the empty border with black | Decided | |
| D-38 | A report section falls back to the other section | Decided | |
| D-39 | `check_dataset`: wrong values are ERROR, missing outputs are WARN | Decided | |
| D-40 | `n_train` counts images or patients | Decided, planned | |
| D-41 | The embedding cache holds features only | Decided | |
| D-42 | One channel count for each dataset | Decided | |
| D-43 | A resample keeps the image centered | Decided | |
| D-44 | A "stable" finding is present | Decided | |
| D-45 | A known-problems list only gets shorter | Decided | |
| D-46 | A prediction names its image by `image_path` | Decided | |
| D-47 | Fine-tune `loss="ce"` needs 2 or more label columns | Decided | |
| D-48 | CheXpert Plus labels come from `impression_fixed.json` | Decided | |
| D-49 | A val split with random steps warns | Decided | |
| D-50 | Both report scoring paths clean the generated text the same way | Decided | |
| D-51 | A MIMIC-Ext-CXR-QBA answer holds the main answer only | Decided by the agent, for Frank Li to confirm | |
| D-52 | An evaluator argument that the class does not use warns, and one it cannot take raises at once | Decided | |

## Table

### D-01 One table is the contract

`harmonize()` returns the table as one pandas `DataFrame`. After `harmonize()`, no class reads the source
CSV or JSON files again. The dataset classes, the app and the evaluators read only the table.

- Required columns: `patient_id`, `study_id`, `image_path`.
- Optional columns: `series_id`, `view_position`, `mask_path`, `bbox`, `bbox_labels`, `report`, `age_years`,
  `sex`, the label columns, the regression columns (`REG_COLS`) and the extra columns (`EXTRA_OUTPUT_COLS`).
- Planned optional columns (D-31): `study_date`, `study_time` and `visit_number`. They are NaN when the
  source has no value.

Why: one format lets a dataset class, a probe or the app work with every dataset.

- Status: From the code (`_select_harmonized_columns` in `radharmony/harmonizer/base.py`).
- Checked by: J-02.
- Source IDs: `study_id` keeps the study ID of the source, also when the source gives one ID to 2
  patients. The harmonizer then warns. Code that groups by study uses `(patient_id, study_id)`, as D-31
  does. Example: BRAX keeps
  the `AccessionNumber`, and 2 of its 23,276 values belong to 2 patients each. Decided (Frank Li,
  2026-10-30, V-H1-9). The other option was the BRAX `Study_<UID>` folder name (23,278 values, 1 patient
  each). PadChest also warns since 2026-11-03: 1 `StudyID` belongs to 2 patients (V-H2-32). Both use
  `warn_shared_study_ids` in `radharmony/harmonizer/_ids.py`. Checked by: `tests/test_padchest.py`.
- MedPMC `answer`: the release index has no caption, so the caption comes from our tag table
  (`scripts/build_medpmc_tags.py`). With no tag table, `answer` is None in every row, and the harmonizer
  warns. The images still work. Decided (Frank Li, 2026-10-30, V-H1-27). The other options were an error,
  and an error with a new argument to turn it off. Checked by
  `tests/test_harmonizer_data.py::test_medpmc_without_tag_table_warns_that_answer_is_empty`.
- MedPMC `include_multipanel=False`: a specialty filter drops the multipanel rows, because their tag comes
  from the caption of the whole figure. Without a specialty filter, the flag does nothing. Example: the
  multipanel radiology rows go from 513,709 to 0. Before 2026-11-03, the flag did nothing, because the tag
  table has no `contains_specialty` column. Decided by the agent, for Frank Li to confirm (2026-11-03,
  V-H1-47). The other options were to drop the flag, and to drop every multipanel row also with no
  specialty filter. Checked by: `tests/test_medpmc.py::test_include_multipanel_false_gives_a_panel_level_set`.
- MIMIC IDs: every harmonizer on MIMIC images writes `patient_id` and `study_id` as digits, the official
  MIMIC IDs (for example `10003502`, not `p10003502`). So a join across these datasets works. Decided (Frank
  Li, 2026-10-31, Q-12 of the 2026-10-31 review). The other option was the source form of each dataset, on
  its wiki page. Since 2026-11-01, GEMeX VQA and MIMIC-Ext-CXR-QBA write digits (Q-12). The `study_ids=` and
  `patient_ids=` filters of MIMIC-Ext-CXR-QBA accept both forms. Checked by:
  `tests/test_harmonizer_id_filter.py::test_mimic_ext_qba_ids_are_digits_and_question_id_names_one_row` and
  `::test_gemex_vqa_ids_are_digits`.
- VQA row id: the `question_id` of MIMIC-Ext-CXR-QBA is `<study_id>_<source question_id>`, because the source
  id is a template name that repeats in every study (V-H1-30). Example: `50037760_D01_describe_finding_015`.
  Decided (Frank Li confirmed on 2026-11-02 the choice of the agent of 2026-11-01). The other option was the
  source id with a separate row number.
- VQA row id of CheXinstruct VQA: `unique_id` of the compiled JSON. The MIMIC-CXR-VQA processor shuffles the
  train questions with no seed, so `scripts/compile_chexinstruct_vqa.py` seeds `random` (`--seed`, default 0).
  2 runs with the same seed then give the same file. The JSON on the NAS came from a run with no seed, and it
  stays. A new run gives the same 701,458 questions, answers, images and splits. Only the `unique_id` and the
  order of the MIMIC-CXR-VQA train rows change, and the `qa_pair` of the closed questions. The harmonizer does
  not read `qa_pair`. Decided (Frank Li,
  2026-11-04). The other options were a seed and a new NAS file, and no seed. Checked by:
  `tests/test_scripts.py::test_compile_chexinstruct_vqa_seed_gives_the_same_file`.
- 3-D `view_position`: the name of the nearest plane ("axial", "coronal" or "sagittal"), from the DICOM
  `ImageOrientationPatient`. The vector goes in the extra column `image_orientation`. Example: CT-RATE.
  Decided (Frank Li, 2026-10-30, V-H1-28). The other options were the vector only, and the plane name with
  no vector column.
- Checked by (3-D `view_position`): `tests/test_harmonizer_data.py::test_ct_rate_view_position_is_the_nearest_plane`
  and `::test_ct_rate_orientation_count_on_nas`.
- Known gaps (3-D `view_position`): 3 harmonizers do not read `ImageOrientationPatient`, so their tables have no
  `view_position`. They are RSNA 2022 cervical spine, RSNA 2023 abdominal trauma and RSNA PE detection. The
  DICOM headers have the tag. Found in the 6c rewrite of the wiki pages (2026-11-04). The fix is in their own
  harmonizers, not in base code.
- RSNA 2024 lumbar spine follows the 3-D rule: `view_position` is "axial" or "sagittal". The series
  description ("Axial T2", "Sagittal T1", "Sagittal T2/STIR") goes to the column `series_description`.
  `series_filter` reads that column, with the same values as before. Decided (Frank Li, 2026-10-31, Q-18 of
  the 2026-10-31 review), with a note from Frank Li: revisit this later. The other option was an exception for
  datasets with a description. Known gap: `view_position` holds the description today (Q-18). The
  description agrees with the orientation in 300 of 300 random series.
- 2-D `view_position`: one shared name list, as for `sex` (D-26). The names are `PA`, `AP` and `LATERAL`, and
  NaN means unknown (for example a blank, `Unknown`, `UNK`, `N/A` or `EXCLUDE`). A source with AP or PA gives
  AP or PA, not `Frontal`. Any other source value (for example an oblique view or `AP_horizontal`) keeps its
  raw value in upper case. One base step does the mapping. Decided (Frank Li, 2026-10-31, Q-13 of the
  2026-10-31 review). The other option was the raw values of each source, listed on each wiki page.
- 2-D `view_position` step: `_build_view_position_names` in `radharmony/harmonizer/base.py` (base code,
  approved by Frank Li on 2026-11-02). `L`, `LL`, `RL` and `LAT` give `LATERAL`, `POSTERO_ANTERIOR` gives
  `PA`, `ANTERO_POSTERIOR` gives `AP`, and the text `None` gives NaN. A class with `VIEW_POSITION_NAMES = False` keeps its own names: CT-RATE and RSNA 2024 lumbar (3-D
  plane or series names) and Emory CXR DICOM. Checked by: `tests/test_view_position.py`. On 2026-11-02 the real
  counts matched the source, for example BRAX `LATERAL` = 9,052 `L` + 1,340 `RL`. A table saved before
  2026-11-02 keeps its old names until `harmonize()` runs again.
- Notes (2-D `view_position` step): since 2026-11-03, base code approved by Frank Li, `ANTERO_POSTERIOR`
  gives `AP` and the text `None` gives NaN (V-H2-43). Before, they gave `ANTERO_POSTERIOR` and `NONE`. The
  ReXGradient train JSON has 1 `ANTERO_POSTERIOR` view and 2 text `None` views. Checked by:
  `tests/test_view_position.py::test_unknown_strings_give_nan_and_an_opt_out_class_keeps_them`.
- CheXpert, CheXpert Plus and CheXphoto `view_position`: a frontal image with `LL` or `RL` in `AP/PA` gives
  NaN, because AP or PA is unknown. CheXpert train has 17 such images. This follows the Emory rule of Q-14.
  Decided (Frank Li confirmed on 2026-11-02 the choice of the agent). The other options were `LATERAL` from
  the shared table, and the source value `LL` or `RL`.
- Known gaps (2-D `view_position`): these harmonizers write `FRONTAL` where the source has AP or PA (Q-13).
  CheXlocalize val has AP or PA in its CSV for 202 images. SIIM-ACR PTX writes `Frontal` for all images,
  but 200 of 200 sampled DICOM files have `ViewPosition` PA or AP. The fixes are in their own harmonizers,
  not in base code.
- MIMIC-CXR (DICOM) `view_position`: only with `dicom_base_dir`. The harmonizer then reads the DICOM tag
  `ViewPosition` of each file, which is slow (377,110 files for all splits). Without it, the harmonized table
  has no `view_position`. Decided (Frank Li, 2026-11-03, found in the 6c rewrite of the wiki page). The other
  options were to read the DICOM files in `base_image_dir`, and to take the view from the MIMIC-CXR-JPG
  metadata file. Before, the docstring said that `dicom_base_dir` falls back to `base_image_dir`, but the code
  never did. Checked by: `tests/test_mimic_cxr.py::test_no_dicom_base_dir_gives_no_view_position`.
- MIMIC-Ext-CXR-QBA `view_position`: `img.view_position` of the question's image in `image_metadata.parquet`,
  with the shared names. With no `image_metadata.parquet`, the harmonizer warns and the table has no
  `view_position`. Decided (Frank Li, 2026-11-04, found in the
  6c rewrite of the wiki page). The other option was no change. In q1M of `A_frontal`, the table and the
  source both give AP for 613,324 questions and PA for 386,676. Checked by:
  `tests/test_mimic_ext_cxr_qba.py::test_view_position_comes_from_the_image_metadata` and
  `::test_no_image_metadata_warns_and_gives_no_view_position`.
- Notes (OpenI view): since 2026-11-02, OpenI corrects the figure number with the DICOM tag (V-H2-40). The
  tag wins when it is AP or a lateral tag. A PA tag counts only on F1, because many lateral images also have
  PA. The 1,313 F1 images with no tag stay `FRONTAL`. Checked by:
  `tests/test_harmonizer_data.py::test_openi_view_takes_the_dicom_tag_when_it_names_ap_or_a_lateral` and
  `::test_openi_view_reads_the_dicom_tag`.
- 2-D `view_position` from the DICOM tag: a release can have no view column. Then the harmonizer reads the
  DICOM `ViewPosition` tag, as D-06 does for age and sex. Example: RSNA Pneumonia Kaggle has no view column,
  but in 300 of its DICOM files the tag is PA (166) or AP (134). Decided by the agent, for Frank Li to
  confirm (2026-11-02, Q-18 of the 2026-11-02 review). The other options were a view from a CSV only, and
  the tag only for datasets that a benchmark filters by view. Since 2026-11-03, RSNA Pneumonia Kaggle train
  and test read the tag. They read it with the sex and the age, in one header pass (`extra_tags` of
  `add_patient_tags` in `radharmony/harmonizer/_dicom_tags.py`). A real run gave train PA 14,511 and AP 12,173 (26,684 images),
  and test PA 1,618 and AP 1,382. Checked by:
  `tests/test_rsna_pneumonia_kaggle.py::test_train_view_position_comes_from_the_dicom_tag` and
  `::test_test_view_position_comes_from_the_dicom_tag`.
- Size columns: the BRAX extra columns `rows` and `columns` give the DICOM size from the source metadata,
  also in the PNG variant. Example: 2752 x 2232 in the table, but the PNG is 1261 x 1024. The BRAX wiki
  page says so. Decided by the agent, for Frank Li to confirm (2026-11-02, Q-10 of the 2026-11-02 review).
  The other options were the names `dicom_rows` and `dicom_columns`, and the PNG size in the PNG variant.
  Notes: since 2026-11-03, `docs/wiki/reference/datasets/cxr-labels/brax.md` says it (Q-10).
- Emory CXR `view_position`: when the `Frontal` column and the DICOM `ViewPosition` tag disagree, `Frontal`
  wins, because it is more accurate (Frank Li). `Frontal` = 0 gives `LATERAL`. `Frontal` = 1 with
  `ViewPosition` Lateral gives NaN, because AP or PA is unknown. Decided (Frank Li, 2026-10-31, Q-14 of the
  2026-10-31 review). The other options were NaN for every conflict, and both columns as in the source.
  Exception: `Frontal` = 0 with `ViewPosition` AP (2,202 images) gives NaN. In 24 random images of this
  group, 9 are upright frontal and 11 are decubitus (AP with the patient on the side). 3 are lateral, and 1
  is unclear. So `LATERAL` is wrong for most of them. Decided (Frank Li, 2026-11-01). The other options were
  AP and `LATERAL`.
  Since the Q-13 step (2026-11-02), the lateral name is `LATERAL`. The rule is for the PNG classes only.
  Emory CXR DICOM keeps `ViewPosition` as in the source (Frank Li, 2026-11-01).
  Checked by: `tests/test_emory_cxr.py::test_view_position_follows_frontal`.
  Notes: a sample of 4 images for each conflict group on 2026-10-31. Frontal=0 with PA: 4 lateral.
  Frontal=1 with Lateral: 4 frontal. Before 2026-11-01, `view_position` came from `ViewPosition` only, so
  a PA filter kept 45,687 lateral images (Q-14).
- Notes (CT-RATE): a real run on 2026-10-30 checked the 47,149 train volumes. 47,111 are axial, 14 coronal
  and 12 sagittal. 12 more are tilted more than 8° from a plane (5 near axial, 5 near coronal, 2 near
  sagittal). All 3,039 volumes of the validation split are axial.

### D-02 Harmonize first, then filter

Run `harmonize()` on the full dataset. Then filter the table with pandas. The `study_ids=` and
`patient_ids=` arguments only make `harmonize()` faster on a big dataset. They must give the same rows
as a filter after `harmonize()`.

<!-- block: run=skip reason="example only, h is any harmonizer" -->
```python
df = h.harmonize()
df = df[df["view_position"] == "PA"]
```

Why: a filter on an image property (view, mask present) must see every image row.

- Status: From the code (`BaseHarmonizer.harmonize` docstring).
- Checked by: `tests/test_harmonizer_id_filter.py::test_study_ids_filter_matches_filter_after`,
  `::test_mimic_ext_qba_id_filter_matches_filter_after`,
  `tests/test_siim_acr_ptx.py::test_id_filter_gives_the_rows_of_a_filter_after`, L-008 and L-014.
- No matching study: a study id that no row has gives 0 rows with the columns of the full table, not an
  error. A split class gets such an id when the study is in another split. Checked by:
  `tests/test_official_splits.py::test_a_study_of_no_row_gives_an_empty_table` and the `no_study` check of
  `tests/fixtures/harmonizer_contract.py` (L-014).
- Notes (no matching study): before 2026-11-01, CheXlocalize, CheXpert Plus and MIMIC-CXR-JPG raised an
  error (V-H1-38).
- VQA harmonizers: only `MIMICExtCXRQBAHarmonizer` gets `study_ids=` and `patient_ids=`, before its slow
  `qa.zip` step. The other VQA harmonizers have no filter. Filter their table with pandas after
  `harmonize()`. Decided (Frank Li, 2026-10-30), option (c). The other options were (a) a filter for all 7
  VQA harmonizers (a base-code change in `base_vqa.py`) and (b) no VQA filter. Findings: V-H2-11, V-H1-16
  and V-H3-10.
- Notes: a real run on 2026-10-29 timed the full VQA `harmonize()`. VQA-RAD train took 0.0 s, CheXinstruct
  VQA train 6.0 s and MedPMC (11.1 million rows) 16.6 s. MIMIC-Ext-CXR-QBA (208,316 studies) did not finish
  in 25 minutes, because it opens 1 file in `qa.zip` for each study. With `max_studies=1000` it took 10.3 s.
  With `study_ids=` of 3 studies, it took 2.8 s (since 2026-10-30).
- A filter argument of a dataset class also filters a given table (`harmonized_df=`, `harmonizer=` or
  `harmonizer_path=`). Example: RSNA 2024 lumbar `series_filter`, in the train and the test class. Before
  2026-11-03, the test class kept all series of a given table (V-D-32). Checked by:
  `tests/test_rsna_2024_lumbar_spine.py::test_series_filter_applies_to_a_given_table`.
- VQA and caption classes: since 2026-11-04, the filters of ROCO, ROCOv2, MedPMC and CheXinstruct VQA also
  filter a given table. Each harmonizer has one `filter_table()` that its `harmonize()` and the dataset class
  both use. Before, a given table kept every row: `ROCOv2TestDataset(modalities=["ct"])` gave 3,615 rows
  from the source files, but 9,927 rows with `harmonized_df=`. A split class with a given table keeps only
  its split, when the table has a `split` column. A MedPMC filter whose column the given table does not have
  raises `ValueError`. Arguments that only change how `harmonize()` makes a column (ROCOv2
  `manual_concepts_only`, MedPMC `tags_path`) do not change a given table. Checked by:
  `tests/test_vqa_filters.py`.
- ROCO `radiology_only`: the dataset default is `None`. `None` gives `True` when the class harmonizes. On a
  given table, `True` keeps only the `radiology` rows, and `None` and `False` keep every row. Decided (Frank Li,
  2026-11-04). The other options were the default `True` that also filters a given table, and no filter on a
  given table at all. With the default `True`, a saved table with `radiology_only=False` loses its
  non-radiology rows.
- Local lists that are not in the release: a harmonizer does not use them, and the table keeps their rows.
  The dataset wiki page names the files, so a user can filter with them. Example: the CT-RATE folder has
  `tables/no_chest_train.txt` (752 volumes) and `tables/no_chest_valid.txt` (37 volumes). Decided by the
  agent, for Frank Li to confirm (2026-11-02, Q-13 of the 2026-11-02 review). The other options were to drop
  these volumes by default, and a flag column such as `has_chest`. Since 2026-11-03, the CT-RATE wiki page
  names the files and shows a filter. On the valid split, the filter removes the 37 listed volumes of 3,039.
- Missing image files: a harmonizer keeps a row whose image file is not on disk. When it can check the disk
  quickly, it warns with the count of its own table, and the user calls `verify_images()` to drop the rows.
  Example: the NAS copy of CheXpert Plus has no folder for 12,471 of 223,228 train rows and 5 of 234 valid
  rows. The same table then comes from every copy of a release. Decided by the agent, for Frank Li to
  confirm (2026-11-03, V-H1-7, V-D-17). The other option was to drop these rows in `harmonize()`. Checked by:
  `tests/test_chexpert_plus.py::test_missing_patient_folders_warn` and
  `::test_a_split_class_counts_the_missing_folders_of_its_own_split`.
- Notes (missing image files): before 2026-11-03, a CheXpert Plus split class counted the rows of both
  splits. Example: 12,476 of 223,462 rows for valid (V-H1-45).
- Related: D-31.

### D-03 One row per image

A harmonizer returns one row per image. A harmonizer can return one row per study, series or volume
instead. In that case, its docstring and its wiki page say so.

Some source files have one line per box or per mask region (for example SIIM-ACR). The harmonizer's own
code joins these lines into one line per image. The base merges raise a `ValueError` when a source file has
more than one line for a key. This applies to labels, view position, masks, boxes and reports.

Why: an image row keeps every image property, and a study or patient table can be built from it.

- Status: From the code. The rule for a repeated key: Decided (Frank Li, 2026-10-29, V-H2-17).
- Checked by: J-02, `tests/test_harmonizer_merge.py` (also `::test_a_repeated_report_key_raises`), and
  `tests/test_dataset_stats.py::test_one_row_per_image` for the datasets with `expected_stats`.
- Notes (report merge): since 2026-11-03, base code approved by Frank Li, the base `_build_report()` also
  raises on a repeated key (V-H2-31). Before, 2 report lines for one study gave 2 rows for each image of
  the study. Only MIMIC-CXR uses this step, and its report list has 0 repeated keys in 227,835 lines.
- Emory CXR: its own code reads the report and label CSVs, which can have 2 different lines for one study. A
  blank report template (only "REPORT", spaces and underscores) counts as no report. When a study has
  another row, the code drops the blank report row and a label row with only `No Finding = 1`. For the
  other conflicts, the first row stays, with a warning. Decided (Frank Li, 2026-10-30, V-H1-12). The other
  options were the first row with a warning, and NaN for every conflict.
- An image of the source list with no annotation file: the harmonizer leaves it out of the table and warns
  with the count. Example: 2 images of the Chest ImaGenome silver `valid.csv` have no scene graph file.
  Decided by the agent, for Frank Li to confirm (2026-11-03, V-H1-37). The other options were a row with
  no boxes, and an error. Checked by:
  `tests/test_chest_imagenome.py::test_silver_warns_about_images_with_no_scene_graph`.
- Notes (Chest ImaGenome): since 2026-11-03, the same rule holds for an image with no DICOM header with `Rows`
  and `Columns` in `base_image_dir`. A box needs the image size (D-07), so the harmonizer leaves the image out
  and warns with the count. Before, it gave no warning. Checked by:
  `tests/test_chest_imagenome.py::test_gold_warns_about_an_image_with_no_readable_size` and
  `::test_silver_warns_about_an_image_with_no_readable_size`.
- Checked by (Emory): `tests/test_harmonizer_data.py::test_emory_prefers_the_real_row_of_a_study` and
  `::test_emory_conflicting_duplicate_rows_on_nas` (8 label and 17 report conflicts left on 2026-10-30).
- Notes (Emory): a real run on 2026-10-30 found 151 studies with 2 report rows. 134 had a blank template
  and a real report, and 17 had 2 real reports. 75 studies had 2 label rows. 67 of them had a "No Finding
  only" row, and 65 of these also had a blank template. The first row was the blank template for 68 reports,
  and the "No Finding only" row for 32 label rows.
- Related: D-31.

### D-31 Study and patient tables come from the image table

The image table stays the only source. One function builds the other levels from it:

<!-- block: run=skip reason="to_level is not built yet (D-31)" -->
```python
df = h.harmonize()                      # image level
df = df[df["view_position"] == "PA"]    # filter at image level first
study_df = to_level(df, "study")        # one row per study
patient_df = to_level(df, "patient")    # one row per patient
```

`harmonize()` does not take a `level=` argument. Every table also gets the columns `study_date`,
`study_time` and `visit_number` (D-01). The rules for each column are in
`docs/proposals/harmonized_df_levels_and_visits.md`.

Why: a filter on an image property must run before the grouping (D-02). One function works for every
harmonizer and for saved and filtered tables.

- Status: Decided, planned (Frank Li, 2026-10-28). We build it after the new evaluation pipeline is done
  and the code findings are fixed.
- Checked by: none.
- Related: D-01, D-02, D-03.

### D-04 Image paths are relative to `base_image_dir`

`image_path` is relative to `base_image_dir`. `base_image_dir` is the deepest folder that is the same for
every image. For example, for `/data/cxr14/images_001/a.png` and `/data/cxr14/images_002/b.png`,
`base_image_dir` is `/data/cxr14` and `image_path` is `images_001/a.png`.

When the dataset authors provide separate train and test folders, each official split class takes its own
folder. For example, VinDr-CXR uses `train/` and `test/`.

A dataset with images from several sources keeps `image_path` relative to the folder of its source, with a
`data_source` column. The dataset class joins the folder of each source when it loads (for example
`source_image_dirs` of CheXinstruct VQA). `image_path` holds a relative path, with one exception: the
copy step (see "Exception (copy step)" below). A `mask_path` that a harmonizer writes is also absolute (see
the last line of this entry).

Why: the user gives one folder, and the same table works on any computer.

- Status: From the code (`.claude/skills/add-dataset/SKILL.md`, "`base_image_dir` Convention"). The rule for
  several sources: Decided (Frank Li, 2026-10-31, Q-11 of the 2026-10-31 review). The other option was
  absolute paths for CheXinstruct VQA, as an exception.
- Checked by: the harmonizer contract (L-008, since 2026-11-03), on 11 fake datasets in CI and on the
  datasets with `expected_stats` in the full run. The proof that the check fails on an absolute path is
  `tests/test_harmonizer_contract.py::test_contract_finds_an_absolute_image_path`. Also
  `tests/test_harmonizer_data.py::test_ms_cxr_t_base_image_dir_is_the_files_folder` (MS-CXR-T only),
  `tests/test_save_load.py::test_verify_images_needs_the_image_folder_for_relative_paths` and
  `::test_series_verify_images_follows_d04` (the 4 3-D harmonizers whose `image_path` is a series folder),
  and `tests/test_vqa_base.py::test_verify_images_needs_the_image_folder_for_relative_paths` (VQA).
- Notes (CheXinstruct VQA): before 2026-11-01, it wrote an absolute `image_path` in all 21,296 test rows, and
  `base_image_dir` was None (Q-11). Checked by: `tests/test_official_splits.py::test_chexinstruct_image_path_is_relative`
  and `::test_chexinstruct_dataset_joins_the_folder_of_each_source`. A table or harmonizer that an older
  version saved has absolute paths. The dataset class makes them relative to the folder first
  (`::test_chexinstruct_dataset_loads_an_old_table_and_a_harmonizer_object`).
- Exception (copy step): after `harmonize(preprocess_output_dir=...)`, `image_path` is the path of the copy
  below `preprocess_output_dir`. It is absolute when that folder is absolute, for example
  `/data/out/a.png.nii`. Base code approved by Frank Li on 2026-11-03: with no `base_image_dir` in the call,
  the copy step reads from `h.base_image_dir`, as `verify_images()` does. Before,
  `h.harmonize(preprocess_output_dir=..., preprocessor=...)` read from the current folder and raised
  "N of N images failed". Keeping the copy paths relative was the other option: rejected because it needs a
  new attribute for the folder of the copies. The mask step (`harmonize(mask_output_dir=...)`) uses the same
  default since 2026-11-03 (base code approved by Frank Li). Checked by:
  `tests/test_harmonizer_preprocess.py::test_copy_step_reads_from_the_folder_of_the_harmonizer` and
  `::test_mask_step_reads_from_the_folder_of_the_harmonizer`.
- Notes: since 2026-10-29 (approved by Frank Li), the base docstrings say that `image_path` is relative.
  `verify_images()` takes `self.base_image_dir` when the harmonizer has it. With relative paths and no
  folder, it raises a `ValueError`. Before, it dropped every row, for example 138 of 138 Montgomery rows
  (V-H2-23, V-H1-22). Since 2026-11-01, the `verify_images()` overrides of RSNA 2024 lumbar, RSNA PE,
  RSNA 2023 abdominal trauma and RSNA 2022 cervical follow the same rule. Before, lumbar dropped all
  6,294 rows although `base_image_dir` was set (V-H3-22). Since 2026-11-02 (base code approved by Frank
  Li), `BaseVQAHarmonizer.verify_images()` follows it too. Before, the ROCOv2 harmonizer had no folder,
  so a default call dropped every row (V-H2-28). Since 2026-11-03, Shenzhen, VinDr-PCXR and SIIM-ACR PTX
  use their own folder argument (`base_dir` or `dicom_dir`) as the default. Before, a default call raised
  the `ValueError` (V-H3-36). Checked by: `tests/test_verify_images.py`.
- Dataset classes: since 2026-11-04 (base code approved by Frank Li), `verify_images()` of a dataset class
  follows the same rule. `BaseRadiologicalDataset`, `BaseVQADataset` and all 7 overrides call one helper,
  `_require_image_folder` in `radharmony/dataset/base.py`. Before, a dataset class with relative paths and
  no `base_image_dir` dropped every row. Example: `CheXpertDataset(harmonized_df=df)`. CheXinstruct VQA with
  several `source_image_dirs` has no single folder, so its own `verify_images()` joins the folder of the
  `data_source` of each row. Before, it dropped every row. Checked by: `tests/test_verify_images.py`.
- RAD-ChestCT folder name: the harmonizer takes `base_image_dir=`, as the other harmonizers and
  `configs/data_paths.json` do. The old name `image_base_dir=` still works, and 2 different folders raise
  `ValueError`. Before 2026-11-03, `image_base_dir=` was the only name, so `verify_images()` with defaults
  raised `ValueError` (V-H2-33). Decided by the agent, for Frank Li to confirm (2026-11-03, V-H2-33). The
  other option was an override of `verify_images()` that reads `image_base_dir`. Checked by:
  `tests/test_radchestct.py`.
- Exception (mask files that a harmonizer writes): their `mask_path` is absolute. The harmonizer writes them to an output
  folder that the user picks, and the dataset class joins `base_image_dir` to `image_path` only. This is the
  case for Montgomery, Shenzhen, QaTa-COV19 and MedSeg COVID CT. Decided by the agent, for Frank Li to
  confirm (2026-11-03, V-H1-50). The other option was a path relative to `base_image_dir`, with a change of
  the join step and of the 4 harmonizers. A move of the output folder needs a new `harmonize()`.

### D-35 `infer_path` looks only in exact places, and missing data raises `FileNotFoundError`

When the user does not give a source file, `infer_path` (`radharmony/utils/infer.py`) looks for it only in
these places, in this order:

1. The path that the user gives. If the user gives a path that does not exist, it raises
   `FileNotFoundError` with that path.
2. `base_dir/<name>`.
3. `parent/<name>`, where `parent` is the folder above `base_dir`.

It does not walk folders. A dataset class whose file is in another folder puts the sub folder in the name.
Example: VinDr-CXR looks for `annotations/image_labels_train.csv` in `parent`.

Why: a folder walk can take a file of another dataset with the same name. On 2026-10-29, the CheXphoto
classes took `ranzcr-clip-catheter-line-classification/train.csv`, because the CheXphoto folder could not
be read. A folder walk can also search all of `/` when a folder is missing, so 2 doc blocks timed out.

- Status: Decided (Frank Li, 2026-10-29), option (a). Frank Li chose it over (b), a shorter folder walk,
  and (c), no change. Findings: V-A-19, V-P-13 and V-P-8.
- Checked by: `tests/test_official_splits.py::test_infer_path_candidate_order_and_sub_path`,
  `::test_infer_path_looks_only_in_exact_places`, `::test_infer_path_raises_for_a_user_path_that_does_not_exist`,
  `::test_infer_path_required_names_the_file_and_the_places`, `::test_a_class_with_no_data_names_the_missing_file`,
  `::test_a_class_names_its_second_missing_file`, `::test_a_test_split_class_does_not_need_the_train_files`,
  `::test_a_test_class_names_the_argument_of_a_wrong_path`, `::test_an_asked_output_needs_its_file`,
  `::test_ranzcr_mask_without_the_annotation_file_names_it`,
  `::test_mimic_cxr_metadata_file_is_not_the_record_list`,
  `::test_chexpert_plus_finds_its_files_2_folders_up_and_takes_impression_fixed`,
  `::test_a_class_with_an_empty_image_folder_names_the_folder` and
  `::test_roco_with_no_split_folder_raises_and_a_missing_one_warns`. Also
  `tests/test_roco.py::test_roco_split_with_no_captions_file_warns_and_names_it` and
  `::test_roco_with_no_captions_file_at_all_raises` (V-H2-45).
- Missing file: when a class does not find a required file, it raises `FileNotFoundError`. The message
  names the file and the places it looked, for example `train_image_level.csv is not in <base_dir> or
  <parent>. Pass image_csv_path=<path>.` Then the doc block runner reports "blocked, needs data". Decided
  (Frank Li, 2026-10-30). The other options were the old `ValueError`s with a new runner rule, and no change.
- File for one output: a file that a class reads for only one output is required when the user asks for that
  output. Example: MIMIC-CXR with `output_cls=True` and no `mimic-cxr-2.0.0-chexpert.csv` raises
  `FileNotFoundError`. Before, the sample had no `cls` key and no warning. Decided (Frank Li, 2026-10-31). The
  other options were a warning and no key, and no change.
- Empty image folder: a class with no index file (a CSV or JSON that lists the images) lists the images in its
  folder. It raises `FileNotFoundError` when it finds no image, and the message names the folder and the file
  type. Example: `No <study>/<series>/*.dcm file in <dir>.` `montgomery_cxr`, `siim_acr_ptx_test` and
  `siim_covid19_test` got this on 2026-10-31. Before, they crashed with an `AttributeError` or a `KeyError`.
  `shenzhen_cxr`, `qata_cov19`, `ranzcr_clip_test` and `rsna_bone_age_test` already did it. Since 2026-11-03,
  `rsna_pneumonia_kaggle_test` does it too (V-H3-28). Its parent folder, as an old docstring said, gave a
  `KeyError` before (V-H3-5). Checked by:
  `tests/test_rsna_pneumonia_kaggle.py::test_test_folder_with_no_dicom_names_the_folder`. Decided (Frank Li,
  2026-10-31). The other options were 0 samples with a warning, and no change.
- Known gaps: the file-for-one-output rule is checked only for MIMIC-CXR, MIMIC-CXR-JPG, RSNA Pneumonia Kaggle,
  CheXpert Plus, RANZCR CLiP, Montgomery, Shenzhen and CheXlocalize. The other classes were not scanned for it
  (no ID).
- Notes (Shenzhen, CheXlocalize): since 2026-11-03, a call for masks raises `FileNotFoundError` that names
  the missing file. For Shenzhen, it is the `Annotations-2/masks/` folder, when the table has a TB image. For
  CheXlocalize, it is `gt_segmentations_*.json`. Before, Shenzhen kept only the normal images, and CheXlocalize raised
  "No mask_path column found" in `harmonize()`. Found in the 6c rewrite of the wiki pages. Checked by:
  `tests/test_harmonizer_preprocess.py::test_shenzhen_masks_with_no_mask_folder_raise` and
  `tests/test_chexlocalize.py::test_output_mask_with_no_mask_file_raises`.
- Notes (Montgomery): since 2026-11-03, a call for masks with no `ManualMask/` raises `FileNotFoundError`
  that names the folder. Before, it raised a `ValueError`, and the wiki said the rows were dropped (V-H2-36).
  Checked by: `tests/test_montgomery_cxr.py::test_masks_without_manual_mask_folder_name_it`.
- Notes: a real run on 2026-10-29 built 70 registry keys with only `base_image_dir` (119 searches). 6 found
  the given path, 14 found the file in `base_dir` and 63 in `parent`. 24 found it in a sibling folder, and 2
  (CheXlocalize) in an uncle folder. 10 found nothing. Each search took 0.6 s or less.
- Notes (fix): on 2026-10-30, `infer_path` looks only in the 2 exact places. 6 datasets put the sub folder in
  the file name: VinDr-CXR `annotations/`, RAD-ChestCT `tables/`, Emory CXR `TABLES/`, ReXGradient
  `download/metadata/`, CheXlocalize `../CheXlocalize/` and CheXpert Plus `../`. CheXpert Plus walked up 4
  folders before 2026-10-31. A scan of the 103 registry keys with only `base_image_dir` found the same files as
  before. Only CheXphoto changed: its folder is restricted, so it raises now. Before, it took the RANZCR
  `train.csv`. On 2026-10-30, 14 dataset classes got `required=True`, and ROCO got its own
  `FileNotFoundError` when it finds no split folder. On 2026-10-31, 15 more classes got `required=True`. Before,
  18 of their keys gave the pandas error "Invalid file path or buffer object type". Examples are
  `chestxray14_bbox`, the MIMIC-CXR and MS-CXR keys, RANZCR CLiP and the RSNA 2022 and 2023 keys. The 4
  CheXInstruct keys gave a `TypeError`, and 5 keys gave another `ValueError`. A test-split class does not
  require the train files. 5 classes set `_TRAIN_FILES_REQUIRED = False`. The RSNA 2022, 2023 and PE test
  classes pass their test file as `csv_path`. A scan on 2026-10-31 built every registry key with an empty image folder and
  called `get_datasets()`. 112 of the 128 keys raise `FileNotFoundError` (the 4 ROCO keys take `base_dir`, not
  `base_image_dir`). 14 keys raise a `ValueError` that names an argument to pass, for example `annotation_dir`
  (Chest ImaGenome). The 2 QaTa-COV19 split keys say that the folder must be in `Train Set/` or `Test Set/`.
  On 2026-11-03, `CheXpertValidDataset` got `required=True` for `valid.csv`. Before, it took the
  `train.csv` search of `CheXpertDataset` when `valid.csv` was missing (V-A-34). Checked by
  `tests/test_app.py::test_chexpert_valid_without_valid_csv_is_an_error`.
- Related: D-04, D-21.

### D-05 Labels

- Label columns are in `LABEL_COLS`, with snake_case names.
- Name a label after the finding, not after the way the source codes it. Example: Montgomery and Shenzhen
  both give `tuberculosis`. Some datasets still use different names for one finding, for example
  ChestX-ray14 `effusion` and CheXpert `pleural_effusion`. Decided (Frank Li, 2026-11-05).
- The label columns are sorted alphabetically by their snake_case names. `BaseHarmonizer.harmonize` sorts
  them. The `cls` vector uses the same order.
- In `cls`, a label keeps the CheXpert-style source codes: 1 (positive), 0 (negative), -1 (uncertain) and
  NaN (not mentioned, or unknown). The probes use only 0 and 1, so they skip -1 and NaN.
  `drop_uncertain=True` drops the rows with -1. A harmonizer can keep raw grades, for example the TAIX-Ray
  grades 0 to 4. Then its dataset class maps them to 0 and 1.
- Not mentioned (CheXpert, CheXpert Plus, MIMIC-CXR and MIMIC-CXR-JPG): the argument `not_mentioned` sets a
  blank label. `"negative"` (the default) gives 0, as most CheXpert papers do. `"nan"` keeps NaN, so the
  probes skip it. Example: Pleural Other is blank in 97% of the CheXpert train rows.
- A regression target goes in `REG_COLS`, not in `LABEL_COLS`.
- Each dataset wiki page gives the policy for uncertain and blank labels. For example, CheXpert maps a
  blank to 0 and keeps `-1`.
- A row for which the labeler gave no label at all (an empty label list) gets NaN in every label. It is
  unknown, not negative. Example: 777 PadChest rows have the raw `Labels` value `['']`, and their reports
  name findings such as scoliosis. The probes skip NaN, so these rows do not count as negatives. A
  CheXpert-style row with every label blank is such a row, also with `not_mentioned="negative"`. Example:
  4,857 MIMIC-CXR-JPG train rows. Before 2026-11-02 they got 0 in every label (V-H1-41).
- A source value that a harmonizer splits into one-hot columns gets NaN in all of these columns when the
  value is blank. Example: a blank RSNA 2024 lumbar severity gives NaN in `normal_mild`, `moderate` and
  `severe`, not 0 in all 3.
- A class that a source does not annotate gets NaN, not 0. Example: the MedSeg dataset2 masks have only the
  values 0, 1 and 2, so `pleural_effusion` is NaN in its 829 rows. Decided by the agent, for Frank Li to
  confirm (2026-11-02, Q-12 of the 2026-11-02 review). The other option was 0 with the current wiki line.
- A trigger word sets only its own finding. Example: the OpenI MeSH term `Lung/hypoinflation` does not set
  `atelectasis`. 189 of the 504 OpenI studies with `atelectasis` = 1 have only this term. Decided by the
  agent, for Frank Li to confirm (2026-11-02, Q-15 of the 2026-11-02 review). The other options were to keep
  the trigger with a wiki line, and a separate hypoinflation column.
- Notes (MedSeg, since 2026-11-03): dataset2 gives NaN in `pleural_effusion`, because its masks have no class 3
  (Q-12). Checked by: `tests/test_harmonizer_data.py::test_medseg_dataset2_pleural_effusion_is_nan`.
- Notes (2026-11-02): OpenI no longer maps `lung/hypoinflation` to `atelectasis` (Q-15). Its reports with
  only the MeSH term "No Indexing", or with no term, get NaN in every label (V-H2-41).
- Notes (2026-11-05): the Montgomery label `tb` became `tuberculosis`, the same name as Shenzhen and the
  public RadHarmony repo. Decided by Frank Li. The other option was to keep `tb`. A saved Montgomery table
  with `tb` has no `tuberculosis` column. With `output_cls=True`, the dataset class warns that the label
  column is missing, and a sample raises `RuntimeError`. Harmonize such a table again. Checked by (the label
  names only): `tests/test_montgomery_cxr.py::test_label_is_tuberculosis`.

Why: a probe can use the label columns of any dataset without code for that dataset.

- Status: From the code. The codes in `cls` (-1 uncertain, NaN not mentioned): Decided (Frank Li, 2026-10-31,
  Q-4 of the 2026-10-31 review). The other options were 1, 0 and NaN only (-1 becomes NaN), and -1 as an
  exception for 3 classes. The empty-label-list rule: Decided (Frank Li, 2026-10-31, Q-15 of the same
  review). The other options were to drop the rows, and to keep them with all labels 0. The one-hot rule:
  Decided (Frank Li, 2026-10-31, Q-17 of the same review). The other option was 0 in all columns, with a
  wiki line. The `not_mentioned` rule: Decided (Frank Li, 2026-11-01, follow-up of Q-4). The other options
  were always 0 and always NaN. 73% of the CheXpert train label cells are blank, so NaN as the default
  would train each probe on the mentioned rows only.
- Checked by: J-04, `tests/test_label_order.py`, L-008, and
  `tests/test_harmonizer_data.py::test_padchest_empty_label_list_is_nan` and
  `::test_rsna_2024_lumbar_blank_severity_is_nan` (since 2026-11-01, Q-15 and Q-17), and
  `::test_chexpert_codes_follow_d05`, `::test_mimic_jpg_keeps_uncertain_as_minus_one`,
  `::test_chexpert_not_mentioned_nan_keeps_the_blank` and `::test_every_chexpert_style_harmonizer_follows_d05`
  (Q-4). For V-H1-41: `::test_chexpert_codes_keep_nan_for_a_row_with_no_label` and
  `::test_a_row_with_no_label_at_all_is_nan_by_default`. For OpenI (Q-15, V-H2-41):
  `::test_openi_mesh_terms_match_whole_words` and `::test_openi_no_indexing_gives_nan_in_every_label`.
- Notes (Q-4): before 2026-11-01, CheXpert, CheXpert Plus, MIMIC-CXR and MIMIC-CXR-JPG mapped -1 to NaN. So
  `cls` lost the difference between "uncertain" and "not mentioned" (Q-4, V-H1-34). Now they share
  `chexpert_codes` in `radharmony/harmonizer/_labels.py`, with the MIMIC-CXR-JPG labeled test class. With the
  defaults, the table is the same as before.
  BRAX ("raw"), Emory CXR and the MIMIC-CXR-JPG labeled test kept -1 already.
- Related: D-08, D-30.

### D-44 A "stable" finding is present

In the phase-grounding labels (`docs/wiki/explanation/phase-grounding.md`), "stable X" means that finding X is still there
and has not changed. So X is present. With no severity word, X gets the default severity 2.

Example: "stable cardiomegaly" gives `heart_size` 2.

Why: a score of 0 would call a present finding absent.

- Status: Decided (Frank Li, 2026-10-31, Q-22 of the 2026-10-31 review). The other options were no default
  for a "stable" finding (score 0), and a separate rule for "stable X".
- Checked by: none.
- Known gaps: the doc calls the score "reasonable" and gives no reason (R-docs-18 of the 2026-10-31 review).
- Known gaps (script): the label script (`extract_mimic_labels.py` of the phase-grounding project,
  outside this repo) has "stable" in its grade-0 word list. So "Stable cardiomegaly." gives `heart_size` 0, and
  the labels parquet does not follow D-44 yet. Found on 2026-11-04 in the 6c rewrite. Frank Li kept D-44
  (2026-11-04): the owner of the script changes it and makes the parquet again (`eval/TODO.md`). The other options were
  to change D-44 to 0, and to leave the question open.
- Related: D-05.

### D-48 CheXpert Plus labels come from `impression_fixed.json`

The CheXpert Plus harmonizers take their labels from `impression_fixed.json`, not from `report_fixed.json`.
Both files are in the release folder, with the same image paths in the same order.

Why: the labels in `report_fixed.json` do not follow the image. A check with pandas only, joined by image
path, gave these numbers on 2026-11-02:

- P(Cardiomegaly=1) is 0.16 when CheXpert v1 says 1, and 0.16 when it does not. With
  `impression_fixed.json`, it is 0.90 and 0.03.
- The images of one study share one report, so they must share labels. In `report_fixed.json`, 31,377 of
  the 33,973 studies with 2 or more images give their images different labels. In `impression_fixed.json`,
  380 do.

CheXpert v1 also takes its labels from the impression.

- Status: Decided (Frank Li, 2026-11-02, Q-7 of the 2026-11-02 review). The other options were to keep
  `report_fixed.json` with a wiki warning, and to leave the question open.
- Checked by: `tests/test_chexpert_plus.py::test_every_chexpert_plus_key_reads_the_labels_of_the_impression`,
  `tests/test_official_splits.py::test_chexpert_plus_finds_its_files_2_folders_up_and_takes_impression_fixed`,
  and the CheXpert Plus `expected_stats` in `tests/test_dataset_stats.py` (full run, real data).
- Notes: since 2026-11-03, the dataset class, the app and `configs/data_paths.json` take
  `impression_fixed.json` (Q-7). The CheXpert Plus rows with no label go from 3,673 to 12 (V-H1-41). With
  the default `drop_uncertain=True`, train has 143,343 images (134,623 before) and valid has 165 (145 before).
  All CheXpert Plus benchmark results must run again.
- Related: D-05, D-35.

### D-51 A MIMIC-Ext-CXR-QBA answer holds the main answer only

The `answer` column of MIMIC-Ext-CXR-QBA holds only the texts of the `main_answer` entries. The texts of
`details` and `related_information` stay in `answer_struct`.

Example: 421,565 of the 1,000,000 q1M questions also have a `details` or `related_information` text. Their
`answer` does not hold it.

Why: `answer` is the short answer that a VQA metric compares. The extra texts make it long.

- Status: Decided by the agent, for Frank Li to confirm (2026-11-02, Q-14 of the 2026-11-02 review). The
  other option was to keep all top-level texts with a wiki line.
- Checked by: `tests/test_mimic_ext_cxr_qba.py::test_answer_holds_only_the_main_answer_texts`.
- Notes: before 2026-11-03, `answer` joined all top-level texts (Q-14), and the wiki said that it joins
  the texts of the annotators (V-H1-43).
- Notes: a question whose `main_answer` text is empty gets `answer` "". A check of 2,000 studies (371,228
  questions) in `qa.zip` on 2026-11-03 found 57 such questions with a `related_information` text. Before,
  their `answer` was that text.
- Related: D-02.

### D-06 Age in years

Every harmonizer whose source has the age of the patient gives the column `age_years`. It is a float in
years. A harmonizer declares the source column and its unit with `AGE_SOURCE_COL` and `AGE_SOURCE_UNIT`.
The code never guesses the unit from the size of the numbers.

DICOM tags: if the age or the sex is only in the DICOM headers, the harmonizer reads the tags `PatientAge`
and `PatientSex`. A missing or empty tag gives NaN. The docstring and the wiki page say that these values
come from the DICOM tags, not from the official release of the dataset.

Why: datasets store age in days, months, years or DICOM age strings. A guess fails on child cohorts.

- Status: Decided (Frank Li, 2026-10-28). The unit rule is from the code (comment on `AGE_SOURCE_COL` in
  `radharmony/harmonizer/base.py`).
- Checked by: L-001 (`tests/test_lessons.py`), `tests/test_dicom_tags.py`, `tests/test_sex_column.py`, and
  `tests/test_dataset_stats.py` for the datasets with `expected_stats`.
- Age groups: when the source gives only an age group, `age_years` is the low edge of the group. BRAX
  gives 5-year groups (`0`, `5`, ..., `80`) and `85 or more`, so `85 or more` gives 85. Decided (Frank Li,
  2026-10-30). The other option was NaN for `85 or more` (2,299 of 40,967 images). The same rule gives 90
  for the `90+` of RSNA 2023 abdominal trauma (90 patients, V-H3-32).
- Chest ImaGenome silver: `age_years` is the low edge of the scene-graph `age_decile`, and `sex` comes from
  its `gender`. The release writes the group 10-20 as `20-Oct`, because a
  spreadsheet read it as a date, so `20-Oct` gives 10. Decided by the agent, for Frank Li to confirm
  (2026-11-03, V-H1-35). The other option was NaN for `20-Oct` (551 of 23,951 valid images). Checked by:
  `tests/test_chest_imagenome.py::test_silver_gives_sex_and_the_low_edge_of_the_age_group`.
- Range: an age below 0 or above 130 years gives NaN (`radharmony/harmonizer/_age.py`). These values are
  errors in the source. Example: ChestX-ray14 has 16 images with an age from 148 to 414. Patient 5567 is 412
  at the first visit and 53 at the next. Decided (Frank Li, 2026-10-30). The other option was to keep the
  raw value.
- VinDr-CXR: a DICOM `PatientAge` of 0 (`000Y` or `000D`) gives NaN, because it means "unknown". The tags
  give age 0 for 905 train and 269 test images, but only 2 to 12 images for each age from 2 to 17. Decided
  (Frank Li, 2026-10-30). This rule is for VinDr-CXR only. A child dataset such as VinDr-PCXR can have real
  newborns. The other options were NaN for age 0 in every DICOM dataset, and no change.
- TAIX-Ray: `Age` 0 days gives `age_years` 0.0, because these are real newborns. The number of images falls
  smoothly over the first days (day 0: 55, day 1: 44, day 2: 36, day 3: 23). The same patients have later
  images at an older age. Decided (Frank Li, 2026-10-31, Q-19 of the 2026-10-31 review). The other option was
  NaN for age 0, as in VinDr-CXR.
- CheXpert demographics workbook: its `AGE_AT_CXR` is one age per patient, so the table does not take it.
  `age_years` comes from the `Age` of each image in the CSV. Before, the column `age` held the workbook age
  on every image, wrong by more than 1 year for 16,169 of 137,783 rows (V-H1-40). Decided (Frank Li
  confirmed on 2026-11-02 the choice of the agent of 2026-11-01). The other option was the workbook age under
  another column name.
- Birth year: when the source has only the birth year and the study date, `age_years` is the study year
  minus the birth year. The age can be 1 year too high. PadChest has `PatientBirth` and `StudyDate_DICOM`,
  and the `PatientAge` of PadChest-GR uses the same formula. Decided (Frank Li, 2026-10-30). The other
  options were no age, and the year difference minus 0.5.
- Checked by (the 3 rules above): `tests/test_harmonizer_data.py::test_brax_age_group_85_or_more_gives_85`,
  `tests/test_dicom_tags.py::test_vindr_cxr_age_0_is_unknown` and
  `tests/test_age_units.py::test_padchest_age_is_study_year_minus_birth_year`.
- Known gaps: MR-RATE gives no `age_years`, because nobody has checked the unit of its column
  `Patient'sAge`. The data is restricted (no ID).
- Notes: a real run on 2026-10-30 gave PadChest an age for 157,601 of its 157,607 rows, from 0 to 105
  (median 62). VinDr-CXR train has ages from 2 to 124, and test from 4 to 90.
- Notes (since 2026-11-03): VinDr-PCXR reads `PatientAge` and `PatientSex` from its DICOM headers (V-H3-31).
  The paper says every patient is under 10. The tags give 72 train and 14 test images an age over 10, up
  to 45 and 51. The table keeps the tag value. RSNA 2023 abdominal trauma reads `Age` and `Sex` from
  `train_demographics_2024.csv` (V-H3-32). Both have `expected_stats` with sex and age (L-021).
- Related: D-26.

### D-26 One `sex` column

Every harmonizer that has the sex of the patient gives one column `sex`. Its values are `"M"`, `"F"` or
NaN. NaN means unknown or other, for example the DICOM value `O`. The source column can stay as an extra
column. A sex that is only in the DICOM headers follows the DICOM rule of D-06. RSNA Bone Age also keeps
`sex_female` (0 or 1), because bone-age models use it as a number.

Why: a user can compare or filter by sex across datasets without code for each dataset.

- Status: Decided (Frank Li, 2026-10-28). Method: each harmonizer declares `SEX_SOURCE_COL`, and one base
  step next to `_build_age_years` maps the values with one shared table. The change to
  `radharmony/harmonizer/base.py` (base code) was approved by Frank Li on 2026-10-28. It was built on the
  same day. On 2026-10-29, Frank Li approved one more base-code change: a dataset and `save()` never use
  `sex` and `age_years` as labels.
- Checked by: `tests/test_sex_column.py`, `tests/test_dicom_tags.py`,
  `tests/test_padchest_gr.py::test_sex_comes_from_patient_sex_dicom`, and `tests/test_dataset_stats.py`
  for the datasets with `expected_stats`.
- CheXpert: `sex` comes from the CSV column `Sex`, on every row. It does not depend on `demographics_path`.
  The workbook column `GENDER` differs for 3 patients (patient51668, patient54170 and patient54565, 1 image
  each). Decided (Frank Li, 2026-10-30). The other options were the workbook when it is given, and NaN when
  the 2 files differ.
- Related: D-06.

### D-07 Box format

A box is `[dim0_min, dim0_max, dim1_min, dim1_max]`, row first. For a 2-D image this is
`[y_min, y_max, x_min, x_max]`. A 3-D box adds `dim2_min, dim2_max`. The values are fractions in [0, 1] of
the size of the image file that `image_path` points to. Each min is less than its max.
`bbox_labels` has one label for each box, in the same order.

A harmonizer clips each box to the image. It drops a box with no area left, with its `bbox_labels` and
`bbox_findings` entries. Examples: a source box with `x1 == x2`, or a box past the image edge. It warns with
the number of dropped boxes. When a box in `ds[i]` has a NaN or inf value, the transform raises a
`ValueError` (MONAI wraps it in a `RuntimeError`).

Why: the first axis of a numpy array is the row. So `img[y0:y1, x0:x1]` uses the box in its own order.

- Status: From the code (`docs/wiki/get-started/concepts.md`, "Box order"). The clip, drop and
  `ValueError` rules: Decided (Frank Li, 2026-10-30). The other options were the `ValueError` rule only, and
  no written rule.
- Checked by: J-03, L-015, `tests/test_bbox_axis_alignment.py`, `tests/test_harmonizer_boxes.py` (with
  `::test_chest_imagenome_gold_drops_boxes_with_no_area`, `::test_chest_imagenome_silver_drops_boxes_with_no_area`,
  `::test_vindr_cxr_train_box_is_clipped_to_the_image`, `::test_clip_boxes_follows_d07` and
  `::test_every_box_harmonizer_clips_its_boxes`),
  `tests/test_padchest_gr.py::test_a_box_past_the_edge_is_clipped_and_a_box_with_no_area_is_dropped`,
  `tests/test_ms_cxr.py::test_box_values_are_python_floats_and_survive_a_csv_round_trip`,
  `tests/test_transforms.py::test_box_that_is_not_finite_raises` and
  `::test_a_table_read_back_from_csv_gives_the_same_boxes`.
- Notes (value type): MS-CXR boxes hold Python floats since 2026-11-03. Before, they held numpy floats, so a
  CSV export wrote `np.float64(...)` text that `json.loads` cannot read (V-H2-35).
- Labels with no class in the source: RSNA 2022 cervical spine boxes get the label `fracture`, one for
  each box. The source CSV marks fracture boxes, but it gives no class or vertebra. Decided (Frank Li
  confirmed on 2026-11-02 the choice of the agent of 2026-11-01, V-H3-23). The other option was no
  `bbox_labels` column.
- Notes: a real run on 2026-10-30 checked the readable box datasets. Chest ImaGenome dropped 1,528 of the
  silver train boxes and 6 of the gold boxes. The other 8 datasets had no box outside [0, 1], with no
  area or not finite. The VinDr-CXR train clip moves 3 boxes that start outside the image (since
  2026-10-30, D-36). On 2026-11-02, NODE21, ChestX-Det10 and VinDr-PCXR became readable. All their source
  boxes are inside [0, 1], with an area, and finite. They are 1,476 boxes in NODE21, 8,339 in ChestX-Det10
  and 6,171 in VinDr-PCXR. Each image has one label per box. PadChest-GR is still restricted, and RSNA Pneumonia
  (adjudicated) is not on the NAS.
- PadChest-GR extra boxes (`extra_bbox`): the same clip and drop rule as `bbox`. Decided (Frank Li confirmed
  on 2026-11-02 the choice of the agent of 2026-11-01, R-code-17). The other option was no check, because the
  extra boxes do not reach `ds[i]`.
- `keep_negatives=False` of NODE21 and ChestX-Det10 drops the images with no finding in the source. An
  image keeps its row when the clip step drops its only box.
- A finding on a region with no box: the harmonizer keeps it out of the boxes and warns with the count.
  The dataset wiki page says so. Example: Chest ImaGenome gold has 227 positive findings (181 images) on the
  region "mediastinum", but its gold box file has only "upper mediastinum". Decided (Frank Li, 2026-11-02,
  Q-8 of the 2026-11-02 review). The other options were the "upper mediastinum" box, a column for
  findings with no box, and to leave the question open. Checked by:
  `tests/test_chest_imagenome.py::test_gold_warns_about_findings_on_a_region_with_no_box` (since
  2026-11-03).
- Inverted source box (a min greater than its max): the harmonizer drops it with the warning above. It
  does not swap the corners. Example: 22 of the 25,989 Chest ImaGenome gold boxes are inverted. Their median
  IoU with the automatic box of the same region is 0.0, and the normal boxes give 1.0. Decided by the agent,
  for Frank Li to confirm (2026-11-02, Q-9 of the 2026-11-02 review). The other options were to keep the
  swap, and to drop only the boxes far from the automatic box. Checked by:
  `tests/test_chest_imagenome.py::test_gold_drops_an_inverted_box_with_the_d07_warning`. Since 2026-11-03,
  the gold table has 25,961 boxes (28 dropped), not 25,983.
- `bbox_labels` use snake case in every harmonizer, as the label columns do (D-05). Example: VinDr-PCXR
  writes `Cardiomegaly`, but VinDr-CXR writes `cardiomegaly`. Decided by the agent, for Frank Li to confirm
  (2026-11-02, Q-17 of the 2026-11-02 review). The other options were the source names on each wiki page,
  and no rule. The shared step is `box_label` in `radharmony/harmonizer/_bbox.py` (lower case, a space gives
  `_`, a `/` stays as in `nodule/mass`). PadChest-GR joins the snake case labels of one sentence with
  `" + "`. Checked by: the `box_labels` check of `tests/fixtures/harmonizer_contract.py` (L-008, also on
  real data in the full run).
- Notes (Q-17, since 2026-11-03): VinDr-PCXR (`Cardiomegaly`), MS-CXR (`Pleural Effusion`), Chest ImaGenome
  (`right lung`) and PadChest-GR (`vascular redistribution`) changed to snake case. The other box harmonizers
  already wrote snake case. A table saved before 2026-11-03 keeps the old names until `harmonize()` runs
  again.
- A source box that gives no box in the table warns with the count, also when the file of the image cannot
  be read. Example: VinDr-PCXR with an unreadable DICOM file (R-code-19 of the 2026-10-31 review). Another
  example is an RSNA 2024 lumbar point on a series that cannot be read (R-code-20). A third is a lumbar point
  on a second GDCM sub-series of the series folder (V-H3-33). The 3-D loader reads only the first
  sub-series. Checked by:
  `tests/test_vindr_pcxr.py::test_an_unreadable_dicom_of_a_boxed_image_warns`,
  `tests/test_rsna_2024_lumbar_spine.py::test_points_on_a_second_sub_series_warn_with_the_count`,
  `::test_a_series_that_cannot_be_read_warns`, and L-020 (no silent `except` in `radharmony/harmonizer/`).
  Notes: a real run of RSNA 2024 lumbar train on 2026-11-03 gave 48,541 boxes. 151 of the 48,692 points are
  not on the loaded volume. All series were read, so these points are on a second sub-series.
- No image folder for a box that needs the image size from a DICOM header: the harmonizer warns and gives
  no box. This is the same as for a file that cannot be read. Example: VinDr-CXR train and test with `bbox_csv_path` and no
  `base_image_dir`. Decided by the agent, for Frank Li to confirm (2026-11-03, V-H3-35). The other option was
  a `ValueError`, as RSNA 2024 lumbar raises. Checked by:
  `tests/test_vindr_cxr.py::test_boxes_with_no_image_folder_warn`. A real run gave the same 8,328 train and
  2,697 test boxes as before with the folder.
- RSNA 2022 cervical spine writes `bbox` and `bbox_labels` as JSON text, as NODE21 does. So a table saved
  with `to_csv` loads back with `output_bbox=True` (V-H3-34). Before 2026-11-03, a Python list became
  `['fracture']`, which is not JSON. At that time, the fix was in the harmonizer, not in the base parser. Checked by:
  `tests/test_rsna_2022_cervical_spine.py::test_a_table_read_back_from_csv_gives_the_boxes`.
- Notes (since 2026-11-03, base code approved by Frank Li): the box step of the transform also reads a
  Python list text, with `ast.literal_eval` (never `eval`). So the table of any box harmonizer, saved with
  `to_csv` and read back, works with `output_bbox=True` (V-D-33). Example: a VinDr-CXR `bbox_labels` is
  `['calcification', 'cardiomegaly']` in the CSV. Before, it raised JSONDecodeError. Checked by:
  `tests/test_transforms.py::test_a_table_read_back_from_csv_gives_the_same_boxes`.
- RSNA 2022 cervical `bbox_epsilon`: the in-plane box comes from the source width and height, so the
  argument has no effect. A value gives a warning. The argument stays, so a harmonizer saved before
  2026-11-03 still loads, with this warning. Decided by the agent, for Frank Li to confirm (2026-11-03, V-H3-24). The other option was
  a box of a fixed size from `bbox_epsilon`. Checked by:
  `tests/test_rsna_2022_cervical_spine.py::test_the_in_plane_box_comes_from_the_source_size`.
- Notes (one clip step): since 2026-11-01, all 15 box harmonizers use `clip_boxes` or `clip_box` in
  `radharmony/harmonizer/_bbox.py` (R-code-17). Before, Chest ImaGenome and VinDr-CXR train clipped and
  dropped. SIIM-COVID19, VinDr-PCXR and the RSNA 2022 and 2024 spine harmonizers only clipped, and
  PadChest-GR raised `ValueError`. A real run gave the same box columns before and after the change for 11
  readable datasets. They are Chest ImaGenome gold and silver, ChestX-ray14, MS-CXR, RAD-ChestCT, RSNA
  2022 cervical, RSNA 2024 lumbar, RSNA Pneumonia (Kaggle), SIIM-COVID19, and VinDr-CXR test and train.
- Related: D-28 (the box in `ds[i]`).

### D-36 Boxes of several readers are grouped by overlap

When 2 or more readers draw boxes on one image, the harmonizer groups the boxes of one class that overlap
(IoU > 0.5) into one lesion. It keeps a lesion that 2 or more readers drew. The box of the lesion is the
mean of the boxes in its group.

Example: VinDr-CXR train, image `03e6ecfa6f6fb33dfeac6ca4f9b459c9`, class Nodule/Mass. 3 readers drew 55
boxes (28, 24 and 3). They form 40 groups, and 14 of them have boxes of 2 or more readers. So the table
has 14 boxes.

Why: the mean of all boxes of a class is on no real lesion when a reader draws more than 1 box. In VinDr-CXR
train this is true for 2,179 of the 9,016 (image, class) pairs. Only in 96 of these 2,179 pairs (4%) is the
mean box near a raw box (IoU > 0.5).

- Status: Decided (Frank Li, 2026-10-29), option (c). The other options: (a) all 28,800 raw boxes, (b)
  weighted boxes fusion with the package `ensemble_boxes`, (d) no change. Findings: V-H3-4.
- Checked by: `tests/test_harmonizer_boxes.py::test_vindr_cxr_train_groups_the_boxes_of_readers_by_overlap`,
  and `tests/test_dataset_stats.py` for VinDr-CXR train (8,328 boxes on 4,167 images, from the raw CSV).
- Notes: the VinDr-CXR test set has 1 agreed set of boxes (no `rad_id`), so it needs no merge. Weighted
  boxes fusion also groups boxes by overlap. With human boxes of equal weight, it gives about the same boxes.
- Related: D-07, D-28.

### D-08 Save and load give the same table

After `h.save(path)` and `load_from_saved(path)`, `pandas.testing.assert_frame_equal` passes on
`harmonized_df`, and `LABEL_COLS` has the same items in the same order.

Why: a saved harmonizer lets you skip a slow harmonize, so it must not change the data.

- Status: From the code. The label rule after `harmonizer_path=` and `harmonized_df=`: Decided (Frank Li, 2026-11-02).
- Checked by: J-02, `tests/test_save_load.py` (for example
  `test_label_order_is_the_same_after_load` and `test_no_labels_stay_no_labels_after_load`), L-008, and
  `tests/test_dataset_stats.py::test_saved_table_matches_harmonize` for the
  datasets with `expected_stats`.
- Dataset load: the dataset class of a key loads its own saved harmonizer with `harmonizer_path=`. It is
  not empty, its image paths are in the table, and its label columns are the harmonizer's. Checked by:
  the `harmonizer_path` and `harmonizer_path_labels` checks of `tests/fixtures/harmonizer_contract.py`, in
  the full run only (`tests/test_dataset_stats.py`, L-014). In CI: `tests/test_save_load.py`
  (`test_dataset_takes_only_the_declared_labels` and `test_harmonized_df_takes_no_extra_columns_as_labels`).
- Label columns: after `harmonizer_path=` or `harmonizer=`, the declared labels of the harmonizer
  (`get_label_cols()`, in snake case) that are in the table. The other columns, for example `age` or
  `bbox_labels`, are not labels. With `harmonized_df=` or `set_harmonized_df()`, a class with empty
  `LABEL_COLS` takes the declared labels of its `_HARMONIZER_CLS` (in snake case) that are in the table.
- Notes: Frank Li approved these 2 base-code changes on 2026-11-02. Before them, the dataset class took
  every column that is not a standard column as a label. Of the 41 real image keys, 16 got wrong label
  columns after `harmonizer_path=`, and the 6 TAIX-Ray keys raised an error. With `harmonized_df=`, 11 of
  the 12 keys with no declared labels got labels, for example `age_months` (the regression target) in RSNA
  Bone Age. The RANZCR test table has no extra columns.
- Related: D-05, D-24.

### D-24 `harmonized_df` equals the last `harmonize()` result

After `df = h.harmonize()`, `pandas.testing.assert_frame_equal(df, h.harmonized_df)` passes. This applies
to every harmonizer, including the official split classes. A harmonizer that removes rows (for example,
to keep one split) removes them from `self.df` too. `self.df` is the source of `h.harmonized_df`.

Why: `save()`, `harmonizer=` and `harmonizer_path=` read `h.harmonized_df`. They must give the same rows
as `harmonize()`.

- Status: Decided (Frank Li, 2026-10-28).
- Checked by: `tests/test_dataset_stats.py::test_saved_table_matches_harmonize` for the datasets with
  `expected_stats`, and L-008 (`tests/test_harmonizer_contract.py` in CI on 11 fake datasets,
  `tests/test_dataset_stats.py::test_harmonizer_contract` in the full run). Also
  `tests/test_mrrate.py::test_a_filter_with_no_series_gives_an_empty_harmonized_table`. Before 2026-11-03, an
  MR-RATE filter with no series gave the raw metadata table, and `harmonized_df` raised `KeyError` (V-H2-30).
- Related: D-08, D-09.

### D-30 A harmonizer changes steps, not `harmonize()`

A harmonizer overrides only the `_build_*` steps and a `_read_source()` step that reads the source files
and returns the table. It overrides `harmonize()` only to take an option (for example `drop_uncertain`) or
to add steps after the base steps. Such an override calls `super().harmonize()`.

Why: when the base `harmonize()` gets a fix, every harmonizer gets it. A copy of the base steps does not.

- Status: Decided (Frank Li, 2026-10-28). The base step `_read_source()` was approved on 2026-10-29. Its
  default reads `csv_path`. Since 2026-11-03 (plan 6b, merges refactor-r1 to r4), every harmonizer uses
  the base steps. Since 2026-11-03, the VQA base also has its own steps, `_read_source()` and
  `_build_vqa_columns()` (base code approved by Frank Li). 8 classes still override `harmonize()`, and each
  calls `super().harmonize()`. MIMIC-CXR, MIMIC-CXR-JPG, CheXpert Plus and PadChest take `drop_uncertain`
  (and, except PadChest, `not_mentioned`). MIMIC-CXR-JPG test sets the default `drop_uncertain=False`.
  CheXpert takes both options and adds a step after. CT-RATE and `_KeepSplitMixin` add a step after.
- Split classes: `_KeepSplitMixin` removes the rows of the other splits in `_read_source()`, before the base
  steps. So `preprocess_output_dir` and `sample_n` see only the split. Before 2026-11-03, it removed them
  after `harmonize()`. Then the image copies held all splits, and `sample_n` could give 0 rows. Checked by:
  `tests/test_official_splits.py::test_keep_split_mixin_samples_and_copies_only_its_own_split` (TAIX-Ray,
  CheXpert Plus and MS-CXR).
- Checked by: L-022, `tests/test_lessons.py::test_l022_every_harmonize_override_calls_the_base_steps`. On
  4f7a98e, the commit before the change, it fails and names the 45 classes that copied the base steps (for
  example BRAX and ChestX-Det10). J-12 checks for other copied code.
- Notes (image copies of an override): since 2026-11-03, base code approved by Frank Li (V-H2-27). A
  `harmonize()` override can skip the base one and never call `preprocess()`. Then, with
  `preprocess_output_dir`, `_wrap_harmonize` runs `preprocess()` after it. The override makes its masks
  first, so the masks still come before the image copies (V-H2-21). A `sample_n` with no
  `preprocess_output_dir` gives a warning in such an override. Since 2026-11-03, the base `harmonize()` also
  warns (base code approved by Frank Li). Checked by
  `tests/test_harmonizer_preprocess.py::test_base_harmonize_with_only_sample_n_warns_once`. Before, such
  overrides (for example Montgomery, RAD-ChestCT, ReXGradient and RANZCR) ignored `preprocess_output_dir`,
  `preprocessor` and `sample_n` with no message. A call with none of these arguments does not change.
  The wrapper reads only keyword arguments. So in an override that skips the base steps, image-copy
  arguments given by position do nothing, with no message. No warning for this: Decided (Frank Li,
  2026-11-03). L-022 stops such overrides in the package, so only a user's own harmonizer can hit it.
  Checked by: `tests/test_montgomery_cxr.py::test_harmonize_runs_the_preprocessor`, the `test_own_harmonize_*`
  tests and `::test_ms_cxr_harmonize_runs_the_preprocessor_once` in `tests/test_harmonizer_preprocess.py`.
- Notes (preprocess output name): since 2026-11-03, base code approved by Frank Li, the copy of
  `a/1.png` is `<preprocess_output_dir>/a/1.png.nii` (V-P-22). The name keeps the file ending, so `a/1.png`
  and `a/1.jpg` give 2 files. Before, both wrote `a/1.nii`. An `image_path` with a `..` part raises
  `ValueError` before any file is written, so every copy stays in the output folder. A copy made before
  2026-11-03 keeps its old name. Checked by: `tests/test_preprocessor.py::test_run_batch_gives_2_files_for_2_images_with_the_same_stem`
  and `::test_a_path_with_dot_dot_raises_and_writes_nothing`.
- Notes (preprocess output): a second `harmonize()` with `preprocess_output_dir` writes every file again. It does not skip a file that
  is already there. The tutorial `notebooks/tutorials/dataset_api_tour.ipynb` says so since 2026-11-03.
  Before, it said that a second call is fast (V-H2-38). Decided by the agent, for Frank Li to confirm
  (2026-11-03, V-H2-38). The other option was a skip of the files that exist, a base-code change in
  `radharmony/preprocessor/base.py`. A skip can keep an old file after a change of the preprocessor settings.
- Notes: on 4f7a98e, 45 of 53 overrides copied the base steps (R-code-8). All 45 copied the base reset step
  (D-24), and 13 copied the base label sort (D-05).
- Related: D-05.

## Splits

### D-09 Official splits are separate classes

When the dataset authors provide an official split, each split has its own harmonizer class and its own
dataset class. A class does not take a `split=` argument and does not mix splits in one table. For example,
ChestX-ray14 has `ChestXray14TrainDataset` and `ChestXray14TestDataset`. Each official split class gives
only its own rows.

An old class or registry key that mixes splits, or takes `split=`, stays and works as before, so old code
still runs. Its docstring and its wiki page say that it mixes splits and is not for benchmarks. The split
classes stand beside it.

Why: a benchmark must use the split that the dataset authors published. With one class for each split, a
user cannot mix train and test images by mistake.

- Status: Decided (Frank Li, 2026-10-28). On the same day, Frank Li decided to keep the old keys as they
  are.
- Checked by: `tests/test_official_splits.py`, `tests/test_save_load.py`, and for the datasets with
  `expected_stats`, `tests/test_dataset_stats.py::test_stat_matches_expected` (the image count) and
  `tests/test_dataset_stats.py::test_patients_in_two_splits`.
- MIMIC-CXR-JPG test: `mimic_cxr_jpg_test` stays the test set of `mimic-cxr-2.1.0-test-set-labeled.csv`
  (685 studies with a JPG, radiologist labels). A new key `mimic_cxr_jpg_test_chexpert` gives the `test`
  split of `mimic-cxr-2.0.0-split.csv` (CheXpert-labeler labels), as the DICOM key
  `mimic_cxr_test` does. Decided (Frank Li, 2026-10-30). The other option was no new key.
- Notes (MIMIC-CXR-JPG test): the `test` split of the split file has 3,269 studies. With the default
  `drop_uncertain=True`, `mimic_cxr_jpg_test_chexpert` gives 2,216 studies, 3,556 images and 287 patients.
- App: the app lists the split keys only. A dataset with official splits gets one entry for each split, and
  the old mixed keys leave the app. They still work in code. A dataset with no official split keeps its one
  key. Decided (Frank Li, 2026-10-31, Q-25 of the 2026-10-31 review). The other options were every registry
  key, and no rule. Exception: Emory CXR DICOM (`emory_cxr_dicom`) stays out of the app. The PNG entry
  shows the same images. Decided (Frank Li, 2026-11-01, A-17). The other option was its own entry.
  Checked by: `tests/test_app.py::test_app_lists_the_split_keys_and_every_dataset`. It reads the old keys
  from the inventory in `docs/wiki/reference/datasets/index.md`. Before 2026-11-02, the app had 17 old mixed entries, for
  example MIMIC-CXR, TAIX-Ray and CT-RATE, and no entry for `mimic_cxr_jpg_test_chexpert` (Q-25, A-17).
- Known gaps:
  - MR-RATE has split harmonizers, but no dataset class and no registry key (no ID). Its data is restricted,
    so a user with access must build and check it.
- Old saved files: a split class loads a file that the old class saved only when every row is in its own
  split. Otherwise it raises `ValueError` with the file name and the split. Example:
  `ChestXDet10TrainDataset(harmonizer_path=<a test file>)` raises. Decided (Frank Li, 2026-10-30). The other
  option was to keep the rows of its own split, which gives 0 rows for a wrong file.
- Source file of another split: when the file name picks the split, a split class raises `ValueError` for
  the file of another split. Examples: RAD-ChestCT `img<split>_...csv`, and ReXGradient
  `<split>_metadata_view_position.json` (since 2026-11-03, V-H2-46). Before, `ReXGradientTestHarmonizer`
  with the train JSON gave the 238,968 train rows. Checked by:
  `tests/test_official_splits.py::test_rexgradient_split_class_rejects_the_json_of_another_split`.
- Notes: 5 split classes load old files (`_LoadOldFileMixin`). An old file never mixes splits. For 3 of
  them, the old class gives the same table as the split class. A run on 2026-10-30 gave 10,675 rows for
  SIIM-ACR train, 6,334 for SIIM-COVID19 train and 26,684 for RSNA Pneumonia Kaggle train. The old
  ChestXDet10 class saves 1 split.
- Notes (since 2026-11-03, V-H1-49): the 3 Chest ImaGenome silver split classes and the 2 CheXphoto
  split classes also load old files. Before, they raised `ValueError` for a file of their own split. The
  CheXphoto table has no `split` column, so the saved `csv_path` names the split (`train.csv` or
  `valid.csv`). Checked by: `tests/test_chexphoto.py::test_valid_class_loads_a_valid_file_of_the_old_class`
  and `tests/test_chest_imagenome.py::test_silver_split_class_loads_a_file_of_the_old_class`.
- MIMIC-Ext-CXR-QBA: 3 split keys since 2026-11-01 (V-H1-29), with the split of each patient from
  `patient_metadata.parquet`. A split class keeps its own patients before the `max_studies` cut, so
  `max_studies` counts the studies of its split. Decided (Frank Li confirmed on 2026-11-02 the choice of the
  agent of 2026-11-01). The other option was the cut first, as `_KeepSplitMixin` does. Then the first 20 studies of the q1M export give 0
  test rows. Checked by: `tests/test_harmonizer_id_filter.py::test_mimic_ext_qba_split_classes_keep_one_split`.
- Related: D-24, D-25, D-34.

### D-25 Images with no labels go in a separate test class

D-09 also applies to a split that has no labels. A harmonizer with label columns gives only the rows that
the authors labeled. Images that the authors give without labels (for example, a Kaggle test set) go in a
separate test class. The test harmonizer has `LABEL_COLS = []`, and the test dataset class does not give
`cls`. Examples: `RSNAPneumoniaKaggleTestHarmonizer`, `SIIMACRPTXTestHarmonizer` and
`SIIMCOVID19TestHarmonizer`.

Why: an unlabeled row in a labeled table can go into train or validation with NaN labels.

- Status: Decided (Frank Li, 2026-10-28).
- Checked by: `tests/test_official_splits.py::test_ranzcr_split_classes` and
  `tests/test_official_splits.py::test_rsna_bone_age_test_images_have_no_labels`.
- Images with no label row: the label merge leaves them out, and the harmonizer warns with the count.
  MIMIC-CXR and MIMIC-CXR-JPG share `warn_images_without_labels` in `radharmony/harmonizer/_labels.py`.
  Example: 15 train images in 8 studies have no row in `mimic-cxr-2.0.0-chexpert.csv`. Decided by the
  agent, for Frank Li to confirm (2026-11-03, V-H1-32). The other option was no warning. Notes: before
  2026-11-03, only the JPG class warned. Checked by:
  `tests/test_mimic_cxr.py::test_mimic_cxr_dicom_warns_about_images_without_labels` and
  `tests/test_harmonizer_data.py::test_mimic_cxr_jpg_warns_about_images_without_labels`.
- Notes:
  - The old key `ranzcr_clip` still adds the 3,582 unlabeled test images by default
    (`include_test_split=True`). D-09 keeps old keys as they are. With `output_cls=True`, it raises a
    `ValueError` (V-H2-25).
  - Kaggle gives only a few sample test files for 2 competitions, so their test classes cannot load the
    full test set. `rsna_abdominal_trauma_2023_test`: `test_series_meta.csv` lists 6 series, but
    `test_images/` has only 3 of them, with 1 slice each. `rsna_2022_cervical_spine_test`: the 3 studies in
    `test.csv` (10197, 10454, 10690) are not in the 3 image folders (22327, 25399, 5876). So the class has
    3 rows, and `ds[0]` raises `RuntimeError`, because no image file exists (checked on 2026-11-04).
- Kaggle code competitions: `rsna_abdominal_trauma_2023_test` and `rsna_2022_cervical_spine_test` stay.
  Their docstrings and wiki pages say that the public download has only sample test files. Each class is
  for a Kaggle notebook of the competition, where the hidden test set has the same layout. Decided (Frank
  Li, 2026-10-30). The other options were to remove both, and a `FileNotFoundError` when the images are not
  on disk.
- Related: D-09.

### D-10 Our own splits group by patient

`get_datasets()` and `get_folds()` split on `patient_id`, not on rows. All images of one patient are in
one split. The split is not stratified by label. A split with no patient raises `ValueError`. The message
names `n_splits` and the number of patients.

Why: images of one patient in train and test make the test score too high. An empty split always means a
wrong call.

- Status: From the code (`docs/wiki/get-started/concepts.md`, "Splits"). The empty-split rule:
  Decided (Frank Li, 2026-10-31, Q-21 of the 2026-10-31 review). The other options were a warning, and no
  check.
- Checked by: J-02.
- Checked by (empty split): `tests/test_dataset.py::test_an_empty_split_raises` and
  `::test_a_fold_with_no_patient_raises`. Before 2026-11-01, `split_data` and `kfold_splits` gave an empty
  split with no error (Q-21). Examples: `n_splits=1`, and 3 patients with `n_splits=10`.
- Related: D-09, D-32.

### D-34 Official splits stay as published, also when they share patients

A split class gives its split as the authors published it, also when the splits share patients. The class
docstring and the wiki page say how many patients the splits share. A user who needs splits with no shared
patient removes them in their own code.

Examples:

- VQA-RAD: 202 of the 203 test images also have train questions. The paper made the test set this way.
- CheXinstruct VQA: the 2 sources made their splits on their own. The `train`, `val` and `test` splits
  share 2,007 patients. 497 of the 734 test patients are also in train.
- MIMIC-CXR-JPG: 3 patients (9 studies, 94 images) of the official `validate` split are also in the
  labeled test set of version 2.1.0 (`mimic_cxr_jpg_test`).

Why: a benchmark must use the split that the authors published (D-09), so that a score can be compared
with published scores. A shared patient is a fact of the dataset, so the docs must say it. A score on such
a test split can be too high.

- Status: Decided (Frank Li, 2026-10-29), option (a). Frank Li chose it over (b), remove the test patients
  from train and validation in every dataset, and (c), a choice for each dataset.
- Checked by: `tests/test_dataset_stats.py::test_patients_in_two_splits`.
- Notes: since 2026-10-30, `mimic_cxr_jpg_validate` keeps the 3 patients (62 images in 43 studies after the
  label filter). Before, it removed them. The MIMIC-CXR DICOM classes follow the official split file, which has no shared patient. No stored
  result uses the VQA-RAD or CheXinstruct VQA splits.
- QaTa-COV19 copies: the table keeps one `patient_id` for each picture (the file stem). The wiki page warns
  that 2 copies of one picture can have 2 IDs, so they can land in train and val (D-10). It also says that
  1 test picture is the same as a train picture. Example: 37 train pairs are near copies under 2 IDs (not pixel-equal), and
  test `sub-S09512` equals train `covid_4134` (correlation 0.9999). Decided by the agent, for Frank Li to
  confirm (2026-11-02, Q-16 of the 2026-11-02 review). The other options were to group the copies as one
  patient, and to drop the test picture that is in train.
- Notes (QaTa-COV19): since 2026-11-03, the wiki page and the 2 docstrings give the copies and the shared
  test picture. Before, they said that the splits have zero overlap (Q-16).
- Related: D-09, D-18.

## Dataset classes and transforms

### D-11 Output flags

`ds[i]` always has `img`. Each `output_*` flag adds only its own keys:

| Flag | Keys |
|---|---|
| `output_cls` | `cls` |
| `output_mask` | `mask` |
| `output_bbox` | `bbox`, `bbox_labels`, and `bbox_findings` if the class supports it |
| `output_report` | `report` |
| `output_reg` | `reg` |

Only `output_mask`, `output_report` and the MS-CXR-T `output_previous` can also remove rows (D-33). If a class does not support a flag, it gives a warning and
does not return the key. It does not raise an error and does not return wrong data.

Why: the same code can request the same outputs from every dataset.

- Status: From the code (`SUPPORTED_OUTPUTS` in `radharmony/dataset/base.py`).
- Checked by:
  - L-010: `tests/test_output_flags.py` in CI (all 5 flags since 2026-10-30), and
    `tests/test_dataset_stats.py::test_output_flag_on_a_real_sample` in the full run.
  - `tests/test_integrity.py::test_requested_output_that_no_sample_has_warns`: `check_dataset` warns when
    no sample has a requested key.
  - `tests/test_output_flags.py::test_a_class_that_supports_reg_names_its_reg_columns`.
  - L-013: `tests/test_output_flags.py::test_output_bbox_keeps_every_key_of_the_flag`. With
    `output_bbox=True`, the default transform keeps `bbox`, `bbox_labels` and `bbox_findings` (where
    supported).
  - The app: `tests/test_app.py::test_bbox_checkbox_follows_the_build_default`.
- `output_reg`: every image key takes it, as the other 4 flags. A dataset with no regression target gives
  the warning and no `reg` key. Decided (Frank Li, 2026-10-30). The other option was `output_reg` only for
  RSNA Bone Age, with `TypeError` elsewhere.
- VQA and caption classes (`BaseVQADataset`, 21 keys): the 5 flags are for image dataset classes only. A VQA
  sample has its own keys (question and answer), so a VQA class raises `TypeError` for these flags. Decided
  (Frank Li, 2026-10-31, Q-24 of the 2026-10-31 review). The other option was to accept the flags with a
  warning (a base-code change in `base_vqa.py`).
- A transform of the user: when a given transform ends with `SelectItemsD` and drops the key of a flag that
  the class supports, `__init__` warns. Example: `output_report=True` with
  `RadiologyTransform2D().get_transform()`, which keeps only `cls` and `img`. No image is loaded for the check.
  Before, the samples had no `report` key and there was no message. Base code in `radharmony/dataset/base.py`,
  approved by Frank Li on 2026-11-04. Decided (Frank Li, 2026-11-04, found in the 6c rewrite of the how-to
  pages). The other options were a `ValueError`, and no change. Checked by:
  `tests/test_output_flags.py::test_a_transform_that_drops_a_requested_key_warns_at_once`.
- VQA `output_struct` and `output_question_type`: when the table has no `answer_struct` or `question_type`
  column, `get_datasets` and `get_folds` warn once, and the samples have no such key. Example: ROCO, MedPMC
  and VQA-RAD with `output_struct=True`. Before, there was no warning. Decided (Frank Li, 2026-11-04). The
  other options were a `ValueError`, and no change. Checked by:
  `tests/test_vqa_base.py::test_output_flag_with_no_column_warns_and_adds_no_key`.
- App VQA rows: the app shows the same sample as the VQA dataset class, one row for each question. A row
  shows only its own question and answer. Decided (Frank Li, 2026-10-31, Q-3 of the 2026-10-31 review). The
  other options were one row for each image with all its questions, and a message with the image count.
  Checked by: `tests/test_app.py::test_gemex_row_shows_only_its_own_question` and
  `::test_vqa_adapter_gives_one_row_per_question`. Before 2026-11-02, a GEMeX VQA row showed all questions of
  its image (Q-3), and so did VQA-RAD, CheXinstruct and MIMIC-Ext-CXR-QBA. Real data: 80 GEMeX rows and 451
  VQA-RAD test rows, each with 1 question now.
- App: when the user picks a dataset, the "BBox" checkbox takes the `output_bbox` default of the
  `_build_*` function of that dataset in `app.py`. It is on for NODE21, ChestX-Det10, PadChest-GR and Chest
  ImaGenome (gold and silver), and off for the others. Decided (Frank Li, 2026-10-30, V-A-13). The other
  options were "BBox" off with a hint in the load message, and no change.
- MS-CXR-T `output_previous`: the sample also has `previous_img`, the prior-visit image. It goes through the
  same transform as `img` in a second call, so a random step (for example a flip) draws again for it.
  Decided (Frank Li confirmed on 2026-11-02 the choice of the agent of 2026-11-01, V-D-31). The other option
  was one random draw for both images, which needs a base-code change (the base load steps name only `img`). A row with no
  `previous_image_path` is dropped, because a pair needs both images. Only a user table can have such a row,
  because the harmonizer always gives the path. Decided (Frank Li confirmed on 2026-11-02 the choice of the
  agent). Checked by:
  `tests/test_output_flags.py::test_ms_cxr_t_previous_image_reaches_the_sample`,
  `::test_ms_cxr_t_previous_image_gets_its_own_random_draw_in_a_worker` and
  `::test_ms_cxr_t_previous_image_needs_its_column`.
- A flag that a class supports, with no data in the table: the class warns and leaves out the keys. This is
  the same as for a flag that it does not support. Example: RSNA 2024 lumbar with `output_bbox=True` and no point CSV (no
  `bbox` column) gives no `bbox` and no `bbox_labels`. Decided by the agent, for Frank Li to confirm
  (2026-11-03, V-D-40). The other option was an empty `(0, 6)` box. At that time, the 3-D box step of the
  base transform failed on an empty box (fixed on 2026-11-03, V-D-41). Before, the warning promised empty boxes, and `ds[0]` raised
  KeyError `bbox`. Checked by:
  `tests/test_rsna_2024_lumbar_spine.py::test_output_bbox_with_no_point_csv_leaves_out_the_box_keys`.
- Notes: on 2026-10-30, `output_reg` was added to 49 `__init__` methods in 48 dataset files. BRAX now
  lists only `cls` in `SUPPORTED_OUTPUTS`, so `output_reg=True` gives a warning, not an empty `reg`.
- Related: D-33.

### D-33 An empty `mask_path` means "not annotated"

An empty `mask_path` (NaN or `""`) means that nobody drew a mask for this image. An annotated image with no
finding gets an all-zero mask, not an empty `mask_path`. With `output_mask=True`, the dataset class drops
the rows that are not annotated. It warns with the number of dropped rows.
A blank or NaN `image_path` always drops the row, because the row has no file (V-P-23, checked by
`tests/test_dataset_outputs.py::test_blank_image_path_drops_the_row`).

Why: if an empty `mask_path` also meant "no finding", `output_mask=True` drops every negative image. Then
the segmentation result is biased. The warning tells the user why the dataset is smaller.

- Status: Decided (Frank Li, 2026-10-28).
- Checked by:
  `tests/test_harmonizer_preprocess.py::test_preprocess_masks_writes_zero_mask_for_annotated_image_with_no_finding`,
  `::test_preprocess_masks_skips_image_that_nobody_annotated`, and
  `tests/test_dataset_outputs.py::test_requested_mask_drops_rows_with_no_mask_and_warns`,
  `::test_requested_report_drops_rows_with_no_report_and_warns`, `::test_output_not_requested_keeps_every_row`
  and `tests/test_harmonizer_preprocess.py::test_shenzhen_tb_image_with_no_mask_file_is_not_annotated`.
- Reports: with `output_report=True`, the dataset class also drops a row with no report (NaN or `""`). It
  warns with the number of dropped rows. Example: PadChest then has 157,590 of 157,607 rows. Decided (Frank
  Li, 2026-10-30, V-D-9). The other options were `""` for a missing report, and NaN with a doc line.
- Placeholder reports: a text that only says that there is no report counts as no report (NaN). Examples: the
  Emory CXR blank template (D-03) and the Montgomery text "NO REPORT (same pt as MCUCXR_0162_1)". Decided
  (Frank Li, 2026-10-31, Q-16 of the 2026-10-31 review). The other option was to keep the text. Checked by:
  `tests/test_harmonizer_data.py::test_montgomery_placeholder_is_no_report` (since 2026-11-01).
- Labeled negatives: an image whose labels say "no finding" for every masked finding is annotated, also when
  the release has no mask file for it. It gets an all-zero mask. Example: CheXlocalize drew masks only for
  positive findings, so its images with all 10 labels negative are labeled negatives. Decided (Frank Li,
  2026-10-31, Q-10 of the 2026-10-31 review). The other option was an empty `mask_path` for them. Checked by:
  `tests/test_official_splits.py::test_chexlocalize_labeled_negative_gets_an_all_zero_mask` (since 2026-11-01).
- Box findings: when the source annotates the findings of only some images, a bool column
  `findings_annotated` tells the annotated images. An empty `bbox_findings` entry of an image that is not
  annotated means "not annotated", not "no finding". Example: the Chest ImaGenome gold attribute file has
  only 500 of the 1,000 images. Decided by the agent, for Frank Li to confirm (2026-11-03, V-H1-14). The
  other options were NaN in `bbox_findings`, which breaks the list of strings in `ds[i]`, and a wiki line
  only. Checked by:
  `tests/test_chest_imagenome.py::test_gold_tells_an_image_with_no_finding_annotation_from_no_finding`.
- Known gaps:
  - MR-RATE `mask_source="nvseg-brain"` gives paths to files that exist only for some series (no ID).
- Notes: since 2026-10-30, the 13 TB-positive Shenzhen images with no finding mask have an empty
  `mask_path`. Since 2026-10-29, the base `preprocess_masks` writes no mask for an image that nobody annotated
  (V-H2-4). RANZCR CLiP train with `output_mask=True` has 9,095 images, not 30,083. CheXlocalize has 667
  of 668 test rows and 234 of 234 val rows since 2026-11-01 (Q-10). Before, its 168 test and 47 val labeled
  negatives had no mask. SIIM-ACR (the RLE `-1` gives an all-zero mask), MedSeg, QaTa-COV19, Montgomery and RSNA 2022
  cervical spine follow this decision (checked on 2026-10-28).
- Notes (reports): a real run on 2026-10-30 counted the rows with no report. PadChest had 17. MIMIC-CXR,
  CheXpert Plus, ReXGradient, Emory CXR, Montgomery and Shenzhen had 0. Open-I had 40 rows (25 studies),
  counted on 2026-11-02 when its folder became readable. Since the D-03 fix, 7 Emory CXR rows (3 studies with only a blank template) have no report.
- Related: D-11.

### D-28 The box in a sample matches the image in that sample

In `ds[i]`, the box always matches `ds[i]["img"]`. Every resize, pad, crop, flip and warp of the image
moves the box in the same way. The box keeps the format of D-07, as fractions of the returned image. A
sample box is an `(N, 4)` (2-D) or `(N, 6)` (3-D) float32 tensor, and `(0, 4)` or `(0, 6)` when there is
no box.

Example: a 2,000 × 1,000 (H × W) image is resized to 512 × 256. Then it is padded equally on the left and
right to 512 × 512. A box with `x_min = 0.0` in the table must have `x_min = 0.25` in `ds[i]`.

Why: a model and a metric use the box together with the returned image, not with the source file.

- Status: Decided (Frank Li, 2026-10-28). Method: exact math on the box corners for resize, pad, crop,
  the `with_spacing` resample (since 2026-11-03),
  flip and transpose. Only the affine and elastic augmentations use the mask round trip. The change to
  `radharmony/dataset/transforms.py` (base code) was approved by Frank Li on 2026-10-28. A warning for the
  encoder box (the known gap below) until a detection evaluator needs it: Decided (Frank Li, 2026-10-31).
  The other options were exact math now, an error, and no warning.
- Checked by: J-03, L-011 (`tests/test_lessons.py::test_l011_box_stays_on_the_white_rectangle`),
  `tests/test_transforms.py`, `tests/test_dataset.py::test_visualize_samples_draws_every_box`,
  `tests/test_encoder_transforms.py::test_encoder_transform_box_is_float32_with_4_columns`,
  `::test_encoder_transform_warns_that_the_box_does_not_follow_the_processor` and
  `::test_encoder_transform_gives_no_box_warning`, `tests/test_transforms.py::test_an_error_in_the_3d_box_step_is_not_hidden`
  and `::test_a_3d_sample_with_no_box_gives_an_empty_box`, L-018 (`tests/test_lessons.py::test_l018_3d_box_stays_on_its_cube`).
- Known gaps: `RadiologyEncoderTransform` does not move the box with the crop or pad of the encoder
  processor (no ID). It warns when `bbox` is in `output_keys` and the processor crops or pads. A custom
  `preprocess` with no `processor=` also warns, because its steps are unknown. A processor that only
  resizes keeps the box fractions, so it gives no warning. Example: the MedSigLIP recipe gives no warning,
  and the RAD-DINO and EVA-X recipes warn.
- Notes: since 2026-10-30, the box of `RadiologyEncoderTransform` is a float32 `(N, 4)` tensor, also for no
  box. Before, it was bfloat16 with shape `(0,)` for no box, and a JSON string box stayed a string. Since
  2026-11-02 (base code approved by Frank Li), an error in the 3-D box step `_ReorientBbox` stops the run.
  Before, `except Exception: pass` hid it, so the image turned and the box kept its old axes (R-code-18).
  On real cervical spine, lumbar spine and RAD-ChestCT samples, that `except` caught no error.
  Since 2026-11-03 (base code approved by Frank Li), a 3-D row with no box gives a `(0, 6)` box (V-D-41).
  Before, `_ReorientBbox` raised IndexError on it. Example: a lumbar series with no labeled point stopped the run.
  Since 2026-11-03 (base code approved by Frank Li), the box moves with the resample of `with_spacing`
  (`pixdim`) with exact math on its corners (V-D-42). Before, the box kept its fractions, but the output covers
  a different area than the input. Example: 15 slices of 4 mm resampled to 1 mm put the box 1.2 voxels off the object.
- Related: D-07, D-12, D-43.

### D-12 3-D orientation

When `base_transpose=True` (the default), every 3-D volume is reoriented to the axis code in `axcodes`
(default `IPL`). Then it is transposed to `(D, H, W)`. The box goes through the same 2 steps. The 2 steps
always stay together.

Why: scanners save volumes in different orientations. One orientation makes the box order of D-07 true
for every volume.

- Status: Decided (Judy, 2026-04-26, `docs/proposals/orientation_normalization.md`).
- Checked by: J-03.
- RAD-ChestCT: its `.npz` arrays have no orientation data, but they are already `IPL` in `(D, H, W)` order.
  The box columns name the axes (`sup_axis0min`/`inf_axis0max`, `ant_axis1min`/`pos_axis1max`,
  `rig_axis2min`/`lef_axis2max`). So the class passes `base_transpose=False`, and the image and the box
  still follow this rule. Decided (Frank Li, 2026-10-31). The other options were code that gives the arrays
  an orientation, and a known gap with no change.
- 2-D slices of a 3-D volume are in the usual axial view. Rows go from anterior to posterior. Columns go
  from patient right to patient left. The mask gets the same step. Example: before 2026-11-03, a MedSeg COVID CT
  slice was `vol[:, :, k]` in NIfTI array order, which is the transpose of this view. Now it is `vol[:, :, k].T`. Decided by the
  agent, for Frank Li to confirm (2026-11-02, Q-11 of the 2026-11-02 review). The other options were the
  array order with a wiki line, and a reorientation with the affine.
- Notes (MedSeg, since 2026-11-03): the slices are in this view, and their folder name has the HU window,
  for example `dataset1_hu-1000_400`. Before, a new window used the old PNGs (V-H1-42). Checked by:
  `tests/test_harmonizer_data.py::test_medseg_slices_use_the_usual_axial_view` and
  `::test_medseg_new_hu_window_writes_new_slices`.
- Related: D-07, D-28.

### D-43 A resample keeps the image centered

A resample to a new voxel spacing covers the same physical area as the input. So the output origin moves by
(new spacing - old spacing) / 2 along each axis. This is the usual way to resample with SimpleITK.

Example: an input of 1,000 pixels at 1 mm covers x = -0.5 to 999.5 mm. At 9 mm, the output must cover the
same area, not -4.5 to 995.5 mm.

Why: a box stored as fractions of the image (D-07) assumes that the output covers the same area as the input.

- Status: Decided (Frank Li, 2026-10-31, Q-20 of the 2026-10-31 review). The other options were a docstring
  line about a shift of up to half a pixel, and no change.
- Checked by: `tests/test_preprocessor.py::test_resample_keeps_the_image_centered`.
- Notes: Frank Li approved the base-code change on 2026-11-02 (Q-20). Before it, `_resize_isotropic` in
  `radharmony/preprocessor/base.py` kept the input origin, so the content moved by up to half an output
  pixel. In the example above, it moved by 4.0 mm. The fix moves the origin in the image direction, so it
  also works for an oblique 3-D image. It changes `CXRPreprocessor` and `CTPreprocessor` output written
  after 2026-11-02. No dataset class uses them.
- CT axes (V-P-25, V-P-26): `CTPreprocessor` first turns the volume to `LPS`. Then the crop and the resize
  take L-R and A-P as the axial plane, for a coronal or sagittal volume too. Then it turns the volume to
  `orientation`. Before, the long side could land on S-I. A bad `orientation` code (for example
  `"LIX"`) raises `ValueError`. Before, it gave LPS output with no error. Checked by
  `tests/test_preprocessor.py::test_ct_long_side_is_axial_for_a_sagittal_volume` and
  `::test_ct_orientation_with_a_bad_code_raises`. No change in `radharmony/preprocessor/base.py`.
- Known gaps: `with_spacing` (MONAI `SpacingD`) keeps the first voxel center, so its output is not centered
  (no ID). Since 2026-11-03 (base code approved by Frank Li), the box moves with it (D-28, V-D-42).
- Related: D-07, D-12, D-28.

### D-13 CXR pixel values

Every CXR that a dataset class gives has bright bone. A `MONOCHROME1` DICOM is inverted one time.

Why: a model must see the same contrast from every dataset.

- Status: From the code. The inversion by GDCM, with the pydicom test as a guard: Decided (Frank Li,
  2026-10-31). The other options were a run-time check in base code and a known gap with no change.
- Checked by: `tests/test_preprocessor.py` (the `CXRPreprocessor` path),
  `tests/test_dataset.py::test_dataset_load_of_signed_monochrome1_dicom_with_window`,
  `::test_dataset_load_uses_the_window_for_any_dicom_file_name` and
  `::test_dataset_load_does_not_read_an_mha_copy_with_pydicom` (the dataset load).
  `tests/test_encoder_transforms.py::test_monai_reader_applies_the_dicom_window`,
  `::test_monai_reader_reads_a_palette_png_as_shown` and `::test_monai_reader_drops_the_alpha_channel`
  (`reader="monai"`). `::test_recipe_image_loader_reads_palette_and_alpha_pngs_as_shown` (the backbone
  recipes). `::test_palette_loader_picks_the_reader_by_file_suffix`,
  `::test_palette_loader_gives_a_new_cache_key` and L-012 (both).
- Notes: GDCM in the ITK reader inverts MONOCHROME1, and the dataset load has no inversion step of its own.
  `test_dataset_load_of_signed_monochrome1_dicom_with_window` fails if GDCM stops the inversion. Its
  expected values come from pydicom, not from GDCM. The dataset load and `CXRPreprocessor` take the GDCM
  maximum `M` from one helper, `stored_value_sum` in `radharmony/utils/dicom.py`. `M` is -1 for signed values. Both find a DICOM file by
  the tag `0028|0004` in the image information, not by its file name (`is_dicom` in the same file). An
  `.mha`, `.nrrd` or NIfTI copy of a DICOM file keeps this tag, so these endings are not DICOM. The base
  `_fix_monochrome` and the unused `_convert_monochrome1_to_2` were removed. All 3 changes are from
  2026-10-29 (approved by Frank Li). A second inversion gave a negative image (V-P-1). Since 2026-10-30, `reader="monai"` of the encoder transforms runs the 2
  DICOM steps of the dataset load (`_fix_missing_highbit` and `_apply_voi_lut`). Since 2026-10-31,
  `reader="monai"` and the backbone recipes read a palette PNG as the colors it shows, not as its raw
  index. Before, 334 of the 662 Shenzhen PNGs were inverted (V-B-11, V-P-19). A palette or RGBA PNG
  becomes RGB, and an LA PNG becomes gray. The mask loaders of the recipes keep `mt.LoadImage`, because a
  palette mask keeps its class numbers in the index. `CXRPreprocessor` also reads an LA PNG as gray (it
  falls back to PIL, because SimpleITK cannot read it, V-P-27). Checked by
  `tests/test_preprocessor.py::test_cxr_reads_a_gray_plus_alpha_png`.
- Notes (a window only for MONOCHROME): the DICOM standard applies a VOI window only to a MONOCHROME1 or
  MONOCHROME2 image. So `_apply_voi_lut` skips an image of another type, for example RGB, as
  `CXRPreprocessor` does.
  Since 2026-11-03, base code approved by Frank Li (V-P-21, V-P-24). Before, an RGB DICOM with window
  tags raised a `ValueError` in the dataset load, in `reader="monai"` and in the recipe loaders
  (`_display_255`). Now an 8-bit RGB DICOM with window tags gives its 3 channels as they are, as with no
  window tags (D-27). The tests use only a synthetic file. We found no real RGB DICOM with window tags. The
  code of `_apply_voi_lut` is in the cache key (D-15). So each 2-D `cache_dir` builds a new cache once.
  An `embedding_cache` file from before 2026-11-03 raises `ValueError` (D-41). The samples of a
  MONOCHROME image are the same. Checked
  by `tests/test_transforms.py::test_an_rgb_dicom_skips_the_window_in_the_dataset_load` and
  `tests/test_encoder_transforms.py::test_an_rgb_dicom_skips_the_window_and_stays_as_it_is`.
- Notes (HighBit step): since 2026-11-03, base code approved by Frank Li. `_fix_missing_highbit` in
  `radharmony/dataset/base.py` skips a color DICOM (more than 1 sample per pixel), as `CXRPreprocessor`
  does. Before, an RGB DICOM with no HighBit tag crashed the load. The code of the step is in the cache key
  (D-15). So each 2-D `cache_dir` builds a new cache once, and an older `embedding_cache` file raises
  `ValueError` (D-41). Checked by `tests/test_dataset.py::test_highbit_step_skips_a_color_dicom`.
- Related: D-14, D-27.

### D-14 Pixel value range

- A 2-D image is scaled to [-1, 1] with the 0.5 and 99.5 percentiles of that image, and clipped to that
  range.
- A 3-D volume is clipped to its `hu_window` when the dataset sets one.

Why: every dataset gives the same value range to the next step.

- Status: From the code (`radharmony/dataset/transforms.py`).
- Checked by: `tests/test_preprocessor.py::test_uniform_image_gives_minus_one_not_nan` (a uniform image
  only).
- Masks: with `RadiologyTransform2D`, a 2-D mask in `ds[i]` holds 0 and 1. The transform maps a value above
  0 to 1, so mask files with 0 and 255 on disk also give 0 and 1. All 2-D masks have 1 class today. Decided (Frank Li,
  2026-10-30), with the base-code change in `radharmony/dataset/transforms.py`. The other options were 0/1
  mask files from the harmonizers, and no change. Checked by
  `tests/test_transforms.py::test_2d_mask_file_with_255_gives_0_and_1`.
- Masks in the segmentation probes: the probes also take masks from other sources. In a binary mask
  (`num_classes=2`), any value above 0 is foreground, for example 1 or 255. A mask with K > 2 classes
  (K = `num_classes`) holds the class indices 0 to K-1. A value above K-1 raises a `ValueError`. The probe
  turns a one-hot mask into indices with `argmax`. Decided (Frank Li, 2026-10-30, V-E-6). The other
  option was a map of the sorted mask values to 0 to K-1. Checked by
  `tests/test_evaluator.py::test_seg_index_mask_targets_do_not_depend_on_the_batch`. Before 2026-11-03, the
  binary probe divided the mask by the batch maximum. So 0, 1 and 2 gave 0, 0 and 1 (V-E-32).
- Notes: before 2026-10-30, Montgomery, Shenzhen, SIIM-ACR, CheXlocalize, QaTa-COV19 and the MedSeg slices
  gave 0 and 255 in `ds[i]` (V-D-10). The segmentation probes take any value above 0 as foreground, so no
  stored result is wrong. A backbone transform (`RadiologyEncoderTransform`) keeps the mask values of the file.
  So a mask with more classes still works there (V-E-6).
- Related: D-27.

### D-42 One channel count for each dataset

Every `ds[i]["img"]` of one 2-D dataset has the same number of channels. A gray modality (CXR, CT slices)
gives 1 channel: an RGB or RGBA file of a gray image becomes gray. A dataset with color figures (MedPMC,
ROCO, ROCOv2, VQA-RAD) gives 3 channels: a gray file is repeated to 3 channels. The dataset class sets the count.

Example: ChestX-ray14 has a few RGBA PNGs. Each gives `(1, 224, 224)`, as the gray PNGs do, so a default
`DataLoader` batch works.

Why: a batch needs one shape, and a model for one modality expects one channel count.

- Status: Decided (Frank Li, 2026-10-31, Q-5 and V-D-27 of the 2026-10-31 review). The other options were 1
  channel for every image (color figures lose their color), and the channels of the file with a doc line.
- Checked by: `tests/test_transforms.py` (`test_a_gray_dataset_gives_1_channel_for_any_png`,
  `test_a_color_dataset_repeats_a_gray_png_to_3_channels`, `test_the_channel_count_is_in_the_cache_key` and
  `test_the_vqa_datasets_set_their_channel_count`).
- Channel step: the step `_SetChannelsD` of `_base_load_2d` drops an alpha channel. With 1 channel, RGB becomes gray
  with the weights of PIL `convert("L")`, or channel 0 if R = G = B. `RadiologyTransform2D(channels=...)` and the class attribute
  `IMG_CHANNELS` of the VQA datasets set the count. MedPMC, ROCO, ROCOv2 and VQA-RAD set 3.
- Notes: Frank Li approved this base-code change on 2026-11-02 (V-D-27). Before it, `_base_load_2d` kept the
  channels of the file. ChestX-ray14 gave `(4, 224, 224)` for 15 of 3,000 sampled train rows, and MedPMC
  mixed 1 and 3 channels, so a default batch failed. The RGBA ChestX-ray14 files are gray images (R = G =
  B, alpha 255), so they keep their gray values. A 1-channel file gives the same tensor as before. The new
  step is in the cached steps, so each `cache_dir` builds a new cache once. An `embedding_cache` file from
  before 2026-11-02 raises `ValueError` (D-41).
- Related: D-13, D-14, D-27.

### D-27 A reader never clips pixel values

A reader is the step that opens the image file. A reader never clips pixel values. Later transform steps
can clip (D-14).

Some readers must give an 8-bit image, for example `reader="pil"` for a Hugging Face processor. Such a
reader scales a 16-bit image to 0 to 255 with the minimum and maximum of that image. Then it converts the
image to RGB. An 8-bit image stays as it is.

The VOI window of a DICOM file (D-13) is not a clip by the reader. It is the display step that the file
defines. The dataset load and `reader="monai"` apply it to a MONOCHROME image after the file is read.

Why: clipping makes most pixels of a 16-bit CXR white. Scaling with the minimum and maximum of each image
also works for a 12-bit image that is stored in a 16-bit file. Division by 257 makes such an image very dark.

- Status: Decided (Frank Li, 2026-10-28).
- Checked by: `tests/test_encoder_transforms.py` (with `::test_recipe_loader_keeps_an_8_bit_png`,
  `::test_recipe_loader_applies_the_dicom_window`, `::test_every_recipe_image_loader_has_the_display_step`
  and `::test_an_rgb_dicom_skips_the_window_and_stays_as_it_is`).
- Notes: since 2026-11-01, the image loader of every backbone recipe follows this rule, with
  `display_255` in `backbones/_utils.py`. Before, it stretched every image with its minimum and maximum,
  also an 8-bit image, and it skipped the DICOM window (V-B-12). A real run gave the same RAD-DINO input
  as `reader="monai"` for 40 VinDr-CXR DICOMs, 40 SIIM-ACR DICOMs and 30 ChestX-ray14 PNGs.
- Related: D-13, D-14.

### D-15 Cache before augmentation

`cache_dir` stores the loaded and preprocessed tensors, before augmentation. Augmentation runs again for
each sample. Each cache folder name has a short key of the cached steps (`_transform_key` in
`radharmony/dataset/base.py`). So a new load or preprocessing step, for example a new `img_size` or reader,
builds a new cache. A new setting of an augmentation, for example the flip strength, does not, because the
augmentation runs after the cache.

Why: augmentation must be random each time, and loading is slow.

- Status: From the code (MONAI `PersistentDataset` in `radharmony/dataset/base.py`). The transform key:
  Decided (Frank Li, 2026-10-29, V-D-8). The code in the key: Decided (Frank Li, 2026-11-02, V-D-37).
- Checked by: `tests/test_dataset.py::test_a_new_image_size_does_not_read_the_old_cache`,
  `::test_the_same_settings_reuse_the_cache_and_augmentation_is_not_in_the_key`,
  `::test_the_transform_key_is_the_same_in_a_new_process`,
  `::test_the_transform_key_has_the_reader_and_takes_a_list`,
  `::test_the_transform_key_ignores_the_train_mode_and_takes_any_tensor`,
  `::test_the_3d_cache_holds_the_preprocessed_volume`, `::test_the_transform_key_has_the_code_of_a_function_step`,
  `::test_the_transform_key_has_the_code_of_a_radharmony_step`,
  `::test_the_code_in_the_transform_key_is_the_same_for_any_hash_seed`,
  `tests/test_dataset_outputs.py::test_augmentation_differs_across_workers_and_epochs`,
  `::test_torch_manual_seed_repeats_the_augmentation_in_the_main_process`,
  `::test_a_sample_with_no_random_step_takes_no_number_from_the_torch_generator` and
  `tests/test_encoder_transforms.py::test_palette_loader_gives_a_new_cache_key`.
- Code in the key: the key has a short hash of code. It covers each function in a cached step, also deep
  inside it, for example the `func` of `Lambdad`. It also covers the `__call__` of each RadHarmony step. So a changed
  lambda or a fixed step gives a new cache. A docstring is not in the hash, so a docstring edit keeps the
  cache. A new Python version changes the bytecode, so its caches build once again. A fix in a helper that a
  step calls does not change the key, so delete the cache then.
- Seed in the main process: `torch.manual_seed(...)` repeats the random augmentation with any
  `num_workers`, also with `num_workers=0` and with `ds[i]`. Decided (Frank Li, 2026-11-02, Q-3 of the
  2026-11-02 review). The other options were a doc line that the seed works only with `num_workers > 0`,
  and to leave the question open. In the main process, `_WorkerSeeded` seeds the random steps for each
  sample with a number from the PyTorch random generator. So a second `torch.manual_seed(0)` with the same
  dataset repeats the augmentation, and the next `ds[0]` without a new seed is a new augmentation.
  Example: `torch.manual_seed(0); ds[0]` gives the same flip in 2 new processes. A torch `DataLoader` takes
  1 number from the PyTorch random generator first, so its first batch is not `ds[0]`, but it also repeats.
  A transform with no random step takes no number, so feature extraction keeps the random numbers of the user.
- Notes (seed in the main process): since 2026-11-03, base code approved by Frank Li (Q-3 of the
  2026-11-02 review). Before, `torch.manual_seed(0); ds[0]` gave a different flip in 2 new processes. Each
  sample with a random step now takes 1 number from the PyTorch random generator. So with `num_workers=0`,
  a seeded training run with augmentation (for example its dropout) gives other random numbers than before
  this change. The draw uses the global generator, not the `generator=` of a DataLoader. A
  `set_random_state` on the transform has no effect on `ds[i]` now. Use `torch.manual_seed`.
- Notes: each DataLoader worker seeds the random steps from its PyTorch worker seed. This seed is new for
  each worker and each epoch (`_WorkerSeeded` in `radharmony/dataset/base.py`, approved by Frank Li on
  2026-10-29). Before, the workers of a torch `DataLoader` repeated the same augmentation in every epoch
  (V-D-7). `BaseVQADataset` uses the same step. A dataset does not read a cache folder made before
  2026-10-29. Delete these old folders to free the disk. Since 2026-11-02, the 3-D cache holds the
  preprocessed volume (V-D-28, base code approved by Frank Li). Before, 2 steps of the 3-D load were not
  MONAI transforms, so the cache stopped after the load. One CT-RATE cache file was 1,053 MB, and now it is
  3 MB. The samples are the same. The 3-D cache folders from before 2026-11-02 are not read again, so delete
  them to free the disk. The code in the key is also from 2026-11-02 (V-D-37, base code approved by Frank
  Li). Before, 2 lambdas `x * 2` and `x * 3` gave one key, and the second read the old tensors. This change
  gives every cache a new key, so no cache folder from before it is read again. Delete them to free the disk.
- Notes (more steps in the key): Frank Li approved this base-code change on 2026-11-02. The key also sees
  a `functools.partial` with its arguments, and the default arguments of a function. It sees a builtin or
  NumPy ufunc by its name, the object of a bound method, and a `staticmethod`. The critic of V-D-37 found
  these 5 gaps.
  Checked by: `tests/test_dataset.py::test_the_transform_key_sees_partials_defaults_builtins_methods_and_staticmethods`.
- Related: D-28.

### D-49 A val split with random steps warns

`get_datasets(n_splits=...)` and `get_folds()` warn when `val_transform` is None and the dataset transform
has a random step. The val split still uses the dataset transform, as `api.md` says.

Example: `ds = CheXpertDataset(..., transform=t_with_random_flip)`, then `ds.get_datasets(n_splits=5)`.
The val images can come out flipped, and the flip changes in each epoch. The warning tells the user to pass
`val_transform=`.

Why: train augmentation must not run on the val split, but the code cannot remove the random steps safely.
For example, a random crop to 224 x 224 also sets the image size.

- Status: Decided (Frank Li, 2026-11-02, Q-4 of the 2026-11-02 review). The other options were to remove
  the random steps from the val transform, a doc line only, and to leave the question open.
- Checked by: `tests/test_dataset.py::test_a_val_split_with_a_random_step_warns`,
  `::test_a_val_split_with_no_random_step_does_not_warn` and
  `::test_a_random_step_is_found_inside_lists_and_compose`.
- Notes: since 2026-11-03, base code approved by Frank Li (Q-4 of the 2026-11-02 review). The check is
  `_warn_random_val` in `radharmony/dataset/base.py`. A random step is a MONAI `RandomizableTrait`, also
  inside a nested `Compose`. `OneOf`, `SomeOf` and `RandomOrder` are random steps.
- Notes (VQA): since 2026-11-03, base code approved by Frank Li, the `get_datasets` and `get_folds` of
  `BaseVQADataset` give the same warning. The default VQA transform has no random step. Checked by
  `tests/test_vqa_base.py::test_a_val_split_with_a_random_step_warns`.
- Notes (official splits): since 2026-11-03, `get_datasets_predefined` of TAIX-Ray and RANZCR CLiP also
  warns. A val or test split with no transform of its own gets the dataset transform. The warning names the
  argument, for example `test_transform`. Checked by
  `tests/test_official_splits.py::test_taix_ray_predefined_val_and_test_warn_about_a_random_step` and
  `::test_ranzcr_predefined_test_warns_about_a_random_step`.
- Related: D-15, D-18.

### D-37 A warp fills the empty border with black

`with_affine` and `with_elastic_deformation` move the image, so they leave an empty border. The border
gets the black value of the image:

- `RadiologyTransform2D` and `RadiologyTransform3D`: -1, because the image is in [-1, 1]. The pad of
  `pad=True` also uses -1. Before 2026-11-03, their docstrings said zero (V-D-43, docstring fix approved by Frank Li).
- `RadiologyEncoderTransform`: the darkest value of each channel of the image. The encoder has already
  normalized this image (`(pixel - mean) / std`), so -1 is a gray there.

Why: the border is outside the body, as the empty area of an X-ray is. A gray border adds an edge that
no real image has. Example: for RAD-DINO, black is -2.055, and -1 is a gray of 27% brightness.

- Status: Decided (Frank Li, 2026-10-30, V-P-16). The other options were the exact encoder black
  `(0 - mean) / std`, which needs the mean and std, and -1 with a doc line.
- Checked by: `tests/test_encoder_transforms.py::test_with_affine_fills_the_border_with_the_encoder_black`
  and `::test_with_affine_fills_the_border_with_the_darkest_value_of_each_channel`.
- Notes: a real run on 2026-10-30 used 50 Shenzhen CXRs and the RAD-DINO processor. The darkest pixel of
  42 images was within 0.05 of black. The worst image had -1.948 (2.7% brightness). For MedSigLIP (mean
  and std 0.5), -1 is black.
- Related: D-15.

## Backbones and evaluators

### D-16 Backbones match their authors

Each `make_<name>()` factory gives the same features as the authors' own code. This includes the input
size, crop, mean, std and the feature token (CLS token, mean pool or pooled output).

Why: the benchmark compares models. A changed input step changes the score of one model only.

- Status: From the code.
- Checked by: J-05, `tests/test_eva_x.py` for EVA-X (it runs where `timm` is installed and the weights are
  cached), and `tests/test_backbones.py` (it runs in the model venv).
- Two input steps: when the authors' code has 2 different input steps, the factory uses the one of their
  downstream (fine-tune) code. For Ark+, this is a bilinear resize to 896 x 896, a center crop of 768 and the
  ImageNet mean and std. The Ark+ pretraining code uses a PIL bicubic resize instead. Decided (Frank Li,
  2026-10-30, V-B-2). The other option was the pretraining input.
- Two features: when the authors' fine-tune code and their linear-probe code use different features, the
  classification feature follows the linear-probe code. For Ark+, this is the projector output (1376 values,
  `generate_embeddings(x, after_proj=True)`, `emb_dim=1376` in `Linearprobing/main_linearprobing.py`), not
  `forward_features` (1536 values). CheXFound follows the same rule (its 4 CLS tokens). Decided (Frank Li,
  2026-11-02, Q-2 of the 2026-11-02 review). The other options were to keep `forward_features` with a note,
  and an argument with `forward_features` as the default. Since 2026-11-03, `make_ark_plus` loads the
  projector (one Linear, 1536 to 1376) and gives its output. The segmentation feature map does not change.
  Checked by `tests/test_backbones.py::test_ark_plus_classification_feature_is_the_projector_output` (where
  the checkpoint is on the machine).
- Notes (CheXFound input): since 2026-11-03, the input follows the authors' eval transform. After the
  resize and the crop, it clips to [0, 1] and scales each channel with its minimum and maximum
  (`RescaleImage`). Before, the input of 4 CheXpert images was in [-0.07, 1.08] (V-B-14). Checked by
  `tests/test_backbones.py::test_chexfound_input_is_rescaled_to_0_1_as_the_authors_do`.
- MedSigLIP resize: the model card gives 2 ways. It recommends `tf.image.resize` (bilinear, no antialias) to
  reproduce its evals, and it allows the bicubic resize of the Transformers processor. `make_medsiglip` keeps
  the processor. Decided (Frank Li, 2026-11-04, found in the 6c rewrite of the backbone pages). The other
  options were a bilinear resize with no antialias as the evals, and an argument with the processor as the
  default.
- PNG files of a generator: `make_chexagent_generator` and `make_maira2_generator` write one PNG for each image.
  With `tmp_dir=None`, the folder is new under the system temp folder, and it goes when Python exits. A folder
  that the user gives stays. Before, the folder stayed in /tmp. Decided (Frank Li, 2026-11-04). The other
  options were a doc line only, and no change. Checked by:
  `tests/test_backbones.py::test_png_folder_of_a_generator_goes_at_exit_unless_the_user_gave_it`.
- Known gaps (MAIRA-2): the current model card (https://huggingface.co/microsoft/maira-2) differs from
  `make_maira2_generator` in 3 points. The card asks for `transformers>=4.48.0,<4.52`, and the `maira2_gen`
  extra pins `>=4.46.0,<4.47`. The card cleans the output with
  `processor.convert_output_to_plaintext_or_grounded_sequence()`, and the recipe uses `strip()`. And
  `normalize_hypothesis` does not remove the box marks of `get_grounding=True`. No fix yet, because the weights
  are gated and not on the test machine. Decided (Frank Li, 2026-11-04): record the gap, fix it with the real
  model (`eval/TODO.md`). The other options were an untested change, and to get the weights first.
- Device of a generator: `make_chexagent_generator` loads with `device_map=device`, not the `"auto"` of the
  model card. So `device="cuda:3"` puts the whole model on GPU 3, as `make_maira2_generator` does. The device
  does not change the output. Decided by the agent, for Frank Li to confirm (2026-11-03, V-B-15). The other
  option was to keep `"auto"` with a doc line. Checked by
  `tests/test_backbones.py::test_chexagent_generator_puts_the_model_on_device`.
- Notes (Ark+): a real run on 2026-10-30 used Shenzhen (662 images) and a linear probe (5 folds, 3 seeds).
  The bicubic features had a mean cosine of 0.995 to the bilinear features. The AUROC was 0.9973 (bilinear),
  0.9969 (bicubic) and 0.9965 (the v0.1 input, 768 with no crop). Ark+ was pretrained with Shenzhen labels,
  so these AUROC values only compare the inputs.
- Notes (CheXFound): since 2026-11-01, the classification feature joins the CLS tokens of the last 4 blocks
  (4096 values), as the authors' linear probe does. A probe used 234 CheXpert valid images (5 labels, 5
  folds). The macro AUROC was 0.774 with the last CLS and 0.803 with the 4 CLS tokens (V-B-13).
- Notes (Ark+ feature): a probe on 2026-11-02 used 600 SIIM-ACR images (300 with pneumothorax), L2 norm,
  logistic regression, 5 folds and 3 seeds. The AUROC was 0.9590 ± 0.0121 with `forward_features` and
  0.9573 ± 0.0121 with the projector output.

### D-17 Frozen-feature probes L2-normalize features

The linear, kNN, SVM and prototype probes and zero-shot use features with L2 norm 1. The feature cache
stores raw features, and the probe normalizes them after load. A user can turn it off with
`l2_normalize=False`.

Why: all probes then use the same features, so a score difference comes from the classifier only.

- Status: Decided (Frank Li, 2026-05-26).
- Checked by: J-07, `tests/test_evaluator.py::test_finetune_frozen_head_sees_l2_normalized_features`,
  `tests/test_evaluator.py::test_prototype_obeys_l2_normalize` and
  `tests/test_evaluator.py::test_zero_shot_warns_for_l2_normalize_false`.
- Zero-shot: it always uses cosine similarity, as CLIP does. `l2_normalize` has no effect there, so
  `ZeroShotEvaluator` gives a warning for `l2_normalize=False`. Decided (Frank Li, 2026-10-30). The other
  options were the doc line only, and a raw dot product for `l2_normalize=False`.
- Segmentation heads: they use the raw patch features. The rule of this entry is for the classification
  probes. A segmentation head trains with a learning rate, and the scale of the features sets which
  learning rate works. The conv and UPerNet heads normalize inside (`LayerNorm2d`, BatchNorm). Decided
  (Frank Li, 2026-10-30, R-code-4). The other options were L2 in the linear head with a default learning
  rate of 1e-1, and L2 in all 3 heads.
- Notes: a real run on 2026-10-30 used Montgomery lungs, RAD-DINO and the linear head (5 folds, 20 epochs).
  The best lung Dice was 0.970 with raw features (lr 1e-2) and 0.969 with L2 (lr 1e-1). At the default lr
  1e-3, raw gave 0.959 and L2 gave 0.819.

### D-18 Test data stays out of training

The code does not use test labels to train, to pick a model, to pick a threshold, to pick a setting or to
stop early. Every probe drops rows with a NaN label. Every random step has a seed.

Why: a score that used test labels is too high.

- Status: From the code.
- Checked by: J-06, L-009 (`tests/test_lessons.py::test_l009_nan_train_labels_equal_rows_with_label`) and
  `tests/test_evaluator.py::test_same_base_seed_gives_same_result`.
- Related: D-29.

### D-29 The threshold comes from validation data

The threshold for F1, accuracy, balanced accuracy, TPR, TNR, PPV, NPV and MCC comes from a validation set.
The validation set is part of the training data, split by patient. The probe picks the threshold with
`threshold_strategy` (for example, `"youden"`). It picks it one time for each seed in fixed-split mode, or
one time for each fold in k-fold mode. Then it uses that threshold on the test set and on every bootstrap
sample.

Why: a threshold that is picked on the test labels makes every threshold metric too high. AUROC and AUPRC
do not use a threshold.

- Status: Decided (Frank Li, 2026-10-28). The validation set is 10% of the training patients
  (`val_fraction=0.1`, the same default as the fine-tune evaluator and the segmentation probes).
- Checked by: J-06, `tests/test_evaluator.py::test_threshold_comes_from_validation_data`, and for
  zero-shot `::test_zero_shot_threshold_metrics_are_nan_without_a_fixed_threshold` and
  `::test_zero_shot_uses_the_fixed_threshold`.
- Fit part: the 5 probes that train (linear, SVM, kNN, prototype, fine-tune) fit on the other 90% of the
  training patients. Example: with `n_train=100` and one image per patient, a probe fits on 90 patients and
  picks the threshold on the other 10. D-40 says what `n_train` counts. A very small `n_train` can give NaN threshold metrics, because the validation
  part then has too few patients. Decided (Frank Li, 2026-10-30). The other option was a second model on
  100% of the patients for the test scores.
- Notes: a real run on 2026-10-30 used Shenzhen TB, RAD-DINO and the linear probe (5 folds, 3 seeds). For
  50 to 200 patients, the AUROC with a fit on 90% and on 100% was 0.003 or less apart. At `n_train=20`, F1
  was NaN.
- Zero-shot: `ZeroShotEvaluator` has no train data, so it has no validation part. Its threshold metrics
  are NaN, unless `threshold_strategy` is `"fixed:<value>"`. It gives one warning that says this. With
  negative prompts, the warning tells the user to pass `"fixed:0"`, which means "closer to the positive
  prompts". AUROC and AUPRC do not change. Decided (Frank Li, 2026-10-30). The other options were a
  threshold from 10% of the test patients, and the threshold from the test labels with new metric names.
- Notes (zero-shot): a real run on 2026-10-30 used MedSigLIP, VinDr-CXR test (3,000 images) and 19
  labels with 20 or more positives. The mean balanced accuracy was 0.798 with the threshold from the test
  labels. It was 0.769 with a threshold from 10% of the test patients, and 0.767 with `fixed:0`.
- Examples: each zero-shot example in `README.md` and in the use-radharmony skill says that the threshold
  metrics are NaN without `threshold_strategy`, or it passes `threshold_strategy="fixed:0"`. Decided by the
  agent, for Frank Li to confirm (2026-11-02, Q-20 of the 2026-11-02 review). The other option was no
  change, because `zero_shot.md` already says it.
- Notes: `README.md` and the use-radharmony skill now say it in a comment. No V- ID: Q-20.
- Segmentation: the 3 segmentation evaluators use no threshold. The head gives each pixel the class with the
  highest score. A `threshold_strategy` other than `"youden"` gives a `UserWarning` (base code in
  `radharmony/evaluator/segmentation/base.py`, approved by Frank Li on 2026-11-04). Before, it had no effect
  and no message. Decided (Frank Li, 2026-11-04). The other options were to remove the argument, and no
  change. Checked by: `tests/test_evaluator.py::test_seg_threshold_strategy_warns_that_it_is_not_used`.
- Related: D-18, D-19.

### D-19 Variance of a result

- Fixed-split mode (`train_dataset=` and `test_dataset=`): `n_seeds` sets the number of train subsamples,
  each with its own seed. `n_bootstrap` sets the number of bootstrap samples of the test set.
- K-fold mode (`dataset=`): the folds give the variance. The evaluator ignores `n_seeds` and
  `n_bootstrap`, and warns if you set them.
- A result row with `bootstrap = -1` is the point estimate.
- The `save_results()` summary takes the mean over the point-estimate rows. It takes the std and the 95%
  CI over the bootstrap rows. With no bootstrap rows, it takes them over the folds or the seeds. With fewer
  than 2 values, the std and the CI are NaN: 1 value has no spread (V-E-31).
- The std is the population std (`np.nanstd`, ddof=0), as `cross_val_score(...).std()` of scikit-learn gives.
  Example: 5 fold AUROCs give 0.043, and the sample std (ddof=1) gives 0.049. Decided (Frank Li, 2026-11-04,
  found in the 6c rewrite): keep it, and say it in the docs. The other options were the sample std
  everywhere, and the sample std over folds or seeds only.
- A metric can be NaN in some folds or seeds, for example a threshold metric of a rare label (D-29). Then
  the summary gives for each metric the number of runs with a value (`<metric>_n`, for example `f1_n = 4`
  of 5 folds). The probe warns one time for each such label, with the number of NaN runs.

Why: a result shows how much the score changes with the train sample and the test sample. A mean over fewer
runs must be visible in the result file, not only in the log.

- Status: From the code (`radharmony/evaluator/base.py`). The summary rule: Decided (Frank Li, 2026-10-29,
  V-E-15). The count and the warning for NaN runs: Decided (Frank Li, 2026-10-31, Q-8 of the 2026-10-31
  review). NaN std and CI for 1 value: Decided (Frank Li, 2026-11-02, V-E-31). The other options were a warning only, and another threshold rule when the validation part has
  too few positives.
- Checked by: `tests/test_evaluator.py::test_summary_mean_uses_point_rows_and_ci_uses_bootstrap_rows`,
  `::test_summary_std_and_ci_need_2_values` and
  `::test_summary_counts_the_runs_with_a_value_and_warns_once_per_label`.
- Notes: Frank Li approved the count and the warning on 2026-11-02 (Q-8), in `_summarize` of
  `radharmony/evaluator/base.py`. So the warning comes from `save_results()`, one time per label. Example:
  `Label 'nodule': f1 is NaN in 1 of 5 runs. Their means use the other runs (D-19).`
- Related: D-29, D-32.

### D-32 The bootstrap draws patients

Each bootstrap sample draws test patients with replacement, and it keeps all images of each drawn patient.
The metric is still computed over images. If the test data has no `patient_id`, the bootstrap draws images.

Example: a patient with 20 follow-up images is drawn as one unit, so the 20 images go in or out together.
For a test set with one image per patient (for example, VinDr-CXR), the result is the same as a draw of
images.

Why: images of one patient are not independent. A draw of images makes the confidence interval too narrow.

- Status: Decided (Frank Li, 2026-10-28).
- Checked by: `tests/test_evaluator.py::test_bootstrap_draws_patients`,
  `tests/test_evaluator.py::test_bootstrap_uses_test_patients`,
  `tests/test_evaluator.py::test_base_has_no_unused_bootstrap_helper` and
  `tests/test_evaluator.py::test_a_table_with_no_patient_id_column_draws_images`.
- Missing ID: when some rows have a NaN `patient_id`, the evaluators raise a `ValueError`. The message gives
  the number of these rows and their first image paths. Decided (Frank Li, 2026-11-02, Q-5 of the
  2026-11-02 review). The other options were to draw each such row as its own patient, to keep the crash,
  and to leave the question open. A NaN ID is a harmonizer bug (D-01).
  Checked by: `tests/test_evaluator.py::test_a_nan_patient_id_raises_with_the_count_and_the_image_paths`.
- Notes (NaN `patient_id`): since 2026-11-03, in `radharmony/evaluator/sampling.py` (not base code, Q-5 of
  the 2026-11-02 review). `patient_ids_of` raises the `ValueError`. The classification
  evaluators and `ReportGenerationEvaluator` read the ids through it. Before, `bootstrap_indices` raised
  `TypeError` ("'<' not supported between instances of 'str' and 'float'"), and `GroupKFold` raised
  `ValueError` ("Input contains NaN").
- Notes (segmentation): since 2026-11-03 (d596a55), `LinearProbeSegEvaluator` and its subclasses take their
  ids from `group_ids`, as the classification probes do. Before, a table with no `patient_id` column raised
  `KeyError` there. Checked by: `tests/test_evaluator.py::test_seg_probe_takes_the_patient_ids_of_the_classification_probes`.
- Notes: the 8 evaluators use `bootstrap_indices` in `radharmony/evaluator/sampling.py`. The old
  `_bootstrap_metrics`, which drew images, was removed on 2026-10-29 (R-code-11).
- Notes (no `patient_id` column): since 2026-11-03, base code approved by Frank Li (V-E-33). In a table with
  no `patient_id` column, each image is its own patient (`group_ids` in `radharmony/evaluator/sampling.py`).
  So the bootstrap draws images, and the k-fold split and the validation part split images. Before, every
  classification evaluator crashed with `KeyError: 'patient_id'`.
- Related: D-01, D-10, D-18, D-19.

### D-40 `n_train` counts images or patients

`n_train_samples` sets the size of the training sample. The argument `n_train_unit` says what it counts.
`"image"` (the default) draws `n_train` images. `"patient"` draws `n_train` patients and keeps all their
images. With one image per patient, both give the same sample.

Example: with 4 images per patient, `n_train=100` gives 100 images from about 47 patients with `"image"`,
and 100 patients with about 400 images with `"patient"`.

Why: label-efficiency studies usually count images, but a study by patient needs whole patients.

- Status: Decided, planned (Frank Li, 2026-10-31, Q-7 of the 2026-10-31 review). The other options were
  images only (with a fix of the D-29 example), and patients only.
- Checked by: none yet.
- Notes: today the probes draw image rows (`_subsample` in each probe, for example
  `radharmony/evaluator/classification/svm_probe.py:111`). `n_train_unit` is not built yet.
- Related: D-29, D-32.

### D-41 The embedding cache holds features only

`embedding_cache` of the classification evaluators stores the raw features of one encoder, one transform,
one `autocast_dtype` and one list of images in one image folder. Each run reads the labels from the
dataset, not from the cache. A cache from other images, another image folder, another encoder, another
transform or another `autocast_dtype` raises `ValueError`.

Example: after a label fix (for example D-05: CheXpert keeps -1), a run with the old cache uses the new
labels and the old features. The features do not depend on the labels.

Why: a cache that also returns its labels makes a later run use old labels with no error.

- Status: Decided (Frank Li, 2026-10-31, Q-6 of the 2026-10-31 review). The other options were a hash of
  the labels that raises `ValueError` when they differ, and labels from the cache with a doc line. The
  fingerprint: Decided (Frank Li, 2026-11-02, V-E-25).
- Checked by: `tests/test_evaluator.py::test_embedding_cache_takes_the_labels_of_the_dataset`,
  `::test_embedding_cache_sees_every_change_of_the_features`,
  `::test_the_encoder_fingerprint_is_the_same_in_every_process`,
  `::test_an_embedding_cache_with_no_image_folder_or_autocast_dtype_raises`,
  `tests/test_lessons.py::test_l017_the_embedding_cache_raises_when_an_input_of_the_features_changes` (L-017),
  and (labels in float32) `tests/test_transforms.py::test_2d_dtype_is_for_the_image_only` and
  `::test_3d_dtype_is_for_the_image_only`.
- Fingerprint: the cache file keeps a fingerprint of the encoder and the transform. It hashes the weights
  and buffers of each torch model in the encoder. This includes a model in a closure, a
  `functools.partial`, a method or a global that a function loads. It also hashes the functions of the encoder, for example its pool
  function. Of the transform, it hashes the cache key (D-15): the steps before the first random step. A
  random step keeps its last draw, so it would change the fingerprint in every run. Another fingerprint
  raises `ValueError`.
- Notes: Frank Li approved the base-code change on 2026-11-02 (Q-6). Before it, the cache stored the label
  matrix and a cache hit returned it, so a label fix was not seen. Now a cache hit reads `cls` from the
  samples before the transform, with no image load. On 48 real CheXpert rows with -1 and NaN, these labels
  equal the labels of a run with no cache. Frank Li also approved the fingerprint on 2026-11-02 (V-E-25).
  Before it, the check used the model class, the weight count and the sum of the first weight. So another
  pool, another transform, 2 function encoders or other later weights read old features. A file from
  before 2026-11-02 raises `ValueError` once. Delete it to build it again. For 300 million weights, the
  fingerprint takes about 1 s.
- Notes (image folder and `autocast_dtype`): since 2026-11-03, base code approved by Frank Li (V-E-34). The
  cache file also keeps the absolute `base_image_dir` and the `autocast_dtype`. Before, a run on TAIX-Ray
  full size read a cache of TAIX-Ray 512, because the 2 tables have the same `image_path` list. A run in
  fp32 also read a cache made in bf16. A file from before 2026-11-03 raises `ValueError` once. Delete it to
  build it again.
- Known gaps:
  - The fingerprint does not see every change of the code. It does not hash the `forward` of a model
    class or a helper that the pool function calls. It does not hash a model in a global dict, or the
    settings of a model from outside RadHarmony. It does not hash a change in a random step of the transform.
  - A cache hit skips the label checks inside a transform, for example the check for NaN labels of RANZCR.
- Notes (label dtype): the transforms cast only `img` (and `bbox_mask`) to `dtype`. `cls`, `reg` and `mask` stay float32, so a
  cache hit and a run with no cache give the same labels. The backbone recipes do this since 2026-11-03 (V-B-16,
  `tests/test_encoder_transforms.py::test_encoder_transform_casts_only_the_image`). `RadiologyTransform2D` and
  `RadiologyTransform3D` do it since 2026-11-03 (base code approved by Frank Li). Before, they cast the labels
  to `dtype` (bf16 by default), so a label 37.3 became 37.25 and a 3-D mask id 257 became 256. Checked by:
  `tests/test_transforms.py::test_2d_dtype_is_for_the_image_only` and `::test_3d_dtype_is_for_the_image_only`.
- Random steps: the frozen-feature evaluators warn when the transform has a random step before feature
  extraction, because the cache keeps one random draw. The evaluator tutorial uses no `with_*` augmentation
  in the transform of a frozen-feature probe. Decided by the agent, for Frank Li to confirm (2026-11-02, Q-19
  of the 2026-11-02 review). The other options were the tutorial fix only, and no change with a doc line.
  Checked by: `tests/test_evaluator.py::test_a_frozen_feature_evaluator_warns_for_a_random_step` and
  `::test_the_random_step_check_keeps_the_random_state`.
- Notes (random steps): since 2026-11-03, base code approved by Frank Li (Q-19). `_load_or_extract` in
  `classification/base.py` warns when the transform has a MONAI `RandomizableTrait` step. It opens a
  nested `Compose`. The warning names the steps. Fine-tune does not call `_load_or_extract`, so it does not
  warn.
- Notes (tutorial): since 2026-11-04 (6c batch 8), `notebooks/tutorials/evaluator_tutorial.ipynb` gives a
  frozen-feature probe a transform with no random step. Before, it had a flip and an intensity jitter.
- Known gaps (log noise): a second frozen-feature probe in the same Jupyter kernel can print some lines in
  `extract_embeddings` (`radharmony/evaluator/classification/base.py`). They start with `Exception ignored in
  _MultiProcessingDataLoaderIter.__del__`. New worker processes
  copy an old data loader iterator. The results are right. Decided (Frank Li, 2026-11-04): record it, no fix
  now (`eval/TODO.md`). The other options were to reproduce and fix it in base code, and no change.
- Related: D-15, D-17.

### D-46 A prediction names its image by `image_path`

Each image gets the id `image_path` with no image file ending, for example `.png`, `.dcm` or `.nii.gz`.
`predict()` of the segmentation probes and of `FinetuneEvaluator` uses this id, and so does the `sample_id`
of `ReportGenerationEvaluator`. A 3-D study folder named by a dotted UID keeps all of its name.
`save_predictions` writes one PNG per image. Its file name is the id with `__` for
each `/`. Example: `patient1/study1/view1_frontal.jpg` gives the id `patient1/study1/view1_frontal` and
the file `patient1__study1__view1_frontal.png`.

Why: the file name alone is not unique. In CheXlocalize, 668 images had only 4 ids, and the PNGs of one
id overwrote each other (V-E-24).

- Status: Decided (Frank Li confirmed on 2026-11-02 the choice of the agent of 2026-11-01, V-E-24). The other
  options were the file name only (the old way), and PNGs in sub-folders that copy the image folders.
- Checked by: `tests/test_evaluator.py::test_prediction_ids_and_pngs_are_one_per_image`,
  `::test_image_ids_cut_only_an_image_file_ending` and `::test_report_sample_id_is_the_image_id`.
- Related: D-04.

### D-47 Fine-tune `loss="ce"` needs 2 or more label columns

With `loss="ce"` and 1 label column, `FinetuneEvaluator.fit()` and `.evaluate()` raise `ValueError`. The
message tells the user to take `loss="bce"`.

Why: a softmax over 1 logit is always 1, so the loss is always 0 and the model does not train. A run gave
AUROC 0.5 in each fold and the score 1.0 for every image, with no warning (V-E-27).

- Status: Decided (Frank Li confirmed on 2026-11-02 the choice of the agent of 2026-11-01, V-E-27). The
  other option was a 2-class softmax for 1 label column.
- Checked by: `tests/test_evaluator.py::test_finetune_ce_with_one_label_column_raises`.
- Related: D-18.

### D-38 A report section falls back to the other section

`ReportGenerationEvaluator` compares the generated text with one section of the real report
(`ref_section`). When the report has no FINDINGS section, `"findings"` uses the IMPRESSION text. When it has
no IMPRESSION section, `"impression"` uses the FINDINGS text. When it has neither, `"findings"` and
`"impression"` use the whole report. `"both"` joins the 2 sections, or uses the whole report when it has
neither. `"full"` always uses the whole report.

Why: each study with a report gets a reference, so no study is dropped.

- Status: Decided (Frank Li, 2026-10-30, V-E-7). The other option was strict sections: an empty
  reference, so the evaluator drops the pair.
- Checked by: `tests/test_evaluator.py::test_report_parser_headers`.
- Headers: a header is a name, an optional "S", then ":", for example `FINDINGS:`. It counts at the start
  of a line in upper or lower case, or inside a line in capitals only. So "new findings: none" in a
  sentence is not a header. Each section stops at the next header of its stop list in
  `radharmony/evaluator/language/_report_parser.py`, also on the same line. The FINDINGS list has
  IMPRESSION and CONCLUSION. Decided (Frank Li, 2026-10-30, V-E-8). The other option was a header at the
  line start only. Checked by `tests/test_evaluator.py::test_report_parser_headers`.
- The fallback for a report with neither section uses the same header rule. So "No new findings: lungs are
  clear." in a sentence keeps the whole report (V-E-28, since 2026-11-03). Before, the fallback cut the
  report at that lowercase word. Checked by `tests/test_evaluator.py::test_report_parser_headers`.
- The IMPRESSION list also has FINDINGS, so a reference never holds the same text 2 times. Example: for
  "IMPRESSION: Stable. FINDINGS: Heart is normal.", `"impression"` gives "Stable." and `"both"` gives "Heart is
  normal. Stable.". Decided (Frank Li, 2026-10-31, Q-9 of the 2026-10-31 review). The other option was to
  keep the old list, because the layout is rare (7 of 20,000 random MIMIC-CXR reports). Checked by
  `tests/test_evaluator.py::test_report_parser_headers` (since 2026-11-01). Before, `"both"` repeated the
  FINDINGS text in this layout (Q-9).
- Notes: a real run on 2026-10-30 used the MIMIC-CXR test split (2,216 studies, one report each). 499
  reports (22.5%) have IMPRESSION only, and 167 have neither section. So with `"findings"`, 22.5% of the
  references are IMPRESSION text. The header rule changed 7 references (0.3%) from the old parser. Each
  one now stops at a header of the stop list.
- Related: D-32.

### D-50 Both report scoring paths clean the generated text the same way

`ReportGenerationEvaluator.evaluate()` and the Stage B script (`notebooks/evaluator/score_reports_radeval.py`)
clean each generated report with one function in `radharmony`. The function removes the CheXagent grounding
tags, the boxes, the ref marks and the bold marks. It is on by default. An argument turns it off.

Example: a CheXagent report with grounding tags around "effusion" gives the same score in both paths.

Why: one set of generations must get one score.

- Status: Decided (Frank Li, 2026-11-02, Q-6 of the 2026-11-02 review). The other options were raw text in
  both paths, a doc line that the 2 paths differ, and to leave the question open.
- Checked by: `tests/test_evaluator.py::test_both_report_scoring_paths_clean_the_hypothesis`.
- Notes: the function is `normalize_hypothesis` in `radharmony/evaluator/language/hypothesis.py`. The file
  imports only `re`. The Stage B script loads it by its path, so the script still runs with no `radharmony`
  install. The argument is `normalize_hyp=False` in `evaluate()` and `--no-normalize-hyp` in Stage B.
  `generate_only()` writes the raw text. `generations.csv` of `evaluate()` holds the cleaned text that was
  scored. Before 2026-11-03, `evaluate()` only removed the white space at the ends (Q-6).
- Related: D-38.

### D-52 An evaluator argument that the class does not use warns, and one it cannot take raises at once

Some evaluators take a shared argument that they do not use. A value other than the default gives a
`UserWarning`. The result does not change.

Example: `ReportGenerationEvaluator(gen, dataset=ds, device="cpu")` warns, because the generator of the
backbone recipe sets its own device. Pass `device` to `make_chexagent_generator()` or `make_maira2_generator()`.

Why: a user who sets an argument expects an effect. Before, these arguments had no effect and no message.

- Status: Decided (Frank Li, 2026-11-04, found in the 6c rewrite of the evaluator pages). The other options
  were to remove the arguments, and no change.
- The arguments: `threshold_strategy` of the 3 segmentation evaluators (D-29, base code in
  `radharmony/evaluator/segmentation/base.py`, approved by Frank Li on 2026-11-04), and `device` of
  `ReportGenerationEvaluator`. `ZeroShotEvaluator` already warned for `n_seeds` other than 1.
- Checked by: `tests/test_evaluator.py::test_seg_threshold_strategy_warns_that_it_is_not_used` and
  `::test_report_device_warns_that_it_is_not_used`.
- An encoder that the class cannot take raises `TypeError` in `__init__`, not later in `evaluate()`.
  `FinetuneEvaluator` moves the encoder and trains its weights, so it needs an `nn.Module`, also with
  `freeze_backbone=True`. Before, a plain function failed in `evaluate()` with `AttributeError` after the data
  loaded. Decided (Frank Li, 2026-11-04). The other options were to accept a plain function with
  `freeze_backbone=True`, and no change. Checked by:
  `tests/test_evaluator.py::test_finetune_with_a_plain_function_raises_type_error_at_once`.
- A zero-shot prompt can be longer than the `model_max_length` of a SigLIP processor (64 tokens for MedSigLIP
  and SigLIP 2, 256 for XraySigLIP). Then `ZeroShotEvaluator` cuts it to that length and warns. Before, the text model
  raised `ValueError` ("Sequence length must be less than max_position_embeddings"). Decided (Frank Li,
  2026-11-04). The other options were a clear `ValueError`, and a cut with no warning. Checked by:
  `tests/test_evaluator.py::test_zero_shot_cuts_a_long_siglip_prompt_and_warns`.
- Related: D-29.

## Repo rules

### D-20 Base code needs approval

A change to base code needs approval first. `CLAUDE.md` lists the base files and the approval steps.
Use a subclass override when you can.

Why: every dataset or evaluator shares base code. A small change can change results everywhere.

- Status: Decided (`CLAUDE.md`).
- Checked by: J-01, `scripts/evalkit/base_guard.py`.

### D-21 One file for data paths

Data paths are in `configs/data_paths.json`. Code, tests and docs do not contain NAS paths.

Why: when a folder moves, you change one file.

- Status: Decided (Frank Li, 2026-10-27).
- Checked by: L-002 and L-006 (`scripts/evalkit/lessons.py`), L-007 (`tests/test_data_paths.py`), and
  `tests/test_evalkit.py::test_l002_checks_the_skills` (L-002 reads `.claude/skills/`).
- Skills: L-002 also checks `.claude/skills/`, because agents read and copy the skills. Decided (Frank Li,
  2026-10-30). The other options were every tracked text file, and no change.
- Checkpoint copies: the local `.ipynb_checkpoints` copies under `docs/` stay. Git ignores them, and every
  check skips them (`blocks.py`, `ste_lint.py`, `wiki_audit.py`, `mkdocs.yml` and L-002). So their old NAS
  paths are not a gap. Decided (Frank Li, 2026-10-31, Q-23 of the 2026-10-31 review). The other option was
  to delete the folders.
- Known gaps: `notebooks/datasets/dataset_integrity_check.ipynb` keeps its own NAS paths in a paths cell,
  and it names 43 of the 128 dataset classes (A-15). `scripts/check_datasets.py` reads the config and
  covers every class.
- Notes: on 2026-10-30, `.claude/skills/` had 0 NAS paths, and the MedSeg and QaTa pages lost their last
  NAS paths.
- A user's own file (Frank Li, 2026-11-04): `configs/data_paths.example.json` is committed. A user copies it to
  `configs/data_paths.json` (out of git in the public repo), or sets `RADHARMONY_DATA_PATHS`, and keeps the
  entries of the datasets to check. A missing file or key raises `MissingPathsError` with a hint. A check skips
  such a dataset, a `tier=nas` claim is skipped, and the tests that need data skip. The dev repo commits a
  complete file (`"complete": true`), so only there the tests check every key. The other options were to do it
  later with the public copy, and no change. Checked by: `tests/test_data_paths_public.py`.
- Check names: each `DatasetSpec` of `scripts/check_datasets.py` is named by its registry key, or
  `<registry key>_<variant>` for a second check of the same class (for example `ranzcr_clip_mask`). Before
  2026-11-04, 16 checks had short names (for example `montgomery`). The old names still work, with a warning
  (`OLD_NAMES`). Decided (Frank Li, 2026-11-04). The other options were to accept both names, and no change.
  Checked by: `tests/test_scripts.py::test_each_check_name_is_a_registry_key_or_a_variant_of_one`.

### D-22 Docs do not hide code bugs

If a doc example fails because the code is wrong, fix or report the code. Do not change the doc to match
the bug.

Why: a doc that matches a bug hides the bug from users and from the checks.

- Status: Decided (Frank Li, 2026-10-27).
- Checked by: J-09.

### D-23 Tests get their expected values from the source

A test gets its expected value from the source data or the spec, not from the code under test.

Why: a value from the code under test makes the test pass on the bug.

- Status: Decided (Frank Li, 2026-10-27).
- Checked by: J-08.

### D-39 `check_dataset`: wrong values are ERROR, missing outputs are WARN

`check_dataset` (`radharmony/integrity.py`) gives ERROR when a value breaks a rule:

- No sample loads. This includes a dataset with 0 samples.
- An image has NaN or Inf pixels.
- A box is not 4 or 6 values in [0, 1] (D-07).
- A box has a min that is not less than its max, for example a wrong axis order (D-07).

It gives WARN when an output is not complete:

- Some samples do not load, or a key is missing in some samples.
- No sample has a requested output.
- `output_bbox=True`, and no sample has `bbox_labels` (D-11).
- Every label, mask or report is empty.

`scripts/check_datasets.py` exits with code 1 when a dataset has ERROR.

Why: wrong values break training with no error message, so a script or CI must stop. Missing outputs can
be normal data.

- Status: Decided (Frank Li, 2026-10-30, V-A-5). The other option was WARN for bad boxes and NaN or Inf
  pixels.
- Checked by: `tests/test_integrity.py::test_bad_values_change_the_status` (with the 2 examples of "Box order"),
  `::test_requested_output_that_no_sample_has_warns`, `::test_boxes_with_no_labels_warn`,
  `::test_zero_samples_is_an_error`.
  `tests/test_scripts.py::test_check_datasets_zero_samples_exits_1` checks the exit code.
  `tests/test_scripts.py::test_check_datasets_covers_dataset` checks which script holds which spec.
- Default run: a spec whose data has `"status": "restricted"` or `"not_on_nas"` in
  `configs/data_paths.json` gets `default=False`. It runs only when the user names it with `--datasets`. So
  a run with no `--datasets` can pass for every user. Decided (Frank Li, 2026-10-30). The other options
  were a run of every spec with a new result BLOCKED, and the `default` values from before 2026-10-29.
- Box order: ERROR for a box with min not less than max. Example: `[0.6, 0.2, 0.4, 0.1]` (a wrong axis order)
  and `[0.5, 0.5, 0.3, 0.3]` (no area). Decided (Frank Li, 2026-10-31, Q-1 of the 2026-10-31 review). The
  other options were WARN, and no check.
- Box labels: WARN when `output_bbox=True` and no sample has `bbox_labels`. Decided (Frank Li, 2026-10-31,
  Q-2 of the 2026-10-31 review). The other option was no check, because `bbox_labels` is an optional table
  column. Since 2026-11-01, every class that gives boxes also gives `bbox_labels` (V-D-29, V-H3-23). A
  real run on 2026-11-01 gave OK with both rules. It checked the 7 readable 2-D box datasets (25 samples
  each) and the 3 3-D box datasets (5 samples each).
- Empty dataset: ERROR for a dataset with 0 samples. Each spec of `check_datasets.py` is a whole dataset
  with its default arguments, so 0 rows means a broken path, join or filter. Decided (Frank Li, 2026-11-02,
  Q-1 of the 2026-11-02 review). The other options were WARN with a line in this entry, and to leave the
  question open.
- Check scripts: `scripts/check_datasets.py` checks every dataset whose class is in the public package.
  `scripts/check_datasets_internal.py` adds only the datasets with internal data (Emory CXR). So MS-CXR,
  MS-CXR-T and MIMIC-Ext-CXR-QBA are in `check_datasets.py`. Decided (Frank Li, 2026-10-30, A-6). The other
  option was to keep them in the internal script.
- Notes: this rule found the bad boxes of SIIM-COVID19 and RAD-ChestCT, which a fix on 2026-10-29
  removed. A real run on 2026-10-30 (50 samples each) gave OK for both. On that day, 35 of the 110 specs
  had `default=False`. 24 were restricted and 6 not on the NAS. The other 5 were Chest ImaGenome silver
  (3, slow) and the 2 Kaggle test classes of D-25.
- Related: D-07, D-14, D-21.

### D-45 A known-problems list only gets shorter

`run_all.py --check-shrink <ref>` fails when a known-problems list has an entry that the list at `<ref>`
did not have, or a higher count. Only a new rule can add its old problems. You name each new rule:
`--check-shrink origin/main --new-rule L-013`.

Why: an old rule with 0 known problems has no entry in the old list, so the check cannot tell it from a
new rule. Example: L-002 (no NAS paths in docs) has no entry today. A new NAS path in a doc page must
fail, not join the list.

- Status: Decided (Frank Li, 2026-11-01, V-A-38). The other options were to read the rule list of each
  check at `<ref>` (RULES.md, `pyproject.toml`, the STE rules), and no change.
- Checked by: `tests/test_evalkit.py::test_shrink_errors`, `::test_shrink_errors_for_an_old_rule_with_no_entry`
  and `::test_new_rules_come_from_the_pr_description`.
- CI: the PR description names each new rule on its own line, for example `New rule: L-013`. The CI step
  passes the PR description with `--pr-body FILE`. `run_all.py` adds the rule of each line to
  `--new-rule`. A line in a code block does not count. Decided (Frank Li, 2026-11-01). The other options
  were a CI failure that a maintainer merges by hand, and a file of new rules in the PR.
- Named rules: a named rule that has an entry in the old list is not new, so its entries still may not
  grow.
- Notes: a list with no file at `<ref>` belongs to a new check, so it cannot grow.
- Related: D-20.

## Open questions

No entry has the status "Open" now. No decided entry has an open part now.

To add an open question, add a D- entry with the next free number and the status "Open". Put it in the
section where it belongs, and add it to the index. A maintainer decides each question. Then the status
changes to "Decided", with the name and the date.
