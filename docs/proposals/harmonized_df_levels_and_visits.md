# Proposal: harmonized table at image, study, or patient level, plus visit number

**Status:** Draft plan, nothing implemented yet
**Date:** 2026-10-04
**Pertains to:** `radharmony/harmonizer/base.py`, `radharmony/dataset/base.py`, a new `radharmony/harmonizer/levels.py`, and the harmonizers that have date or order info.

## Goal

1. Let users get the harmonized table at **image**, **study**, or **patient** level. Today it is image level only (one row per image).
2. Add a **visit number** per study (1 = the patient's first study, 2 = second, ...) for longitudinal work. Build it from the study date and time when the dataset has them.

## Main idea

Keep the image-level table as the single source of truth. Build the study and patient tables **from** it with one generic function.

```python
df = h.harmonize()                          # image level, same as today
df = df[df["view_position"] == "PA"]        # user filters at image level
study_df = to_level(df, "study")            # one row per study
patient_df = to_level(df, "patient")        # one row per patient
```

Why this way:

- It follows the package rule: harmonize first, then filter. Filters on image properties (view, mask present) must happen before grouping, so `harmonize()` itself should not take a `level=` argument.
- It works for all harmonizers, including the 36 with their own `harmonize()`, because it only needs the output table.
- It works on saved tables and on user-filtered tables. The save format does not change.
- `to_level` lives in a new module, so it is not a base code change.

## Part 1: study and patient level

### Example (fake IDs)

Image level (today):

| patient_id | study_id | image_path | view_position | pleural_effusion | report | study_date | study_time | visit_number |
|---|---|---|---|---|---|---|---|---|
| 1001 | 501 | p1001/s501/a.jpg | PA | 1.0 | "Small effusion..." | 2150-01-03 | 08:10:00 | 1 |
| 1001 | 501 | p1001/s501/b.jpg | LATERAL | 1.0 | "Small effusion..." | 2150-01-03 | 08:10:00 | 1 |
| 1001 | 502 | p1001/s502/c.jpg | AP | 0.0 | "No effusion..." | 2150-02-11 | 14:02:30 | 2 |

`to_level(df, "study")`:

| patient_id | study_id | n_images | image_path | view_position | pleural_effusion | report | study_date | study_time | visit_number |
|---|---|---|---|---|---|---|---|---|---|
| 1001 | 501 | 2 | [a.jpg, b.jpg] | [PA, LATERAL] | 1.0 | "Small effusion..." | 2150-01-03 | 08:10:00 | 1 |
| 1001 | 502 | 1 | [c.jpg] | [AP] | 0.0 | "No effusion..." | 2150-02-11 | 14:02:30 | 2 |

`to_level(df, "patient")`:

| patient_id | n_studies | study_id | visit_number | study_date | study_time | pleural_effusion | image_path |
|---|---|---|---|---|---|---|---|
| 1001 | 2 | [501, 502] | [1, 2] | [2150-01-03, 2150-02-11] | [08:10:00, 14:02:30] | [1.0, 0.0] | [[a.jpg, b.jpg], [c.jpg]] |

### Rules

**Study level** groups by `(patient_id, study_id)`:

- `image_path` is always a list, even with one image. Add `n_images`.
- Images inside a study are sorted by `image_path`, so the output is the same every run.
- `report` and **label columns** are always **one value per study**. Finding labels come from the report (or a study-level label file), so they belong to the study, like the report. Datasets with truly per-image labels (VinDr-CXR, ChestX-ray14, TAIX-Ray) have one image per study, so nothing is grouped there.
- Every other column follows one rule. It is checked across the **whole table**, not per row, so a column never mixes single values and lists:
  - Same value inside every study: keep one value. Example: study date.
  - Differs inside some study: list, in the same order as `image_path`. Examples: view_position, mask_path, bbox, series_id.

**Report or labels that differ inside a study.** This only happens with data problems. We keep the report and labels of the **first image** (sorted by `image_path`), so the report and its labels always come from the same image and match. We warn with the count. Known cases:

- CheXpert-Plus has one report row per image. 33,971 of 187,711 study folders (18%) have more than one distinct report text. Checked 2026-10-04: it is the **same report, de-identified separately for each image**. Only the fake values differ: dates (`3/8/2012` vs `3-27-2017`), phone numbers, people's names, and hospital or company names. The medical text is the same. So the study folder is the right study, and taking the first image's report is safe. Do **not** use the accession as the study id: it is also a fake value per image (and 14,123 accession values appear in more than one folder). `patient_report_date_order` is also per image (e.g. 5 and 6 in one folder), so use the min per folder for visit order.
- BRAX: labels differ inside 2 of 23,276 studies.

**Patient level** groups the study-level table by `patient_id`:

- Every per-study column becomes a list in **visit order**: study_id, visit_number, modality, report (one per visit), labels (one per visit).
- Per-image columns become a list of lists, one inner list per visit: image_path, view_position.
- Columns that are the same for the whole patient (e.g. sex) stay one value.
- Add `n_studies`.
- Labels are **not** merged across studies, because a patient's labels change over time.

### What each dataset looks like at each level

This works on every table. For some datasets a level adds nothing:

| Case | Datasets | Study level | Patient level |
|---|---|---|---|
| study_id is really the image id | ChestX-ray14, TAIX-Ray, RANZCR CLiP | same as image level | groups real patients |
| patient_id equals study_id or the image id | VinDr-CXR, VinDr-PCXR, RSNA Pneumonia, RSNA PE, RSNA cervical, RSNA lumbar, RSNA abdominal, bone age, Montgomery, Shenzhen | groups images per study if any | same as study level |
| 3-D, one row per volume or series | CT-RATE (several reconstructions per scan), RSNA lumbar and abdominal (several series per study) | groups series or volumes | groups studies |
| VQA, one row per question | VQA-RAD, GEMeX, MIMIC-Ext-CXR-QBA, ROCO | **out of scope** (rows are questions, not images) | out of scope |

`to_level` warns when the same `image_path` appears twice inside a study (see "Problems found" below).

### Modality

Every row gets a standard `modality` column, so the patient level can show which modalities each patient had.

- **Values:** short codes based on DICOM: `XR` (radiograph, covers DICOM `CR` and `DX`), `CT`, `MR`, `US`, ... When the source has the DICOM tag, also keep the raw code in `dicom_modality` (Emory CHORUS reads it today but does not output it).
- **Source:** a class attribute on each harmonizer, e.g. `MODALITY = "XR"` (one line each: CXR datasets `XR`; CT-RATE, RAD-ChestCT, RSNA PE, cervical, abdominal `CT`; RSNA lumbar `MR`). The base fills the column from it when the harmonizer did not set it per row. A multi-modality dataset sets it per row from the DICOM tag.
- **Study level:** one value per study (general rule; a study with mixed series, e.g. PET/CT, becomes a list).
- **Patient level:** one value per visit, in visit order, lined up with `study_id`.

Example, patient level:

| patient_id | n_studies | study_id | visit_number | modality | image_path |
|---|---|---|---|---|---|
| 1001 | 3 | [501, 900, 502] | [1, 2, 3] | [XR, CT, XR] | [[a.jpg, b.jpg], [ct900/], [c.jpg]] |

Find patients who had both an X-ray and a CT:

```python
patient_df[patient_df["modality"].apply(lambda m: {"XR", "CT"} <= set(m))]
```

**When does a patient have more than one modality?** Every dataset today has one modality (CHORUS keeps only its X-ray subset). It happens when:

1. a multi-modality dataset is added (e.g. the full CHORUS DICOM tree), or
2. tables from several harmonizers are combined for the same patients:

```python
df = pd.concat([xr_df.assign(source="xr_set"), ct_df.assign(source="ct_set")])
df = add_visit_number(df)            # renumber so X-ray and CT visits are ordered together
patient_df = to_level(df, "patient")
```

Combining needs the same `patient_id` format in both tables and `study_id` values that do not clash. `to_level` checks this and warns. Today the same MIMIC study has two ID formats across harmonizers (see "Problems found" #7), so those tables would not join without a fix. The optional `source` column (dataset name) records where each row came from.

## Part 2: study date, time, and visit number

### New standard columns

| Column | Type | Meaning |
|---|---|---|
| `study_date` | pandas datetime (date only) | Study date (DICOM `StudyDate`). |
| `study_time` | time (`HH:MM:SS`), may be empty | Study time (DICOM `StudyTime`). Empty when the source has only a date, so a date-only study never gets a fake midnight time. |
| `visit_number` | int (nullable) | 1, 2, 3... order of the study within the patient. |
| `days_from_first` | float (optional) | Days since the patient's first study. Useful because MIMIC and Emory dates are shifted per patient: the year is fake but the gaps are real. |

**Study date and time are recorded in the harmonized table at every level** (see the example tables in Part 1):

- Image level: every image row has its study's date and time (the same value for all images of a study).
- Study level: one value per study.
- Patient level: one value per visit, in visit order, lined up with `study_id`.
- The columns are always present, even for datasets without dates (then empty). So tables from different datasets have the same columns and can be combined.
- One value per study: the earliest image of the study. MIMIC images in one study can differ by minutes (4,591 studies).
- Dates are kept as the source gives them. MIMIC years (2110 to 2208) are shifted on purpose; we do not change them.

### How visit number is computed

One generic function, `add_visit_number(df)`, in the same new module:

1. Take one row per study.
2. Inside each patient, sort studies by `study_date`, then `study_time`, then by an order column if the dataset has one (`visit_order`, e.g. ChestX-ray14 `Follow-up #`), then by `study_id` to break ties. An empty time counts as a tie with any time on the same day.
3. Number them 1..N. Each study gets its own number, even when two studies share a date. Warn with the tie count.
4. If there is no date and no order: `visit_number = 1` when the patient has one study, otherwise empty (NaN). We do not guess the order from the study id.

Example of ties (date-only data, e.g. PadChest):

| patient_id | study_id | study_date | study_time | visit_number |
|---|---|---|---|---|
| 7 | 30 | 2012-03-01 | (empty) | 1 |
| 7 | 31 | 2012-03-01 | (empty) | 2 (tie broken by study_id, counted in the warning) |
| 7 | 35 | 2013-06-20 | (empty) | 3 |

**Filtering and visit number.** Each harmonizer computes `visit_number` on the full table, so it means "the Nth study in this patient's full history". If a user keeps only some studies, visit 3 stays 3. To renumber inside the filtered table, call `add_visit_number(df)` again.

### Where each dataset gets its date, time, or order

From the data survey (headers read on NAS, 2026-10-04):

| Source | Datasets | Filled | Notes |
|---|---|---|---|
| Date + time | MIMIC-CXR-JPG (`StudyDate`, `StudyTime` in metadata.csv), MIMIC-CXR DICOM (same CSV, new optional arg; or the DICOM header when `dicom_base_dir` is set), MS-CXR and MS-CXR-T (join MIMIC metadata), Chest ImaGenome Silver (`StudyDateTime` in the JSON it already parses) | `study_date`, `study_time` | MIMIC: only 4 exact ties out of 227,835 studies. Dates are shifted per patient. |
| Date only | PadChest (`StudyDate_DICOM`), BRAX, TAIX-Ray, ReXGradient, Emory CXR, Emory CHORUS (these 5 already output `study_date`), CT-RATE (`StudyDate` in the metadata file it already merges) | `study_date` | Same-day ties happen (PadChest 2,279 groups, TAIX-Ray 4,918). Emory CXR could get `study_time` for about 79% of images from its acquisition table. |
| Order column | ChestX-ray14 (`Follow-up #`), CheXpert-Plus (`patient_report_date_order`, take the min per study), Chest ImaGenome (`StudyOrder`) | none (dates not released); `visit_number` from the order | ChestX-ray14 follow-up is not strictly by time: age goes down in about 6% of steps. |
| Nothing, but repeat patients | SIIM COVID-19 (dates hashed), RANZCR CLiP, RAD-ChestCT, CheXpert v1.0 (`studyN` matches true order only 86% of the time) | none | `visit_number` empty for multi-study patients. CheXpert could later borrow CheXpert-Plus order. |
| Nothing, one study per patient | VinDr-CXR, SIIM-ACR, RSNA Pneumonia, RSNA PE, RSNA cervical, RSNA lumbar, RSNA abdominal, bone age, Montgomery, Shenzhen | none | `visit_number = 1`. SIIM-ACR and RSNA Pneumonia DICOMs have placeholder dates (`19010101`): treat as empty. |
| Not checked (no read permission) | VinDr-PCXR, OpenI | probably none | |

Each harmonizer with date or order info sets `study_date` and `study_time` (or `visit_order`) and calls `add_visit_number(self.df)`. This is explicit in each harmonizer, no hidden base logic.

## Must fix first: new columns become labels

Done on 2026-11-02 for `harmonizer=`, `harmonizer_path=`, `harmonized_df=` and `set_harmonized_df()` (D-08): the dataset takes the declared labels of the harmonizer. `save()` was fixed on 2026-10-29 (V-H1-15).

Today, when a dataset gets its table through `harmonizer=` or `harmonizer_path=`, every column that is not in a fixed list is treated as a label (`dataset/base.py` `_infer_label_cols_from_df` / `_sync_label_cols_from_harmonizer`). This ignores the class `LABEL_COLS`.

Example (checked by running it): a harmonizer with `EXTRA_OUTPUT_COLS = ["study_date", "visit_number"]` and `LABEL_COLS = ["Effusion"]` gives a dataset with `LABEL_COLS = ["bbox_labels", "effusion", "study_date", "visit_number"]`. With `output_cls=True`, `visit_number` becomes an extra label in `cls`.

This already happens today for `study_date` and `age` in BRAX, TAIX-Ray, ReXGradient, Emory CXR and Emory CHORUS. `save()` has the same problem for harmonizers with empty `LABEL_COLS`.

Fix:

- Use the harmonizer's own label list when it has one (`h._label_columns_for_output(df)`).
- Keep inference only as a fallback, and exclude known non-label columns: core columns, `EXTRA_OUTPUT_COLS`, `REG_COLS`, and the new visit and modality columns.
- Merge the four separate "non-label" column sets into one: `_HARMONIZED_CORE_COLS` (harmonizer/base.py, removed on 2026-10-29), `_HARMONIZED_NON_LABEL_COLS` (dataset/base.py), datathon26 `_NON_LABEL_COLS`, and the unused `_VQA_CORE_COLS`.

## Base code changes needed (each needs your approval before editing)

| File | Change | Phase |
|---|---|---|
| `radharmony/dataset/base.py` | Label inference fix above | 0 |
| `radharmony/harmonizer/base.py` | One shared non-label column set; `save()` uses it | 0 |
| `radharmony/harmonizer/base.py` | Add `study_date`, `study_time`, `visit_number`, `days_from_first`, `modality`, `dicom_modality`, `source` to `_OPTIONAL_COLS`, so they reach the output without each harmonizer listing them in `EXTRA_OUTPUT_COLS`. Add the `MODALITY` class attribute and fill `modality` from it when a harmonizer did not set it. | 2 |
| `radharmony/harmonizer/base.py` (optional) | Convenience method `h.get_harmonized_df(level="study")` that calls `to_level(h.harmonized_df, ...)` | 1 |
| `radharmony/evaluator/classification/base.py` | Store `image_path` and `study_id` per row in the embedding cache; warn on old caches | 5b |

Not base code: the new `radharmony/harmonizer/levels.py` (`to_level`, `add_visit_number`, `pick_view`, `add_prior`, `pair_by_date`), `radharmony/evaluator/fusion.py`, `radharmony/dataset/slot_dataset.py`, and the per-dataset harmonizer edits.

## Phases

| Phase | Work | Tests |
|---|---|---|
| 0 | Create the branch. Fix label inference and merge the non-label sets. | Extra columns are not labels; `save()`/`load_from_saved()` round trip keeps the right `LABEL_COLS`. |
| 1 | `to_level()` in `levels.py`. Docs: new section in `docs/wiki/api.md`. | Synthetic MIMIC-like table: study and patient rows, single-value vs list columns, report and labels kept once per study (from the same image, with a warning when they differ), duplicate image warning, deterministic order. |
| 2 | `add_visit_number()`, new standard columns, `MODALITY` attribute. Set `MODALITY` on every harmonizer (one line each). | Ties, missing dates, single-study patients, renumber after filter. Combined X-ray + CT table: patient row has `modality = [XR, CT, XR]` in visit order; clashing `study_id` warns. |
| 3a | Date + time datasets: MIMIC-CXR-JPG, MIMIC-CXR, MS-CXR, MS-CXR-T, Chest ImaGenome Silver. | Real-data check: MIMIC visit numbers match `StudyDate`/`StudyTime` order on a sample of patients. |
| 3b | Date-only datasets: PadChest, BRAX, TAIX-Ray, ReXGradient, CT-RATE, Emory CXR, Emory CHORUS. Convert the existing `study_date` values (int `20120301` or text `2012-03-01` today) to pandas dates. | Real-data spot check per dataset. |
| 3c | Order-column datasets: ChestX-ray14, CheXpert-Plus. Single-study datasets get `visit_number = 1`. | Spot check. |
| 4 | Docs: a "Visit info" line on each dataset wiki page; harmonized table contract in `architecture.md`. | Wiki audit tool. |
| 5a | Table helpers for training: `pick_view`, `add_prior`, `pair_by_date` (see Phase 5). | See Phase 5. |
| 5b | Embedding path: store `image_path` in the embedding cache; `slot_features` / `pool_features`. | See Phase 5. |
| 5c | `SlotDataset`: end-to-end training with one image per slot. | See Phase 5. |

Phase 1 to 4 change only the DataFrame. Datasets, transforms, evaluators and the app keep working on image-level rows. Splits already group by `patient_id`, so they are fine at any level.

## Phase 5: training on study and patient tables

Goal: use the study and patient tables to train models that take several inputs per sample: several views (frontal + lateral), the prior study (longitudinal), or several modalities (X-ray + CT), plus report and table features (age, sex).

**Main idea: one column per input ("slot").** Helpers turn a study or patient table into a table where each slot column holds **one** image path. The dataset then only loads. Choosing images stays in the DataFrame, where it is easy to inspect and filter.

```python
t = to_level(df, "study")
t = pick_view(t, "img_frontal", views={"PA", "AP"})
t = pick_view(t, "img_lateral", views={"LATERAL", "LL"})
t = add_prior(t, cols=["img_frontal", "report"])    # adds img_frontal_prior, report_prior, days_since_prior
```

| patient_id | study_id | visit_number | img_frontal | img_lateral | img_frontal_prior | report_prior | days_since_prior |
|---|---|---|---|---|---|---|---|
| 1001 | 501 | 1 | a.jpg | b.jpg | (empty) | (empty) | (empty) |
| 1001 | 502 | 2 | c.jpg | (empty) | a.jpg | "Small effusion..." | 39 |

Order of work: 5a, then 5b (cheap, no image-loading changes), then 5c.

### 5a. Table helpers (in `levels.py`, not base code)

| Helper | What it does |
|---|---|
| `pick_view(study_df, name, views)` | New column `name` with the first image (sorted by `image_path`) whose `view_position` is in `views`; empty if none. |
| `add_prior(study_df, cols, k=1)` | Adds `<col>_prior` from visit `visit_number - k` of the same patient, and `days_since_prior`. `k=-1` gives the **next** visit, e.g. to use a later label as a prognosis target (only when asked). |
| `pair_by_date(df, a="XR", b="CT", max_days=30)` | For each study of modality `a`, the study of modality `b` of the same patient with the closest date, within `max_days`. Columns get the modality as suffix (`img_xr`, `img_ct`). |

### 5b. Embedding path (two steps)

1. Run a frozen encoder per modality at image level, with the existing evaluator code (`extract_embeddings` / `embedding_cache`).
2. Match embeddings to slot columns by `image_path`, then train a small fusion model on the arrays.

```python
X, present = slot_features(t, emb, slots=["img_frontal", "img_lateral", "img_frontal_prior"])
# X: [n_studies, 3, D], present: [n_studies, 3] (0 = image missing, row of zeros in X)
X_pool = pool_features(to_level(df, "study"), emb)   # [n_studies, D], mean over all images of the study
```

Needed:

- The embedding cache must store `image_path` (and `study_id`) for each row. Today it stores only `feats`, `labels`, `patient_ids`, matched by row order (`evaluator/classification/base.py` `_load_or_extract`). **Base code change.** Old caches without `image_path` must be re-extracted; warn when one is loaded.
- `slot_features` and `pool_features` in a new module (e.g. `radharmony/evaluator/fusion.py`).

Why first: no change to image loading, works for 2-D and 3-D, one encoder per modality, and it is fast to try ideas.

### 5c. `SlotDataset` (end-to-end training, new file `radharmony/dataset/slot_dataset.py`)

```python
ds = SlotDataset(
    t,
    slots={"img_frontal": t2d, "img_lateral": t2d, "img_frontal_prior": t2d},   # one transform per slot
    base_image_dir="/path/to/files/",        # or {source: dir} for combined tables
    output_cls=True, output_report=True, extra_cols=["days_since_prior"],
)
train_ds, val_ds = ds.get_datasets(n_splits=5)      # split by patient_id, like today
```

- Each sample: one tensor per slot, plus `<slot>_present` (1 or 0). A missing image becomes a blank tensor with the slot's shape. Every sample then has the same keys and shapes, so the normal PyTorch collate works.
- Each slot uses its own transform (2-D for X-ray, 3-D for CT). The dataset runs the existing `RadiologyTransform2D` / `3D` on `{"img": path}` and renames the output to the slot name. So `transforms.py` (base code) does **not** change.
- Cache each loaded image by path, not per sample. Otherwise an image used as "current" in one sample and as "prior" in another is loaded and cached twice.
- Labels: study-level `cls` from the table. Patient-level targets come from a chosen visit (e.g. via `add_prior(k=-1)`).
- Not in 5c: a variable number of images per sample (padded list + custom collate). Add later only if needed.

### Use cases this unlocks

- **MAIRA-2 with priors:** current frontal, current lateral, prior frontal, prior report. Our recipe passes `None` for the prior inputs today (`backbones/maira2_gen.py`).
- **MS-CXR-T progression:** current + prior image. Today the prior image is never loaded ("Problems found" #4).
- **Report generation once per study:** one sample per study with all its views, instead of the same reference report scored once per image.
- **X-ray + CT** pairs for multi-modality models.

### Rules

- **Split by patient.** A patient's visits must not end up in both train and validation.
- **No future leak.** To predict at visit k, use only visits before k. `add_prior` looks back by default; looking forward needs `k=-1` on purpose.
- **Shifted dates.** MIMIC and Emory shift dates per patient. Differences inside one patient are real; dates from two different sources can only be compared if they share the same shift (e.g. the same institution's export). `pair_by_date` only compares studies of the same patient.

### Tests

| Part | Tests |
|---|---|
| 5a | `pick_view` with a missing view; `add_prior` on visit 1 (no prior) and with a gap; `pair_by_date` respects `max_days` and never pairs across patients. |
| 5b | Synthetic embeddings + table: `slot_features` order and `present` mask; `pool_features` mean; loading an old cache without `image_path` warns. |
| 5c | Tiny PNGs: every sample has all slot keys; a missing slot gives zeros and `present = 0`; a batch collates; a 2-D slot and a 3-D slot in one sample; each image loaded once. |

## Branch and repos

- Work in the public repo on a new branch `df-levels-visits` from `main`. The dev repo gets it later with `git pull public main`.
- The PR starts with the `## ⚠️ Base code change` section (files, what and why, can affect, approved by).
- One PR per phase, or one PR for phases 0 to 2 and one per group in phase 3.

## Problems found during the survey (separate small fixes, not part of this plan)

From reading the code and data. I have not verified these by running them unless noted.

1. **Duplicate rows for one image.** SIIM-ACR has one row per RLE (11,582 rows for 10,675 images). RSNA Pneumonia Kaggle has one row per class-info line (30,227 rows for 26,684 images). Study level would list the same image twice. (Fixed on 2026-10-28, V-H3-1 and V-H3-2: one row per image.)
2. **ChestX-ray14 Train/Test** filter only the returned DataFrame. `h.harmonized_df`, `save()` and the `harmonizer=` preset still see all 112,120 images. (Fixed on 2026-10-28, V-H1-23.)
3. **CheXpert with `drop_uncertain=True`** may drop all lateral images: `dropna()` runs before the AP/PA fill, and AP/PA is empty for laterals. (Fixed on 2026-10-28, V-H1-1.)
4. **MS-CXR-T** `output_previous` is stored but never used, so the prior image is never loaded.
5. **SIIM-ACR patient_id is fake** (it is the study folder, not the DICOM `PatientID`), so patient-level splits do not group real patients. (Checked on 2026-10-28: the DICOM `PatientID` is unique for each of the 10,675 images, so the folder loses no patient grouping. SIIM COVID-19 now uses the DICOM `PatientID`, V-H3-3.)
6. **`docs/wiki/api.md`, the `harmonized_df` section,** describes `LABEL_COLS` inference in a way that does not match the code.
7. Same MIMIC study has two ID formats: bare numbers (MIMIC-CXR, MS-CXR) vs `p.../s...` prefixes (GEMeX, MIMIC-Ext-CXR-QBA).

## Open questions

Decided (2026-10-04): reports and labels are one value per study (labels come from the report). If they differ inside a study, take the first image's values and warn.

1. Same-day studies: separate visit numbers broken by `study_id` (proposed), or the same number?
2. Existing `study_date` columns (BRAX, TAIX-Ray, ReXGradient, Emory CXR, Emory CHORUS) keep their name but change type to pandas dates (today int `20120301` or text). OK, or keep the old format for one release?
3. Add `days_from_first` now, or later?
4. Add the convenience method `h.get_harmonized_df(level=...)` on the base harmonizer (base change), or only the standalone `to_level()`?
5. Patient level: per-image columns as a list of lists (proposed), or one flat list plus a matching `study_id` list?
6. Modality codes: short codes with `XR` for all radiographs (proposed), or the raw DICOM codes (`CR`, `DX`, `CT`, ...) as the main column?
7. Phase 5b: also ship a simple fusion evaluator (join slot embeddings + linear probe, patient-grouped folds), or only the arrays?
8. Phase 5 order: embedding path (5b) before `SlotDataset` (5c) (proposed), or `SlotDataset` first?
