# Wiki outline for 6c (draft for review)

Status: approved by Frank Li on 2026-11-03, with the 5 decisions below.

## Why change the structure

- The nav has 42 dataset pages in one flat list. A reader cannot find "all datasets with masks" or "all CT datasets".
- Each page mixes 3 jobs: a tutorial (load your first dataset), a how-to (save and reuse a table), and a reference (the argument table). Example: 26 dataset pages repeat a "Load from saved harmonized CSV" section with almost the same text.
- The download steps are in 2 places: a "Download" section on 40 dataset pages, and `acquiring-datasets.md` (398 lines).
- `datasets/chexinstruct_vqa.md` is not in the nav at all.

## Rules for the new structure

- One job per page (Diátaxis): tutorial, how-to, reference or explanation.
- One home for each fact. Other pages link to it.
- Every code block has a run note and runs in its tier (ci, local or gpu).
- Every count in prose has a claim note, or it goes.
  Decided (Frank Li, 2026-11-03): a count of the harmonized table comes from `expected_stats` in
  `configs/datasets/<key>.json` (`scripts/evalkit/dataset_stat.py`, tier=ci). A count of a source file
  (for example the rows of `train.csv`) comes from the source file on the NAS (tier=nas). A dataset with no
  `expected_stats` gets them, from the raw files read with pandas. The other options were table counts
  only, no counts until the fix phase, and counts with no check.
- Every argument table has an `<!-- args: <module.Class> -->` line above it. The wiki audit (part 2) reads this line instead of guessing the class.

## New nav

Old file names are on the right. "new" means a page to write. "merge" means content that moves into the page and then the old page or section goes.

```
Home                                   index.md (shorter: 1 example, 4 links)
Get started (tutorials)
  Install                              quickstart.md (install part)
  Your first benchmark in 10 minutes   new (1 dataset + 1 backbone + linear probe)
  Concepts                             new, from architecture.md (harmonizer, dataset,
                                       evaluator, harmonized table, box order, splits)
How-to guides
  Get the data                         acquiring-datasets.md (per source: Kaggle,
                                       PhysioNet, HuggingFace, BIMCV, Stanford AIMI)
  Load a dataset and pick outputs      api.md (constructor pattern, outputs)
  Save and reuse a harmonized table    api.md + merge the 26 "Load from saved" sections
  Split data (official, random, folds) api.md "Splitting data"
  Cache and pre-warm                   api.md "Caching", "Pre-warming the cache"
  Run a probe                          evaluator/index.md "Quick start"
  Run segmentation                     new, from the 3 seg evaluator pages
  Run report generation                evaluator/report_generation.md (usage part)
  Compare backbones                    new
  Add a dataset                        new, points to the add-dataset skill
                                       + merge app-extend.md
  Add a backbone                       backbones/index.md "Authoring a new recipe"
                                       + backbones/custom_backbones.md
  Use the app                          app.md
Reference
  Dataset catalog                      datasets.md (one table: key, modality, task,
                                       outputs, size, license, page)
  Datasets: chest X-ray, labels        12 pages (list below)
  Datasets: chest X-ray, boxes/masks   16 pages
  Datasets: CT, MRI and other          8 pages
  Datasets: VQA, reports, text         7 pages
  Evaluators                           evaluator/*.md (10 pages, argument tables)
  Backbones                            backbones/*.md (14 pages)
  API: datasets                        api.md (data dict keys, base arguments)
  API: transforms                      transforms.md
  Data paths (internal)                nas_paths.md (generated, dev only)
Explanation
  Design decisions                     design.md (as it is: the decision log)
  Label policy                        new, short: 1/0/NaN, -1 uncertain, blank labels,
                                      snake case names, box order; links to design.md
  MIMIC-CXR label extraction           phase-grounding.md
  DICOM loading                        docs/dicom_loading_issues.md (moved in)
```

## Dataset groups (all 42 pages)

- **Chest X-ray, labels (12):** brax, chestxray14, chexpert, chexpert_plus, chexphoto, emory_cxr (internal), mimic_cxr, mimic_cxr_jpg, ms_cxr_t (internal), padchest, rexgradient, taix_ray.
- **Chest X-ray, boxes and masks (16):** chest_imagenome, chestxdet10, chexlocalize, montgomery_cxr, ms_cxr (internal), node21, padchest_gr, qata_cov19, ranzcr_clip, rsna_pneumonia, rsna_pneumonia_kaggle, shenzhen_cxr, siim_acr_ptx, siim_covid19, vindr_cxr, vindr_pcxr.
- **CT, MRI and other (8):** ct_rate, medseg_covid_ct, radchestct, rsna_2022_cervical_spine, rsna_2024_lumbar_spine, rsna_abdominal_trauma_2023, rsna_bone_age, rsna_pe_detection.
- **VQA, reports and text (7):** chexinstruct_vqa, gemex_vqa, medpmc, mimic_ext_cxr_qba, openi_cxr, roco, vqa_rad.
  medpmc is a new page: MedPMC had no page before (decided by Frank Li, 2026-11-04).

A page goes in the group of its main output. A dataset with labels and boxes (for example VinDr-CXR) goes in boxes and masks. The catalog table shows every output, so a reader can also filter there. The exact split is open question 3.

The internal pages (emory_cxr, ms_cxr, ms_cxr_t, nas_paths) stay in the dev repo only. The public copy leaves them out.

## Dataset page template

```
# <Name>
<!-- blocks: run=local data=<registry key> -->

At a glance (one table)
  Registry keys | Task and outputs | Images, studies, patients (claim notes)
  Modality, 2-D or 3-D | License | Download (link to "Get the data#<source>")

## Quick use              one block: dataset class, get_datasets, one sample
## How it is harmonized   5 to 8 short points (see the example below)
## Labels                 the label columns and what each one means
## Reports, Boxes or      only for an output that needs more words (optional)
   Prior image
## Splits                 official split classes, sizes (claim notes)
## Arguments              <!-- args: radharmony.dataset.<Class> -->
   ### Shared arguments (inherited from BaseRadiologicalDataset)
## Harmonizer             one block; extra columns; <!-- args: ... --> table
## Known data problems    problems in the source data, each with its design.md entry (D-xx) when one exists
```

"How it is harmonized" says what the harmonizer does with the source files, in the order of
the steps: the files it reads, what one row is, how the IDs and `image_path` are made, how
the raw label values map, the view, age and sex, and which rows it drops. It names the
`design.md` entry of each rule. Example for CheXpert:

```
## How it is harmonized

- Source: `train.csv` or `valid.csv`, one row for each image.
- IDs: `Path` `CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg` gives
  `patient_id` `patient00001`, `study_id` `patient00001_study1`, and `image_path`
  `patient00001/study1/view1_frontal.jpg`, relative to `base_image_dir` (`.../train/`) (D-04).
- Labels (14 columns, D-05): `1` gives 1 and `0` gives 0. A blank gives 0
  (`not_mentioned="negative"`) or NaN (`"nan"`). A row with no label at all keeps NaN.
- Uncertain `-1`: with `drop_uncertain=True` (the default), the row is dropped.
- View: from `AP/PA`. It is blank on every lateral image.
- Age and sex: from `Age` (years) and `Sex` (D-06, D-26).
- With `demographics_path`, the demographics workbook is joined on `patient_id`.
```

"Known data problems" lists problems in the source files that the authors released, not
bugs in our code. Example: the NAS copy of CheXpert Plus misses 1,777 of 64,725 patient
folders. The harmonizer warns, and `verify_images()` drops those rows.

Gone from each page: "Download" (now in "Get the data"), "Load from saved harmonized CSV" (now one how-to), "Example paths" (the NAS paths are in `nas_paths.md`).

## Evaluator and backbone page templates

- **Evaluator:** At a glance (registry key, task, splits it supports), Quick use, Arguments (with `args:` line and the shared subsection), Output, Notes.
- **Backbone:** At a glance (model card link, input size, normalization, pooled token, install extra), Quick use (gpu block), Segmentation mode, Notes. The 4 facts in "At a glance" are the ones that rule J-05 checks.

## How the move works

1. Add `mkdocs-redirects` (a new docs dependency) so old URLs go to the new pages.
2. Move and merge the pages, with no change of the text. Check: `mkdocs build --strict` passes, and the wiki audit gives the same 200 findings.
3. Rewrite one section at a time: datasets in 4 groups, evaluators, backbones, core pages, then README and the notebooks. A section is done when the wiki audit, the block runner (local tier) and the STE linter give 0 for it. Then a critic agent checks facts and clarity.
4. Wiki audit part 2: the `args:` lines replace the class guess. The section-title and shared-subsection checks become one template check.

## Open questions for Frank Li

1. **Redirects.** Add `mkdocs-redirects` to the docs extra? Without it, old links to pages break.
   Decided (Frank Li, 2026-11-03): yes. The other options were to keep the old file paths, or to accept broken links.
2. **design.md.** Keep it as one 2,025-line page under Explanation, or split it into one page per group (labels, boxes, splits, ...)? Recommendation: keep it as one page now. A split can come later.
   Decided (Frank Li, 2026-11-03): keep one page. The other option was one page for each group now.
3. **Dataset groups.** Group by main output (as above), or by modality only (chest X-ray, CT, MRI, other)? Recommendation: by main output, because a user picks a dataset for a task.
   Decided (Frank Li, 2026-11-03): by main output. The other option was by modality only.
4. **Download steps.** One home in "Get the data", with only a link on each dataset page? Recommendation: yes. The account and tool steps are the same for all datasets of one source.
   Decided (Frank Li, 2026-11-03): one home in "Get the data". A dataset page keeps the link and only
   its own special steps. The other option was the full steps on each page.
5. **Tutorial.** "Your first benchmark in 10 minutes" needs a dataset that a new user can download fast. Candidates: Montgomery (138 images, no account) or Shenzhen (662 images, no account). Recommendation: Montgomery.
   Decided (Frank Li, 2026-11-03): Montgomery. The tutorial says that 138 images give a noisy AUROC.
   The other options were Shenzhen, or both (train on one, test on the other).
