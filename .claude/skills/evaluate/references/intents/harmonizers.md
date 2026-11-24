# Intent: harmonizers

A harmonizer reads one raw dataset release and returns `harmonized_df`. The rest of RadHarmony trusts it.

**Design decisions that apply** (`docs/wiki/design.md`). Test each one, and read its known gaps:
- D-01 to D-10: the table and the splits
- D-24: `harmonized_df` equals the `harmonize()` result
- D-25: unlabeled images
- D-26: `sex`
- D-30: steps, not `harmonize()`
- D-33: empty `mask_path`
- D-34: official splits that share patients
- D-35: `infer_path`
- D-36: boxes of several readers
- D-44: "stable" findings
- D-48: CheXpert Plus labels
- D-51: MIMIC-Ext-CXR-QBA answers
- D-31 is not built yet. Do not test it.
- Any other entry in the design.md sections "Table" and "Splits" that applies to this part.

**Other test steps:**

1. **Rows.** The row count matches the raw source after the documented filters, for example
   `drop_uncertain`, frontal-only or a split class. A merge does not add rows (no many-to-many join). The
   counts in the dataset wiki page and in `configs/datasets/<name>.json` are true. (One row per image: D-03.)
2. **Ids and paths.** `patient_id`, `study_id` and `image_path` are never null. `base_image_dir` plus
   `image_path` is a real file or folder (D-04). The same patient has the same `patient_id` in every row.
3. **Splits.** In a source split, no patient is in two splits, unless the source says so. (Split classes:
   D-09. Our own splits: D-10.)
4. **Labels.** Each raw label value maps as the documented policy says: positive, negative, uncertain
   (`-1`), blank or NaN. The prevalence of each label equals the raw source under that policy. A
   multi-label image has every one of its labels. (Names and order: D-05.)
5. **Boxes.** A box is on the image that the source gives it to. (Format: D-07.)
6. **Masks.** `mask_path` points to a mask for the same image, with the same height and width. (Empty
   `mask_path`: D-33.)
7. **Reports.** `report` is the text of that study, not the text of another study.

**How to test:** compare `harmonized_df` with the raw CSV or JSON, which you read yourself with pandas.
Pick random rows, open the image and check the size, and recompute boxes and labels by hand. Use small
`study_ids` subsets for big datasets.
