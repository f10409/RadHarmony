# Intent: dataset classes and transforms

A dataset class wraps a harmonizer and returns training samples.

**Design decisions that apply** (`docs/wiki/design.md`). Test each one, and read its known gaps:
- D-05: the `cls` order
- D-10: splits
- D-11: output flags
- D-12: 3-D orientation
- D-13: CXR pixel values
- D-14: value range
- D-15: cache
- D-27: readers
- D-28: the box in a sample
- D-33: empty `mask_path`
- D-35: `infer_path`
- D-37: warp border
- D-42: channel count
- D-43: a resample keeps the image centered
- D-49: random steps on a val split
- Any other entry in the design.md section "Dataset classes and transforms" that applies to this part.

**Other test steps:**

1. **Image.** The image has the documented shape, channel count and dtype. The tensor layout matches what
   `radharmony/dataset/transforms.py` promises.
2. **Labels.** `cls[j]` is the label in column `LABEL_COLS[j]` of the harmonized row of the same sample.
   The label names that the dataset reports (for example with `get_label_names` or `LABEL_COLS`) match the
   vector.
3. **Mask.** The mask has the same height and width as the image. It is resized with nearest neighbor,
   not interpolated. The values are {0, 1} or the class ids. A mask that the harmonizer makes from RLE or
   JSON lands on the right pixels: decode one RLE by hand and compare.
4. **Boxes.** The box in `ds[i]` marks the same pixels as in the source image. Test a non-square image,
   because a pad moves the box (D-28).
5. **Loaders.** A `DataLoader` with `num_workers > 0` and the default collate works on samples with a
   different number of boxes and with reports.
6. **Augmentation.** Train augmentation never runs on the val or test split. A fixed seed gives the same
   sample two times.
7. **Cache.** `cache_dir` and `harmonizer_path` give the same data as a new `harmonize()` (D-15, D-08).

**How to test:** run `scripts/check_datasets.py --datasets <key> --max-samples 20 --num-workers 0`. It reads
the data paths file (`references/parts.md`, "Data and environments"). For a key with no entry, it stops with a hint. Then
build the dataset class with the data folders of your task. Call `subsample_dataset(ds, 20)` (it changes `ds`
in place) and `check_dataset(ds, num_workers=0)` from `radharmony.integrity`. Then add your own checks on single
samples.

**Large VQA sets.** A full `harmonize()` of a large VQA set can take more than 10 minutes. CheXinstruct VQA
has 701,458 rows, and MIMIC-Ext-CXR-QBA and MedPMC have more. Of the VQA sets, only MIMIC-Ext-CXR-QBA has a
`study_ids=` and `patient_ids=` filter (D-02).

Use a constructor filter: `max_studies=` (MIMIC-Ext-CXR-QBA), `question_types=` or `data_sources=`
(CheXinstruct VQA) or `question_subtypes=` (GEMeX VQA). For MedPMC, `subsets=` alone is not enough. Add
`radiology_only=True` or `specialties=`. Or write a small table yourself, with the columns `patient_id`,
`study_id`, `question_id`, `image_path`, `question` and `answer`. Pass it to the dataset class as
`harmonized_df=`.
