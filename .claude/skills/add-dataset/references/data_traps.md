# Data traps

Traps that earlier datasets had. Read the section for each trap of the new source. The design entry of each
trap is the rule. This page shows how to follow it.

## `base_image_dir`

`base_image_dir` is the deepest folder that holds every image. `image_path` is relative to it (D-04). When
the wiki page says "`base_image_dir`: the `train/` folder", the user gives the path of `train/`, not of the
download root.

Right: `image_path` is relative to the folder that the user gives.

```python
# base_image_dir = /data/rsna-pneumonia-detection-challenge/stage_2_train_images/
self.df["image_path"] = self.df["patientId"].astype(str) + ".dcm"
```

Wrong: a fixed folder name in `image_path` makes the user give a higher folder.

```python
# base_image_dir = /data/rsna-pneumonia-detection-challenge/
self.df["image_path"] = "stage_2_train_images/" + self.df["patientId"].astype(str) + ".dcm"
```

- **Official splits.** Each split class takes its own image folder, for example `train_images/` and
  `test_images/` (D-09). Do not share one root for all splits.
- **A prefix in the source.** When the source path starts with a fixed prefix, remove it in
  `_build_image_path()`. Examples: CheXpert `Path` starts with `CheXpert-v1.0/train/`, and MIMIC-CXR
  `path` starts with `files/`.
- **Sibling image folders.** When the images are in several sibling folders inside one folder,
  `image_path` keeps the sibling folder name. Examples: ChestX-ray14 (`images_001/` to `images_012/`) and
  RSNA Bone Age val (`boneage-validation-dataset-1/` and `-2/`). `base_image_dir` is still the deepest
  common folder. Say so in the class docstring and on the wiki page.
- **The tables of the source files.** `infer_path` looks only in `base_image_dir` and its parent folder (D-35). It does
  not search sub folders. When `base_image_dir` is deeper, put the extra folders in the file name, for
  example `"../labels.csv"` or `"annotations/labels.csv"`.
- **The catalog.** A dataset with official splits gets a row in the table of the section
  "`base_image_dir` convention" of `docs/wiki/reference/datasets/index.md`.

## Boxes

A box is `[dim0_min, dim0_max, dim1_min, dim1_max]` in [0, 1], row first (D-07). For a 2-D image this is
`[y_min, y_max, x_min, x_max]`.

- **Check the axis order on the images, not in the docs.** Sources give other orders, for example
  `[x0, y0, x1, y1]` (PadChest-GR) or `(x, y, w, h)` (NODE21). A box read in the wrong order gives a
  transposed box, and its values are still valid. A quick check works when the report names a side. Take the boxes of one side.
  The share of these boxes with the center on that side of the midline is far above 50%. The patient's left
  is the image right.
- **Pixel boxes.** Divide by the image size. For DICOM, read `Rows` and `Columns` with
  `pydicom.dcmread(path, stop_before_pixels=True)`. Read the headers only for the images with boxes.
- **More than one box for each image.** Override `_build_bbox()`. Store a list of boxes in one row, and
  one `bbox_labels` entry for each box (VinDr-CXR). One row stays one image (D-03). The harmonizer
  groups the boxes of several readers by overlap (D-36).
- **Clip.** When the `bbox` column is final, call `clip_boxes(self.df, "<dataset name>")` from
  `radharmony/harmonizer/_bbox.py` (D-07, L-015). It clips each box to [0, 1], drops a box with no area
  with its labels, and warns. Call it before a filter that reads the boxes.
- **No finding.** A row with no box gets `[]`. Its sample has a box tensor of shape `(0, 4)` or `(0, 6)`.
  The dataset class needs no code for it.

## Boxes on 3-D volumes

A 3-D box is `[d_min, d_max, y_min, y_max, x_min, x_max]` in [0, 1], normalized by the shape of the
volume.

- **The axis order** is the order after `OrientationD("IPL")` and `TransposeD([0, 3, 2, 1])` (D-12). Dim 0
  is I (the slice axis). Dim 1 is P (the height). Dim 2 is L (the width).
- **A box over many slices**, for example the lung region of RadChestCT. See `_preprocess_bbox_df` in
  `radharmony/harmonizer/radchestct/radchestct.py`.
- **A 2-D box on one slice**, for example RSNA 2024 Lumbar Spine and RSNA 2022 Cervical Spine. The depth range
  holds the slice: `eps_z = 0.5 / n_instances`. Take the slice rank from the file order of the 3-D loader,
  `itk.GDCMSeriesFileNames`, the call that ITKReader uses. Neither `ImagePositionPatient[2]` nor
  `InstanceNumber` gives this order for every series. See `_loader_file_order` in
  `radharmony/harmonizer/rsna_2024_lumbar_spine/rsna_2024_lumbar_spine.py`.
- **A thin box** is one voxel deep after the resample. The app (`_overlay_bbox` in `app.py`) makes it larger
  on the screen. The harmonizer needs no code for it.
- **Dense boxes** (one box for each vertebra or level): add the app name of the dataset to
  `_BBOX_FOCUS_FIRST_DATASETS` in `app.py`. The app then centers the 3 panels on one box, shows only that
  box, and shows the "Next bbox" button. A dataset with one large box for each volume (RadChestCT) does not
  go in this set.

## More than one mask for each image

Some sources give one mask for each finding, for example a JSON file with
`{image: {finding: rle}}` (CheXlocalize). The mask steps of `BaseHarmonizer` (`_build_mask_path`,
`preprocess_masks`, `_decode_mask`) give one mask for each row. Do not repeat a row for each finding: that
repeats its labels and breaks one row for each image (D-03).

- Override `_build_mask_path()`. Store a key for each row (for example `f"{study_id}_{file name}"`), or
  `None` when the image has no mask.
- Override `_decode_mask(key, width, height)`. Decode the masks of all findings for that key and join them
  with a bitwise OR, as SIIM-ACR does for the rows of several readers.
- Load a large JSON file only when the user asks for masks (`output_mask=True`).
- COCO RLE (`{"size": [H, W], "counts": ...}`): decode with `pycocotools.mask.decode`. `rle_to_mask` in
  `radharmony/utils/mask_utils.py` is the SIIM-ACR RLE, a different format.
- This needs no base code change. A mask with one channel for each finding is a base code change. Then
  follow `CLAUDE.md`, section "Base code needs explicit approval".

## Two mask files for each image

Some sources give 2 files for one mask, for example a left and a right lung mask. Use the base mask step:

- In `_build_mask_path()`, store a key for each row in `mask_path`, for example the image file name. Before
  the mask step, `mask_path` holds this key, not a file.
- In `_decode_mask(key, width, height)`, read the 2 files of that key and join them with a bitwise OR.
  Return a 0/1 `uint8` array.
- `harmonize(mask_output_dir=...)` calls `preprocess_masks`. It writes one PNG for each image and puts its
  path in `mask_path`.
- The dataset class raises a clear error for `output_mask=True` with no `mask_output_dir`, as in "RLE masks".

## A source with no table

Some sources have no CSV. The images and the text files are the source, and the labels are in the file
names (for example Montgomery and Shenzhen CXR).

- Call `super().__init__(csv_path=None)`. Override `_read_source()`: list the image folder and read the
  other files into one table. Raise `FileNotFoundError` when the folder has no image, with the folder and
  the file type in the message (D-35, "Empty image folder").
- Override `_harmonizer_init_snapshot()` and remove `csv_path` from it, because it is not an argument of the
  class: `snap.pop("csv_path", None)`.
- Read a file that an ID needs in `_read_source()` (see "The order of the steps" in `harmonizer.md`).

## RLE masks

When the table holds RLE strings in `mask_path`, MONAI reads each string as a file name and fails with
`OSError: File name too long`. `harmonize(mask_output_dir=...)` writes the masks as PNG files first. So the
dataset class raises a clear error:

```python
if output_mask and mask_output_dir is None and harmonizer_path is None \
        and harmonized_df is None and harmonizer is None:
    raise ValueError(
        "output_mask=True needs mask_output_dir. The masks of My Dataset are RLE strings. "
        "The class writes them as PNG files first. Pass mask_output_dir='/path/to/masks'."
    )
```

The check skips a preset table or harmonizer, because its `mask_path` can hold PNG files already.
`MontgomeryCXRDataset` has the same check for its joined masks. An image with no mask follows D-33.

## DICOM series folders

When `image_path` is a folder (a DICOM series), override `verify_images` in the harmonizer and in the
dataset class. Both base methods use `os.path.isfile`, so they drop every row of a folder.

## Other traps

- **MRI.** Use `RadiologyTransform3D(..., hu_window=None)`, the default. The percentile step of the 3-D load
  scales the values.
- **Grades and classes.** For a target with grades or classes (for example a severity), make one 0/1 column
  for each pair of target and class. A blank source value gives NaN in all these columns (D-05). See
  RSNA 2024 Lumbar Spine.
- **A file in the release with no row, or a row with no file.** Count them for `expected_stats._comment`.
  Ask the user if the class drops them.
