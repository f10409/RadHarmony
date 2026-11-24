# RSNA 2022 Cervical Spine

<!-- claim: id=rsna_2022_cervical_spine.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_2022_cervical_spine rsna_2022_cervical_spine_train.images" expect="2,019" -->
<!-- claim: id=rsna_2022_cervical_spine.train.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_2022_cervical_spine rsna_2022_cervical_spine_train.patients" expect="2,019" -->
| | |
|---|---|
| Registry keys | `rsna_2022_cervical_spine_train`, `rsna_2022_cervical_spine_bbox`, `rsna_2022_cervical_spine_test` |
| Task and outputs | Classification, 8 labels (`cls`). Vertebra masks (`mask`) on 87 studies. 3-D fracture boxes (`bbox`) in the bbox class. The test split has images only |
| Images in the table | train 2,019 volumes, bbox 235 volumes, test 3 volumes (sample files only). One study is one patient |
| Modality | Neck CT, 3-D, DICOM series. One row is one study, which is one volume |
| License | The rules of the Kaggle competition. Accept them before the download |
| Download | [Get the data: RSNA 2022 Cervical Spine](../../../how-to/get-the-data.md#rsna-2022-cervical-spine) |

<!-- claim: id=rsna_2022_cervical_spine.train.images.again tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_2022_cervical_spine rsna_2022_cervical_spine_train.images" expect="2,019" -->
<!-- claim: id=rsna_2022_cervical_spine.bbox.boxes tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_2022_cervical_spine rsna_2022_cervical_spine_bbox.boxes" expect="7,217" -->
RSNA 2022 Cervical Spine Fracture Detection is a 2022 Kaggle challenge. The train split has 2,019 CT studies
of the neck. Each study has 8 labels: a fracture label for each vertebra from C1 to C7, and
`patient_overall`. 235 studies also have 7,217 fracture boxes, each on one slice. 87 studies have a
segmentation of the vertebrae. The test images are hidden (D-25).

## Quick use

`base_image_dir` is `train_images/` or `test_images/`. The classes find `train.csv`,
`train_bounding_boxes.csv` and `test.csv` in `base_image_dir` or in the folder above it. The train class
finds `segmentations/` next to `base_image_dir`.

```
rsna-2022-cervical-spine-fracture-detection/
  train.csv
  train_bounding_boxes.csv
  test.csv
  segmentations/<StudyInstanceUID>.nii
  train_images/<StudyInstanceUID>/<slice_number>.dcm
  test_images/<StudyInstanceUID>/<slice_number>.dcm
```

The bbox class reads the header of each slice file of its 235 studies. This takes about 1 minute. On the
public download, `test_ds[0]` raises `RuntimeError` (see Splits).

```python
import torch
from radharmony.dataset import (
    RSNA2022CervicalSpineBboxDataset,
    RSNA2022CervicalSpineTestDataset,
    RSNA2022CervicalSpineTrainDataset,
)

train = RSNA2022CervicalSpineTrainDataset(
    base_image_dir="/data/rsna-2022-cervical-spine-fracture-detection/train_images/",
    output_cls=True,
    output_mask=True,  # keeps only the 87 studies with a segmentation
    dtype=torch.float32,
)
train_ds, val_ds = train.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 112, 112, 112), 1 channel, then (D, H, W)
# sample["cls"]: Tensor(8,)
# sample["mask"]: Tensor(1, 112, 112, 112), the vertebra numbers

boxes = RSNA2022CervicalSpineBboxDataset(
    base_image_dir="/data/rsna-2022-cervical-spine-fracture-detection/train_images/",
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
)
box_train_ds, box_val_ds = boxes.get_datasets(n_splits=5)
# box_train_ds[0]["bbox"]: Tensor(N, 6), one row for each slice box
# box_train_ds[0]["bbox_labels"]: a list of N names, all 'fracture'

test_ds = RSNA2022CervicalSpineTestDataset(
    base_image_dir="/data/rsna-2022-cervical-spine-fracture-detection/test_images/",
    dtype=torch.float32,
).get_datasets()
# test_ds[0] has only "img", for a Kaggle submission
```

The image steps: MONAI `ITKReader` reads all DICOM files of the study folder as one volume. The class keeps
the default `base_transpose=True`. So the transform turns each volume to the axis code `IPL` and then to
`(D, H, W)` (D-12). It clips the values to `hu_window` (default `(-200, 1800)`, for bone) (D-14). Then it
scales the 0.5 and 99.5 percentiles to -1 and 1. It resizes the longest side to 112, and pads the volume
with -1 to 112 x 112 x 112 (D x H x W).

The bbox class has no `hu_window` argument. Its transform clips the values to `(-1000, 1000)`.

## How it is harmonized

<!-- claim: id=rsna_2022_cervical_spine.train.images.view tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_2022_cervical_spine rsna_2022_cervical_spine_train.images" expect="2,019" -->
- Source: `train.csv` has one row for each study. The folder `train_images/<StudyInstanceUID>/` holds one
  `.dcm` file for each slice.
- IDs: the `StudyInstanceUID` `1.2.826.0.1.3680043.6200` gives `patient_id`, `study_id` and `image_path`
  `1.2.826.0.1.3680043.6200`. `image_path` is the study folder, relative to `base_image_dir` (D-04). The
  source has no patient ID column. The DICOM `PatientID` is the last part of the `StudyInstanceUID` (for
  example `6200`), so one study is one patient.
- Labels (D-05): the columns `C1` to `C7` become `c1` to `c7`, and `patient_overall` stays. 1 gives 1 and 0
  gives 0. The source has no uncertain or blank label.
- View (D-01): the DICOM headers have `ImageOrientationPatient`, but the harmonizer does not read it. So the
  harmonized table has no `view_position`. D-01 lists this as a known gap. All 2,019 studies are axial.
- Age and sex: the source files and the DICOM headers have neither, so the harmonized table has no
  `age_years` or `sex` (D-06, D-26).
- Dropped rows: the harmonizer drops no row. `harmonize()` does not check that the study folders exist.

## Labels

The 8 label columns, in the order of `cls`:

| Column | Meaning |
|---|---|
| `c1` | A fracture of the C1 vertebra |
| `c2` | A fracture of the C2 vertebra |
| `c3` | A fracture of the C3 vertebra |
| `c4` | A fracture of the C4 vertebra |
| `c5` | A fracture of the C5 vertebra |
| `c6` | A fracture of the C6 vertebra |
| `c7` | A fracture of the C7 vertebra |
| `patient_overall` | A fracture of any cervical vertebra |

In the train split, `patient_overall` is 1 for 961 studies.

## Boxes

Only `RSNA2022CervicalSpineBboxDataset` has boxes. It has one row for each of the 235 studies with a box.

<!-- claim: id=rsna_2022_cervical_spine.bbox.boxes.again tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_2022_cervical_spine rsna_2022_cervical_spine_bbox.boxes" expect="7,217" -->
- Source: `train_bounding_boxes.csv` has 7,217 rows. Each row is a pixel box on one slice: `x`, `y`,
  `width`, `height` and `slice_number` (the name of the slice file). The source has no finding name or
  vertebra for a box. All 235 studies have `patient_overall` 1.
- The harmonizer reads the header of each slice file, so it needs `base_image_dir`. It sorts the slices of a
  study by `ImagePositionPatient[2]`, from low to high, as the 3-D loader does. The rank of the box slice in
  this order gives the depth: from `rank / n_slices` to `(rank + 1) / n_slices`. It divides `y` and
  `y + height` by `Rows` (the height), and `x` and `x + width` by `Columns` (the width).
- The box is `[dim0_min, dim0_max, dim1_min, dim1_max, dim2_min, dim2_max]`, in fractions of the volume
  size (D-07). In the harmonized table, the order is slice, `y`, `x`, in the slice order of the 3-D loader.
  The class turns the box with the volume (D-12, D-28). These series are axial, so the slice is the D axis.
- Example: study `1.2.826.0.1.3680043.10051` has 272 slices. Its first box has `x` 219.27715, `y` 216.71419,
  `width` 17.304399999999987, `height` 20.385169999999988 and `slice_number` 133. Slice 133 has the rank 139.
  The box is `[0.5110294117647058, 0.5147058823529411, 0.42326990234375, 0.4630846875, 0.42827568359375,
  0.46207333984375]`.
- `bbox_labels`: `fracture` for each box, because the source has no finding name (D-07).
- Readers (D-36): the source has one set of boxes for each study, so the harmonizer does not group boxes.
- `bbox` and `bbox_labels` are JSON text in the harmonized table, so they stay the same after a CSV save and
  load.
- The sample: `bbox` is a float32 tensor `(N, 6)`, in fractions of the returned volume (D-28). `bbox_labels`
  is a list of N names. Example: `(2, 6)` and `['fracture', 'fracture']`. A study has 2 to 167 boxes. A
  fracture on 10 slices gives 10 thin boxes, not one box.
- Dropped boxes: the harmonizer clips each box to the volume. It drops a box with no area after the clip,
  and warns (D-07). It also drops a box on a slice with no DICOM file in the study folder, and warns. The
  source files have no such box.

## Masks

- Source: `segmentations/` has one NIfTI file `<StudyInstanceUID>.nii` for 87 studies. It covers the whole
  volume. Its values are the vertebra numbers
  ([Lin et al., Radiology: Artificial Intelligence 2023](https://doi.org/10.1148/ryai.230256)). 0 is the
  background, 1 to 7 are C1 to C7, and 8 to 19 are T1 to T12. Every mask has C1 to C7, but not every mask has
  all thoracic vertebrae.
- `mask_path` is the full path of the file, for example
  `/data/rsna-2022-cervical-spine-fracture-detection/segmentations/1.2.826.0.1.3680043.10633.nii`. Without
  `segmentation_dir`, the class looks for `segmentations/` next to `base_image_dir`.
- An empty `mask_path` means "not annotated" (D-33). With `output_mask=True`, the class drops these rows and
  warns. Only the 87 studies with a file stay.
- The sample: `mask` is a float32 tensor `(1, 112, 112, 112)` with the vertebra numbers of the file. For
  example, one study has the values 0 to 10. The transform turns and resizes it with the volume. The resize
  takes the nearest value, so the numbers stay whole. The 0 and 1 rule of D-14 is for 2-D masks only.

## Splits

<!-- claim: id=rsna_2022_cervical_spine.splits.train tier=ci cmd="python scripts/evalkit/dataset_stat.py rsna_2022_cervical_spine rsna_2022_cervical_spine_train.images" expect="2,019" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `RSNA2022CervicalSpineTrainDataset` | `rsna_2022_cervical_spine_train` | 2,019 | `train.csv` and `train_images/` |
| `RSNA2022CervicalSpineBboxDataset` | `rsna_2022_cervical_spine_bbox` | 235 | The train studies in `train_bounding_boxes.csv`. They are also in the train class |
| `RSNA2022CervicalSpineTestDataset` | `rsna_2022_cervical_spine_test` | 3 | `test.csv` and `test_images/`, with no labels (D-25). Use it only for a Kaggle submission |
| `RSNA2022CervicalSpineDataset` | `rsna_2022_cervical_spine` | | The old key. It reads `train.csv`, so it has the train studies. Use the split classes for a benchmark (D-09) |

The source has no labeled test split. `get_datasets(n_splits=5)` makes a train and val split from the train
class. It groups by patient (D-10). The source has no patient ID, so one study is one patient, and nobody can
check that a patient is in only 1 split. The bbox class has train studies only, so do not use it to test a
model that you trained on the train class.

This is a Kaggle code competition (D-25). The public download has only sample test files. The 3 studies of
`test.csv` are not in the 3 folders of `test_images/`. So `test_ds` has 3 rows, but `test_ds[0]` raises
`RuntimeError`, because the study folder does not exist. Use the test class in a Kaggle notebook of the
competition, where the hidden test set has the same layout.

## Arguments

The arguments are the same for `RSNA2022CervicalSpineTrainDataset` and `RSNA2022CervicalSpineTestDataset`.
The test class reads `test.csv` as `csv_path`, and it ignores `segmentation_dir`.

<!-- args: radharmony.dataset.RSNA2022CervicalSpineTrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | `train_images/` or `test_images/`. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `train.csv` or `test.csv`. Without it, the class looks in `base_image_dir` and the folder above it. A missing file raises `FileNotFoundError` (D-35) |
| `segmentation_dir` | `str` | No | `None` | The folder of the NIfTI masks. Without it, the class uses `segmentations/` next to `base_image_dir` when the folder exists |
| `hu_window` | `tuple` or `None` | No | `(-200, 1800)` | The HU window. The transform clips each volume to it before the percentile scale (D-14). `None` gives no clip. It has no effect with your own `transform` |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform3D` with `img_size=112` and `hu_window=hu_window` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample. In `RSNA2022CervicalSpineTestDataset`: not supported. The class warns and adds no key (D-25) |
| `output_mask` | `bool` | No | `False` | Add the `mask` tensor to the sample. The class drops the rows with an empty `mask_path`, and warns (D-33). In `RSNA2022CervicalSpineTestDataset`: not supported. The class warns and adds no key (D-25) |
| `output_report` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11). Use `RSNA2022CervicalSpineBboxDataset` for the boxes |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The labels stay float32 |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

### RSNA2022CervicalSpineBboxDataset

<!-- args: radharmony.dataset.RSNA2022CervicalSpineBboxDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | `train_images/`. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `train.csv`. Without it, the class looks in `base_image_dir` and the folder above it |
| `bbox_csv_path` | `str` | No | `None` | `train_bounding_boxes.csv`. Without it, the class looks in `base_image_dir` and the folder above it |

#### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform3D` with `img_size=112` and `hu_window=(-1000, 1000)` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_report` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Add `bbox` and `bbox_labels` to the sample |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The labels stay float32 |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import RSNA2022CervicalSpineTrainHarmonizer

h = RSNA2022CervicalSpineTrainHarmonizer(
    csv_path="/data/rsna-2022-cervical-spine-fracture-detection/train.csv",
    base_image_dir="/data/rsna-2022-cervical-spine-fracture-detection/train_images/",
    segmentation_dir="/data/rsna-2022-cervical-spine-fracture-detection/segmentations/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'mask_path', 'c1', 'c2', ..., 'c7', 'patient_overall']
```

`RSNA2022CervicalSpineTestHarmonizer` takes `csv_path` (`test.csv`) and `base_image_dir`. Its harmonized
table has only `patient_id`, `study_id` and `image_path`. `test.csv` has one row for each study and
prediction type, so the harmonizer keeps one row for each study.

<!-- args: radharmony.harmonizer.RSNA2022CervicalSpineTrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `train.csv` |
| `base_image_dir` | `str` | No | `None` | `train_images/`. `save()` keeps it |
| `segmentation_dir` | `str` | No | `None` | The folder of the NIfTI masks. Without it, the harmonized table has no `mask_path` |

`RSNA2022CervicalSpineBboxHarmonizer` makes the harmonized table of the bbox class. This harmonized table
has `bbox` and `bbox_labels`, and no `mask_path`.

<!-- args: radharmony.harmonizer.RSNA2022CervicalSpineBboxHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `train.csv` |
| `bbox_csv_path` | `str` | Yes | | `train_bounding_boxes.csv` |
| `base_image_dir` | `str` | No | `None` | `train_images/`. The harmonizer reads the slice order and size from it. Without it, the harmonizer orders the slices by `slice_number` and uses the size 512 x 512 (H x W). Both can be wrong |
| `bbox_epsilon` | `float` | No | `None` | Not used. With a value, the harmonizer warns. The argument stays, so an old saved harmonizer still loads (D-07) |
| `bbox_z_epsilon` | `float` | No | `None` | Half the depth of a box, as a fraction. `None` gives `0.5 / n_slices`, so the box is one slice deep |
| `num_workers` | `int` | No | `8` | The number of threads that read the slice headers |

## Known data problems

- Test files: the public download has only sample test files (D-25).
- Slice size: most studies are 512 x 512 (H x W). `1.2.826.0.1.3680043.22678` and
  `1.2.826.0.1.3680043.8858` are 768 x 768, and `1.2.826.0.1.3680043.23400` is 512 x 519 (H x W). Of these
  3, only `1.2.826.0.1.3680043.22678` has boxes (see Boxes).
- Slice order: in about 9 of 10 studies, the slice number goes up as `ImagePositionPatient[2]` goes
  down. So the file name order puts a box at the wrong end of the volume (see Boxes).
