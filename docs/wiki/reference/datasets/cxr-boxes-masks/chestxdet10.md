# ChestX-Det10

<!-- claim: id=chestxdet10.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chestxdet10 chestxdet10_train.images" expect="3,001" -->
| | |
|---|---|
| Registry keys | `chestxdet10_train`, `chestxdet10_test` |
| Task and outputs | Detection of 10 findings: boxes (`bbox`) and image labels (`cls`) |
| Images in the table | train 3,001, test 542. One image is one patient |
| Modality | Chest X-ray, 2-D, PNG |
| License | The GitHub repo states no license. Its README asks you to cite the paper |
| Download | [Get the data: other datasets](../../../how-to/get-the-data.md#other-datasets) |

ChestX-Det10 is a box dataset from the Deepwise AI Lab
([Liu, Lin and Yu, arXiv:2006.10550](https://arxiv.org/abs/2006.10550)). The images come from NIH ChestX-ray14,
with new file names. 3 radiologists drew boxes of 10 findings. 2 of them drew the boxes of each image, and the
third chose among their boxes and could add boxes.

The official metric is the recall at 0.05, 0.1 and 0.2 false positives per image (IoU 0.5), not mAP. The
authors' `eval.py` computes it. RadHarmony does not compute it.

## Quick use

`base_image_dir` is the dataset root. The class finds `train.json` or `test.json` in this folder.

```
ChestX-Det10/
  train.json
  test.json
  eval.py
  train-old/36200.png
  test_data/36199.png
```

```python
import torch
from radharmony.dataset import ChestXDet10TrainDataset, ChestXDet10TestDataset

train = ChestXDet10TrainDataset(
    base_image_dir="/data/ChestX-Det10/",
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
)
train_ds, val_ds = train.get_datasets(n_splits=5)  # 1/5 of the images go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(10,)
# sample["bbox"]: Tensor(N, 4), one row for each box. N is 0 for an image with no finding
# sample["bbox_labels"]: a list of N names, for example ['effusion']

test_ds = ChestXDet10TestDataset(
    base_image_dir="/data/ChestX-Det10/",
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
).get_datasets()
```

## How it is harmonized

- Source: `train.json` or `test.json`. One record is one image, with `file_name`, `syms` (the findings) and
  `boxes`.
- IDs: the source has no patient ID. The record with `file_name` `36302.png` in `test.json` gives
  `patient_id` `36302`, `study_id` `36302` and `image_path` `test_data/36302.png`. So one image is one patient.
  `image_path` is relative to `base_image_dir` (D-04).
- Image folders: the train images are in `train-old/` and the test images are in `test_data/`. These are
  the folder names in the source zip files.
- Labels (D-05): a finding in `syms` gives 1, and the other labels give 0. The source has no uncertain or
  blank label.
- View, age and sex: the source has none of them, so the harmonized table has no `view_position`, `age_years`
  or `sex` (D-01).
- Extra columns: `split` (`"train"` or `"test"`) and `n_boxes` (the number of boxes of the image).
- Dropped rows: none by default. With `keep_negatives=False`, the harmonizer drops the images with no finding
  (681 in train and 83 in test).

## Labels

The 10 label columns, in the order of `cls`:

| Column | Meaning |
|---|---|
| `atelectasis` | Part of a lung is collapsed |
| `calcification` | A calcification |
| `consolidation` | Airspace consolidation |
| `effusion` | Pleural effusion |
| `emphysema` | Emphysema |
| `fibrosis` | Pulmonary fibrosis |
| `fracture` | A rib or other bone fracture |
| `mass` | A lung mass |
| `nodule` | A lung nodule |
| `pneumothorax` | Pneumothorax |

<!-- claim: id=chestxdet10.paper.test_atelectasis_boxes source="Liu, Lin and Yu, arXiv 2020" expect="51" -->
A label is 1 when the image has 1 or more boxes of the finding. The paper counts boxes, so its counts are
larger. Example: in the test split, `atelectasis` is 1 on 48 images, and these images have 51 atelectasis
boxes.

## Boxes

<!-- claim: id=chestxdet10.train.boxes tier=ci cmd="python scripts/evalkit/dataset_stat.py chestxdet10 chestxdet10_train.boxes" expect="6,863" -->
<!-- claim: id=chestxdet10.train.rows_with_bbox tier=ci cmd="python scripts/evalkit/dataset_stat.py chestxdet10 chestxdet10_train.rows_with_bbox" expect="2,320" -->
<!-- claim: id=chestxdet10.test.boxes tier=ci cmd="python scripts/evalkit/dataset_stat.py chestxdet10 chestxdet10_test.boxes" expect="1,476" -->
- Source: `boxes` holds pixel boxes `[x1, y1, x2, y2]` (top left, bottom right). `syms` holds one finding
  for each box, in the same order. Each image is 1024 x 1024 (H x W). The train split has 6,863 boxes on
  2,320 images, and the test split has 1,476 boxes on 459 images.
- The harmonizer reads the size of each image file, so it needs `base_image_dir`. It divides `y1` and `y2` by
  the height, and `x1` and `x2` by the width.
- The box is `[y_min, y_max, x_min, x_max]`, in fractions of the image size (D-07).
- Example: `36302.png` has `boxes` `[[799, 666, 937, 761]]` and `syms` `["Effusion"]`. It gives `bbox`
  `[[0.650390625, 0.7431640625, 0.7802734375, 0.9150390625]]` and `bbox_labels` `["effusion"]`.
- `bbox_labels`: the snake_case name of the finding, as in the label columns (D-07).
- Readers (D-36): the source has one final set of boxes for each image, so the harmonizer does not group
  boxes.
- `bbox` and `bbox_labels` are JSON text in the harmonized table, so they stay the same after a CSV save and
  load.
- The sample: `bbox` is a float32 tensor `(N, 4)`, in fractions of the returned image (D-28). `bbox_labels` is
  a list of N names. Example: `(1, 4)` and `['effusion']`. An image with no box gives `(0, 4)` and `[]`. A
  train image has up to 36 boxes, and a test image has up to 14.
- Dropped boxes: the harmonizer clips each box to the image. It drops a box with no area after the clip, and
  warns (D-07). The source files have no such box.

## Splits

<!-- claim: id=chestxdet10.splits.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chestxdet10 chestxdet10_train.images" expect="3,001" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `ChestXDet10TrainDataset` | `chestxdet10_train` | 3,001 | `train.json` and `train-old/` |
| `ChestXDet10TestDataset` | `chestxdet10_test` | 542 | `test.json` and `test_data/`. Use it as the test split |
| `ChestXDet10Dataset` | `chestxdet10` | | The old key. It reads the split that `split` names (default `"train"`). Use the split classes for a benchmark (D-09) |

`get_datasets(n_splits=5)` makes a train and val split from the train class. It groups by patient (D-10). The
source has no patient ID, so one image is one patient, and nobody can check that a patient is in only 1 split.
No file name is in both splits.

A split class gives only its own split. `ChestXDet10TrainDataset(split="test")` raises `ValueError`, and the
test class does the same. A file that the old `ChestXDet10Harmonizer` saved still loads with
`harmonizer_path` in the class of its split. A file with rows of the other split raises `ValueError`.

## Arguments

The arguments are the same for `ChestXDet10TrainDataset` and `ChestXDet10TestDataset`.

<!-- args: radharmony.dataset.ChestXDet10TrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The dataset root with `train-old/` and `test_data/`. Not needed with `harmonizer_path`, which holds it |
| `split` | `str` | No | `None` | Only the split of the class: `"train"` or `"test"`. Another value raises `ValueError` |
| `csv_path` | `str` | No | `None` | `train.json` or `test.json`. Without it, the class uses the file in `base_image_dir` |
| `keep_negatives` | `bool` | No | `True` | Keep the images with no finding. With `False`, the harmonizer drops them |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
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
from radharmony.harmonizer import ChestXDet10TrainHarmonizer

h = ChestXDet10TrainHarmonizer(
    csv_path="/data/ChestX-Det10/train.json",
    base_image_dir="/data/ChestX-Det10/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'bbox', 'bbox_labels', 'atelectasis', ..., 'split', 'n_boxes']
```

`ChestXDet10TestHarmonizer` takes the same arguments, with `test.json`.

<!-- args: radharmony.harmonizer.ChestXDet10TrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `train.json` or `test.json` |
| `base_image_dir` | `str` | No | `None` | The dataset root. The harmonizer reads the image sizes from it. Without it, the harmonizer looks for the images in the working folder |
| `split` | `str` | No | The split of the class (`"train"` or `"test"`) | Only the split of the class. Another value raises `ValueError` |
| `keep_negatives` | `bool` | No | `True` | Keep the images with no finding |

## Known data problems

- Train folder: the train images unzip to `train-old/`, not `train/`.
- File names: the names are not the NIH ChestX-ray14 names. Example: `36199.png`, where NIH uses names such
  as `00001075_000.png`. So a ChestX-ray14 copy does not give the images or the NIH patient IDs.
- RGBA files: 40 PNG files are RGBA, and the others are gray. The RGBA files are gray images (R = G = B,
  alpha 255), so the dataset gives 1 channel for every image (D-42).
- Shared boxes: one box can have 2 or more findings (paper, section 2). Then the source repeats the box with
  each finding. Example: `36367.png` has the box `[243, 662, 407, 886]` for `Fibrosis` and for
  `Consolidation`.
