# NODE21

<!-- claim: id=node21.images tier=ci cmd="python scripts/evalkit/dataset_stat.py node21 node21.images" expect="4,882" -->
<!-- claim: id=node21.positives tier=ci cmd="python scripts/evalkit/dataset_stat.py node21 node21.positives.nodule" expect="1,134" -->
<!-- claim: id=node21.original.images tier=nas cmd="python scripts/evalkit/dataset_stat.py --source node21_original base_image_dir/cxr_images/original_data/metadata.csv unique:img_name" expect="4,882" -->
| | |
|---|---|
| Registry keys | `node21`, `node21_original` |
| Task and outputs | Nodule detection: boxes (`bbox`) and 1 image label (`cls`) |
| Images in the table | 4,882 (1,134 with nodules) in each key. One image is one patient |
| Modality | Chest X-ray, 2-D, MHA |
| License | CC BY-NC-ND 4.0: no commercial use and no changed copies |
| Download | [Get the data: other datasets](../../../how-to/get-the-data.md#other-datasets) |

<!-- claim: id=node21.intro.positives tier=ci cmd="python scripts/evalkit/dataset_stat.py node21 node21.positives.nodule" expect="1,134" -->
<!-- claim: id=node21.intro.boxes tier=ci cmd="python scripts/evalkit/dataset_stat.py node21 node21.boxes" expect="1,476" -->
NODE21 is the data of the [NODE21 nodule challenge](https://node21.grand-challenge.org/)
([Sogancioglu et al., arXiv:2401.02192](https://arxiv.org/abs/2401.02192)). The frontal chest X-rays come
from ChestX-ray14, PadChest, JSRT and Open-I. Radiologists drew 1,476 nodule boxes on 1,134 images. The
other images are controls with no nodule. The source has 2 copies of each image: `proccessed_data` and
`original_data`.

The official metric is a free-response ROC (FROC): the sensitivity at each mean number of false positives per
image. It is not mAP. RadHarmony does not compute it. The `ct_patches/` folder holds 3-D CT nodule patches for
the generation task of the challenge. No class reads them.

## Quick use

`base_image_dir` is the dataset root with `cxr_images/`. The class finds `metadata.csv` in the folder of its
variant.

```
NODE21/
  cxr_images/
    proccessed_data/metadata.csv
    proccessed_data/images/n0001.mha
    original_data/metadata.csv
    original_data/filenames_orig_and_new.csv
    original_data/images/n0001.mha
  ct_patches/
```

```python
import torch
from radharmony.dataset import Node21Dataset

ds = Node21Dataset(
    base_image_dir="/data/NODE21/",
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)  # 1/5 of the images go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(1,)
# sample["bbox"]: Tensor(N, 4), one row for each box. N is 0 for a control
# sample["bbox_labels"]: a list of N names, for example ['nodule']
```

`Node21OriginalDataset` takes the same arguments and reads `original_data/`.

## How it is harmonized

<!-- claim: id=node21.metadata.rows tier=nas cmd="python scripts/evalkit/dataset_stat.py --source node21 base_image_dir/cxr_images/proccessed_data/metadata.csv rows" expect="5,224" -->
- Source: `metadata.csv` of the variant. One row is one box (5,224 rows in `proccessed_data`). A control has 1
  row with `label` 0 and `x`, `y`, `width` and `height` 0. The harmonizer makes one row for each image, and a
  control gets no box.
- IDs: the source has no patient ID. `img_name` `n0239.mha` gives `patient_id` `n0239`, `study_id` `n0239`
  and `image_path` `cxr_images/proccessed_data/images/n0239.mha`. So one image is one patient. A name starts
  with `n` for an image with nodules and with `c` for a control. `image_path` is relative to `base_image_dir`
  (D-04).
- Variant: `variant="processed"` (the default) reads `proccessed_data/`, and `variant="original"` reads
  `original_data/`. The column `variant` holds the value.
- Labels (D-05): `nodule` is 1 when the image has a row with `label` 1, and 0 if not. The source has no
  uncertain or blank label.
- Source dataset: `source_dataset` comes from `original_data/filenames_orig_and_new.csv`. It is
  `chestxray14` (617 images), `padchest` (314), `jsrt` (149) or `openi` (54). The file lists only the images
  with nodules, so a control gets `unknown`.
- View, age and sex: the source has none of them, so the harmonized table has no `view_position`, `age_years`
  or `sex` (D-01).
- Dropped rows: none by default. With `keep_negatives=False`, the harmonizer drops the controls.

## Labels

The 1 label column of `cls`:

| Column | Meaning |
|---|---|
| `nodule` | The image has 1 or more lung nodules |

## Boxes

- Source: `x`, `y`, `width` and `height`, in the pixels of the image. The source does not say which corner is
  the origin. With the top left corner, the boxes fall on nodules in the lungs, so the harmonizer uses that
  corner.
- The harmonizer reads the size of each image file (with SimpleITK), so it needs `base_image_dir`. It divides
  `y` and `y + height` by the height, and `x` and `x + width` by the width.
- The box is `[y_min, y_max, x_min, x_max]`, in fractions of the image size (D-07).
- Example: in `proccessed_data`, `n0239.mha` has `x` 776, `y` 579, `width` 92 and `height` 94, on a
  1024 x 1024 (H x W) image. It gives `bbox` `[[0.5654296875, 0.6572265625, 0.7578125, 0.84765625]]` and `bbox_labels`
  `["nodule"]`. In `original_data`, the same nodule has `y` 538, so `bbox` starts with `0.525390625`.
- `bbox_labels`: `nodule` for each box (D-07).
- Readers (D-36): the source has one set of boxes for each image, so the harmonizer does not group boxes.
- `bbox` and `bbox_labels` are JSON text in the harmonized table, so they stay the same after a CSV save and
  load.
- The sample: `bbox` is a float32 tensor `(N, 4)`, in fractions of the returned image (D-28). `bbox_labels` is
  a list of N names. Example: `(1, 4)` and `['nodule']`. An image with no box gives `(0, 4)` and `[]`. An image
  has 0 to 3 boxes.
- Dropped boxes: the harmonizer clips each box to the image. It drops a box with no area after the clip, and
  warns (D-07). The source files have no such box, in either variant.

## Splits

<!-- claim: id=node21.splits.images tier=ci cmd="python scripts/evalkit/dataset_stat.py node21 node21.images" expect="4,882" -->
<!-- claim: id=node21.splits.original.images tier=nas cmd="python scripts/evalkit/dataset_stat.py --source node21_original base_image_dir/cxr_images/original_data/metadata.csv unique:img_name" expect="4,882" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `Node21Dataset` | `node21` | 4,882 | `cxr_images/proccessed_data/` |
| `Node21OriginalDataset` | `node21_original` | 4,882 | `cxr_images/original_data/` |

NODE21 has no official split. `get_datasets(n_splits=5)` makes a train and val split. It groups by patient
(D-10). The source has no patient ID, so one image is one patient, and nobody can check that a patient is in
only 1 split. The public source is the train data of the challenge. The
challenge test set stays on Grand Challenge, so you cannot compute a NODE21 test score from these files.

## Arguments

The arguments are the same for `Node21Dataset` and `Node21OriginalDataset`.

<!-- args: radharmony.dataset.Node21Dataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The dataset root with `cxr_images/`. Not needed with `harmonizer_path`, which holds it |
| `variant` | `str` | No | `None` | `"processed"` or `"original"`. `None` gives the variant of the class: `"processed"` for `node21` and `"original"` for `node21_original` |
| `csv_path` | `str` | No | `None` | `metadata.csv`. Without it, the class uses the file in the folder of the variant |
| `keep_negatives` | `bool` | No | `True` | Keep the controls. With `False`, the harmonizer drops them |

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
from radharmony.harmonizer import Node21Harmonizer

h = Node21Harmonizer(base_image_dir="/data/NODE21/")
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'bbox', 'bbox_labels', 'nodule', 'variant', 'n_boxes', 'source_dataset']
```

One harmonizer serves both variants. Pass `variant="original"` for `original_data/`.

<!-- args: radharmony.harmonizer.Node21Harmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | No | `None` | `metadata.csv`. Without it, the harmonizer uses the file in `base_image_dir`, in the folder of the variant. Give `csv_path` or `base_image_dir`, or it raises `ValueError` |
| `base_image_dir` | `str` | No | `None` | The dataset root. The harmonizer reads the image sizes and `filenames_orig_and_new.csv` from it |
| `variant` | `str` | No | `"processed"` | `"processed"` or `"original"`. Another value raises `ValueError` |
| `keep_negatives` | `bool` | No | `True` | Keep the controls |

## Known data problems

- Folder name: the source spells the folder `proccessed_data`, with 2 c.
- Variants: every `proccessed_data` image is 1024 x 1024 (H x W) with 16-bit pixels. The `original_data`
  images have many sizes, for example 1024 x 1024 and 2991 x 3000 (H x W). They also have several pixel
  types: 8-bit, 16-bit and 64-bit float.
- Box coordinates: a box is valid only on the image of its variant. The processing resized and padded the
  images, so the coordinates moved (example in [Boxes](#boxes)).
- Source dataset: `filenames_orig_and_new.csv` lists only the images with nodules. So for a control, you
  cannot find the same image in ChestX-ray14, PadChest or Open-I. You cannot check that a control is not also
  in your other data.
