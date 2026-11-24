# ChestX-ray14

<!-- claim: id=chestxray14.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chestxray14 chestxray14_train.images" expect="86,524" -->
<!-- claim: id=chestxray14.train.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py chestxray14 chestxray14_train.patients" expect="28,008" -->
<!-- claim: id=chestxray14.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chestxray14 chestxray14_test.images" expect="25,596" -->
<!-- claim: id=chestxray14.test.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py chestxray14 chestxray14_test.patients" expect="2,797" -->
<!-- claim: id=chestxray14.bbox.images tier=nas cmd="python scripts/evalkit/dataset_stat.py --source chestxray14_bbox bbox_csv_path 'unique:Image Index'" expect="880" -->
| | |
|---|---|
| Registry keys | `chestxray14_train`, `chestxray14_test`, `chestxray14_bbox` |
| Task and outputs | Classification, 15 labels (`cls`). Boxes (`bbox`) with `chestxray14_bbox` |
| Images in the table | train 86,524 (28,008 patients), test 25,596 (2,797 patients), boxes 880 |
| Modality | Chest X-ray, 2-D, PNG |
| License | NIH Clinical Center release. Cite Wang et al., CVPR 2017 |
| Download | [Get the data: ChestX-ray14](../../../how-to/get-the-data.md#chestx-ray14) |

<!-- claim: id=chestxray14.all.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chestxray14 chestxray14_train.images+chestxray14_test.images" expect="112,120" -->
<!-- claim: id=chestxray14.all.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py chestxray14 chestxray14_train.patients+chestxray14_test.patients" expect="30,805" -->
<!-- claim: id=chestxray14.bbox.boxes tier=nas cmd="python scripts/evalkit/dataset_stat.py --source chestxray14_bbox bbox_csv_path rows" expect="984" -->
ChestX-ray14 (NIH Chest X-ray Dataset) has 112,120 frontal chest X-rays of 30,805 patients from the NIH
Clinical Center. A text-mining tool read each report and gave 14 findings or "No Finding". The official
split puts each patient in train or in test. Radiologists drew 984 boxes of 8 findings on 880 test
images.

## Quick use

`base_image_dir` is the `CXR14/` folder. The class finds `Data_Entry_2017.csv` and the split lists in
this folder. The images are in 12 folders, `images_001/` to `images_012/`.

```
CXR14/
  Data_Entry_2017.csv
  BBox_List_2017.csv
  train_val_list.txt
  test_list.txt
  images_001/images/00000001_000.png
  ...
  images_012/images/00030805_000.png
```

```python
import torch
from radharmony.dataset import ChestXray14TrainDataset, ChestXray14TestDataset

train = ChestXray14TrainDataset(
    base_image_dir="/data/NIH_CXR/CXR14/",
    output_cls=True,
    dtype=torch.float32,
)
train_ds, val_ds = train.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(15,)

test_ds = ChestXray14TestDataset(
    base_image_dir="/data/NIH_CXR/CXR14/",
    output_cls=True,
    dtype=torch.float32,
).get_datasets()
```

## How it is harmonized

- Source: `Data_Entry_2017.csv`. One row is one image (D-03). A split class keeps only the images in its
  list, `train_val_list.txt` or `test_list.txt` (D-09).
- IDs: the row with `Image Index` `00000003_000.png`, `Patient ID` 3 and `Follow-up #` 0 gives
  `patient_id` `3` and `study_id` `3_0`. So one study is one image (D-01).
- `image_path`: the harmonizer looks for the file in the 12 image folders. Example:
  `images_001/images/00000003_000.png`. It is relative to `base_image_dir` (D-04).
- Labels (15 columns, D-05): `Finding Labels` holds names with a `|` between them, for example
  `Atelectasis|Effusion`. A name in the list gives 1, and all other labels give 0. The source has no
  uncertain or blank label.
- View (D-01): from `View Position`. It is `PA` or `AP` for every image.
- Age and sex: `age_years` comes from `Patient Age`, in years (D-06). `sex` is `M` or `F`, from
  `Patient Gender` (D-26).
- Dropped rows: the harmonizer drops the rows that are not in the split list. It drops no row for its
  labels.

## Labels

The 15 label columns, in the order of `cls`:

| Column | Meaning |
|---|---|
| `atelectasis` | Part of a lung is collapsed |
| `cardiomegaly` | The heart is large |
| `consolidation` | Airspace consolidation |
| `edema` | Pulmonary edema |
| `effusion` | Pleural effusion |
| `emphysema` | Emphysema |
| `fibrosis` | Pulmonary fibrosis |
| `hernia` | Hernia |
| `infiltration` | Infiltration |
| `mass` | A lung mass |
| `no_finding` | No finding |
| `nodule` | A lung nodule |
| `pleural_thickening` | Pleural thickening |
| `pneumonia` | Pneumonia |
| `pneumothorax` | Pneumothorax |

## Boxes

`ChestXray14BboxDataset` gives the 880 images with boxes. Each image has 1 or more boxes.

```python
import torch
from radharmony.dataset import ChestXray14BboxDataset

bbox_ds = ChestXray14BboxDataset(
    base_image_dir="/data/NIH_CXR/CXR14/",
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
).get_datasets()
sample = bbox_ds[0]
# sample["bbox"]: Tensor(1, 4), one row for each box
# sample["bbox_labels"]: ['infiltration']
```

- Source: `BBox_List_2017.csv`. A box is `x`, `y`, `w`, `h` in the pixels of the 1024 x 1024 PNG.
- The harmonizer divides `x` and `w` by the PNG width, and `y` and `h` by the PNG height. So
  `ChestXray14BboxHarmonizer` needs `base_image_dir`, and without it `harmonize()` raises `ValueError`.
- The box is `[dim0_min, dim0_max, dim1_min, dim1_max]`, so `[y_min, y_max, x_min, x_max]`. The
  harmonizer clips each box to the image (D-07).
- A box label uses the snake_case name of its label column (D-07). For example, `Infiltrate` gives
  `infiltration`. The 8 box labels are `atelectasis`, `cardiomegaly`, `effusion`, `infiltration`, `mass`,
  `nodule`, `pneumonia` and `pneumothorax`.

## Splits

<!-- claim: id=chestxray14.splits.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chestxray14 chestxray14_train.images" expect="86,524" -->
<!-- claim: id=chestxray14.splits.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chestxray14 chestxray14_test.images" expect="25,596" -->
<!-- claim: id=chestxray14.splits.all.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chestxray14 chestxray14_train.images+chestxray14_test.images" expect="112,120" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `ChestXray14TrainDataset` | `chestxray14_train` | 86,524 | `train_val_list.txt` |
| `ChestXray14TestDataset` | `chestxray14_test` | 25,596 | `test_list.txt`. Use it as the test split |
| `ChestXray14BboxDataset` | `chestxray14_bbox` | 880 | The images in `BBox_List_2017.csv` |
| `ChestXray14Dataset` | `chestxray14` | 112,120 minus the 880 with boxes | All images. It mixes the 2 splits, so do not use it for a benchmark (D-09) |

`get_datasets(n_splits=5)` makes a train and val split from the train class. It groups by patient (D-10). No
patient is in 2 splits (train and test). All 880 images with boxes are in `test_list.txt`. The test class
keeps these 880 images, without boxes. Do not train on `chestxray14_bbox` and then test on
`chestxray14_test`, because the same images are in both.

`ChestXray14Dataset` drops the 880 images with boxes when it finds `BBox_List_2017.csv`. It finds the
file from `bbox_csv_path`, or in `base_image_dir` or the folder above it.

## Arguments

The arguments are the same for `ChestXray14TrainDataset`, `ChestXray14TestDataset` and
`ChestXray14BboxDataset`.

<!-- args: radharmony.dataset.ChestXray14TrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `CXR14/` folder with `images_001/` to `images_012/`. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `Data_Entry_2017.csv`. Without it, the class looks in `base_image_dir` and the folder above it |
| `bbox_csv_path` | `str` | No | `None` | `BBox_List_2017.csv`. Only `ChestXray14BboxDataset` uses it. The train and test classes ignore it |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_report` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Add `bbox` and `bbox_labels`, with `ChestXray14BboxDataset` only. In the other classes: not supported. The class warns and adds no key (D-11) |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The labels stay float32 |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import ChestXray14TrainHarmonizer

h = ChestXray14TrainHarmonizer(
    csv_path="/data/NIH_CXR/CXR14/Data_Entry_2017.csv",
    base_image_dir="/data/NIH_CXR/CXR14/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'view_position', 'age_years', 'sex', 'atelectasis', ...]
```

`ChestXray14TestHarmonizer` takes the same arguments. `ChestXray14BboxHarmonizer` also needs
`bbox_csv_path`, and its harmonized table adds `bbox` and `bbox_labels`.

<!-- args: radharmony.harmonizer.ChestXray14TrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `Data_Entry_2017.csv` |
| `bbox_csv_path` | `str` | No | `None` | The train and test harmonizers ignore it |
| `base_image_dir` | `str` | No | `None` | Needed by `harmonize()`. The `CXR14/` folder with the split list. Without it, `harmonize()` raises an error |

## Known data problems

- Age: 16 source ages are above 130 (148 to 414), so they are errors (D-06). Example: patient 5567 is
  412 at the first visit and 53 at the next.
- RGBA files: some PNGs are RGBA, but the image is gray (D-42).
- Box labels: `BBox_List_2017.csv` writes `Infiltrate`, but `Data_Entry_2017.csv` writes `Infiltration`.
- Box CSV header: the header has the commas of `Bbox [x,y,w,h]` inside it, and it ends with 3 empty
  fields. So the header has 9 fields, but each data row has 6.
- Image size: `OriginalImage[Width` and `Height]` in `Data_Entry_2017.csv` give the size before the
  authors resized the image to the PNG. The boxes use the PNG size, so these columns do not match the
  boxes.
