# Chest ImaGenome

<!-- claim: id=chest_imagenome.gold.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chest_imagenome_gold chest_imagenome_gold.images" expect="1,000" -->
<!-- claim: id=chest_imagenome.gold.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py chest_imagenome_gold chest_imagenome_gold.patients" expect="500" -->
<!-- claim: id=chest_imagenome.silver.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chest_imagenome_silver chest_imagenome_silver_train.images+chest_imagenome_silver_valid.images+chest_imagenome_silver_test.images" expect="237,844" -->
<!-- claim: id=chest_imagenome.silver.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py chest_imagenome_silver chest_imagenome_silver_train.patients+chest_imagenome_silver_valid.patients+chest_imagenome_silver_test.patients" expect="62,945" -->
| | |
|---|---|
| Registry keys | `chest_imagenome_gold`, `chest_imagenome_silver_train`, `chest_imagenome_silver_valid`, `chest_imagenome_silver_test` |
| Task and outputs | Boxes of anatomical regions (`bbox`), with the findings of each region (`bbox_findings`). No `cls` |
| Images in the table | gold 1,000 (500 patients), silver 237,844 in 3 splits (62,945 patients) |
| Modality | Chest X-ray, 2-D, DICOM from MIMIC-CXR |
| License | PhysioNet credentialed access, as for MIMIC-CXR |
| Download | [Get the data: other datasets](../../../how-to/get-the-data.md#other-datasets). The images come from [MIMIC-CXR (DICOM)](../../../how-to/get-the-data.md#mimic-cxr-dicom) |

Chest ImaGenome ([Wu et al., NeurIPS 2021 Datasets and Benchmarks](https://arxiv.org/abs/2108.00316)) is a
scene graph dataset for the MIMIC-CXR images. For each frontal image, it gives boxes of anatomical regions
(for example the right lung or the trachea) and the findings in each region. It has no image files: the
images come from the MIMIC-CXR DICOM files. RadHarmony reads 2 sets:

<!-- claim: id=chest_imagenome.gold.boxes tier=ci cmd="python scripts/evalkit/dataset_stat.py chest_imagenome_gold chest_imagenome_gold.boxes" expect="25,961" -->
<!-- claim: id=chest_imagenome.silver.boxes tier=ci cmd="python scripts/evalkit/dataset_stat.py chest_imagenome_silver chest_imagenome_silver_train.boxes+chest_imagenome_silver_valid.boxes+chest_imagenome_silver_test.boxes" expect="8,497,868" -->
- Gold: 1,000 images. People checked the boxes. The harmonized table has 25,961 boxes of 26 regions.
- Silver: the full set, made by an automatic pipeline, with official train, valid and test splits. The
  harmonized table has 8,497,868 boxes of up to 36 regions.

The findings stay with their boxes. RadHarmony does not make an image label (`cls`) from them.

## Quick use

`base_image_dir` is the `files/` folder of MIMIC-CXR (DICOM), because the harmonizer reads the image size
from each DICOM header. `annotation_dir` is the Chest ImaGenome folder.

```
MIMIC-CXR-V2-AWS/files/                <- base_image_dir
  p10/p10000935/s50578979/d0b71acc-b5a62046-bbb5f6b8-7b173b85-65cdf738.dcm
CHEST-IMAGENOME/                       <- annotation_dir
  gold_dataset/
    gold_bbox_coordinate_annotations_1000images.csv
    gold_object_attribute_with_coordinates.txt
  silver_dataset/
    scene_graph.zip
    splits/train.csv, valid.csv, test.csv, images_to_avoid.csv
  utils/cxr-record-list_view.csv
```

```python
import torch
from radharmony.dataset import ChestImaGenomeGoldDataset, ChestImaGenomeSilverValidDataset

gold = ChestImaGenomeGoldDataset(
    base_image_dir="/data/MIMIC-CXR-V2-AWS/files/",
    annotation_dir="/data/CHEST-IMAGENOME/",
    dtype=torch.float32,
)
train_ds, val_ds = gold.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["bbox"]: Tensor(N, 4), one row for each region (N = 26 for most gold images)
# sample["bbox_labels"]: a list of N names
# sample["bbox_findings"]: a list of N strings, one for each box

valid_ds = ChestImaGenomeSilverValidDataset(
    base_image_dir="/data/MIMIC-CXR-V2-AWS/files/",
    annotation_dir="/data/CHEST-IMAGENOME/",
    dtype=torch.float32,
).get_datasets()
# valid_ds[0]["bbox"]: Tensor(N, 4) (N = 36 for most silver images)
```

`output_bbox` is `True` by default, because the boxes are the only output.

## How it is harmonized

- Source, gold: `gold_bbox_coordinate_annotations_1000images.csv` (one row for each image and region) and
  `gold_object_attribute_with_coordinates.txt` (the findings). `utils/cxr-record-list_view.csv` gives the
  patient, study, path and view of each `dicom_id`.
- Source, silver: the split CSV (`train.csv`, `valid.csv` or `test.csv`) gives the patient, study, path and
  view. `scene_graph.zip` has one JSON file for each image, with its boxes and findings. The harmonizer
  leaves out the images in `images_to_avoid.csv`.
- IDs: one row is one image (D-03). The `dicom_id` `00046130-fd952ef0-57f2948d-491a16b4-5db3a18c` gives
  `patient_id` `15184836`, `study_id` `59382057`, and `image_path`
  `p15/p15184836/s59382057/00046130-fd952ef0-57f2948d-491a16b4-5db3a18c.dcm`. The IDs are the MIMIC digits
  (D-01). `image_path` is relative to `base_image_dir` (D-04).
- Findings: the harmonizer keeps a finding of a region when its relation is `yes` and its category is
  `anatomicalfinding`, `disease`, `tubesandlines` or `device`. It leaves out the categories `nlp` and
  `technicalassessment`.
- View (D-01): `AP` or `PA`, from `ViewPosition` of the record list (gold) or the split CSV (silver).
- Age and sex: silver only. `age_years` comes from the scene graph `age_decile`, in years (D-06). It is the
  low edge of the group: `50-60` gives 50, and `>=90` gives 90. `sex` is `M`, `F` or NaN, from the scene
  graph `gender` (D-26). The gold harmonized table has no age and no sex.
- Extra columns: `bbox_findings`, `image_width` and `image_height`. Gold adds `findings_annotated`, and
  silver adds `split`.
- Dropped rows: the silver harmonizer drops an image of the split list that has no scene graph file, and
  warns. The harmonizer also drops an image when it cannot read the size in its DICOM header,
  and warns with the count (D-07).

## Labels

The harmonized table has no label columns and no `cls`. The labels are in the boxes:

- `bbox_labels`: the region of each box. See [Boxes](#boxes).
- `bbox_findings`: the findings of each region, joined with `|`, in the same order as the boxes. An empty
  string means no finding. In gold, it can also mean "not annotated" (see [Boxes](#boxes)). Example:
  `left_hilar_structures` with `lung opacity|vascular congestion`. The full list of findings is in
  `semantics/attribute_relations_v1.txt` of `annotation_dir`.

## Boxes

- Source: `original_x1`, `original_y1`, `original_x2` and `original_y2`, in the pixels of the DICOM image.
- The harmonizer reads the size of each image file (the DICOM tags `Rows` and `Columns`), so it needs
  `base_image_dir`. It divides `y1` and `y2` by the height, and `x1` and `x2` by the width.
- The box is `[y_min, y_max, x_min, x_max]`, in fractions of the image size (D-07). Example: the image
  `00046130-fd952ef0-57f2948d-491a16b4-5db3a18c` (the IDs example above) is 2544 x 3056 (H x W). Its gold
  `aortic arch` box is 1691, 900, 1978, 1173 (x1, y1, x2, y2). It gives `[0.3538, 0.4611, 0.5533, 0.6473]`.
- `bbox_labels`: the snake_case name of the region. For example, `aortic arch` gives `aortic_arch` (D-07).
- Readers (D-36): the source has one set of boxes for each image, so the harmonizer does not group boxes.
- The sample: `bbox` is a float32 tensor `(N, 4)`, in fractions of the returned image (D-28). `bbox_labels`
  and `bbox_findings` are lists of N strings. Example: `(26, 4)`, `['aortic_arch', 'cardiac_silhouette', ...]`
  and `['', '', ...]` for a gold image. Most silver images give `(36, 4)`. Every image has at least one box.
- Dropped boxes: the harmonizer clips each box to the image. It drops a box with no area after the clip, and
  warns (D-07). It also drops an inverted box (`x1 > x2` or `y1 > y2`), and it does not swap the corners. It
  drops the label and the findings of a dropped box too. Known data problems gives the counts.
- Findings with no box (gold): the harmonizer leaves out a finding on a region that has no gold box, and
  warns (D-07).
- Images with no finding annotation (gold): the column `findings_annotated` is True for the gold images that
  have finding annotations. For the other gold images, an empty `bbox_findings` entry means "not annotated",
  not "no finding" (D-33). Filter with `df[df["findings_annotated"]]` before you count findings.

## Splits

<!-- claim: id=chest_imagenome.splits.gold tier=ci cmd="python scripts/evalkit/dataset_stat.py chest_imagenome_gold chest_imagenome_gold.images" expect="1,000" -->
<!-- claim: id=chest_imagenome.splits.train tier=nas cmd="python scripts/evalkit/dataset_stat.py --source chest_imagenome_silver_train annotation_dir/silver_dataset/splits/train.csv rows" expect="166,521" -->
<!-- claim: id=chest_imagenome.splits.valid tier=nas cmd="python scripts/evalkit/dataset_stat.py --source chest_imagenome_silver_valid annotation_dir/silver_dataset/splits/valid.csv rows" expect="23,953" -->
<!-- claim: id=chest_imagenome.splits.test tier=nas cmd="python scripts/evalkit/dataset_stat.py --source chest_imagenome_silver_test annotation_dir/silver_dataset/splits/test.csv rows" expect="47,393" -->
<!-- claim: id=chest_imagenome.splits.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chest_imagenome_silver chest_imagenome_silver_train.images" expect="166,504" -->
<!-- claim: id=chest_imagenome.splits.valid.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chest_imagenome_silver chest_imagenome_silver_valid.images" expect="23,951" -->
<!-- claim: id=chest_imagenome.splits.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chest_imagenome_silver chest_imagenome_silver_test.images" expect="47,389" -->
<!-- claim: id=chest_imagenome.splits.silver tier=ci cmd="python scripts/evalkit/dataset_stat.py chest_imagenome_silver chest_imagenome_silver_train.images+chest_imagenome_silver_valid.images+chest_imagenome_silver_test.images" expect="237,844" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `ChestImaGenomeGoldDataset` | `chest_imagenome_gold` | 1,000 | The gold box file |
| `ChestImaGenomeSilverTrainDataset` | `chest_imagenome_silver_train` | 166,504 | `splits/train.csv` (166,521 rows) |
| `ChestImaGenomeSilverValidDataset` | `chest_imagenome_silver_valid` | 23,951 | `splits/valid.csv` (23,953 rows) |
| `ChestImaGenomeSilverTestDataset` | `chest_imagenome_silver_test` | 47,389 | `splits/test.csv` (47,393 rows). Use it as the test split |
| `ChestImaGenomeSilverDataset` | `chest_imagenome_silver` | 237,844 | All images. It mixes the 3 splits, so do not use it for a benchmark (D-09) |

A silver split class drops the images of its split file that have no scene graph file (see Known data
problems). In silver, no patient is in 2 splits. Gold has no official split. `get_datasets(n_splits=5)` makes
a train and val split. It groups by patient (D-10).

A silver split class fixes `split` to its own split. A call with another `split` value raises `ValueError`.
It also loads a harmonizer that `ChestImaGenomeSilverHarmonizer` saved, but only when every row is in its
split. If not, it raises `ValueError` (D-09).

## Arguments

The arguments are the same for the gold class and the 3 silver split classes. `ChestImaGenomeSilverDataset`
also takes `split`: `"all"` (the default), `"train"`, `"valid"` or `"test"`.

<!-- args: radharmony.dataset.ChestImaGenomeGoldDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `files/` folder of MIMIC-CXR (DICOM), with `p10/` to `p19/` in it. Not needed with `harmonizer_path`, which holds it |
| `annotation_dir` | `str` | Yes | `None` | The Chest ImaGenome folder, with `gold_dataset/`, `silver_dataset/` and `utils/` in it. Not needed with `harmonizer_path`, which holds it |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11). The harmonized table has no label columns |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_report` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `True` | Add `bbox`, `bbox_labels` and `bbox_findings` to the sample |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img` |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import ChestImaGenomeGoldHarmonizer

h = ChestImaGenomeGoldHarmonizer(
    annotation_dir="/data/CHEST-IMAGENOME/",
    base_image_dir="/data/MIMIC-CXR-V2-AWS/files/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'view_position', 'bbox', 'bbox_labels', 'bbox_findings',
#  'image_width', 'image_height', 'findings_annotated']
```

`ChestImaGenomeSilverTrainHarmonizer`, `ChestImaGenomeSilverValidHarmonizer` and
`ChestImaGenomeSilverTestHarmonizer` take the same arguments. `ChestImaGenomeSilverHarmonizer` also takes
`split`. A full silver split reads one DICOM header for each image, so it is slow. Save the harmonizer to
use it again.

<!-- args: radharmony.harmonizer.ChestImaGenomeGoldHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `annotation_dir` | `str` | Yes | | The Chest ImaGenome folder |
| `base_image_dir` | `str` | No | `None` | Needed by `harmonize()`. The `files/` folder of MIMIC-CXR (DICOM). Without it, `harmonize()` raises `ValueError` |

## Known data problems

- Gold regions: the gold box file has no `mediastinum` box, but the gold attribute file puts 227 findings
  on `mediastinum` (181 images). So the harmonized table does not have them (D-07).
- Gold box corners: 22 gold boxes are inverted (`x1 > x2` or `y1 > y2`). With the corners swapped, they
  still do not cover their region (D-07). With the 6 gold boxes that have no area, the harmonizer drops 28
  gold boxes.
- Silver boxes: some silver boxes have no area inside the image, for example 235 in the valid split. No
  silver box is inverted.
- Gold findings: the gold attribute file has only 500 of the 1,000 gold images (D-33).
- Silver age group: the source writes the group 10-20 as `20-Oct`, because a spreadsheet read it as a date.
  The harmonizer reads it as 10 (D-06).
- Silver scene graphs: 17 train, 2 valid and 4 test images of the split files have no file in
  `scene_graph.zip`.
