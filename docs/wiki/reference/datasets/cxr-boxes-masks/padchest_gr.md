# PadChest-GR

<!-- claim: id=padchest_gr.source.images tier=nas cmd="python scripts/evalkit/dataset_stat.py --source padchest_gr:tables csv_path unique:ImageID" expect="4,555" -->
<!-- claim: id=padchest_gr.source.patients tier=nas cmd="python scripts/evalkit/dataset_stat.py --source padchest_gr:tables csv_path unique:PatientID" expect="4,459" -->
| | |
|---|---|
| Registry keys | `padchest_gr_train`, `padchest_gr_validation`, `padchest_gr_test` |
| Task and outputs | Classification, 154 labels (`cls`). Boxes (`bbox`) with the sentence of each box. Reports (`report`) |
| Images in the table | The source tables give 4,555 images (1 for each study) of 4,459 patients. The paper is [Castro et al., arXiv:2411.05085](https://arxiv.org/abs/2411.05085). The RadHarmony tests cannot read the images, so no test checks the harmonized table |
| Modality | Chest X-ray, 2-D, PNG |
| License | The source has no license file. The terms are on the BIMCV download page |
| Download | [Get the data: PadChest-GR](../../../how-to/get-the-data.md#padchest-gr) |

PadChest-GR (grounded reports) is a BIMCV release of [PadChest](../cxr-labels/padchest.md) images. The
authors split each report into sentences. Each sentence is abnormal or not, and it has the PadChest labels of
its finding. Radiologists drew boxes for the findings that have a place on the image. Each sentence is in
English and in Spanish.

## Quick use

`base_image_dir` is the source folder. Its name ends with a space (see Known data problems). The class finds
`master_table.csv` and `grounded_reports_20240819.json` in this folder. The images are in `PadChest_GR/`.

```
BIMCV-Padchest-GR /
  grounded_reports_20240819.json
  master_table.csv
  PadChest_GR/<ImageID>.png
  PadChest_GR_progression_prior_studies/
```

```python
import torch
from radharmony.dataset import PadChestGRTrainDataset, PadChestGRTestDataset

train_ds = PadChestGRTrainDataset(
    base_image_dir="/data/PadChest-GR/BIMCV-Padchest-GR /",
    padchest_csv_path="/data/PadChest/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz",
    output_cls=True,
    output_bbox=True,
    output_report=True,
    dtype=torch.float32,
).get_datasets()
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(154,)
# sample["bbox"]: Tensor(N, 4), N is 0 for an image with no box
# sample["bbox_labels"]: a list of N names
# sample["bbox_findings"]: a list of N sentences
# sample["report"]: the report text

test_ds = PadChestGRTestDataset(
    base_image_dir="/data/PadChest-GR/BIMCV-Padchest-GR /",
    output_cls=True,
    dtype=torch.float32,
).get_datasets()
```

`PadChestGRValidationDataset` loads the validation split.

## How it is harmonized

- Source: `grounded_reports_20240819.json` has one item for each image, with its sentences, labels and
  boxes. `master_table.csv` has one row for each label of an abnormal sentence, and one `Normal` row for
  an image with no abnormal sentence. Each row has the patient, split, sex and age. The
  harmonizer joins them on `ImageID`. One row is one image (D-03).
- Checks: the patient, split, sex, age and date must be the same on all rows of an image in
  `master_table.csv`. If not, `harmonize()` raises `ValueError`.
- IDs: the `ImageID` `251488034557732338959601580328898734705_wigcpj.png` gives `image_path`
  `PadChest_GR/251488034557732338959601580328898734705_wigcpj.png`, relative to `base_image_dir` (D-04).
  The row of this image has a `StudyID`, which gives `study_id` `251488034557732338959601580328898734705`.
  It also has a `PatientID`, which gives `patient_id` `243089588168270594079245995953840813067`. They are
  the same IDs as in PadChest.
- Labels (154 columns, D-05): a label of an abnormal sentence gives 1 in its column. All other
  columns are 0. An image with no abnormal sentence has 0 in all columns. A label that is not in the
  list of 154 raises `ValueError`.
- View (D-01): the PadChest-GR files have no view. With `padchest_csv_path`, the view comes from the
  PadChest column `Projection`. It is `PA`, `AP` or `AP_HORIZONTAL`. An image that is not in that file
  raises `ValueError`. Without `padchest_csv_path`, the harmonized table has no `view_position`.
- Age and sex: `age_years` comes from `PatientAge`, in years (D-06). `sex` is `M` or `F`, from
  `PatientSex_DICOM` (D-26).
- Report: the harmonizer joins the sentences of an image into one text. The language is English
  (`report_language="en"`) or Spanish (`report_language="es"`).
- Dropped rows: a split class keeps only the rows of its split. The harmonizer drops no other row.

## Labels

The 154 label columns are PadChest labels in snake_case, for example `apical_pleural_thickening`. They are
the labels of the abnormal sentences of the source. The list, in the order of `cls`, is `LABEL_COLUMNS` in
`radharmony/harmonizer/padchest_gr/padchest_gr.py`.

An image with no box can have a label. Example: a sentence with the label `vertebral_fracture` never has a
box. So `n_boxes == 0` does not mean that all labels are 0.

## Boxes

- Source: each sentence in the JSON file has `boxes`, a list of `[x0, y0, x1, y1]` fractions of the image.
  So x comes first.
- The source values are already fractions, so the harmonizer only changes the order to `[y0, y1, x0, x1]`.
  Example: the sentence "Minimal biapical pleural thickening." has the box
  `[0.27670749, 0.10440252, 0.48518978, 0.18993711]`. This gives
  `[0.10440252, 0.18993711, 0.27670749, 0.48518978]`.
- The box is `[y_min, y_max, x_min, x_max]`, in fractions of the image size (D-07).
- `bbox_labels` has the labels of the sentence of each box, in snake_case. A sentence with 2 or 3 labels gives
  one name with ` + ` between the labels, for example `suture_material + surgery_breast + metal` (D-07).
- `bbox_findings` has the English sentence of each box. It has the same order as `bbox`.
- The harmonized table holds `bbox`, `bbox_labels` and `bbox_findings` as JSON text, because the lists of
  the images have different lengths. An image with no box has `"[]"`. `n_boxes` is the number of boxes.
- Readers (D-36): the source gives one set of boxes in `boxes`, and a second set in `extra_boxes`,
  so the harmonizer does not group boxes. The extra boxes go to `extra_bbox` and `n_extra_boxes`.
  They are not in the sample.
- The sample: `bbox` is a float32 tensor `(N, 4)`, in fractions of the returned image (D-28). `bbox_labels`
  is a list of N names, and `bbox_findings` is a list of N sentences. Example: `(2, 4)` and
  `['apical_pleural_thickening', 'apical_pleural_thickening']`. An image with no box gives `(0, 4)` and `[]`.
- Dropped boxes: the harmonizer clips each box and each extra box to the image. It drops a box with no area
  after the clip, with its label and sentence, and warns (D-07). The source JSON file has no such box.

## Reports

- `report` is the text of all sentences of the image, in the language of `report_language`. Example: "Minimal
  biapical pleural thickening. Slight blunting of the posterior left costophrenic angle. No other
  significant alterations."
- `findings` is JSON text with one item for each sentence. An item has `sentence_en`, `sentence_es`,
  `abnormal`, `labels`, `locations`, `progression` and `n_boxes`.
- A normal image also has a report, with sentences that are not abnormal.

## Splits

<!-- claim: id=padchest_gr.splits.train tier=nas cmd="python scripts/evalkit/dataset_stat.py --source padchest_gr:tables csv_path unique:ImageID@split=train" expect="3,185" -->
<!-- claim: id=padchest_gr.splits.validation tier=nas cmd="python scripts/evalkit/dataset_stat.py --source padchest_gr:tables csv_path unique:ImageID@split=validation" expect="455" -->
<!-- claim: id=padchest_gr.splits.test tier=nas cmd="python scripts/evalkit/dataset_stat.py --source padchest_gr:tables csv_path unique:ImageID@split=test" expect="915" -->
<!-- claim: id=padchest_gr.splits.all tier=nas cmd="python scripts/evalkit/dataset_stat.py --source padchest_gr:tables csv_path unique:ImageID" expect="4,555" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `PadChestGRTrainDataset` | `padchest_gr_train` | 3,185 | `split` is `train` in `master_table.csv` |
| `PadChestGRValidationDataset` | `padchest_gr_validation` | 455 | `split` is `validation` |
| `PadChestGRTestDataset` | `padchest_gr_test` | 915 | `split` is `test`. Use it as the test split |
| `PadChestGRDataset` | `padchest_gr` | 4,555 | All images. It mixes the 3 splits, so do not use it for a benchmark (D-09) |

The counts above come from `master_table.csv` of the source. They match the 70%, 10% and 20% of the paper.

No patient is in 2 splits.

Every PadChest-GR image is also in [PadChest](../cxr-labels/padchest.md), with the same `ImageID` and
`PatientID`. The PadChest-GR split does not follow PadChest. So a PadChest-GR test patient can be in your
PadChest train data. Drop these patients before you train on PadChest and test on PadChest-GR.

## Arguments

The arguments are the same for the 3 split classes and `PadChestGRDataset`. A split class sets `split`
itself. If you give another split name, it raises `ValueError`.

<!-- args: radharmony.dataset.PadChestGRTrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The source folder with the JSON file, `master_table.csv` and `PadChest_GR/`. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `master_table.csv`. Without it, the class uses the file in `base_image_dir` |
| `json_path` | `str` | No | `None` | `grounded_reports_20240819.json`. Without it, the class uses the file in `base_image_dir` |
| `padchest_csv_path` | `str` | No | `None` | The PadChest file `PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz`. It adds `view_position` |
| `split` | `str` | No | `None` | `PadChestGRDataset` only: `"train"`, `"validation"` or `"test"`. `None` keeps all images |
| `report_language` | `str` | No | `"en"` | The language of `report`: `"en"` or `"es"` |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_report` | `bool` | No | `False` | Add the `report` text to the sample. The class drops the rows with no report, and warns (D-33) |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Add `bbox`, `bbox_labels` and `bbox_findings` to the sample |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The labels stay float32 |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

The harmonizer reads only the JSON file, `master_table.csv` and, with `padchest_csv_path`, the PadChest
label file. So it runs without the images.

```python
from radharmony.harmonizer import PadChestGRTrainHarmonizer

h = PadChestGRTrainHarmonizer(
    base_image_dir="/data/PadChest-GR/BIMCV-Padchest-GR /",
    padchest_csv_path="/data/PadChest/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'view_position', 'bbox', 'bbox_labels', 'report', 'age_years',
#  'sex', 'abnormal_foreign_body', ..., 'split', 'n_boxes', 'n_extra_boxes', 'bbox_findings', 'extra_bbox',
#  'findings', 'patient_age', 'study_year', 'prior_image_id']
```

`PadChestGRValidationHarmonizer`, `PadChestGRTestHarmonizer` and `PadChestGRHarmonizer` take the same
arguments as `PadChestGRTrainHarmonizer`.

Extra columns:

- `split`, `n_boxes`, `n_extra_boxes`, `bbox_findings`, `extra_bbox` and `findings`.
- `patient_age`: the age in years, as in `age_years`.
- `study_year`: from `StudyDate`.
- `prior_image_id`: the `ImageID` of the prior image, if the source names one. The prior images are in
  `PadChest_GR_progression_prior_studies/`. The classes do not read them.

<!-- args: radharmony.harmonizer.PadChestGRTrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | No | `None` | `master_table.csv`. Without it, the harmonizer uses the file in `base_image_dir` |
| `base_image_dir` | `str` | No | `None` | The source folder. Without it, give `csv_path` and `json_path`, or the harmonizer raises `ValueError` |
| `json_path` | `str` | No | `None` | `grounded_reports_20240819.json`. Without it, the harmonizer uses the file in `base_image_dir` |
| `padchest_csv_path` | `str` | No | `None` | The PadChest label file. It adds `view_position` |
| `split` | `str` | No | `None` | `PadChestGRHarmonizer` only: `"train"`, `"validation"` or `"test"`. `None` keeps all images |
| `report_language` | `str` | No | `"en"` | The language of `report`: `"en"` or `"es"` |

## Known data problems

<!-- claim: id=padchest_gr.paper.extra source="Castro et al., arXiv 2024" expect="5,242" -->
- Folder name: the source folder `BIMCV-Padchest-GR ` ends with a space.
- Split zip: the images are in a zip of 37 parts. `unzip` cannot read it. Use `7z x PadChest_GR.zip.001`
  (see [Get the data](../../../how-to/get-the-data.md#padchest-gr)).
- Row count: a count of the `master_table.csv` rows is not a count of images or of sentences. Each label of
  an abnormal sentence has its own row. A sentence that is not abnormal has no row.
- Extra boxes: the JSON file does not say what `extra_boxes` is. The paper says that each abnormal sentence
  has up to 2 independent sets of boxes, and that 5,242 findings have extra boxes.
- Vague labels: some labels are vague or do not name a body part, for example `dai`, `end on vessel`,
  `chronic changes`, `metal`, `surgery` and `abnormal foreign body`. The harmonizer keeps them.
