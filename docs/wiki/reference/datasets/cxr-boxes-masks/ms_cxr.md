# MS-CXR

<!-- claim: id=ms_cxr.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py ms_cxr ms_cxr_train.images" expect="737" -->
<!-- claim: id=ms_cxr.val.images tier=ci cmd="python scripts/evalkit/dataset_stat.py ms_cxr ms_cxr_val.images" expect="155" -->
<!-- claim: id=ms_cxr.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py ms_cxr ms_cxr_test.images" expect="155" -->
<!-- claim: id=ms_cxr.all.boxes tier=ci cmd="python scripts/evalkit/dataset_stat.py ms_cxr ms_cxr_train.boxes+ms_cxr_val.boxes+ms_cxr_test.boxes" expect="1,448" -->
| | |
|---|---|
| Registry keys | `ms_cxr_train`, `ms_cxr_val`, `ms_cxr_test` |
| Task and outputs | Phrase grounding. Boxes of 8 findings (`bbox`), and 8 labels (`cls`) |
| Images in the table | train 737, val 155, test 155. 1,448 boxes in all |
| Modality | Chest X-ray, 2-D, JPEG from MIMIC-CXR-JPG 2.0.0 |
| License | PhysioNet credentialed access, as for MIMIC-CXR |
| Download | [Get the data: other datasets](../../../how-to/get-the-data.md#other-datasets). The images come from [MIMIC-CXR-JPG](../../../how-to/get-the-data.md#mimic-cxr-jpg) |

<!-- claim: id=ms_cxr.source.images tier=nas cmd="python scripts/evalkit/dataset_stat.py --source ms_cxr_train csv_path unique:dicom_id" expect="1,047" -->
<!-- claim: id=ms_cxr.source.boxes tier=nas cmd="python scripts/evalkit/dataset_stat.py --source ms_cxr_train csv_path rows" expect="1,448" -->
MS-CXR is a phrase grounding benchmark ([Boecking et al., ECCV 2022](https://arxiv.org/abs/2204.09817)).
Radiologists linked a phrase of the report to a box on the image. The source is one CSV file with 1,448
phrase and box pairs on 1,047 MIMIC-CXR images. Each box has one of 8 findings. The source has an official
train, val and test split.

## Quick use

`base_image_dir` is the `files/` folder of MIMIC-CXR-JPG 2.0.0. The CSV is in its own folder, so pass
`csv_path`.

```
mimic-cxr-jpg/2.0.0/files/        <- base_image_dir
  p10/p10233088/s54276838/675d792f-a3521e48-5eec8573-1e81d644-e60c34f8.jpg
ms-cxr/
  MS_CXR_Local_Alignment_v1.1.0.csv   <- csv_path
```

```python
import torch
from radharmony.dataset import MSCXRTrainDataset, MSCXRValDataset, MSCXRTestDataset

args = dict(
    base_image_dir="/data/mimic-cxr-jpg/2.0.0/files/",
    csv_path="/data/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv",
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
)
train_ds = MSCXRTrainDataset(**args).get_datasets()
val_ds = MSCXRValDataset(**args).get_datasets()
test_ds = MSCXRTestDataset(**args).get_datasets()
sample = val_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(8,)
# sample["bbox"]: Tensor(N, 4), one row for each box
# sample["bbox_labels"]: a list of N names
```

## How it is harmonized

- Source: `MS_CXR_Local_Alignment_v1.1.0.csv`. One row is one phrase and its box. The harmonizer makes one
  row for each image (`dicom_id`), with a list of boxes (D-03).
- IDs: the `path`
  `files/p10/p10233088/s54276838/675d792f-a3521e48-5eec8573-1e81d644-e60c34f8.jpg` gives `patient_id`
  `10233088`, `study_id` `54276838`, and `image_path`
  `p10/p10233088/s54276838/675d792f-a3521e48-5eec8573-1e81d644-e60c34f8.jpg`. The IDs are the MIMIC digits
  (D-01). `image_path` is relative to `base_image_dir` (D-04).
- DICOM images: `base_image_dir` can also be the `files/` folder of MIMIC-CXR (DICOM). The harmonizer looks
  for the first image as `.jpg`, then as `.dcm`. If only `.dcm` exists, every `image_path` ends in `.dcm`.
- Labels (8 columns, D-05): a label is 1 when the image has a box of this finding, and 0 when it has none.
  So 0 means "no box of this finding", not "no finding".
- View, age and sex: the source has none of them, so the harmonized table has no `view_position`,
  `age_years` or `sex` (D-01).
- Extra columns: `split`, `label_text` (one phrase for each box), `image_width` and `image_height`.
- Dropped rows: a split class keeps the rows of its split. The harmonizer drops no other row.

## Labels

The 8 label columns, in the order of `cls`. The box labels use the same names.

| Column | Meaning |
|---|---|
| `atelectasis` | Part of a lung is collapsed |
| `cardiomegaly` | The heart is large |
| `consolidation` | Airspace consolidation |
| `edema` | Pulmonary edema |
| `lung_opacity` | A lung opacity |
| `pleural_effusion` | Pleural effusion |
| `pneumonia` | Pneumonia |
| `pneumothorax` | Pneumothorax |

## Boxes

<!-- claim: id=ms_cxr.boxes.train tier=ci cmd="python scripts/evalkit/dataset_stat.py ms_cxr ms_cxr_train.boxes" expect="1,020" -->
- Source: the CSV columns `x`, `y`, `w` and `h`, in the pixels of the image. The columns `image_width` and
  `image_height` give the image size, so the harmonizer does not read the image files for it.
- The harmonizer divides `y` and `h` by the height, and `x` and `w` by the width.
- The box is `[y_min, y_max, x_min, x_max]`, in fractions of the image size (D-07). Example: the image of the
  first CSV row is 2021 x 1767 (H x W). Its box `x` 196, `y` 1136, `w` 532, `h` 315 gives
  `[0.5621, 0.7180, 0.1109, 0.4120]`.
- `bbox_labels`: the snake_case name of `category_name`. For example, `Pleural Effusion` gives
  `pleural_effusion` (D-07).
- Readers (D-36): each box has its own phrase, so the harmonizer does not group boxes. The phrases are in
  `label_text`, in the order of the boxes. They are in the harmonized table only, not in the sample.
- Box counts: train has 1,020 boxes, val 212 and test 216. Every image has at least one box.
- The sample: `bbox` is a float32 tensor `(N, 4)`, in fractions of the returned image (D-28). `bbox_labels`
  is a list of N names. Example: `(1, 4)` and `['pneumonia']`.
- Dropped boxes: the harmonizer clips each box to the image. It drops a box with no area after the clip, and
  warns (D-07). The source files have no such box.

## Splits

<!-- claim: id=ms_cxr.splits.all tier=ci cmd="python scripts/evalkit/dataset_stat.py ms_cxr ms_cxr_train.images+ms_cxr_val.images+ms_cxr_test.images" expect="1,047" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `MSCXRTrainDataset` | `ms_cxr_train` | 737 | The rows with `split` `train` |
| `MSCXRValDataset` | `ms_cxr_val` | 155 | The rows with `split` `val` |
| `MSCXRTestDataset` | `ms_cxr_test` | 155 | The rows with `split` `test`. Use it as the test split |
| `MSCXRDataset` | `ms_cxr` | 1,047 | All images. It mixes the 3 splits, so do not use it for a benchmark (D-09) |

No patient is in 2 splits.

## Arguments

The arguments are the same for `MSCXRTrainDataset`, `MSCXRValDataset`, `MSCXRTestDataset` and
`MSCXRDataset`.

<!-- args: radharmony.dataset.MSCXRTrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `files/` folder of MIMIC-CXR-JPG, or of MIMIC-CXR (DICOM). Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `MS_CXR_Local_Alignment_v1.1.0.csv`. Without it, the class looks in `base_image_dir` and the folder above it. The CSV is usually in its own folder, so pass it |

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
from radharmony.harmonizer import MSCXRTrainHarmonizer

h = MSCXRTrainHarmonizer(
    csv_path="/data/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv",
    mimic_base_dir="/data/mimic-cxr-jpg/2.0.0/files/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'bbox', 'bbox_labels', 'atelectasis', ..., 'pneumothorax',
#  'split', 'label_text', 'image_width', 'image_height']
```

`MSCXRValHarmonizer`, `MSCXRTestHarmonizer` and `MSCXRHarmonizer` (all splits) take the same arguments.

<!-- args: radharmony.harmonizer.MSCXRTrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `MS_CXR_Local_Alignment_v1.1.0.csv` |
| `mimic_base_dir` | `str` | No | `None` | The `files/` folder of MIMIC-CXR-JPG or MIMIC-CXR (DICOM). The harmonizer checks one file in it to choose `.jpg` or `.dcm`. Without it, `image_path` ends in `.jpg` |

## Known data problems

- No IDs: version 1.1.0 of the CSV has no `subject_id` or `study_id` column. The IDs come from `path`.
