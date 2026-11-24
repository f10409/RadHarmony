# CheXpert Plus

<!-- claim: id=chexpert_plus.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chexpert_plus chexpert_plus_train.images" expect="143,343" -->
<!-- claim: id=chexpert_plus.train.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py chexpert_plus chexpert_plus_train.patients" expect="50,728" -->
<!-- claim: id=chexpert_plus.valid.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chexpert_plus chexpert_plus_valid.images" expect="165" -->
<!-- claim: id=chexpert_plus.valid.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py chexpert_plus chexpert_plus_valid.patients" expect="142" -->
| | |
|---|---|
| Registry keys | `chexpert_plus_train`, `chexpert_plus_valid` |
| Task and outputs | Classification, 14 labels (`cls`), and the report text (`report`) |
| Images in the table | train 143,343 (50,728 patients), valid 165 (142 patients) |
| Modality | Chest X-ray, 2-D, DICOM |
| License | Stanford AIMI research use agreement |
| Download | [Get the data: CheXpert Plus](../../../how-to/get-the-data.md#chexpert-plus) |

<!-- claim: id=chexpert_plus.csv_rows tier=nas cmd="python scripts/evalkit/dataset_stat.py --source chexpert_plus_train csv_path rows" expect="223,462" -->
<!-- claim: id=chexpert_plus.csv_patients tier=nas cmd="python scripts/evalkit/dataset_stat.py --source chexpert_plus_train csv_path unique:deid_patient_id" expect="64,725" -->
<!-- claim: id=chexpert_plus.train.images.again tier=ci cmd="python scripts/evalkit/dataset_stat.py chexpert_plus chexpert_plus_train.images" expect="143,343" -->
CheXpert Plus adds a radiology report and demographics to each CheXpert study.
`df_chexpert_plus_240401.csv` has 223,462 images of 64,725 patients, in the splits `train` and `valid`.
The 14 labels come from the impression of the report. The default `drop_uncertain=True` drops each row
with a `-1` label, so 143,343 train images stay.

## Quick use

`base_image_dir` is the `DICOM/Uncompressed/` folder. The class finds `df_chexpert_plus_240401.csv` and
`impression_fixed.json` in `base_image_dir`, in the folder above it, or in the folder 2 levels above it
(D-35).

```
chexpertplus/
  df_chexpert_plus_240401.csv
  impression_fixed.json
  DICOM/Uncompressed/train/patient42142/study5/view1_frontal.dcm
  DICOM/Uncompressed/valid/patient64680/study1/view1_frontal.dcm
```

```python
import torch
from radharmony.dataset import CheXpertPlusTrainDataset, CheXpertPlusValidDataset

train = CheXpertPlusTrainDataset(
    base_image_dir="/data/chexpertplus/DICOM/Uncompressed/",
    output_cls=True,
    output_report=True,
    dtype=torch.float32,
)
train.verify_images()  # drop the rows with no DICOM file
train_ds, val_ds = train.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(14,)
# sample["report"]: str

test = CheXpertPlusValidDataset(
    base_image_dir="/data/chexpertplus/DICOM/Uncompressed/",
    output_cls=True,
    output_report=True,
    dtype=torch.float32,
)
test.verify_images()
test_ds = test.get_datasets()
```

## How it is harmonized

- Source: `df_chexpert_plus_240401.csv`. One row is one image. A split class keeps only the rows with its
  split name (`train` or `valid`) in the column `split` (D-09).
- IDs: the `path_to_image` `train/patient42142/study5/view1_frontal.jpg` gives `patient_id`
  `patient42142` and `study_id` `patient42142_study5`. `image_path` is the `path_to_dcm`
  `train/patient42142/study5/view1_frontal.dcm`, relative to `base_image_dir` (D-04).
- Labels (D-48, D-05): the 14 labels come from `impression_fixed.json`, not from `report_fixed.json`. The
  harmonizer joins them on `path_to_image`. `1` gives 1 and `0` gives 0. A blank gives 0
  (`not_mentioned="negative"`) or NaN (`not_mentioned="nan"`). A row with every label blank gets NaN in
  every label.
- View (D-01): `AP` or `PA` from the column `ap_pa`, and `LATERAL` for a lateral image. A frontal image
  with `LL` or `RL` in `ap_pa` gives NaN.
- Age and sex: `age_years` comes from `age`, in years (D-06). `sex` is `M`, `F` or NaN, from `sex`
  (D-26). The value `Unknown` gives NaN.
- Dropped rows: with `drop_uncertain=True` (the default), the harmonizer drops a row with `-1` in any
  label. With `False`, `-1` stays in `cls`. The harmonizer keeps a row with no DICOM file and warns with
  the count (D-02). `verify_images()` drops these rows.

## Labels

The 14 label columns, in the order of `cls`:

| Column | Meaning |
|---|---|
| `atelectasis` | Part of a lung is collapsed |
| `cardiomegaly` | The heart is large |
| `consolidation` | Airspace consolidation |
| `edema` | Pulmonary edema |
| `enlarged_cardiomediastinum` | The mediastinum is wide |
| `fracture` | A rib or other bone fracture |
| `lung_lesion` | A lung lesion |
| `lung_opacity` | A lung opacity |
| `no_finding` | No finding |
| `pleural_effusion` | Pleural effusion |
| `pleural_other` | Another pleural finding |
| `pneumonia` | Pneumonia |
| `pneumothorax` | Pneumothorax |
| `support_devices` | A tube, line or other device |

## Reports

With `output_report=True`, `sample["report"]` is the full text of the CSV column `report`. The section
columns of the CSV (for example `section_findings` and `section_impression`) are not in the harmonized
table. The report generation evaluator finds the sections from the headers in the text (D-38).

## Splits

<!-- claim: id=chexpert_plus.splits.train tier=ci cmd="python scripts/evalkit/dataset_stat.py chexpert_plus chexpert_plus_train.images" expect="143,343" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `CheXpertPlusTrainDataset` | `chexpert_plus_train` | 143,343 | The rows with `split` = `train` |
| `CheXpertPlusValidDataset` | `chexpert_plus_valid` | 165 | The rows with `split` = `valid`. Use it as the test split |
| `CheXpertPlusDataset` | `chexpert_plus` | | All images. It mixes the 2 splits, so do not use it for a benchmark (D-09) |

No patient is in 2 splits. `get_datasets(n_splits=5)` makes a train and val split from the train class.
It groups by patient (D-10).

## Arguments

The arguments are the same for `CheXpertPlusTrainDataset`, `CheXpertPlusValidDataset` and
`CheXpertPlusDataset`.

<!-- args: radharmony.dataset.CheXpertPlusTrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `DICOM/Uncompressed/` folder, with `train/` and `valid/` in it. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `df_chexpert_plus_240401.csv`. Without it, the class looks in `base_image_dir`, the folder above it and the folder 2 levels above it |
| `label_json_path` | `str` | No | `None` | `impression_fixed.json` (D-48). Without it, the class looks in the same places. `output_cls=True` needs it |
| `drop_uncertain` | `bool` | No | `True` | Drop the rows with an uncertain label (`-1`). With `False`, `-1` stays in `cls` |
| `not_mentioned` | `str` | No | `"negative"` | A blank label. `"negative"` gives 0. `"nan"` keeps NaN, so the probes skip it (D-05) |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_report` | `bool` | No | `False` | Add the `report` text to the sample. The class drops the rows with no report, and warns (D-33) |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The labels stay float32 |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import CheXpertPlusTrainHarmonizer

h = CheXpertPlusTrainHarmonizer(
    csv_path="/data/chexpertplus/df_chexpert_plus_240401.csv",
    label_json_path="/data/chexpertplus/impression_fixed.json",
    base_image_dir="/data/chexpertplus/DICOM/Uncompressed/",
)
df = h.harmonize()  # harmonize() takes drop_uncertain and not_mentioned, as the dataset does
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'view_position', 'report', 'age_years', 'sex',
#  'atelectasis', ..., 'support_devices', 'age', 'race', 'ethnicity', 'split']
```

`CheXpertPlusValidHarmonizer` takes the same arguments. The extra columns `age`, `race` and `ethnicity`
come from the CSV as they are. `split` is `train` or `valid`.

<!-- args: radharmony.harmonizer.CheXpertPlusTrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `df_chexpert_plus_240401.csv` |
| `label_json_path` | `str` | No | `None` | `impression_fixed.json`. With `None`, the harmonized table has no label columns |
| `base_image_dir` | `str` | No | `None` | The `DICOM/Uncompressed/` folder. The harmonizer uses it to count the missing patient folders |

## Known data problems

- Labels: the labels in `report_fixed.json` do not follow the image. The images of one study share one
  report, but they often get different labels (D-48).
- 12 train images have no label at all in `impression_fixed.json` (D-05).
- The NAS copy of the DICOM release misses the folders of some patients, among them 4 valid patients (5
  rows). The counts above include these rows (D-02).
