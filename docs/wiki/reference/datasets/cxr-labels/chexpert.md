# CheXpert

<!-- claim: id=chexpert.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chexpert chexpert_train.images" expect="138,358" -->
<!-- claim: id=chexpert.train.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py chexpert chexpert_train.patients" expect="49,408" -->
<!-- claim: id=chexpert.valid.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chexpert chexpert_valid.images" expect="234" -->
<!-- claim: id=chexpert.valid.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py chexpert chexpert_valid.patients" expect="200" -->
| | |
|---|---|
| Registry keys | `chexpert_train`, `chexpert_valid` |
| Task and outputs | Classification, 14 labels (`cls`) |
| Images in the table | train 138,358 (49,408 patients), valid 234 (200 patients) |
| Modality | Chest X-ray, 2-D, JPEG |
| License | Stanford CheXpert research use agreement |
| Download | [Get the data: CheXpert](../../../how-to/get-the-data.md#chexpert) |

<!-- claim: id=chexpert.train.csv_rows tier=nas cmd="python scripts/evalkit/dataset_stat.py --source chexpert_train csv_path rows" expect="223,414" -->
<!-- claim: id=chexpert.train.images.again tier=ci cmd="python scripts/evalkit/dataset_stat.py chexpert chexpert_train.images" expect="138,358" -->
CheXpert is a chest X-ray dataset from Stanford. For `train.csv`, the CheXpert labeler (a rule-based tool)
read each report and gave 14 labels. For `valid.csv`, 3 radiologists gave the labels. `train.csv` has
223,414 images. The default `drop_uncertain=True` drops each row with a `-1` label, so 138,358 train images
stay. The official test set is in [CheXlocalize](../cxr-boxes-masks/chexlocalize.md).

## Quick use

`base_image_dir` is the `train/` or `valid/` folder. The class finds `train.csv` or `valid.csv` in
the folder above it.

```
CheXpert-v1.0/
  train.csv
  valid.csv
  train/patient00001/study1/view1_frontal.jpg
  valid/patient64541/study1/view1_frontal.jpg
```

```python
import torch
from radharmony.dataset import CheXpertTrainDataset, CheXpertValidDataset

train = CheXpertTrainDataset(
    base_image_dir="/data/CheXpert-v1.0/train/",
    output_cls=True,
    dtype=torch.float32,
)
train_ds, val_ds = train.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(14,)

test_ds = CheXpertValidDataset(
    base_image_dir="/data/CheXpert-v1.0/valid/",
    output_cls=True,
    dtype=torch.float32,
).get_datasets()
```

## How it is harmonized

- Source: `train.csv` or `valid.csv`. One row is one image.
- IDs: the `Path` `CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg` gives `patient_id`
  `patient00001`, `study_id` `patient00001_study1`, and `image_path`
  `patient00001/study1/view1_frontal.jpg`. `image_path` is relative to `base_image_dir` (D-04).
- Labels (D-05): `1` gives 1 and `0` gives 0. A blank gives 0 (`not_mentioned="negative"`) or NaN
  (`not_mentioned="nan"`). A row with every label blank gets NaN in every label. `valid.csv` has only 0
  and 1.
- View (D-01): `AP` or `PA` from the column `AP/PA`, and `LATERAL` for a lateral image. A frontal image
  with `LL` or `RL` in `AP/PA` gives NaN.
- Age and sex: `age_years` comes from `Age`, in years (D-06). `sex` is `M`, `F` or NaN, from `Sex` (D-26).
- Race and ethnicity: only with the harmonizer argument `demographics_path`. The harmonizer joins the
  workbook on `patient_id`.
- Dropped rows: with `drop_uncertain=True` (the default), the harmonizer drops a row with `-1` in any
  label. With `False`, `-1` stays in `cls`.

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

## Splits

<!-- claim: id=chexpert.splits.train tier=ci cmd="python scripts/evalkit/dataset_stat.py chexpert chexpert_train.images" expect="138,358" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `CheXpertTrainDataset` | `chexpert_train` | 138,358 | `train.csv` |
| `CheXpertValidDataset` | `chexpert_valid` | 234 | `valid.csv`. Use it as the test split |

`get_datasets(n_splits=5)` makes a train and val split from the train class. It groups by patient (D-10).

## Arguments

The arguments are the same for `CheXpertTrainDataset` and `CheXpertValidDataset`.

<!-- args: radharmony.dataset.CheXpertTrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `train/` or `valid/` folder. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `train.csv` or `valid.csv`. Without it, the class looks in the folder above `base_image_dir` |
| `drop_uncertain` | `bool` | No | `True` | Drop the rows with an uncertain label (`-1`). With `False`, `-1` stays in `cls` |
| `not_mentioned` | `str` | No | `"negative"` | A blank label. `"negative"` gives 0. `"nan"` keeps NaN, so the probes skip it (D-05) |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_report` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The labels stay float32 |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import CheXpertTrainHarmonizer

h = CheXpertTrainHarmonizer(
    csv_path="/data/CheXpert-v1.0/train.csv",
    base_image_dir="/data/CheXpert-v1.0/train/",
)
df = h.harmonize()  # harmonize() takes drop_uncertain and not_mentioned, as the dataset does
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'view_position', 'age_years', 'sex', 'atelectasis', ...]
```

`CheXpertValidHarmonizer` takes the same arguments, with `valid.csv` and the `valid/` folder.

<!-- args: radharmony.harmonizer.CheXpertTrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `train.csv` or `valid.csv` |
| `base_image_dir` | `str` | No | `None` | The `train/` or `valid/` folder |
| `demographics_path` | `str` | No | `None` | `Chexpert race labels.xlsx`. It adds the columns `race` and `ethnicity` |

## Known data problems

- Sex: for 3 patients, `Sex` in the CSV and `GENDER` in the demographics workbook differ (patient51668,
  patient54170 and patient54565). The harmonized table keeps the CSV value (D-26).
- Age: the workbook column `AGE_AT_CXR` holds one age for each patient, not for each study (D-06).
- Age 0: 3 rows of `train.csv` have `Age` 0. 1 of them stays after the default drop (D-06).
