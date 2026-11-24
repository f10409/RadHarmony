# CheXphoto

<!-- claim: id=chexphoto.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chexphoto chexphoto_train.images" expect="20,034" -->
<!-- claim: id=chexphoto.train.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py chexphoto chexphoto_train.patients" expect="2,279" -->
<!-- claim: id=chexphoto.valid.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chexphoto chexphoto_valid.images" expect="702" -->
<!-- claim: id=chexphoto.valid.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py chexphoto chexphoto_valid.patients" expect="200" -->
| | |
|---|---|
| Registry keys | `chexphoto_train`, `chexphoto_valid` |
| Task and outputs | Classification, 14 labels (`cls`) |
| Images in the table | train 20,034 (2,279 patients), valid 702 (200 patients) |
| Modality | Chest X-ray, 2-D, JPEG |
| License | Stanford research use agreement |
| Download | [Get the data: CheXphoto](../../../how-to/get-the-data.md#other-datasets) |

<!-- claim: id=chexphoto.train.csv_rows tier=nas cmd="python scripts/evalkit/dataset_stat.py --source chexphoto_train base_image_dir/train.csv rows" expect="32,521" -->
<!-- claim: id=chexphoto.train.images.again tier=ci cmd="python scripts/evalkit/dataset_stat.py chexphoto chexphoto_train.images" expect="20,034" -->
CheXphoto ([Phillips et al., ML4H 2020](https://arxiv.org/abs/2007.06199)) tests a model on photos of
chest X-rays. It takes CheXpert X-rays and gives phone photos of a screen (`natural`) and synthetic copies
(`synthetic`). The 14 labels are the CheXpert labels, with no change. So you can train on
[CheXpert](chexpert.md) and test here. `train.csv` has 32,521 images. The default `drop_uncertain=True`
drops each row with a `-1` label, so 20,034 train images stay.

The authors keep the test split for their CodaLab competition, so there is no test class. Redivis gives
`train` and `valid` as 2 file index tables. Download both.

## Quick use

`base_image_dir` is the folder with `natural/` and `synthetic/` in it. The class finds `train.csv` or
`valid.csv` in this folder.

```
CheXphoto/
  train.csv
  valid.csv
  natural/iphone/patient10657/study1/view3_lateral.jpg
  natural/oneplus/patient64590/study1/view2_lateral.jpg
  synthetic/digital/patient00623/study1/view2_lateral.jpg
```

```python
import torch
from radharmony.dataset import CheXphotoTrainDataset, CheXphotoValidDataset

train = CheXphotoTrainDataset(
    base_image_dir="/data/CheXphoto/",
    output_cls=True,
    dtype=torch.float32,
)
train_ds, val_ds = train.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(14,)

test_ds = CheXphotoValidDataset(
    base_image_dir="/data/CheXphoto/",
    output_cls=True,
    dtype=torch.float32,
).get_datasets()
```

## How it is harmonized

- Source: `train.csv` or `valid.csv`. One row is one image: one photo or one synthetic copy of an X-ray.
  The steps are the [CheXpert](chexpert.md) steps, after the harmonizer renames the columns to the
  CheXpert names (for example `AP_PA` to `AP/PA`). Only the IDs differ.
- IDs: the `Path` `CheXphoto-v1.0/train/synthetic/digital/patient00623/study1/view2_lateral.jpg` gives
  `patient_id` `patient00623`, `study_id` `patient00623_study1`, `subset` `synthetic` and
  `capture_method` `digital`. `image_path` is `synthetic/digital/patient00623/study1/view2_lateral.jpg`,
  relative to `base_image_dir` (D-04).
- Labels (D-05): `1` gives 1 and `0` gives 0. A blank gives 0 (`not_mentioned="negative"`) or NaN
  (`not_mentioned="nan"`). A row with every label blank gets NaN in every label.
- View (D-01): `AP` or `PA` from the column `AP_PA`, and `LATERAL` for a lateral image. `AP_PA` is blank
  on every lateral image. A frontal image with `LL` in `AP_PA` gives NaN.
- Age and sex: `age_years` comes from `Age`, in years (D-06). `sex` is `M` or `F`, from `Sex` (D-26).
- Subset: with `subset="natural"` or `subset="synthetic"`, the harmonizer keeps only the rows of that
  subset.
- Dropped rows: with `drop_uncertain=True` (the default), the harmonizer drops a row with `-1` in any
  label. With `False`, `-1` stays in `cls`. `valid.csv` has no `-1`.

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

<!-- claim: id=chexphoto.splits.train tier=ci cmd="python scripts/evalkit/dataset_stat.py chexphoto chexphoto_train.images" expect="20,034" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `CheXphotoTrainDataset` | `chexphoto_train` | 20,034 | `train.csv`: `natural/iphone`, `natural/nokia`, `synthetic/digital` and `synthetic/photographic` |
| `CheXphotoValidDataset` | `chexphoto_valid` | 702 | `valid.csv`: `natural/oneplus`, `synthetic/digital` and `synthetic/photographic`. Use it as the test split |
| `CheXphotoDataset` | `chexphoto` | | The old key. It reads `train.csv` by default, so it gives the train images (D-09) |

Valid has 234 X-rays of 200 patients. Each X-ray is in valid 3 times: 1 phone photo and 2 synthetic
copies. To compare the capture types, make one class with `subset="natural"` (234 images) and one with
`subset="synthetic"` (468 images). The column `capture_method` gives the device or the transform.

No patient is in 2 splits. Every patient has images in `natural` and in `synthetic`.
`get_datasets(n_splits=5)` groups by patient, so all images of one patient go to the same split (D-10).

## Arguments

The arguments are the same for `CheXphotoTrainDataset` and `CheXphotoValidDataset`.

<!-- args: radharmony.dataset.CheXphotoTrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The folder with `natural/` and `synthetic/` in it. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `train.csv` or `valid.csv`. Without it, the class takes the CSV of its split in `base_image_dir` |
| `subset` | `str` | No | `"all"` | `"all"`, `"natural"` or `"synthetic"` |
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
from radharmony.harmonizer import CheXphotoTrainHarmonizer

h = CheXphotoTrainHarmonizer(
    csv_path="/data/CheXphoto/train.csv",
    base_image_dir="/data/CheXphoto/",
)
df = h.harmonize()  # harmonize() takes drop_uncertain and not_mentioned, as the dataset does
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'view_position', 'age_years', 'sex', 'atelectasis', ...,
#  'support_devices', 'subset', 'capture_method']
```

`CheXphotoValidHarmonizer` takes the same arguments, with `valid.csv`. The extra column `subset` is
`natural` or `synthetic`. `capture_method` is `iphone`, `nokia`, `oneplus`, `digital` or
`photographic`. A split harmonizer can load a file that `CheXphotoHarmonizer` saved. The saved
`csv_path` must be the CSV of the same split. If not, the harmonizer raises `ValueError` (D-09).

<!-- args: radharmony.harmonizer.CheXphotoTrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `train.csv` or `valid.csv` |
| `base_image_dir` | `str` | No | `None` | The folder with `natural/` and `synthetic/` in it |
| `subset` | `str` | No | `"all"` | `"all"`, `"natural"` or `"synthetic"` |

## Known data problems

- `Path` starts with `CheXphoto-v1.0/<split>/`, but the download has no such folders (D-04).
- In `Path`, the patient folder is not at the same position as in CheXpert, because of the 2 extra folders
  `<subset>/<capture method>`.
- The CSV column names have `_` where the CheXpert names have a space or `/`.
- The paper describes a `film` subset in valid, but version 1.0 has no film images.
