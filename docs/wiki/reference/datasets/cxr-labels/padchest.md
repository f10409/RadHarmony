# PadChest

<!-- claim: id=padchest.images tier=ci cmd="python scripts/evalkit/dataset_stat.py padchest padchest.images" expect="157,607" -->
<!-- claim: id=padchest.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py padchest padchest.patients" expect="66,587" -->
<!-- claim: id=padchest.studies tier=ci cmd="python scripts/evalkit/dataset_stat.py padchest padchest.studies" expect="107,439" -->
| | |
|---|---|
| Registry keys | `padchest` |
| Task and outputs | Classification, 193 labels (`cls`). The report text in Spanish (`report`) |
| Images in the table | 157,607 (66,587 patients, 107,439 studies). By default, the harmonizer drops the rows with `exclude` or `suboptimal_study` |
| Modality | Chest X-ray, 2-D, PNG |
| License | BIMCV PadChest research use agreement |
| Download | [Get the data: PadChest](../../../how-to/get-the-data.md#padchest) |

<!-- claim: id=padchest.source.rows tier=nas cmd="python scripts/evalkit/dataset_stat.py --source padchest csv_path rows" expect="160,861" -->
<!-- claim: id=padchest.source.patients tier=nas cmd="python scripts/evalkit/dataset_stat.py --source padchest csv_path unique:PatientID" expect="67,625" -->
PadChest is a chest X-ray dataset from Hospital San Juan de Alicante, Spain. BIMCV released it. Its 193
labels come from a text-mining tool on the Spanish reports, plus a manual review. The source CSV has
160,861 images of 67,625 patients.

The source CSV also holds the report of each image. The reports are in Spanish. The authors cut each word
to its stem, for example `hallazg` for `hallazgo`.

Download each numbered zip by itself, because the zip of the whole folder does not resume. Unzip
`N.zip` into `images/N/`.

## Quick use

`base_image_dir` is the `images/` folder. The class finds the label CSV in this folder or in the folder
above it. The PNG files are in 52 folders: `0/` to `50/`, and `54/`.

```
PadChest/
  PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz
  images/
    0/20536686640136348236148679891455886468_k6ga29.png
    1/
    ...
    50/
    54/
```

```python
import torch
from radharmony.dataset import PadChestDataset

ds = PadChestDataset(
    base_image_dir="/data/PadChest/images/",
    output_cls=True,
    output_report=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(193,)
# sample["report"]: ' sin hallazg patolog edad pacient .'
```

## How it is harmonized

- Source: `PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz` (or `.csv`). One row is one image
  (D-03).
- IDs: `patient_id` is `PatientID`, for example `839860488694292331637988235681460987`. `study_id` is
  `StudyID`, for example `20536686640136348236148679891455886468` (D-01). `image_path` is
  `<ImageDir>/<ImageID>`, for example `0/20536686640136348236148679891455886468_k6ga29.png`. It is
  relative to `base_image_dir` (D-04).
- Labels (193 columns, D-05): `Labels` holds a Python list as text, for example
  `['cardiomegaly', 'pleural effusion']`. A label in the list gives 1, and all other labels give 0. The
  harmonizer removes a space at the start of a label, so `' pneumonia'` and `'pneumonia'` give one column.
  A row with the value `['']` gets NaN in all 193 labels, not 0. The probes skip NaN.
- Uncertain: PadChest has no `-1`. Two labels mark bad images: `exclude` and `suboptimal_study`. With
  `drop_uncertain=True` (the default), the harmonizer drops the rows with either label.
- View (D-01): from `Projection`. `L` gives `LATERAL`, and `AP_horizontal` gives `AP_HORIZONTAL`. `EXCLUDE`
  and `UNK` give NaN. `COSTAL` stays.
- Age and sex: the source has no age. `age_years` comes from the year of `StudyDate_DICOM` minus
  `PatientBirth`, in years (D-06). `sex` is `M`, `F` or NaN, from `PatientSex_DICOM`. The value `O` gives
  NaN (D-26).
- Report: `report` is the `Report` column. With `output_report=True`, the dataset drops the rows with no
  report, and warns (D-33).
- Dropped rows: the harmonizer always drops the 103 rows with a blank `Labels` value. It also drops a row
  whose `Labels` value is not a valid Python list.

## Labels

The harmonized table has 193 label columns, in alphabetical order. The full list is
`PadChestDataset.LABEL_COLS`. The table below shows the most common labels, with the number of images
with 1, and the 2 image-quality labels.

<!-- claim: id=padchest.positives.normal tier=ci cmd="python scripts/evalkit/dataset_stat.py padchest padchest.positives.normal" expect="50,607" -->
<!-- claim: id=padchest.positives.copd_signs tier=ci cmd="python scripts/evalkit/dataset_stat.py padchest padchest.positives.copd_signs" expect="23,171" -->
<!-- claim: id=padchest.positives.cardiomegaly tier=ci cmd="python scripts/evalkit/dataset_stat.py padchest padchest.positives.cardiomegaly" expect="14,764" -->
<!-- claim: id=padchest.positives.unchanged tier=ci cmd="python scripts/evalkit/dataset_stat.py padchest padchest.positives.unchanged" expect="14,163" -->
<!-- claim: id=padchest.positives.aortic_elongation tier=ci cmd="python scripts/evalkit/dataset_stat.py padchest padchest.positives.aortic_elongation" expect="10,660" -->
<!-- claim: id=padchest.positives.pleural_effusion tier=ci cmd="python scripts/evalkit/dataset_stat.py padchest padchest.positives.pleural_effusion" expect="9,665" -->
<!-- claim: id=padchest.positives.scoliosis tier=ci cmd="python scripts/evalkit/dataset_stat.py padchest padchest.positives.scoliosis" expect="8,245" -->
<!-- claim: id=padchest.positives.pneumonia tier=ci cmd="python scripts/evalkit/dataset_stat.py padchest padchest.positives.pneumonia" expect="7,837" -->
<!-- claim: id=padchest.positives.interstitial_pattern tier=ci cmd="python scripts/evalkit/dataset_stat.py padchest padchest.positives.interstitial_pattern" expect="7,654" -->
<!-- claim: id=padchest.positives.chronic_changes tier=ci cmd="python scripts/evalkit/dataset_stat.py padchest padchest.positives.chronic_changes" expect="7,269" -->
<!-- claim: id=padchest.positives.infiltrates tier=ci cmd="python scripts/evalkit/dataset_stat.py padchest padchest.positives.infiltrates" expect="6,977" -->
<!-- claim: id=padchest.positives.nsg_tube tier=ci cmd="python scripts/evalkit/dataset_stat.py padchest padchest.positives.nsg_tube" expect="5,196" -->
<!-- claim: id=padchest.positives.nodule tier=ci cmd="python scripts/evalkit/dataset_stat.py padchest padchest.positives.nodule" expect="3,738" -->
<!-- claim: id=padchest.positives.pacemaker tier=ci cmd="python scripts/evalkit/dataset_stat.py padchest padchest.positives.pacemaker" expect="3,457" -->
<!-- claim: id=padchest.positives.endotracheal_tube tier=ci cmd="python scripts/evalkit/dataset_stat.py padchest padchest.positives.endotracheal_tube" expect="2,344" -->
| Column | Meaning | Images with 1 |
|---|---|---|
| `normal` | No finding | 50,607 |
| `copd_signs` | Signs of COPD | 23,171 |
| `cardiomegaly` | The heart is large | 14,764 |
| `unchanged` | No change from the prior study | 14,163 |
| `aortic_elongation` | The aorta is long | 10,660 |
| `pleural_effusion` | Pleural effusion | 9,665 |
| `scoliosis` | Scoliosis | 8,245 |
| `pneumonia` | Pneumonia | 7,837 |
| `interstitial_pattern` | Interstitial pattern | 7,654 |
| `chronic_changes` | Chronic changes | 7,269 |
| `infiltrates` | Infiltrates | 6,977 |
| `nsg_tube` | A nasogastric tube | 5,196 |
| `nodule` | A lung nodule | 3,738 |
| `pacemaker` | A pacemaker | 3,457 |
| `endotracheal_tube` | An endotracheal tube | 2,344 |
| `exclude` | The image is bad. With `drop_uncertain=True`, it is 0 or NaN in every row | 0 |
| `suboptimal_study` | The image quality is too low. With `drop_uncertain=True`, it is 0 or NaN in every row | 0 |

## Splits

<!-- claim: id=padchest.splits.images tier=ci cmd="python scripts/evalkit/dataset_stat.py padchest padchest.images" expect="157,607" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `PadChestDataset` | `padchest` | 157,607 | All images of the source CSV |

PadChest has no official split. `get_datasets(n_splits=5)` makes a train and val split. It groups by
patient (D-10).

## Arguments

<!-- args: radharmony.dataset.PadChestDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `images/` folder with `0/` to `54/`. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | The label CSV (`.csv.gz` or `.csv`). Without it, the class looks in `base_image_dir` and the folder above it |
| `drop_uncertain` | `bool` | No | `True` | Drop the rows with the label `exclude` or `suboptimal_study`. With `False`, they stay |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_report` | `bool` | No | `False` | Add the `report` text to the sample. The class drops the rows with no report, and warns (D-33) |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11). `Localizations` names regions, not boxes |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The labels stay float32 |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import PadChestHarmonizer

h = PadChestHarmonizer(
    csv_path="/data/PadChest/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz",
    base_image_dir="/data/PadChest/images/",
)
df = h.harmonize()  # harmonize() takes drop_uncertain, as the dataset does
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'view_position', 'report', 'age_years', 'sex',
#  'abnormal_foreign_body', ..., 'volume_loss']
```

`harmonize(drop_uncertain=False)` keeps the rows with `exclude` or `suboptimal_study`.

<!-- args: radharmony.harmonizer.PadChestHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | The label CSV (`.csv.gz` or `.csv`) |
| `base_image_dir` | `str` | No | `None` | The `images/` folder |

## Known data problems

- Empty label lists: 777 rows have the `Labels` value `['']`. Their reports name findings such as
  scoliosis, so an empty list does not mean normal (D-05).
- Shared study ID: 1 `StudyID` belongs to 2 patients. The harmonizer warns. Group by `patient_id` and
  `study_id` together (D-01).
- Age: the source has only the birth year, not the birth date. So an age can be 1 year too high. 6 rows
  have no birth year (D-06).
- No report: 17 images of the default harmonized table have a blank `Report` (D-33).
- Image folders: the folders `51/`, `52/` and `53/` do not exist, and folder `50/` holds 1 image. This is
  the layout of the source, not a download error.
