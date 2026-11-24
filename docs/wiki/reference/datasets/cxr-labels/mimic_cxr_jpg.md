# MIMIC-CXR-JPG

<!-- claim: id=mimic_cxr_jpg.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr_jpg mimic_cxr_jpg_train.images" expect="291,240" -->
<!-- claim: id=mimic_cxr_jpg.train.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr_jpg mimic_cxr_jpg_train.patients" expect="60,024" -->
<!-- claim: id=mimic_cxr_jpg.validate.images tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr_jpg mimic_cxr_jpg_validate.images" expect="2,320" -->
<!-- claim: id=mimic_cxr_jpg.validate.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr_jpg mimic_cxr_jpg_validate.patients" expect="459" -->
<!-- claim: id=mimic_cxr_jpg.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr_jpg mimic_cxr_jpg_test.images" expect="1,029" -->
<!-- claim: id=mimic_cxr_jpg.test.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr_jpg mimic_cxr_jpg_test.patients" expect="296" -->
<!-- claim: id=mimic_cxr_jpg.test_chexpert.images tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr_jpg mimic_cxr_jpg_test_chexpert.images" expect="3,556" -->
<!-- claim: id=mimic_cxr_jpg.test_chexpert.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr_jpg mimic_cxr_jpg_test_chexpert.patients" expect="287" -->
| | |
|---|---|
| Registry keys | `mimic_cxr_jpg_train`, `mimic_cxr_jpg_validate`, `mimic_cxr_jpg_test`, `mimic_cxr_jpg_test_chexpert` |
| Task and outputs | Classification, 14 labels (`cls`) |
| Images in the table | train 291,240 (60,024 patients), validate 2,320 (459 patients), test 1,029 (296 patients), test_chexpert 3,556 (287 patients) |
| Modality | Chest X-ray, 2-D, JPEG |
| License | PhysioNet credentialed access (CITI training and a signed DUA) |
| Download | [Get the data: MIMIC-CXR-JPG](../../../how-to/get-the-data.md#mimic-cxr-jpg) |

<!-- claim: id=mimic_cxr_jpg.source.images tier=nas cmd="python scripts/evalkit/dataset_stat.py --source mimic_cxr_jpg_train csv_path rows" expect="377,110" -->
<!-- claim: id=mimic_cxr_jpg.source.studies tier=nas cmd="python scripts/evalkit/dataset_stat.py --source mimic_cxr_jpg_train csv_path unique:study_id" expect="227,835" -->
<!-- claim: id=mimic_cxr_jpg.source.patients tier=nas cmd="python scripts/evalkit/dataset_stat.py --source mimic_cxr_jpg_train csv_path unique:subject_id" expect="65,379" -->
MIMIC-CXR-JPG is the JPEG copy of [MIMIC-CXR](mimic_cxr.md). It is faster to load than the DICOM files.
`mimic-cxr-2.0.0-metadata.csv` has 377,110 images in 227,835 studies of 65,379 patients. The source has no
report text, so use MIMIC-CXR for reports.

The labels of `mimic_cxr_jpg_train`, `mimic_cxr_jpg_validate`
and `mimic_cxr_jpg_test_chexpert` come from the CheXpert labeler. The labels of `mimic_cxr_jpg_test` come
from radiologists. The counts above use the default `drop_uncertain` of each class.

## Quick use

`base_image_dir` is the `files/` folder. The class finds `mimic-cxr-2.0.0-metadata.csv`,
`mimic-cxr-2.0.0-chexpert.csv` and `mimic-cxr-2.1.0-test-set-labeled.csv` in `base_image_dir` or in the
folder above it. A split class finds `mimic-cxr-2.0.0-split.csv` next to the metadata file.

```
mimic-cxr-jpg/2.0.0/
  mimic-cxr-2.0.0-metadata.csv
  mimic-cxr-2.0.0-chexpert.csv
  mimic-cxr-2.0.0-split.csv
  mimic-cxr-2.1.0-test-set-labeled.csv
  files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
```

```python
import torch
from radharmony.dataset import (
    MIMICCXRJPGTestDataset,
    MIMICCXRJPGTrainDataset,
    MIMICCXRJPGValidateDataset,
)

kw = dict(base_image_dir="/data/mimic-cxr-jpg/2.0.0/files/", output_cls=True, dtype=torch.float32)
train_ds = MIMICCXRJPGTrainDataset(**kw).get_datasets()
val_ds = MIMICCXRJPGValidateDataset(**kw).get_datasets()
test_ds = MIMICCXRJPGTestDataset(**kw).get_datasets()  # the labeled test set, -1 stays
sample = test_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(14,)
```

## How it is harmonized

- Source: `mimic-cxr-2.0.0-metadata.csv`. One row is one JPEG image. A split class keeps the images of
  its split in `mimic-cxr-2.0.0-split.csv`, by `dicom_id` (D-09).
- IDs: `subject_id` 10000032 gives `patient_id` `10000032`, and `study_id` 50414267 gives `50414267`.
  With the `dicom_id`, they give `image_path`
  `p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg`, relative to
  `base_image_dir` (D-04).
- Labels (D-05): `mimic-cxr-2.0.0-chexpert.csv` has one row for each study. The harmonizer joins it on
  `patient_id` and `study_id`, so all images of one study get the same labels. `1` gives 1 and `0` gives
  0. A blank gives 0 (`not_mentioned="negative"`) or NaN (`not_mentioned="nan"`). A study with every label
  blank gets NaN in every label.
- Labeled test set: `mimic-cxr-2.1.0-test-set-labeled.csv` has no `subject_id`, and it names Lung Opacity
  `Airspace Opacity`. So the harmonizer joins it on `study_id` only, and renames the column to
  `lung_opacity`.
- View (D-01): from the column `ViewPosition`. A shared step maps the values to `AP`, `PA` and `LATERAL`.
  A blank gives NaN. A rare value keeps its name in upper case, for example `AP AXIAL` or `LAO`.
- Age and sex: the source has no age and no sex.
- Dropped rows: with `drop_uncertain=True`, the harmonizer drops a study with `-1` in any label, with all
  its images. `True` is the default for all classes, except `MIMICCXRJPGTestDataset`. With `False`, `-1`
  stays in `cls`. The harmonizer also drops an image whose study has no row in the label file, and warns
  with the count (D-25). `mimic_cxr_jpg_test` gives no warning, because it uses this join to keep the
  test studies only.

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

<!-- claim: id=mimic_cxr_jpg.splits.train tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr_jpg mimic_cxr_jpg_train.images" expect="291,240" -->
<!-- claim: id=mimic_cxr_jpg.splits.validate tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr_jpg mimic_cxr_jpg_validate.images" expect="2,320" -->
<!-- claim: id=mimic_cxr_jpg.splits.test tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr_jpg mimic_cxr_jpg_test.images" expect="1,029" -->
<!-- claim: id=mimic_cxr_jpg.splits.test_chexpert tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_cxr_jpg mimic_cxr_jpg_test_chexpert.images" expect="3,556" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `MIMICCXRJPGTrainDataset` | `mimic_cxr_jpg_train` | 291,240 | The `train` images of `mimic-cxr-2.0.0-split.csv` |
| `MIMICCXRJPGValidateDataset` | `mimic_cxr_jpg_validate` | 2,320 | The `validate` images of the split file |
| `MIMICCXRJPGTestDataset` | `mimic_cxr_jpg_test` | 1,029 | The labeled test set, `mimic-cxr-2.1.0-test-set-labeled.csv` (685 studies). Use it as the test split |
| `MIMICCXRJPGTestCheXpertDataset` | `mimic_cxr_jpg_test_chexpert` | 3,556 | The `test` images of the split file, with the labels from the CheXpert labeler |
| `MIMICCXRJPGDataset` | `mimic_cxr_jpg` | | All images. It mixes the 3 splits, so do not use it for a benchmark (D-09) |

`mimic_cxr_jpg_test_chexpert` has the same images and labels as the DICOM key `mimic_cxr_test`. All its
287 patients are also in `mimic_cxr_jpg_test`. No train patient is in validate or in a test class.

The splits stay as the authors published them (D-34). 3 patients are in validate and also in the labeled
test set. Validate has 62 images (43 studies) of these patients. If you tune on validate and test on
`mimic_cxr_jpg_test`, remove them in your own code.

## Arguments

The arguments are the same for `MIMICCXRJPGTrainDataset`, `MIMICCXRJPGValidateDataset` and
`MIMICCXRJPGTestCheXpertDataset`. `MIMICCXRJPGTestDataset` has no `split_csv_path`, and its
`drop_uncertain` is `False` by default. Its `label_csv_path` is `mimic-cxr-2.1.0-test-set-labeled.csv`. A
missing file raises `FileNotFoundError`.

<!-- args: radharmony.dataset.MIMICCXRJPGTrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `files/` folder. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `mimic-cxr-2.0.0-metadata.csv`. Without it, the class looks in `base_image_dir` and the folder above it |
| `label_csv_path` | `str` | No | `None` | `mimic-cxr-2.0.0-chexpert.csv`. Without it, the class looks in the same places. With `output_cls=True`, a missing file raises `FileNotFoundError` |
| `split_csv_path` | `str` | No | `None` | `mimic-cxr-2.0.0-split.csv`. Without it, the class looks next to `csv_path` and `label_csv_path`. A missing file raises `FileNotFoundError` |
| `drop_uncertain` | `bool` | No | `True` | Drop the studies with an uncertain label (`-1`). With `False`, `-1` stays in `cls` |
| `not_mentioned` | `str` | No | `"negative"` | A blank label. `"negative"` gives 0. `"nan"` keeps NaN, so the probes skip it (D-05) |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_report` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11). Use [MIMIC-CXR](mimic_cxr.md) for reports |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The labels stay float32 |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import MIMICCXRJPGTrainHarmonizer

h = MIMICCXRJPGTrainHarmonizer(
    csv_path="/data/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv",
    label_csv_path="/data/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv",
    base_image_dir="/data/mimic-cxr-jpg/2.0.0/files/",
)
df = h.harmonize()  # harmonize() takes drop_uncertain and not_mentioned, as the dataset does
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'view_position', 'atelectasis', ..., 'support_devices']
```

`MIMICCXRJPGValidateHarmonizer` and `MIMICCXRJPGTestCheXpertHarmonizer` take the same arguments.
`MIMICCXRJPGTestHarmonizer` takes the same arguments, except `split_csv_path`. Its `harmonize()` keeps
`-1` by default.

<!-- args: radharmony.harmonizer.MIMICCXRJPGTrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `mimic-cxr-2.0.0-metadata.csv` |
| `label_csv_path` | `str` | No | `None` | `mimic-cxr-2.0.0-chexpert.csv`. With `None`, the harmonized table has no label columns |
| `base_image_dir` | `str` | No | `None` | The `files/` folder |
| `split_csv_path` | `str` | No | `None` | `mimic-cxr-2.0.0-split.csv`. Without it, the class looks next to `csv_path` and `label_csv_path` |

## Known data problems

- The labeled test file has 687 studies. 2 of them (54107786 and 58592606) have no JPEG in the metadata
  file.
- 676 studies of the labeled test file are in the official `test` split, and 9 studies (3 patients) are in
  `validate` (D-34).
- 15 train images in 8 studies have no row in `mimic-cxr-2.0.0-chexpert.csv` (D-25).
- Some images have a blank `ViewPosition` (D-01).
