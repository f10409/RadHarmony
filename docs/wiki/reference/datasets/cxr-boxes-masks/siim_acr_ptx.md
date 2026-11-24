# SIIM-ACR Pneumothorax

<!-- claim: id=siim_acr_ptx.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py siim_acr_ptx siim_acr_ptx_train.images" expect="10,675" -->
<!-- claim: id=siim_acr_ptx.train.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py siim_acr_ptx siim_acr_ptx_train.patients" expect="10,675" -->
<!-- claim: id=siim_acr_ptx.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py siim_acr_ptx siim_acr_ptx_test.images" expect="1,377" -->
<!-- claim: id=siim_acr_ptx.test.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py siim_acr_ptx siim_acr_ptx_test.patients" expect="1,377" -->
| | |
|---|---|
| Registry keys | `siim_acr_ptx_train`, `siim_acr_ptx_test` |
| Task and outputs | Classification, 1 label (`cls`). Pneumothorax masks (`mask`). The test split has images only |
| Images in the table | train 10,675 (10,675 patients), test 1,377 (1,377 patients) |
| Modality | Chest X-ray, 2-D, DICOM |
| License | The rules of the Kaggle competition. Accept them before the download |
| Download | [Get the data: SIIM-ACR Pneumothorax](../../../how-to/get-the-data.md#siim-acr-pneumothorax) |

<!-- claim: id=siim_acr_ptx.train.images.again tier=ci cmd="python scripts/evalkit/dataset_stat.py siim_acr_ptx siim_acr_ptx_train.images" expect="10,675" -->
<!-- claim: id=siim_acr_ptx.train.positives tier=ci cmd="python scripts/evalkit/dataset_stat.py siim_acr_ptx siim_acr_ptx_train.positives.pneumothorax" expect="2,379" -->
<!-- claim: id=siim_acr_ptx.train.csv_rows tier=nas cmd="python scripts/evalkit/dataset_stat.py --source siim_acr_ptx_train csv_path rows" expect="11,582" -->
<!-- claim: id=siim_acr_ptx.test.images.again tier=ci cmd="python scripts/evalkit/dataset_stat.py siim_acr_ptx siim_acr_ptx_test.images" expect="1,377" -->
SIIM-ACR Pneumothorax Segmentation is a 2019 Kaggle challenge. RadHarmony reads the stage 1 files. The
train split has 10,675 images, and 2,379 of them have a pneumothorax. `train-rle.csv` has 11,582 rows,
because it has one row for each pneumothorax region. Each row holds one region as a run-length encoded
(RLE) mask. The 1,377 test images have no labels and no masks.

## Quick use

`base_image_dir` is `dicom-images-train/` or `dicom-images-test/`. The class finds `train-rle.csv` in
`base_image_dir` or in the folder above it.

```
SIIM_ACR_Pneumothorax/
  train-rle.csv
  dicom-images-train/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
  dicom-images-test/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
```

<!-- claim: id=siim_acr_ptx.train.images.quick tier=ci cmd="python scripts/evalkit/dataset_stat.py siim_acr_ptx siim_acr_ptx_train.images" expect="10,675" -->
The first run with `output_mask=True` decodes the masks and writes 10,675 PNG files to `mask_output_dir`.
A later run uses these files.

```python
import torch
from radharmony.dataset import SIIMACRPTXTrainDataset, SIIMACRPTXTestDataset

train = SIIMACRPTXTrainDataset(
    base_image_dir="/data/SIIM_ACR_Pneumothorax/dicom-images-train/",
    output_cls=True,
    output_mask=True,
    mask_output_dir="/data/siim_acr_ptx_masks/",
    mask_num_cores=8,  # threads that decode the masks
    dtype=torch.float32,
)
train_ds, val_ds = train.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(1,), 1 is pneumothorax and 0 is no pneumothorax
# sample["mask"]: Tensor(1, 224, 224), 1 is pneumothorax and 0 is background

test_ds = SIIMACRPTXTestDataset(
    base_image_dir="/data/SIIM_ACR_Pneumothorax/dicom-images-test/",
    dtype=torch.float32,
).get_datasets()
# test_ds[0] has only "img", for a Kaggle submission
```

## How it is harmonized

- Source: `train-rle.csv` and the DICOM files in `dicom-images-train/`. The CSV has one row for each
  region. The harmonizer joins the rows of one image into one row (D-03).
- IDs: the harmonizer finds each DICOM file below `base_image_dir` and joins it on `ImageId`. The first
  folder of the path (the `StudyInstanceUID`) gives `patient_id`, for example
  `1.2.276.0.7230010.3.1.2.8323329.5597.1517875188.959089`. The second folder (the `SeriesInstanceUID`)
  gives `study_id`, for example `1.2.276.0.7230010.3.1.3.8323329.5597.1517875188.959088`. `image_path` is
  the path of the file, relative to `base_image_dir` (D-04), for example
  `1.2.276.0.7230010.3.1.2.8323329.5597.1517875188.959089/1.2.276.0.7230010.3.1.3.8323329.5597.1517875188.959088/1.2.276.0.7230010.3.1.4.8323329.5597.1517875188.959090.dcm`.
- Patients: `patient_id` is the `StudyInstanceUID` folder, not the DICOM `PatientID`. Both are different
  for each image, so the split is the same with either one.
- Label (D-05): `pneumothorax` is 0 when the RLE text is `-1`, and 1 for any other text.
- View (D-01): the harmonized train table has `view_position` `FRONTAL` for every image. The harmonizer does
  not read the DICOM tag `ViewPosition`, which is PA or AP for every file. D-01 lists this as a known gap.
  The harmonized test table has no `view_position`.
- Age and sex: they come from the DICOM headers. `age_years` comes from `PatientAge`, in years (D-06).
  `sex` is `M` or `F`, from `PatientSex` (D-26).
- Dropped rows: the harmonizer drops the 37 train DICOM files that have no row in `train-rle.csv`.

## Labels

| Column | Meaning |
|---|---|
| `pneumothorax` | Pneumothorax. 1 when the image has a mask, 0 when the RLE text is `-1` |

## Masks

- Source: the column ` EncodedPixels` of `train-rle.csv` (the name starts with a space). Each row is the
  RLE text of one region on the 1024 x 1024 image (H x W), for example `407576 2 1021 7 1015 10 ...`. The
  pixels go down each column first. Each start is relative to the end of the run before it. `-1` means no
  pneumothorax.
- Without `mask_output_dir`, `mask_path` holds the RLE text. An image with 2 or more regions joins their
  texts with `|`. 624 train images have 2 or more regions.
- With `mask_output_dir`, the harmonizer decodes each image into 1 PNG. A pixel is 1 when it is in one of
  the regions. An image with `-1` gets an all-zero PNG, so the negative images stay. The PNG path copies
  `image_path`, for example `<mask_output_dir>/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.png`.
  `mask_path` is the full path of that PNG. The harmonizer does not write a PNG again when the file exists.
- An empty `mask_path` means "not annotated" (D-33). With `output_mask=True`, the class drops these rows
  and warns. Every train image has an RLE text or `-1`, so no `mask_path` is empty, and no row is
  dropped.
- `output_mask=True` with no `mask_output_dir` raises `ValueError`.
- The sample: `mask` is a float32 tensor `(1, 224, 224)` with the values 0 and 1 (D-14). The transform
  resizes it with the image.

## Splits

<!-- claim: id=siim_acr_ptx.splits.train tier=ci cmd="python scripts/evalkit/dataset_stat.py siim_acr_ptx siim_acr_ptx_train.images" expect="10,675" -->
<!-- claim: id=siim_acr_ptx.splits.test tier=ci cmd="python scripts/evalkit/dataset_stat.py siim_acr_ptx siim_acr_ptx_test.images" expect="1,377" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `SIIMACRPTXTrainDataset` | `siim_acr_ptx_train` | 10,675 | `train-rle.csv` and `dicom-images-train/` |
| `SIIMACRPTXTestDataset` | `siim_acr_ptx_test` | 1,377 | `dicom-images-test/`, with no labels (D-25). Use it only for a Kaggle submission |
| `SIIMACRPTXDataset` | `siim_acr_ptx` | | The old key. It reads only the train files, so it gives the train images. Use the split classes for a benchmark (D-09) |

The source has no labeled test split. `get_datasets(n_splits=5)` makes a train and val split from the train
class. It groups by patient (D-10). No patient is in 2 splits.

## Arguments

`SIIMACRPTXTestDataset` takes the same arguments, except `mask_output_dir` and `mask_num_cores`.

<!-- args: radharmony.dataset.SIIMACRPTXTrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | `dicom-images-train/` or `dicom-images-test/`. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `train-rle.csv`. Without it, the class looks in `base_image_dir` and the folder above it. The test class ignores it |
| `mask_output_dir` | `str` | No | `None` | The folder for the mask PNG files. Needed with `output_mask=True`. Without it, the class raises `ValueError` |
| `mask_num_cores` | `int` | No | `1` | The number of threads that decode the masks |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample. In `SIIMACRPTXTestDataset`: not supported. The class warns and adds no key (D-25) |
| `output_mask` | `bool` | No | `False` | Add the `mask` tensor to the sample. It needs `mask_output_dir`. The class drops the rows with an empty `mask_path`, and warns (D-33). In `SIIMACRPTXTestDataset`: not supported. The class warns and adds no key (D-25) |
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
from radharmony.harmonizer import SIIMACRPTXTrainHarmonizer

h = SIIMACRPTXTrainHarmonizer(
    csv_path="/data/SIIM_ACR_Pneumothorax/train-rle.csv",
    dicom_dir="/data/SIIM_ACR_Pneumothorax/dicom-images-train/",
)
df = h.harmonize()  # harmonize(mask_output_dir=...) also decodes the masks to PNG files
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'view_position', 'mask_path', 'age_years', 'sex', 'pneumothorax']
```

`SIIMACRPTXTestHarmonizer` takes only `dicom_dir`. Its harmonized table has `patient_id`, `study_id`,
`image_path`, `age_years` and `sex`. With `harmonizer_path`, `SIIMACRPTXTrainDataset` also loads a file that the old
`SIIMACRPTXHarmonizer` saved.

<!-- args: radharmony.harmonizer.SIIMACRPTXTrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `train-rle.csv` |
| `dicom_dir` | `str` | Yes | | `dicom-images-train/`. The harmonizer finds every `.dcm` file below it |

## Known data problems

- Age: 2 train images have the `PatientAge` 413 and 148, so they are errors. Their `age_years` is NaN
  (D-06).
