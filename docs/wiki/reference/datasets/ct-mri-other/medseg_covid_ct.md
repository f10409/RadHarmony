# MedSeg COVID-19 CT

| | |
|---|---|
| Registry keys | `medseg_covid_ct`, `medseg_covid_ct_val` |
| Task and outputs | Segmentation of COVID-19 lesions: lesion masks (`mask`) and 4 slice labels (`cls`). `medseg_covid_ct_val` has images only |
| Images in the table | dataset1 100 slices, dataset2 829 slices (9 volumes), val 10 slices |
| Modality | Chest CT, 2-D axial slices of 3-D NIfTI volumes. One row is one slice |
| License | CC0 (the figshare records) |
| Download | [Get the data: other datasets](../../../how-to/get-the-data.md#other-datasets) |

MedSeg (medicalsegmentation.com) published 2 COVID-19 chest CT sets on figshare. "MedSeg Covid Dataset 1"
([figshare 13521488](https://figshare.com/articles/dataset/13521488)) has 100 axial slices from 43
patients. "MedSeg Covid Dataset 2" ([figshare 13521509](https://figshare.com/articles/dataset/13521509))
has 9 whole volumes. The argument `subset` picks one of them: `"dataset1"` or `"dataset2"`.

The masks mark ground-glass opacity, consolidation and pleural effusion. It is like
[QaTa-COV19](../cxr-boxes-masks/qata_cov19.md), but on CT. Its labels are not all 1: many slices have no
lesion.

## Quick use

The download needs no account. This script gets the files of both records from figshare:

<!-- block: run=skip reason="downloads the MedSeg files from figshare" -->
```python
import json
import urllib.request

for aid in (13521488, 13521509):
    det = json.load(urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{aid}"))
    for f in det["files"]:
        urllib.request.urlretrieve(f["download_url"], f["name"])
```

Put the files of each record in its own folder. Unzip `rp_im.zip` and `rp_msk.zip` of dataset2 in that
folder. `base_image_dir` is the folder of one record.

```
MedSeg-COVID-CT/
  figshare_13521488/tr_im.nii.gz
  figshare_13521488/tr_mask.nii.gz
  figshare_13521488/val_im.nii.gz
  figshare_13521509/rp_im/1.nii.gz
  figshare_13521509/rp_msk/1.nii.gz
```

```python
import torch
from radharmony.dataset import MedSegCOVIDCTDataset, MedSegCOVIDCTValDataset

ds = MedSegCOVIDCTDataset(
    base_image_dir="/data/MedSeg-COVID-CT/figshare_13521509/",
    slice_output_dir="/data/medseg_slices/",  # any folder you can write to
    subset="dataset2",
    output_cls=True,
    output_mask=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)  # 1/5 of the volumes go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(4,), pleural_effusion is NaN in dataset2
# sample["mask"]: Tensor(1, 224, 224), 1 is lesion and 0 is background

test_ds = MedSegCOVIDCTValDataset(
    base_image_dir="/data/MedSeg-COVID-CT/figshare_13521488/",
    slice_output_dir="/data/medseg_slices/",
    dtype=torch.float32,
).get_datasets()
# test_ds[0] has only "img": 10 slices with no labels, to look at predictions
```

The first run writes 1 image PNG and 1 mask PNG for each slice to
`<slice_output_dir>/<subset>_hu<low>_<high>/`, for example `dataset2_hu-1000_400/`. The harmonizer does not
write a PNG again when the file exists. The first run also writes a MONAI cache to `./cache`.
`cache_dir=None` gives no cache.

`hu_window` (default `(-1000.0, 400.0)`, a lung window) acts in the harmonizer. It clips each slice to the
window and saves it as an 8-bit PNG. The raw HU values stay only in the `.nii.gz` files. The window is in
the folder name, so a new window writes new PNGs. Then the transform scales the 0.5 and 99.5 percentiles
of the PNG to -1 and 1 (D-14).

## How it is harmonized

- Source: in dataset1, `tr_im.nii.gz` and `tr_mask.nii.gz` hold 100 slices of 512 x 512 (H x W). In
  dataset2, `rp_im/<n>.nii.gz` and `rp_msk/<n>.nii.gz` hold the 9 volumes, with slices of 630 x 630 (H x W).
  One row is one axial slice. In dataset1, the harmonizer also reads `Test-Images-Clinical-Details.csv`.
  It holds the SIRM case, the age and the sex of each `tr_im` slice.
- Slices (D-12): the harmonizer writes each slice in the usual axial view. Rows go from anterior to
  posterior, and columns go from patient right to patient left. The mask gets the same step.
- IDs: in dataset2, slice 7 of `rp_im/1.nii.gz` gives `patient_id` `1`, `study_id` `dataset2_1_0007` and
  `image_path` `dataset2_1_0007.png`. In dataset1, slice 0 gives `study_id` `dataset1_tr_0000`, and its
  SIRM case gives `patient_id` `42`. The class sets `base_image_dir` to the `images/` folder of the slices,
  and `image_path` is relative to it (D-04).
- Patients (D-10): in dataset2, one volume is one patient. In dataset1, `patient_id` is the SIRM case from
  `Test-Images-Clinical-Details.csv`, so the 100 slices are 43 patients. Without this file, the harmonizer
  raises `FileNotFoundError`. The `val_im` slices are not in the file, so each val slice is its own patient.
- Labels (D-05): they come from the class values in the mask slice: 1 ground-glass opacity, 2
  consolidation and 3 pleural effusion. A label is 1 when the slice has its value. Else it is 0.
  `any_lesion` is 1 when the slice has a value above 0. The dataset2 masks have only 0, 1 and 2, so
  `pleural_effusion` is NaN in each dataset2 row.
- View (D-01): the source has no view, so the harmonized table has no `view_position`.
- Age and sex (D-06, D-26): in dataset1, they come from `Test-Images-Clinical-Details.csv`. Only the first
  row of a case has them, and they apply to all slices of the case. Sex 0 gives `"M"` and 1 gives `"F"`.
  The dataset2 volumes and the val slices have no age or sex. So their harmonized tables have no
  `age_years` or `sex`.
- Dropped rows: the harmonizer drops no row.

## Labels

The 4 label columns, in the order of `cls`:

| Column | Meaning |
|---|---|
| `any_lesion` | The slice has a lesion of any class |
| `consolidation` | The slice has consolidation (mask value 2) |
| `ground_glass` | The slice has ground-glass opacity (mask value 1) |
| `pleural_effusion` | The slice has pleural effusion (mask value 3). NaN in dataset2 |

The slices with each label:

| Subset | `any_lesion` | `consolidation` | `ground_glass` | `pleural_effusion` |
|---|---|---|---|---|
| dataset1 (100 slices) | 99 | 78 | 96 | 25 |
| dataset2 (829 slices) | 373 | 233 | 298 | NaN |

## Masks

- Source: `tr_mask.nii.gz` and `rp_msk/<n>.nii.gz` hold one class value for each voxel: 0 background, 1
  ground-glass opacity, 2 consolidation and 3 pleural effusion (dataset1 only). The harmonizer does not
  read the lung masks in `rp_lung_msk/`.
- The harmonizer writes 1 binary mask PNG for each slice to the `masks/` folder of the slices. A pixel is
  255 when its class value is above 0. Else it is 0. So the mask joins the 3 classes. The labels tell the
  classes apart.
- `mask_path` is the full path of that PNG, for example
  `/data/medseg_slices/dataset2_hu-1000_400/masks/dataset2_1_0007.png`. The extra column `lesion_pixels`
  holds the number of lesion pixels in the slice.
- An empty `mask_path` means "not annotated" (D-33). With `output_mask=True`, the class drops these rows
  and warns. Every slice has a mask, so no row is dropped. A slice with no lesion has an all-zero mask:
  1 slice in dataset1 and 456 slices in dataset2.
- The sample: `mask` is a float32 tensor `(1, 224, 224)` with the values 0 and 1 (D-14). The transform
  resizes it with the image.

## Splits

| Class | Registry key | Images | Source |
|---|---|---|---|
| `MedSegCOVIDCTDataset` | `medseg_covid_ct` | 100 (dataset1) or 829 (dataset2) | `tr_im.nii.gz` and `tr_mask.nii.gz` (dataset1), or `rp_im/` and `rp_msk/` (dataset2) |
| `MedSegCOVIDCTValDataset` | `medseg_covid_ct_val` | 10 | `val_im.nii.gz` of dataset1, with no labels and no masks (D-25). Use it only to look at predictions |

MedSeg COVID-19 CT has no official split. `get_datasets(n_splits=5)` makes a train and val split. It groups
by patient (D-10). In dataset2, a patient is a volume, so no volume is in 2 splits. In dataset1, a
patient is a SIRM case, so no case is in 2 splits.

`MedSegCOVIDCTValDataset` writes its slices to `<slice_output_dir>/dataset1_val_hu<low>_<high>/images/`.

## Arguments

<!-- args: radharmony.dataset.MedSegCOVIDCTDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The folder of one record: with `tr_im.nii.gz` and `Test-Images-Clinical-Details.csv` (dataset1), or with `rp_im/` (dataset2). Not needed with `harmonizer_path`, which holds it |
| `slice_output_dir` | `str` | Yes | `None` | The folder for the slice PNGs. Without it, the class raises `ValueError`. Not needed with `harmonizer_path`, which holds it |
| `subset` | `str` | No | `"dataset1"` | `"dataset1"` or `"dataset2"` |
| `hu_window` | `tuple` | No | `(-1000.0, 400.0)` | The HU window of the PNG slices. It is part of the slice folder name |

`MedSegCOVIDCTValDataset` takes the same arguments, but no `subset`. Its `base_image_dir` is the folder with
`val_im.nii.gz`.

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample. In `MedSegCOVIDCTValDataset`: not supported. The class warns and adds no key (D-25) |
| `output_mask` | `bool` | No | `False` | Add the `mask` tensor to the sample. The class drops the rows with an empty `mask_path`, and warns (D-33). In `MedSegCOVIDCTValDataset`: not supported. The class warns and adds no key (D-25) |
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
from radharmony.harmonizer import MedSegCOVIDCTHarmonizer

h = MedSegCOVIDCTHarmonizer(
    base_dir="/data/MedSeg-COVID-CT/figshare_13521509/",
    slice_output_dir="/data/medseg_slices/",  # any folder you can write to
    subset="dataset2",
)
df = h.harmonize()  # writes the slice PNGs the first time
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'mask_path', 'any_lesion', ..., 'subset', 'volume', ...]
```

The extra columns are `subset`, `volume` (`tr` in dataset1, `1` to `9` in dataset2), `slice_index` and
`lesion_pixels`. `MedSegCOVIDCTValHarmonizer` takes `base_dir`, `slice_output_dir` and `hu_window`. Its
harmonized table has no label column and no `mask_path`.

<!-- args: radharmony.harmonizer.MedSegCOVIDCTHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_dir` | `str` | Yes | | The folder of one record: with `tr_im.nii.gz` and `Test-Images-Clinical-Details.csv` (dataset1), or with `rp_im/` (dataset2) |
| `slice_output_dir` | `str` | No | `None` | The folder for the slice PNGs. Without it, `harmonize()` raises `ValueError` |
| `subset` | `str` | No | `"dataset1"` | `"dataset1"` or `"dataset2"`. Another value raises `ValueError` |
| `hu_window` | `tuple` | No | `(-1000.0, 400.0)` | The HU window of the PNG slices |

## Known data problems

- dataset1 age and sex: in `Test-Images-Clinical-Details.csv`, SIRM cases 16, 18 and 19 have no age or
  sex, and cases 17 and 45 have no sex. The last row, "Not used", lists SIRM cases, not slices.
- dataset2 volume sizes: the 9 volumes have 39 to 418 slices, and volume 3 has 418 of the 829 slices. So a
  split by volume can give splits of very different sizes.
- dataset2 slice order: volume 1 has the orientation code `LPI`, and the other volumes have `LPS`. The view
  of a slice is the same, but `slice_index` 0 is at the other end of the chest.
- val slices: `val_im.nii.gz` has no masks.
