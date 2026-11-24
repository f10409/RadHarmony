# Shenzhen Hospital CXR

<!-- claim: id=shenzhen_cxr.images tier=ci cmd="python scripts/evalkit/dataset_stat.py shenzhen_cxr shenzhen_cxr.images" expect="662" -->
<!-- claim: id=shenzhen_cxr.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py shenzhen_cxr shenzhen_cxr.patients" expect="662" -->
| | |
|---|---|
| Registry keys | `shenzhen_cxr` |
| Task and outputs | Classification, 1 label (`cls`). TB region masks (`mask`), only with the `Annotations-2/` download. Report text (`report`) |
| Images in the table | 662 (662 patients) |
| Modality | Chest X-ray, 2-D, PNG |
| License | NLM research use. Do not share the data outside your group. Cite Jaeger et al. and Candemir et al., IEEE TMI 2014 |
| Download | [Get the data: Shenzhen Hospital CXR](../../../how-to/get-the-data.md#shenzhen-hospital-cxr) |

<!-- claim: id=shenzhen_cxr.tuberculosis tier=ci cmd="python scripts/evalkit/dataset_stat.py shenzhen_cxr shenzhen_cxr.positives.tuberculosis" expect="336" -->
The Shenzhen Hospital CXR Set is a tuberculosis (TB) screening set from the U.S. National Library of
Medicine (NLM) and Shenzhen No.3 People's Hospital, China. It has 662 chest X-rays from outpatient care:
336 with TB (numbers 0327 to 0662) and 326 normal. A Philips DR system took them, and an image is about
3000 x 3000 pixels (H x W). Each image has a clinical reading with the sex, the age and a short diagnosis.
It comes from the same NLM project as [Montgomery County CXR](montgomery_cxr.md), with the same folders
and the same form of file name.

A later download, `Annotations-2/`, adds 19 TB finding columns and masks of the TB regions for 323 of the
336 TB images. It is a separate download: [Get the data](../../../how-to/get-the-data.md#shenzhen-hospital-cxr)
has a Python script for it. The harmonizer does not read the older `Annotations/` folder or
`shenzhen_consensus_roi.csv` (250 TB boxes on 68 images).

## Quick use

`base_image_dir` is the `CXR_png/` folder. The class finds `ClinicalReadings/` and `Annotations-2/` next
to it.

```
Shenzhen-Hospital-CXR-Set/
  CXR_png/CHNCXR_0001_0.png
  ClinicalReadings/CHNCXR_0001_0.txt
  Annotations-2/Statistics_ShenzhenDataset_US.csv
  Annotations-2/masks/CHNCXR_0327_1_Small_Infiltrate_(non-linear)_1.png
```

```python
import torch
from radharmony.dataset import ShenzhenCXRDataset

ds = ShenzhenCXRDataset(
    base_image_dir="/data/Shenzhen-Hospital-CXR-Set/CXR_png/",
    output_cls=True,
    output_mask=True,  # needs Annotations-2/
    mask_output_dir="/data/shenzhen_tb_masks/",  # any folder you can write to
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(1,), 1 is TB and 0 is normal
# sample["mask"]: Tensor(1, 224, 224), 1 is a TB region and 0 is background
```

For classification only, leave out `output_mask` and `mask_output_dir`. Then the class keeps all 662 images.

## How it is harmonized

- Source: the PNG files in `CXR_png/`. One file is one row. The source has no label CSV.
- IDs: the file `CHNCXR_0001_0.png` gives `patient_id` `0001`, `study_id` `0001` and `image_path`
  `CHNCXR_0001_0.png`. `image_path` is relative to `base_image_dir` (D-04). Each image number is one
  patient and one study.
- Label (D-05): the file name gives `tuberculosis`. A file name that ends in `_1` gives 1 (TB), and `_0`
  gives 0 (normal).
- Age and sex: they come from the first line of `ClinicalReadings/<file name>.txt`, for example
  `male 45yrs`. `age_years` comes from `patient_age`, in years (D-06). An age in months or days becomes
  years, so `female 64days` gives 0.175. `sex` is `M` or `F`, from `patient_sex` (D-26).
- Report: `report` holds the whole reading, with the first line, for example `male 45yrs\nnormal`.
- TB findings: the harmonizer joins the 19 finding columns of
  `Annotations-2/Statistics_ShenzhenDataset_US.csv` on the file name. They are NaN for a normal image.
- View (D-01): the source has no view, so the harmonized table has no `view_position`.
- Dropped rows: the harmonizer drops no row.

## Labels

| Column | Meaning |
|---|---|
| `tuberculosis` | Tuberculosis. 1 is TB, 0 is normal |

The 19 TB finding columns are extra columns, not labels, so `cls` does not hold them. They come from
`Annotations-2/`. Each value is the number of regions of that finding on the image (0 to 75), not a 0 or 1
flag. For each column, the table below shows the number of TB images with 1 or more regions and the total number
of regions.

| Column | Images | Regions |
|---|---|---|
| `Small_Infiltrate_(non-linear)` | 145 | 219 |
| `Clustered_Nodule_(2mm-5mm_apart)` | 113 | 605 |
| `Linear_Density` | 90 | 339 |
| `Pleural_Thickening_(non-apical)` | 75 | 113 |
| `Single_Nodule_(non-calcified)` | 71 | 193 |
| `Apical_Thickening` | 68 | 92 |
| `Moderate_Infiltrate_(non-linear)` | 66 | 97 |
| `Calcified_Nodule` | 42 | 171 |
| `Pleural_Effusion` | 34 | 36 |
| `Other` | 29 | 71 |
| `Cavity` | 22 | 101 |
| `Adenopathy` | 22 | 30 |
| `Retraction` | 19 | 22 |
| `Severe_Infiltrate_(Consolidation)` | 18 | 20 |
| `Miliary` | 15 | 27 |
| `Unknown` | 14 | 19 |
| `Thickening_of_interlobar_fissure` | 8 | 8 |
| `Calcification_(other_than_nodule&lymphnod)` | 6 | 6 |
| `Calcified_lymph_node` | 5 | 8 |

## Masks

- Source: `Annotations-2/masks/` holds 1 binary PNG for each finding region of a TB image. The file name
  gives the image, the finding and a region number, for example
  `CHNCXR_0330_1_Pleural_Effusion_3.png`. A normal image has no mask file.
- The extra column `finding_masks` holds a JSON list of the mask files of each TB image, for example
  `["../Annotations-2/masks/CHNCXR_0327_1_Small_Infiltrate_(non-linear)_1.png"]`. The paths are relative to
  `base_image_dir`.
- With `mask_output_dir`, the harmonizer joins the masks of each TB image into 1 "TB region" mask. A pixel
  is 1 when it is in one of the masks. A normal image gets an all-zero mask of the image size. The
  harmonizer writes 1 PNG for each image to `mask_output_dir`. `mask_path` is the full path of that PNG.
  The harmonizer does not write a PNG again when the file exists.
- An empty `mask_path` means "not annotated" (D-33). With `output_mask=True`, the class drops these rows
  and warns. 13 TB images have no mask file, so their `mask_path` is empty, and 649 of the 662 images
  stay.
- With `mask_output_dir` and no `Annotations-2/masks/` folder, the class raises `FileNotFoundError` with the
  folder name (D-35).
  Download `Annotations-2/` before you use `output_mask=True`.
- `output_mask=True` with no `mask_output_dir` raises `ValueError`.
- The sample: `mask` is a float32 tensor `(1, 224, 224)` with the values 0 and 1 (D-14). The transform
  resizes it with the image.

## Reports

`report` holds the whole reading. The first line has the sex and the age. The next lines have the
diagnosis, for example `normal` or `bilateral PTB` (PTB is pulmonary tuberculosis). Every image has a
reading, so `output_report=True` keeps all 662 images.

## Splits

<!-- claim: id=shenzhen_cxr.splits.images tier=ci cmd="python scripts/evalkit/dataset_stat.py shenzhen_cxr shenzhen_cxr.images" expect="662" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `ShenzhenCXRDataset` | `shenzhen_cxr` | 662 | The PNG files in `CXR_png/` |

Shenzhen Hospital CXR has no official split. `get_datasets(n_splits=5)` makes a train and val split. It
groups by patient (D-10). The source has no patient ID, so one image is one patient, and nobody can check
that a patient is in only 1 split.

## Arguments

<!-- args: radharmony.dataset.ShenzhenCXRDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `CXR_png/` folder. `ClinicalReadings/` and `Annotations-2/` are next to it. Not needed with `harmonizer_path`, which holds it |
| `mask_output_dir` | `str` | No | `None` | The folder for the mask PNG files. Needed with `output_mask=True`. Without it, the class raises `ValueError` |
| `mask_num_cores` | `int` | No | `1` | Not used. The class has it so that its arguments match SIIM-ACR Pneumothorax |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample |
| `output_mask` | `bool` | No | `False` | Add the `mask` tensor to the sample. It needs `mask_output_dir`. The class drops the rows with an empty `mask_path`, and warns (D-33) |
| `output_report` | `bool` | No | `False` | Add the `report` text to the sample. The class drops the rows with no report, and warns (D-33) |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11). The harmonizer does not read the TB boxes |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The labels stay float32 |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import ShenzhenCXRHarmonizer

h = ShenzhenCXRHarmonizer(base_dir="/data/Shenzhen-Hospital-CXR-Set/CXR_png/")
df = h.harmonize()  # harmonize(mask_output_dir=...) also writes the TB region masks and adds mask_path
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'report', 'age_years', 'sex', 'tuberculosis', 'patient_age', ...]
```

The extra columns are `patient_age` (float years), `patient_sex`, the 19 TB finding columns and
`finding_masks`.

<!-- args: radharmony.harmonizer.ShenzhenCXRHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_dir` | `str` | Yes | | The `CXR_png/` folder. `ClinicalReadings/` and `Annotations-2/` are next to it |

## Known data problems

- Age text: the first line of a reading has several forms, for example `femal 32yrs` (a typo, in 3 files),
  `male35yrs` (no space), `male 16month` and `female 64days`. `CHNCXR_0018_0` says `male 42` with no unit,
  so its age is NaN. The harmonizer does not guess the unit (D-06).
- PNG type: 635 of the 662 PNGs are palette images and 27 are RGB. All are gray, so `img` has 1 channel
  (D-42).
