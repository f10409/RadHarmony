# Montgomery County CXR

<!-- claim: id=montgomery_cxr.images tier=ci cmd="python scripts/evalkit/dataset_stat.py montgomery_cxr montgomery_cxr.images" expect="138" -->
<!-- claim: id=montgomery_cxr.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py montgomery_cxr montgomery_cxr.patients" expect="135" -->
| | |
|---|---|
| Registry keys | `montgomery_cxr` |
| Task and outputs | Classification, 1 label (`cls`). Lung masks (`mask`). Report text (`report`) |
| Images in the table | 138 (135 patients) |
| Modality | Chest X-ray, 2-D, PNG |
| License | NLM research use. Do not share the data outside your group. Cite Jaeger et al. and Candemir et al., IEEE TMI 2014 |
| Download | [Get the data: Montgomery County CXR](../../../how-to/get-the-data.md#montgomery-county-cxr) |

<!-- claim: id=montgomery_cxr.tuberculosis tier=ci cmd="python scripts/evalkit/dataset_stat.py montgomery_cxr montgomery_cxr.positives.tuberculosis" expect="58" -->
The Montgomery County CXR Set is a tuberculosis (TB) screening set from the U.S. National Library of
Medicine (NLM) and Montgomery County, Maryland. It has 138 chest X-rays: 58 with TB and 80 normal. Each
image has a left and a right lung mask, and a clinical reading with the age, the sex and a short report.
The download is small and needs no account, so it is a good first dataset. With 138 images, a result can
change a lot from one split to the next.

The source also has `montgomery_consensus_roi.csv`, with 109 TB boxes on 55 images. The harmonizer does not
read it.

## Quick use

Get the data first: [Get the data: Montgomery County CXR](../../../how-to/get-the-data.md#montgomery-county-cxr).

`base_image_dir` is the `CXR_png/` folder, not `MontgomerySet/`. The class finds `ClinicalReadings/` and
`ManualMask/` next to `CXR_png/`.

```
MontgomerySet/
  CXR_png/MCUCXR_0001_0.png
  ClinicalReadings/MCUCXR_0001_0.txt
  ManualMask/leftMask/MCUCXR_0001_0.png
  ManualMask/rightMask/MCUCXR_0001_0.png
```

```python
import torch
from radharmony.dataset import MontgomeryCXRDataset

ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/",
    output_cls=True,
    output_mask=True,
    mask_output_dir="/data/montgomery_lung_masks/",  # any folder you can write to
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(1,), 1 is TB and 0 is normal
# sample["mask"]: Tensor(1, 224, 224), 1 is lung and 0 is background
```

The class makes `mask_output_dir` and writes 1 lung mask PNG for each image to it. The first run also
writes a MONAI cache to `./cache`. `cache_dir=None` gives no cache. For classification only, leave out
`output_mask` and `mask_output_dir`.

## How it is harmonized

- Source: the PNG files in `CXR_png/`. One file is one row. The source has no label CSV.
- IDs: the file `MCUCXR_0001_0.png` gives `patient_id` `MCUCXR_0001`, `study_id` `MCUCXR_0001` and
  `image_path` `MCUCXR_0001_0.png`. `image_path` is relative to `base_image_dir` (D-04). One study is one
  image.
- Same patient: 3 readings say that the patient is the same as in another image, for example
  `(same pt as MCUCXR_0113_1)` in `MCUCXR_0117_1.txt`. The harmonizer gives each image of the group the
  smallest ID of the group as `patient_id`. So the 138 images give 135 patients.
- Label (D-05): the file name gives `tuberculosis`. A file name that ends in `_1` gives 1 (TB), and `_0` gives 0
  (normal).
  Before 2026-11-05, the column was `tb`. If you saved a harmonized table before then, harmonize it again
  ([Save and reuse a harmonized table](../../../how-to/save-and-reuse.md)).
- Age and sex: they come from the reading `ClinicalReadings/<file name>.txt`. `age_years` comes from
  `Patient's Age`, in years (D-06). For example, `027Y` gives 27. `sex` is `M`, `F` or NaN, from
  `Patient's Sex` (D-26).
- Report: `report` holds the other lines of the reading. A text that only says "NO REPORT" gives NaN
  (D-33).
- View (D-01): the source has no view, so the harmonized table has no `view_position`.
- Dropped rows: the harmonizer drops no row.

## Labels

| Column | Meaning |
|---|---|
| `tuberculosis` | Tuberculosis. 1 is TB, 0 is normal |

## Masks

- Source: `ManualMask/leftMask/` and `ManualMask/rightMask/` hold 1 binary PNG for each image, with the
  same file name as the image. People drew them by hand, and a radiologist supervised the work. A mask
  has the size of its image, 4020 x 4892 or 4892 x 4020 pixels (H x W).
- When `ManualMask/` is there, the harmonized table always has 2 extra columns, `mask_path_left` and
  `mask_path_right`, for example `../ManualMask/leftMask/MCUCXR_0001_0.png`. They are relative to
  `base_image_dir`. Without `mask_output_dir`, the harmonized table has no `mask_path`.
- With `mask_output_dir`, the harmonizer joins the left and right masks into 1 lung mask. A pixel is 1 when
  it is in one of the 2 masks. It writes 1 PNG for each image to `mask_output_dir`. `mask_path` is the full
  path of that PNG. The harmonizer does not write a PNG again when the file exists.
- An empty `mask_path` means "not annotated" (D-33). With `output_mask=True`, the class drops these rows
  and warns. All 138 images have both masks, so no row is dropped.
- `output_mask=True` with no `mask_output_dir` raises `ValueError`. With `output_mask=True` and no
  `ManualMask/` folder, the class raises `FileNotFoundError` with the folder name (D-35).
- The sample: `mask` is a float32 tensor `(1, 224, 224)` with the values 0 and 1 (D-14). The transform
  resizes it with the image.

## Reports

`report` holds the free text of the reading, in one line. Every normal image has the text `normal`. A TB
image has a short note, for example `MCUCXR_0104_1`:

```
inactive TB scars LUL, unchanged for 2Y, pt previously treated DOT (17Y ago) for pan-sensitive disease
```

With `output_report=True`, the class drops the 1 image with no report (`MCUCXR_0166_1`), and warns (D-33).

## Splits

<!-- claim: id=montgomery_cxr.splits.images tier=ci cmd="python scripts/evalkit/dataset_stat.py montgomery_cxr montgomery_cxr.images" expect="138" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `MontgomeryCXRDataset` | `montgomery_cxr` | 138 | The PNG files in `CXR_png/` |

Montgomery County CXR has no official split. `get_datasets(n_splits=5)` makes a train and val split. It
groups by patient (D-10). The source has no patient ID. Only 3 readings link the images of one patient, so
nobody can check that a patient is in only 1 split.

## Arguments

<!-- args: radharmony.dataset.MontgomeryCXRDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `CXR_png/` folder. `ClinicalReadings/` and `ManualMask/` are next to it. Not needed with `harmonizer_path`, which holds it |
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
from radharmony.harmonizer import MontgomeryCXRHarmonizer

h = MontgomeryCXRHarmonizer(base_dir="/data/MontgomerySet/CXR_png/")
df = h.harmonize()  # harmonize(mask_output_dir=...) also writes the lung masks and adds mask_path
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'report', 'age_years', 'sex', 'tuberculosis', 'age', 'mask_path_left', ...]
```

The extra columns are `age` (the source age in whole years), `mask_path_left` and `mask_path_right`.

<!-- args: radharmony.harmonizer.MontgomeryCXRHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_dir` | `str` | Yes | | The `CXR_png/` folder. `ClinicalReadings/` and `ManualMask/` are next to it |

## Known data problems

- Sex: the reading of `MCUCXR_0080_0` has the sex `O`. So its `sex` is NaN (D-26).
