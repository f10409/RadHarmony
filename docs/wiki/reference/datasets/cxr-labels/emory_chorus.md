# Emory CHoRUS (X-ray subset)

<!-- claim: id=emory_chorus.images tier=ci cmd="python scripts/evalkit/dataset_stat.py emory_chorus emory_chorus.images" expect="10,290" -->
<!-- claim: id=emory_chorus.studies tier=ci cmd="python scripts/evalkit/dataset_stat.py emory_chorus emory_chorus.studies" expect="4,889" -->
<!-- claim: id=emory_chorus.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py emory_chorus emory_chorus.patients" expect="1,397" -->
| | |
|---|---|
| Registry keys | `emory_chorus` |
| Task and outputs | Images only (`img`). No labels, masks, boxes or reports |
| Images in the table | 10,290 chest images, 4,889 studies and 1,397 patients (`chest_only=True`) |
| Modality | Chest X-ray, 2-D, DICOM (all images are `DX`) |
| License | Internal Emory data. It is not public, and you must not share it |
| Download | No download. The data paths are in `configs/data_paths.json` |

<!-- claim: id=emory_chorus.source.images tier=nas cmd="python scripts/evalkit/dataset_stat.py --source emory_chorus manifest_csv_path rows" expect="13,611" -->
<!-- claim: id=emory_chorus.source.patients tier=nas cmd="python scripts/evalkit/dataset_stat.py --source emory_chorus manifest_csv_path unique:person_id" expect="1,496" -->
Emory CHoRUS is an internal dataset of Emory. It links a clinical database with DICOM images of many
modalities. RadHarmony reads only the X-ray subset: 13,611 images of 1,496 patients. By default, the class
keeps only the chest X-rays. The other images show the abdomen, the spine, the skull or the limbs.

The database uses OMOP CDM, a standard format for clinical tables, with the patients, conditions,
measurements, procedures and visits. The class names write the dataset name as `CHORUS`.

The subset has no labels, masks, boxes or reports. Use it for image-only work, for example to extract
features or to look at the images of a patient. The page is in the label group only because the dataset
has chest X-rays.

## Quick use

`base_image_dir` is the `images_dicom_xray/` folder. The source has no table of the images. So the first
run reads the header of each DICOM file and writes a manifest CSV to `manifest_csv_path`. The manifest CSV
is a table with one row for each DICOM file. The next runs read the manifest.

```
Emory_CHORUS/
  images_dicom_xray/<person_id>/Images/<StudyInstanceUID>/<SeriesInstanceUID>/<file>.dcm
  chorus_xray_manifest.csv   # the first run writes it
```

```python
import torch
from radharmony.dataset import EmoryCHORUSDataset

ds = EmoryCHORUSDataset(
    base_image_dir="/data/Emory_CHORUS/images_dicom_xray/",
    manifest_csv_path="/data/Emory_CHORUS/chorus_xray_manifest.csv",
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample keys: img
# sample["img"]: Tensor(1, 224, 224)
```

## How it is harmonized

- Source: the manifest CSV. One row is one DICOM file. The scan reads each header with pydicom, with
  `num_workers` processes. It skips a file that pydicom cannot read. Without `manifest_csv_path`, each run
  scans the folder again, and this is slow.
- Chest filter: with `chest_only=True` (the default), the harmonizer keeps a row when `Modality` is `DX`
  or `CR`, and `BodyPartExamined` or `StudyDescription` contains `CHEST`. See
  [Known data problems](#known-data-problems).
- IDs: the file
  `<person_id>/Images/<StudyInstanceUID>/<SeriesInstanceUID>/<file>.dcm` gives `patient_id` `<person_id>`
  (the name of the patient folder), `study_id` `<StudyInstanceUID>` and `series_id` `<SeriesInstanceUID>`.
  `image_path` is the whole path, relative to `base_image_dir` (D-04). A study can have more than one
  image.
- View (D-01): from the DICOM `ViewPosition`. `AP` and `PA` stay. `LL`, `LAT` and `LATERAL` give
  `LATERAL`. Oblique views, for example `LAO`, keep their source value.
- Age and sex: `age_years` comes from the DICOM `PatientAge`, in whole years (D-06). `sex` is `M`, `F` or
  NaN, from `PatientSex` (D-26).
- Dropped rows: only the chest filter drops rows. With `chest_only=False`, the table has all rows of the
  manifest.

## Labels

The subset has no label columns, so `LABEL_COLS` is empty. The OMOP tables have the condition codes of
each patient, but RadHarmony does not read them.

## Splits

<!-- claim: id=emory_chorus.splits.images tier=ci cmd="python scripts/evalkit/dataset_stat.py emory_chorus emory_chorus.images" expect="10,290" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `EmoryCHORUSDataset` | `emory_chorus` | 10,290 | The manifest CSV, chest images only |

The source has no official split. `get_datasets(n_splits=5)` makes a train and val split. It groups by
patient (D-10).

## Arguments

<!-- args: radharmony.dataset.EmoryCHORUSDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `images_dicom_xray/` folder. Not needed with `harmonizer_path`, `harmonized_df` or `harmonizer` |
| `manifest_csv_path` | `str` | No | `None` | The manifest CSV. The class reads it when it exists. Otherwise the class scans the DICOM files and writes it. With `None`, the class scans the DICOM files at each run and writes no file |
| `chest_only` | `bool` | No | `True` | Keep only the chest X-rays. `False` keeps all images of the X-ray subset |
| `num_workers` | `int` | No | `12` | The number of processes that read the DICOM headers in a scan |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` with 224 pixels |
| `output_cls` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_report` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img` with the default transform. With your own `transform`, it does nothing |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import EmoryCHORUSHarmonizer

h = EmoryCHORUSHarmonizer(
    base_image_dir="/data/Emory_CHORUS/images_dicom_xray/",
    csv_path="/data/Emory_CHORUS/chorus_xray_manifest.csv",  # the manifest CSV
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'series_id', 'view_position', 'age_years', 'sex',
#  'body_part', 'study_date', 'age', 'manufacturer']
```

The harmonizer calls the manifest CSV `csv_path`. The dataset class calls it `manifest_csv_path`.

The extra columns are `body_part` (`BodyPartExamined`), `study_date` (`StudyDate`), `age` (the source
text, for example `45`) and `manufacturer` (`Manufacturer`). Use them to filter the table without a new
scan.

<!-- args: radharmony.harmonizer.EmoryCHORUSHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | | The `images_dicom_xray/` folder |
| `csv_path` | `str` | No | `None` | The manifest CSV. The harmonizer reads it when it exists. Otherwise it scans the DICOM files and writes it |
| `chest_only` | `bool` | No | `True` | Keep only the chest X-rays |
| `num_workers` | `int` | No | `12` | The number of processes that read the DICOM headers in a scan |

## Known data problems

- Manufacturer: one maker has more than one spelling, for example `CARESTREAM HEALTH` and
  `Carestream Health`, or `GE Healthcare` with and without quote marks. The table keeps the source text.
- Study date: `study_date` has the form `2020-01-31 00:00:00`, not the DICOM form `20200131`. The DICOM
  headers of this subset store the date in this form, and the harmonizer does not change it.
- Chest filter: the filter reads the header text, not the image. So an image with the body part `ABDOMEN`
  or `RIB` stays when its study description names the chest.
