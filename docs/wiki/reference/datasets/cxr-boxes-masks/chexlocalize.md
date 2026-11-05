# CheXlocalize

<!-- claim: id=chexlocalize.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chexlocalize chexlocalize.images" expect="668" -->
<!-- claim: id=chexlocalize.test.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py chexlocalize chexlocalize.patients" expect="500" -->
<!-- claim: id=chexlocalize.val.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chexlocalize chexlocalize_val.images" expect="234" -->
<!-- claim: id=chexlocalize.val.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py chexlocalize chexlocalize_val.patients" expect="200" -->
| | |
|---|---|
| Registry keys | `chexlocalize`, `chexlocalize_val` |
| Task and outputs | Classification, 14 labels (`cls`). Segmentation, one mask for each image (`mask`) |
| Images in the table | test 668 (500 patients), val 234 (200 patients) |
| Modality | Chest X-ray, 2-D, JPEG |
| License | Stanford AIMI research use agreement, as for CheXpert |
| Download | [Get the data: other datasets](../../../how-to/get-the-data.md#other-datasets) |

<!-- claim: id=chexlocalize.test.mask_images tier=nas cmd="python scripts/evalkit/dataset_stat.py --source chexlocalize mask_json_path rows" expect="499" -->
CheXlocalize adds segmentation masks to the CheXpert test set and the CheXpert valid set. Radiologists
drew the masks of 10 findings. The test split is the official CheXpert test set, so the
[CheXpert](../cxr-labels/chexpert.md) page points here. The mask file of the test split has 499 of the 668
test images. RadHarmony joins the 10 masks of an image into one mask, so a sample has one "any finding"
mask.

## Quick use

`base_image_dir` is the `test/` or `val/` folder. The class finds the label CSV in the folder above it. It
finds the mask file in the `CheXlocalize/` folder next to the `CheXpert/` folder (D-35). The images and the
masks are 2 downloads. Put them side by side, as below.

```
chexlocalize/
  CheXpert/
    test_labels.csv
    val_labels.csv
    test/patient64741/study1/view1_frontal.jpg
    val/patient64541/study1/view1_frontal.jpg
  CheXlocalize/
    gt_segmentations_test.json
    gt_segmentations_val.json
```

```python
import torch
from radharmony.dataset import CheXlocalizeDataset, CheXlocalizeValDataset

val_ds = CheXlocalizeValDataset(
    base_image_dir="/data/chexlocalize/CheXpert/val/",
    mask_output_dir="/data/chexlocalize_masks/val/",  # any folder you can write to
    output_cls=True,
    output_mask=True,
    dtype=torch.float32,
).get_datasets()
sample = val_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(14,)
# sample["mask"]: Tensor(1, 224, 224), values 0 and 1

test_ds = CheXlocalizeDataset(
    base_image_dir="/data/chexlocalize/CheXpert/test/",
    mask_output_dir="/data/chexlocalize_masks/test/",  # any folder you can write to
    output_cls=True,
    output_mask=True,
    dtype=torch.float32,
).get_datasets()
```

## How it is harmonized

- Source: `test_labels.csv` or `val_labels.csv`. One row is one image.
- IDs: the `Path` `test/patient64741/study1/view1_frontal.jpg` gives `patient_id` `patient64741`,
  `study_id` `patient64741_study1`, and `image_path` `patient64741/study1/view1_frontal.jpg`. `image_path`
  is relative to `base_image_dir` (D-04). In `val_labels.csv`, the harmonizer also removes
  `CheXpert-v1.0/valid/` from the start of `Path`. With a CSV of another split, it raises `ValueError`.
- Labels (14 columns, D-05): `1` gives 1 and `0` gives 0. Both CSV files have only 0 and 1.
- View (D-01): `FRONTAL` or `LATERAL`, from the file name. `val_labels.csv` has AP or PA in its column
  `AP/PA`, but the harmonizer does not read it. D-01 lists this as a known gap.
- Age and sex: only in the val split. `age_years` comes from `Age`, in years (D-06). `sex` is `M` or `F`,
  from `Sex` (D-26). `test_labels.csv` has no age and no sex.
- Masks: only with `mask_json_path`. See [Masks](#masks).
- Dropped rows: the harmonizer drops no row. With `output_mask=True`, the class drops a row with an empty
  `mask_path`, and warns (D-33).

## Labels

The 14 label columns, in the order of `cls`. The masks cover the 10 findings with "Yes" in the column
Mask.

| Column | Meaning | Mask |
|---|---|---|
| `atelectasis` | Part of a lung is collapsed | Yes |
| `cardiomegaly` | The heart is large | Yes |
| `consolidation` | Airspace consolidation | Yes |
| `edema` | Pulmonary edema | Yes |
| `enlarged_cardiomediastinum` | The mediastinum is wide | Yes |
| `fracture` | A rib or other bone fracture | No |
| `lung_lesion` | A lung lesion | Yes |
| `lung_opacity` | A lung opacity | Yes |
| `no_finding` | No finding | No |
| `pleural_effusion` | Pleural effusion | Yes |
| `pleural_other` | Another pleural finding | No |
| `pneumonia` | Pneumonia | No |
| `pneumothorax` | Pneumothorax | Yes |
| `support_devices` | A tube, line or other device | Yes |

## Masks

<!-- claim: id=chexlocalize.val.mask_images tier=nas cmd="python scripts/evalkit/dataset_stat.py --source chexlocalize_val mask_json_path rows" expect="187" -->
<!-- claim: id=chexlocalize.test.rows_with_mask tier=ci cmd="python scripts/evalkit/dataset_stat.py chexlocalize chexlocalize.rows_with_mask" expect="667" -->
- Source: `gt_segmentations_test.json` and `gt_segmentations_val.json`. The key of an image is
  `<study_id>_<file name without .jpg>`, for example `patient64744_study1_view1_frontal`. Each image in the
  file has 10 masks in COCO RLE, one for each masked finding. The mask of an absent finding is empty. The
  test file has 499 images, and the val file has 187.
- The harmonizer joins the 10 masks of an image with a bitwise OR. So the mask shows the area of any
  finding. It does not tell the findings apart.
- `mask_path`: with `output_mask=True`, the class passes the mask file to the harmonizer. It then writes the
  masks as PNG files to `mask_output_dir`. Each PNG has the size of its image and the values 0
  and 255. Example: `<mask_output_dir>/patient64541/study1/view1_frontal.png`. Without `output_mask=True`,
  the class does not read the mask file, so the harmonized table has no `mask_path` column.
- An image that is not in the mask file: if all 10 masked findings are 0, the image is a labeled negative. It
  gets an all-zero mask (D-33). The test split has 168 labeled negatives, and the val split has 47.
- An empty `mask_path` means "not annotated" (D-33). With `output_mask=True`, the class drops these rows and
  warns. Only 1 test image has an empty `mask_path`, so 667 test and 234 val samples stay.
- The sample: `mask` is a float32 tensor `(1, 224, 224)` with the values 0 and 1 (D-14). The transform
  resizes it with the image.

If you need one mask for each finding, read the JSON file with `pycocotools`. The class does not give them.

## Splits

<!-- claim: id=chexlocalize.splits.test tier=ci cmd="python scripts/evalkit/dataset_stat.py chexlocalize chexlocalize.images" expect="668" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `CheXlocalizeDataset` | `chexlocalize` | 668 | `test_labels.csv`, the CheXpert test set. Use it as the test split |
| `CheXlocalizeValDataset` | `chexlocalize_val` | 234 | `val_labels.csv`, the CheXpert valid set |

No patient is in 2 splits. The val split has the same 234 images as `chexpert_valid` on the
[CheXpert](../cxr-labels/chexpert.md) page.

## Arguments

The arguments are the same for `CheXlocalizeDataset` and `CheXlocalizeValDataset`.

<!-- args: radharmony.dataset.CheXlocalizeDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `test/` or `val/` folder. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `test_labels.csv` or `val_labels.csv`. Without it, the class looks in `base_image_dir` and the folder above it |
| `mask_json_path` | `str` | No | `None` | `gt_segmentations_test.json` or `gt_segmentations_val.json`. Without it, the class looks in `base_image_dir`, the folder above it, and the `CheXlocalize/` folder next to the `CheXpert/` folder. With `output_mask=True`, a missing file raises `FileNotFoundError`. Without it, the class does not read the file |
| `mask_output_dir` | `str` | No | `None` | The folder for the mask PNG files. Needed with `output_mask=True`. Without it, the class raises `ValueError` |
| `mask_num_cores` | `int` | No | `1` | The number of threads that decode the masks |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample |
| `output_mask` | `bool` | No | `False` | Add the `mask` tensor to the sample. It needs `mask_output_dir`. The class drops the rows with an empty `mask_path`, and warns (D-33) |
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
from radharmony.harmonizer import CheXlocalizeHarmonizer

h = CheXlocalizeHarmonizer(
    csv_path="/data/chexlocalize/CheXpert/test_labels.csv",
    mask_json_path="/data/chexlocalize/CheXlocalize/gt_segmentations_test.json",
    base_image_dir="/data/chexlocalize/CheXpert/test/",
)
df = h.harmonize(mask_output_dir="/data/chexlocalize_masks/test/")  # writes the mask PNGs
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'view_position', 'mask_path', 'atelectasis', ...]
```

`CheXlocalizeValHarmonizer` takes the same arguments, with `val_labels.csv`, `gt_segmentations_val.json`
and the `val/` folder. Its harmonized table also has `age_years` and `sex`.

<!-- args: radharmony.harmonizer.CheXlocalizeHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `test_labels.csv` or `val_labels.csv` |
| `mask_json_path` | `str` | No | `None` | The mask file. Without it, the harmonized table has no `mask_path` column |
| `base_image_dir` | `str` | No | `None` | The `test/` or `val/` folder. `harmonize(mask_output_dir=...)` reads the image sizes from it to make the mask PNGs |

## Known data problems

- Missing masks: 1 test image, `patient65086/study1/view1_frontal.jpg`, has 5 positive masked findings in
  `test_labels.csv`, but it is not in `gt_segmentations_test.json`. So it has no mask (D-33).
- Finding name: the mask files write `Airspace Opacity` for the label `Lung Opacity`.
