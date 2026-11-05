# RSNA Pneumonia

| | |
|---|---|
| Registry keys | `rsna_pneumonia` |
| Task and outputs | Classification, 3 labels (`cls`). Boxes (`bbox`) of lung opacities |
| Images in the table | Not counted. The RadHarmony tests cannot read the source files, so no test checks the harmonized table |
| Modality | Chest X-ray, 2-D, DICOM |
| License | The terms of the RSNA download page |
| Download | [Get the data: RSNA Pneumonia (official release)](../../../how-to/get-the-data.md#rsna-pneumonia-official-release) |

The RSNA Pneumonia Detection Challenge (2018) has chest X-rays from
[ChestX-ray14](../cxr-labels/chestxray14.md), with 3 labels and boxes on the lung opacities. This page is
about the adjudicated JSON release from the RSNA website. The JSON file puts the annotations in label groups.
For example, `Team 1` and `Team 2` are reader teams, and `Calculated` is the final consensus. The Kaggle
release of the same challenge has its own page: [RSNA Pneumonia (Kaggle)](rsna_pneumonia_kaggle.md). The
source files are not available to the maintainers, so this page has no real example.

## Quick use

`base_image_dir` is the folder with the `<StudyInstanceUID>/` folders. The class finds the JSON file in this
folder or in the folder above it.

```
rsna/
  pneumonia-challenge-annotations-adjudicated-kaggle_2018.json
  <StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
```

```python
import torch
from radharmony.dataset import RSNAPneumoniaDataset

ds = RSNAPneumoniaDataset(
    base_image_dir="/data/rsna/",
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(3,)
# sample["bbox"]: Tensor(N, 4), N is 0 for an image with no box
# sample["bbox_labels"]: a list of N names
```

## How it is harmonized

- Source: the JSON file `pneumonia-challenge-annotations-adjudicated-kaggle_2018.json`. `csv_path` is this
  JSON file. The harmonizer keeps the annotations of one label group, by default `Calculated` (the final
  consensus). One row is one image (`SOPInstanceUID`).
- IDs: `patient_id` and `study_id` are the `StudyInstanceUID` of the annotation. `image_path` is
  `<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`, relative to `base_image_dir` (D-04).
- Labels (3 columns, D-05): an annotation with a label gives 1 in the column of that label. The other columns
  stay 0. In `Calculated`, the label IDs `L_v8n`, `L_yd0` and `L_o8w` give, in this order,
  `lung_opacity`, `no_lung_opacity_/_not_normal` and `normal`.
- View, age and sex: the source has none of them, so the harmonized table has no `view_position`,
  `age_years` or `sex` (D-01).
- Boxes: from the annotations with a `data` field. See [Boxes](#boxes).
- Dropped rows: the harmonizer drops an image with no annotation in the label group.

## Labels

The 3 label columns, in the order of `cls`:

| Column | Meaning |
|---|---|
| `lung_opacity` | A lung opacity that can be pneumonia. Only these annotations have boxes |
| `no_lung_opacity_/_not_normal` | No lung opacity, but the image is not normal |
| `normal` | A normal chest X-ray |

## Boxes

- Source: an annotation of a lung opacity has `data` with `x`, `y`, `width` and `height` of the box, in
  pixels. The annotation also has `width` and `height` of the image.
- The harmonizer divides `data.y` and `data.height` by the image `height`, and `data.x` and `data.width` by
  the image `width`. So the box is `[data.y / height, (data.y + data.height) / height, data.x / width,
  (data.x + data.width) / width]`.
- The box is `[y_min, y_max, x_min, x_max]`, in fractions of the image size (D-07).
- `bbox_labels` has the snake_case name of the label of each box, `lung_opacity`. An image with no box has
  `[]` in `bbox` and in `bbox_labels` (D-07).
- Readers (D-36): the harmonizer uses the boxes of 1 label group only, so it does not group the
  boxes of several teams.
- The sample: `bbox` is a float32 tensor `(N, 4)`, in fractions of the returned image (D-28). `bbox_labels`
  is a list of N names. An image with no box gives `(0, 4)` and `[]`.
- Dropped boxes: the harmonizer clips each box to the image. It drops a box with no area after
  the clip, and warns (D-07).

## Splits

| Class | Registry key | Images | Source |
|---|---|---|---|
| `RSNAPneumoniaDataset` | `rsna_pneumonia` | Not counted | The images in the label group of the JSON file |

RSNA Pneumonia has no official split. `get_datasets(n_splits=5)` makes a train and val split. It groups by
patient (D-10). Here `patient_id` is the `StudyInstanceUID`, so the split groups by study.

## Arguments

<!-- args: radharmony.dataset.RSNAPneumoniaDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The folder with the `<StudyInstanceUID>/` folders. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | The JSON file. Without it, the class looks in `base_image_dir` and the folder above it |
| `label_group` | `str` | No | `"Calculated"` | The label group of the JSON file to use. `"Calculated"` is the final consensus |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_report` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Add `bbox` and `bbox_labels` to the sample |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The labels stay float32 |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import RSNAPneumoniaHarmonizer

h = RSNAPneumoniaHarmonizer(
    csv_path="/data/rsna/pneumonia-challenge-annotations-adjudicated-kaggle_2018.json",
    base_image_dir="/data/rsna/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'bbox', 'bbox_labels', 'lung_opacity', ...]
```

<!-- args: radharmony.harmonizer.RSNAPneumoniaHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | The JSON file `pneumonia-challenge-annotations-adjudicated-kaggle_2018.json` |
| `base_image_dir` | `str` | No | `None` | The folder with the `<StudyInstanceUID>/` folders |
| `label_group` | `str` | No | `"Calculated"` | The label group of the JSON file to use. Another name that is not in the file raises `ValueError` |

## Known data problems

None known. The maintainers could not read the source files to check them.
