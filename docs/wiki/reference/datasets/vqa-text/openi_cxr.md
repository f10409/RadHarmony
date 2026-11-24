# OpenI Indiana University CXR (IU X-Ray)

<!-- claim: id=openi_cxr.images tier=ci cmd="python scripts/evalkit/dataset_stat.py openi_cxr openi_cxr.images" expect="7,470" -->
<!-- claim: id=openi_cxr.studies tier=ci cmd="python scripts/evalkit/dataset_stat.py openi_cxr openi_cxr.studies" expect="3,851" -->
| | |
|---|---|
| Registry keys | `openi_cxr` |
| Task and outputs | Classification, 8 labels (`cls`). The report text (`report`) |
| Images in the table | 7,470 (3,851 reports). One report is one study and one patient |
| Modality | Chest X-ray, 2-D, PNG |
| License | Open access from the National Library of Medicine (NLM). Read the terms of the Open-i download page |
| Download | [Get the data: OpenI IU CXR](../../../how-to/get-the-data.md#openi-iu-cxr) |

<!-- claim: id=openi_cxr.paper.reports source="Demner-Fushman et al., JAMIA 2016" expect="3,955" -->
<!-- claim: id=openi_cxr.studies.again tier=ci cmd="python scripts/evalkit/dataset_stat.py openi_cxr openi_cxr.studies" expect="3,851" -->
<!-- claim: id=openi_cxr.images.again tier=ci cmd="python scripts/evalkit/dataset_stat.py openi_cxr openi_cxr.images" expect="7,470" -->
OpenI IU X-Ray has chest X-rays and radiology reports from the Indiana University hospital network
([Demner-Fushman et al., JAMIA 2016](https://doi.org/10.1093/jamia/ocv080)). The NLM released it on Open-i.
The source has 3,955 XML reports. 104 reports have no image, so 3,851 reports with 7,470 images stay. A
report has about 2 images, a frontal and a lateral. The 8 labels come from the MeSH terms of each report.

## Quick use

`base_image_dir` is the dataset root, with `ecgen-radiology/` and `images/`. Put the PNG files of
`NLMCXR_png.tgz` in `images/`. The DICOM files of `NLMCXR_dcm.tgz` are optional. The harmonizer reads their
`ViewPosition` tag to correct the view.

```
OpenI-IU-CXR/
  ecgen-radiology/1.xml ... 3999.xml   # the numbers have gaps
  images/CXR1_1_IM-0001-3001.png
  1/1_IM-0001-3001.dcm          # optional DICOM files, one folder for each report
```

```python
import torch
from radharmony.dataset import OpenICXRDataset

ds = OpenICXRDataset(
    base_image_dir="/data/OpenI-IU-CXR/",
    output_cls=True,
    output_report=True,
    dtype=torch.float32,
)
train_ds, val_ds = ds.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 224, 224)
# sample["cls"]: Tensor(8,)
# sample["report"]: 'The cardiac silhouette and mediastinum size are within normal limits. ...'
```

`img` has 1 channel. The PNG files are RGB, and the class makes them gray (D-42).

## How it is harmonized

- Source: the XML reports `ecgen-radiology/<n>.xml`. There is no CSV. Each `parentImage` of a report gives
  one row, so one row is one image (D-03).
- IDs: report `1.xml` gives `patient_id` `CXR1` (from `uId`) and `study_id` `1` (from `pmcId`). The source
  has no patient ID, so one report is one patient. `image_path` is `images/<parentImage id>.png`, for example
  `images/CXR1_1_IM-0001-3001.png`. It is relative to `base_image_dir` (D-04).
- Extra columns: `dicom_path`, `findings`, `impression`, `indication` and `mesh_major`. For `dicom_path`,
  the `parentImage id` `CXR1_1_IM-0001-3001` gives `1/1_IM-0001-3001.dcm`, and `CXR2_IM-0652-1001` gives
  `2/2_IM-0652-1001.dcm`.
- Labels (D-05): from the MeSH major terms of the report. See [Labels](#labels).
- Report: the FINDINGS and IMPRESSION text of the report. The images of one report get the same text. See
  [Reports](#reports).
- View (D-01): the figure number gives a first view. `F1` gives `FRONTAL`, `F2` gives `LATERAL`, and the
  others give `OTHER`. The DICOM tag `ViewPosition` corrects it. `AP` gives `AP`, and `LL`, `RL`, `LAT`,
  `LATERAL` or `LARGE LATERAL` give `LATERAL`. `PA` gives `PA` only on an `F1` image, because many lateral
  images also have the tag `PA`. Without the DICOM files, the figure number alone gives the view, and the
  harmonizer warns.
- Age and sex: the source has neither, so the harmonized table has no `age_years` or `sex` (D-01).
- Dropped rows: the harmonizer drops a report with no `parentImage` (104 reports). It also skips an XML file
  that it cannot read. The source has no such file.

## Labels

The 8 label columns, in the order of `cls`, with the MeSH triggers and the images with label 1:

<!-- claim: id=openi_cxr.labels.no_finding tier=ci cmd="python scripts/evalkit/dataset_stat.py openi_cxr openi_cxr.positives.no_finding" expect="2,696" -->
| Column | Meaning | MeSH triggers | Images |
|---|---|---|---|
| `atelectasis` | Part of a lung is collapsed | "atelectasis", "pulmonary atelectasis" | 616 |
| `cardiomegaly` | The heart is large | "cardiomegaly" | 655 |
| `consolidation` | Airspace consolidation | "consolidation", "pneumonia", "airspace disease" | 290 |
| `edema` | Pulmonary edema | "pulmonary edema", "pulmonary congestion", "edema" | 183 |
| `no_finding` | No finding | "normal" | 2,696 |
| `pleural_effusion` | Pleural effusion | "pleural effusion", "effusion/pleural" | 286 |
| `pneumothorax` | Pneumothorax | "pneumothorax" | 44 |
| `support_devices` | A tube, line or other device | "catheter", "tube", "medical device", "stent" | 425 |

<!-- claim: id=openi_cxr.labels.nan tier=ci cmd="python scripts/evalkit/dataset_stat.py openi_cxr openi_cxr.uncertain.no_finding" expect="172" -->
- A trigger matches whole words of a MeSH term. The match ignores upper and lower case. Part of a word does
  not match: `tube` does not match `Tuberculosis`, and `normal` does not match `Abnormal`. A final `s` can
  follow, so `tube` matches `Tubes`. A term that matches gives 1, and no match gives 0.
- `support_devices`: a pacemaker study has the term "Implanted Medical Device" or "Medical Device". A device
  term with the word "breast" is a breast implant, so it does not set `support_devices`.
- A report with no MeSH term, or with only the term "No Indexing", gets NaN in every label. 172 images have
  this (D-05).
- `mesh_major` keeps the MeSH major terms as JSON text, for example `["normal"]`.

## Reports

<!-- claim: id=openi_cxr.reports tier=ci cmd="python scripts/evalkit/dataset_stat.py openi_cxr openi_cxr.rows_with_report" expect="7,430" -->
- `report` is the FINDINGS text and the IMPRESSION text, joined with a space. Example:
  `The cardiac silhouette and mediastinum size are within normal limits. ... Normal chest x-XXXX.`
- The harmonizer leaves out a section that is empty, `None` or `None.`. A report with neither section gets
  `report` None.
- `findings`, `impression` and `indication` keep each section alone.
- 7,430 images have a report. With `output_report=True`, the class drops the other rows and warns (D-33).

## Splits

<!-- claim: id=openi_cxr.splits.images tier=ci cmd="python scripts/evalkit/dataset_stat.py openi_cxr openi_cxr.images" expect="7,470" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `OpenICXRDataset` | `openi_cxr` | 7,470 | `ecgen-radiology/*.xml` and `images/` |

OpenI IU X-Ray has no official split. `get_datasets(n_splits=5)` makes a train and val split. It groups by
patient (D-10). The source has no patient ID, so one report is one patient, and nobody can check that a
patient is in only 1 split.

## Arguments

<!-- args: radharmony.dataset.OpenICXRDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The dataset root, with `ecgen-radiology/` and `images/`. Not needed with `harmonizer_path`, which holds it |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform2D` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_report` | `bool` | No | `False` | Add the `report` text to the sample. The class drops the rows with no report, and warns (D-33) |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The labels stay float32 |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import OpenICXRHarmonizer

h = OpenICXRHarmonizer(base_dir="/data/OpenI-IU-CXR/")
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'view_position', 'report', 'atelectasis', ...,
#  'support_devices', 'dicom_path', 'findings', 'impression', 'indication', 'mesh_major']
```

<!-- args: radharmony.harmonizer.OpenICXRHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_dir` | `str` | Yes | | The dataset root, with `ecgen-radiology/` and `images/`. Without `ecgen-radiology/`, `harmonize()` raises `FileNotFoundError` |

## Known data problems

- Labels: the MeSH terms do not always list the findings of the report text. Example: report `157.xml` has
  the impression "Chest. Resolving pulmonary interstitial edema and pulmonary venous hypertension." and the
  only MeSH term "normal". So its images get `no_finding` 1 and `edema` 0.
- View: the figure number is not always right. The DICOM tag shows that 97 `F1` images are lateral, and
  229 `F2` images are AP (D-01).
- Reports: the source replaced some words with `XXXX` to remove personal data, for example
  `Normal chest x-XXXX.`
