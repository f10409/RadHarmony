# ROCO and ROCOv2

<!-- claim: id=rocov2.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py rocov2 rocov2_train.images" expect="59,962" -->
<!-- claim: id=rocov2.validation.images tier=ci cmd="python scripts/evalkit/dataset_stat.py rocov2 rocov2_validation.images" expect="9,904" -->
<!-- claim: id=rocov2.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py rocov2 rocov2_test.images" expect="9,927" -->
| | |
|---|---|
| Registry keys | ROCO: `roco_train`, `roco_validation`, `roco_test`. ROCOv2: `rocov2_train`, `rocov2_validation`, `rocov2_test` |
| Task and outputs | Image captioning: the caption (`answer`). `question` is empty |
| Images in the table | ROCOv2: train 59,962 images, validation 9,904 images, test 9,927 images. One image has one caption. ROCO: no count, because the RadHarmony tests cannot read the source files |
| Modality | Figures from PubMed Central articles: CT, X-ray, MRI, ultrasound, angiography and more. 2-D, JPEG. With `radiology_only=False`, ROCO also gives figures that are not radiology |
| License | ROCO: the roco-dataset repository states no license. ROCOv2: CC BY-NC-SA 4.0 |
| Download | [Get the data: other datasets](../../../how-to/get-the-data.md#other-datasets) |

ROCO ([Pelka et al., LABELS 2018](https://doi.org/10.1007/978-3-030-01364-6_20)) and ROCOv2
([Rückert et al., Scientific Data 2024](https://doi.org/10.1038/s41597-024-03496-6)) pair figures from PubMed
Central articles with their captions. ROCOv2 is the next version of ROCO. It has only radiology figures, and
each image has UMLS concepts (CUIs). The classes give the caption as `answer`, not as `report`.
`ReportGenerationEvaluator` reads `report`, so it cannot score these classes.

## Quick use

### ROCO

Clone [razorx89/roco-dataset](https://github.com/razorx89/roco-dataset) and run its `scripts/fetch.py`. It
downloads the images from PubMed Central. `base_dir` is the folder of the clone.

```
roco-dataset/
  data/train/radiology/
    captions.txt          # ROCO_00001<TAB><caption>
    keywords.txt          # ROCO_00001<TAB><TAB><keyword><TAB><keyword>...
    images/ROCO_00001.jpg
  data/train/non-radiology/...
  data/validation/...
  data/test/...
```

```python
import torch
from radharmony.dataset import ROCOTrainDataset, ROCOTestDataset

train_ds = ROCOTrainDataset(base_dir="/data/roco-dataset/", dtype=torch.float32).get_datasets()
test_ds = ROCOTestDataset(base_dir="/data/roco-dataset/", dtype=torch.float32).get_datasets()
sample = train_ds[0]
# sample keys: img, question, answer, patient_id, study_id, question_id
# sample["img"]: Tensor(3, 224, 224)
# sample["question"]: ''
```

### ROCOv2

The source stores the images in parquet files. Run `scripts/materialize_rocov2.py --base-dir /data/ROCOv2/`
once. It writes each image to `images/<split>/<image_id>.jpg`.

```
ROCOv2/
  hf/data/train-00000-of-00027.parquet ...   # image, image_id, caption, cui
  hf/data/validation-*.parquet, hf/data/test-*.parquet
  train_concepts_manual.csv, valid_concepts_manual.csv, test_concepts_manual.csv
  images/test/ROCOv2_2023_test_000001.jpg    # made by scripts/materialize_rocov2.py
```

```python
import torch
from radharmony.dataset import ROCOv2TrainDataset, ROCOv2ValidationDataset, ROCOv2TestDataset

train_ds = ROCOv2TrainDataset(base_dir="/data/ROCOv2/", dtype=torch.float32).get_datasets()
val_ds = ROCOv2ValidationDataset(base_dir="/data/ROCOv2/", dtype=torch.float32).get_datasets()
test_ds = ROCOv2TestDataset(base_dir="/data/ROCOv2/", dtype=torch.float32).get_datasets()
sample = test_ds[0]
# sample keys: img, question, answer, patient_id, study_id, question_id
# sample["img"]: Tensor(3, 224, 224)
# sample["answer"]: 'CT chest axial view showing a huge ascending aortic aneurysm (*).'
```

## How it is harmonized

### ROCO

- Source: `captions.txt` and `keywords.txt` of each split and subset folder. One row of the harmonized table
  is one image and its caption (D-03). With `radiology_only=True` (the default), the harmonizer reads only
  the `radiology/` folders.
- IDs: the source has no patient or study ID, so `patient_id`, `study_id` and `question_id` are all the ROCO
  ID, for example `ROCO_00002`. One image is one patient.
- `image_path`: `data/<split>/<subset>/<image_subdir>/<ROCO ID>.jpg`, for example
  `data/train/radiology/images/ROCO_00002.jpg`. It is relative to `base_dir` (D-04).
- Caption: `answer` is the text after the ID. A caption with a tab in it keeps the tab. `question` is `""`.
- Keywords: `keywords` joins the keywords of `keywords.txt` with tabs. Use `.split("\t")` for a list. The
  harmonizer does not read `cuis.txt` and `semtypes.txt`.
- View, age and sex: the source has none of them, so the harmonized table has no `view_position`,
  `age_years` or `sex` (D-01).
- Missing folders (D-35): a split folder with no `captions.txt` gives no rows and a warning that names it.
  When no folder has a `captions.txt`, the harmonizer raises `FileNotFoundError`.
- Dropped rows: the harmonizer drops a `captions.txt` row with no ID or no caption.

### ROCOv2

- Source: the parquet files `hf/data/<split>-*.parquet` (columns `image_id`, `caption` and `cui`) and the
  manual concept file of the split. One row of the harmonized table is one image and its caption (D-03).
- IDs: the source has no patient ID, so `patient_id`, `study_id` and `question_id` are all `image_id`, for
  example `ROCOv2_2023_test_000001`. One image is one patient. Several figures can come from one article.
- `image_path`: `<split>/<image_id>.jpg`, for example `test/ROCOv2_2023_test_000001.jpg`. It is relative to
  `base_image_dir`, which is `<base_dir>/images` by default (D-04).
- Caption: `answer` is `caption`. `question` is `""`.
- Concepts: with `manual_concepts_only=True` (the default), `cuis` comes from `<split>_concepts_manual.csv`
  (`valid_` for the validation split). An image that the file does not list takes the CUIs of the parquet.
  With no file for the split, or with `False`, every image takes the CUIs of the parquet.
- Modality: the harmonizer maps 13 CUIs to 7 modalities. `modality` is the one modality of the CUIs, or
  None when they name none or more than one.
- View, age and sex: the source has none of them, so the harmonized table has no `view_position`,
  `age_years` or `sex` (D-01).
- Dropped rows: by default, the harmonizer drops no row. The argument `modalities` keeps only the modalities
  in its list, and `require_modality=True` drops the rows with no `modality`.

## Captions

The text columns of the harmonized table:

| Column | Dataset | Meaning | Source |
|---|---|---|---|
| `question` | Both | Always `""`, because this is a captioning set | None |
| `answer` | Both | The figure caption | ROCO: `captions.txt`. ROCOv2: `caption` of the parquet |
| `split` | Both | `train`, `validation` or `test` | The folder (ROCO) or the parquet file name (ROCOv2) |
| `keywords` | ROCO | The keywords, joined with tabs. `""` when the file has no row for the image | `keywords.txt` |
| `subset` | ROCO | `radiology` or `non-radiology` | The folder |
| `cuis` | ROCOv2 | A list of UMLS CUIs | The manual concept file, or the parquet `cui` |
| `modality_source` | ROCOv2 | `manual` when the manual concept file lists the image, else `auto` | The harmonizer |
| `modality` | ROCOv2 | `ct`, `mri`, `xray`, `ultrasound`, `angiography`, `pet_spect` or `mammography`. None when the CUIs name no modality or more than one | The harmonizer, from `cuis` |
| `n_modality_cuis` | ROCOv2 | The number of modalities that the CUIs name | The harmonizer, from `cuis` |

ROCOv2 modalities: in the source files, no image names 2 modalities and no image has the mammography CUI. In
the test split, 105 images get no `modality`, and 44 get `pet_spect`.

Example row of `rocov2_test`:

| Column | Value |
|---|---|
| `question_id` | `ROCOv2_2023_test_000001` |
| `answer` | `CT chest axial view showing a huge ascending aortic aneurysm (*).` |
| `cuis` | `['C0040405']` |
| `modality` | `ct` |
| `modality_source` | `manual` |

The sample holds `img`, `question`, `answer`, `patient_id`, `study_id` and `question_id`. `img` is
`(3, 224, 224)`: the figures can be in color, so the classes set `IMG_CHANNELS = 3` (D-42). A gray file gives
3 equal channels. The extra columns (`keywords`, `cuis`, `modality` and the others) are only in the
harmonized table, not in the sample.

## Splits

<!-- claim: id=rocov2.splits.train tier=ci cmd="python scripts/evalkit/dataset_stat.py rocov2 rocov2_train.images" expect="59,962" -->
<!-- claim: id=rocov2.splits.validation tier=ci cmd="python scripts/evalkit/dataset_stat.py rocov2 rocov2_validation.images" expect="9,904" -->
<!-- claim: id=rocov2.splits.test tier=ci cmd="python scripts/evalkit/dataset_stat.py rocov2 rocov2_test.images" expect="9,927" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `ROCOTrainDataset` | `roco_train` | | `data/train/` |
| `ROCOValidationDataset` | `roco_validation` | | `data/validation/` |
| `ROCOTestDataset` | `roco_test` | | `data/test/`. Use it as the test split |
| `ROCODataset` | `roco` | | All images. It mixes the 3 splits, so do not use it for a benchmark (D-09) |
| `ROCOv2TrainDataset` | `rocov2_train` | 59,962 | `hf/data/train-*.parquet` |
| `ROCOv2ValidationDataset` | `rocov2_validation` | 9,904 | `hf/data/validation-*.parquet` |
| `ROCOv2TestDataset` | `rocov2_test` | 9,927 | `hf/data/test-*.parquet`. Use it as the test split |
| `ROCOv2Dataset` | `rocov2` | | All images. It mixes the 3 splits, so do not use it for a benchmark (D-09) |

The RadHarmony tests cannot read the ROCO source files, so no test checks the harmonized ROCO table, and
the table above gives no ROCO count.

<!-- claim: id=rocov2.patients_in_two_splits tier=ci cmd="python scripts/evalkit/dataset_stat.py rocov2 patients_in_two_splits" expect="0" -->
The ROCOv2 splits share 0 patients (D-34). Each ROCOv2 image ID holds its split name, so no image is in 2
splits. `get_datasets(n_splits=5)` makes a train and val split from the train class. It
groups by patient (D-10). The source has no patient ID, so one image is one patient, and nobody can check
that a patient is in only 1 split. Several figures of one article can go to both train and val.

## Arguments

### ROCO

The arguments are the same for the 3 ROCO split classes and `ROCODataset`.

<!-- args: radharmony.dataset.ROCOTrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_dir` | `str` | Yes | `None` | The roco-dataset folder, with `data/` in it. Not needed with `harmonizer_path`, which holds it |
| `splits` | `list[str]` | No | `None` | In a split class, `None` gives its split, for example `["train"]`, and another value raises `ValueError` (D-09). In the mixed class, `None` gives all 3 splits |
| `radiology_only` | `bool` | No | `None` | `True` reads only the `radiology/` folders, and `False` also reads `non-radiology/`. `None` gives `True` when the class harmonizes. On a given table, `True` keeps only the `radiology` rows, and `None` and `False` keep every row (D-02) |
| `image_subdir` | `str` | No | `"images"` | The image folder in each split and subset folder. It must match `--subdir` of the roco-dataset `scripts/fetch.py` |

#### Shared arguments (inherited from BaseVQADataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives the VQA steps. They load `img` with 3 channels (D-42), and resize and pad it to 224 x 224 (H x W). They have no `SelectItemsD`, so the text keys stay in the sample |
| `output_struct` | `bool` | No | `False` | Add `answer_struct` to the sample. The harmonized table has no `answer_struct` column, so the class warns and adds no key (D-11) |
| `output_question_type` | `bool` | No | `False` | Add `question_type` to the sample. The harmonized table has no `question_type` column, so the class warns and adds no key (D-11) |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The text keys stay strings |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer (`.pkl`) of the same split. A file of another class raises `ValueError`. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

### ROCOv2

The arguments are the same for the 3 ROCOv2 split classes and `ROCOv2Dataset`.

<!-- args: radharmony.dataset.ROCOv2TrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_dir` | `str` | Yes | `None` | The ROCOv2 folder, with `hf/data/` and the manual concept files in it. Not needed with `harmonizer_path`, which holds it |
| `base_image_dir` | `str` | No | `None` | The folder of the image files. With `harmonizer_path`, `None` gives the `base_image_dir` of the saved harmonizer. Else, or when the saved harmonizer has none, it gives `<base_dir>/images`. With `harmonizer_path` and no `base_dir`, the class takes `base_dir` from the saved harmonizer |
| `splits` | `list[str]` | No | `None` | In a split class, `None` gives its split, for example `["train"]`, and another value raises `ValueError` (D-09). In the mixed class, `None` gives all 3 splits |
| `modalities` | `list[str]` | No | `None` | Keep only these modalities, for example `["ct", "mri"]`. An unknown name raises `ValueError`. It also filters a given table (D-02) |
| `require_modality` | `bool` | No | `False` | Drop the images with no `modality`. It also filters a given table (D-02) |
| `manual_concepts_only` | `bool` | No | `True` | Take the CUIs of the manual concept file where it lists the image. `False` takes the CUIs of the parquet for every image. Only `harmonize()` uses it, so it does not change a given table |

#### Shared arguments (inherited from BaseVQADataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives the VQA steps. They load `img` with 3 channels (D-42), and resize and pad it to 224 x 224 (H x W). They have no `SelectItemsD`, so the text keys stay in the sample |
| `output_struct` | `bool` | No | `False` | Add `answer_struct` to the sample. The harmonized table has no `answer_struct` column, so the class warns and adds no key (D-11) |
| `output_question_type` | `bool` | No | `False` | Add `question_type` to the sample. The harmonized table has no `question_type` column, so the class warns and adds no key (D-11) |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The text keys stay strings |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer (`.pkl`) of the same split. A file of another class raises `ValueError`. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

### ROCO

```python
from radharmony.harmonizer import ROCOTrainHarmonizer

h = ROCOTrainHarmonizer(base_dir="/data/roco-dataset/")
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'question_id', 'image_path', 'question', 'answer', 'split', 'keywords', 'subset']
```

`ROCOValidationHarmonizer` and `ROCOTestHarmonizer` take the same arguments.

<!-- args: radharmony.harmonizer.ROCOTrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_dir` | `str` | Yes | `None` | The roco-dataset folder, with `data/` in it. Without it, `harmonize()` raises `ValueError` |
| `splits` | `list[str]` | No | `None` | `None` gives the split of the class. Another value raises `ValueError` |
| `radiology_only` | `bool` | No | `True` | Read only the `radiology/` folders |
| `image_subdir` | `str` | No | `"images"` | The image folder in each split and subset folder |

### ROCOv2

```python
from radharmony.harmonizer import ROCOv2TestHarmonizer

h = ROCOv2TestHarmonizer(base_dir="/data/ROCOv2/", base_image_dir="/data/ROCOv2/images/")
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'question_id', 'image_path', 'question', 'answer', 'split', 'cuis', 'modality',
#  'modality_source', 'n_modality_cuis']
```

`ROCOv2TrainHarmonizer` and `ROCOv2ValidationHarmonizer` take the same arguments. Only the dataset class sets
`base_image_dir` to `<base_dir>/images` when it is `None`. The harmonizer keeps `None`.

<!-- args: radharmony.harmonizer.ROCOv2TestHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_dir` | `str` | Yes | `None` | The ROCOv2 folder. Without it, `harmonize()` raises `ValueError`. A split with no parquet file raises `FileNotFoundError` |
| `base_image_dir` | `str` | No | `None` | The folder of the image files. `harmonize()` does not read it. `verify_images()` and a saved harmonizer use it |
| `splits` | `list[str]` | No | `None` | `None` gives the split of the class. Another value raises `ValueError` |
| `modalities` | `list[str]` | No | `None` | Keep only these modalities |
| `require_modality` | `bool` | No | `False` | Drop the images with no `modality` |
| `manual_concepts_only` | `bool` | No | `True` | Take the CUIs of the manual concept file where it lists the image |

## Known data problems

<!-- claim: id=rocov2.paper.train source="Rückert et al., Scientific Data 2024" expect="59,958" -->
<!-- claim: id=rocov2.train.images.again tier=ci cmd="python scripts/evalkit/dataset_stat.py rocov2 rocov2_train.images" expect="59,962" -->
- ROCOv2 train: the paper gives 59,958 train images. The Hugging Face parquet files have 59,962.
- ROCOv2 validation: the caption of `ROCOv2_2023_valid_003902` is only a narrow no-break space (U+202F).

<!-- claim: id=rocov2.source.valid_manual tier=nas cmd="python scripts/evalkit/dataset_stat.py --source rocov2_validation base_dir/valid_concepts_manual.csv rows" expect="9,945" -->
<!-- claim: id=rocov2.source.test_manual tier=nas cmd="python scripts/evalkit/dataset_stat.py --source rocov2_test base_dir/test_concepts_manual.csv rows" expect="9,972" -->
- ROCOv2 manual concept files: `valid_concepts_manual.csv` lists 9,945 images and `test_concepts_manual.csv`
  lists 9,972, more than the parquet files have. `ROCOv2_2023_valid_003902` is not in
  `valid_concepts_manual.csv`, so its `modality_source` is `auto`.
- ROCO: none known. The maintainers could not read the source files to check them.
