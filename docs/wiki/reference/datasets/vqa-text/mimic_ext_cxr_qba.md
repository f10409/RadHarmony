# MIMIC-Ext-CXR-QBA

<!-- claim: id=mimic_ext_cxr_qba.validate.images tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_ext_cxr_qba mimic_ext_cxr_qba_validate.images" expect="1,641" -->
<!-- claim: id=mimic_ext_cxr_qba.validate.rows tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_ext_cxr_qba mimic_ext_cxr_qba_validate.rows" expect="7,620" -->
<!-- claim: id=mimic_ext_cxr_qba.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_ext_cxr_qba mimic_ext_cxr_qba_test.images" expect="2,850" -->
<!-- claim: id=mimic_ext_cxr_qba.test.rows tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_ext_cxr_qba mimic_ext_cxr_qba_test.rows" expect="12,569" -->
| | |
|---|---|
| Registry keys | `mimic_ext_cxr_qba_train`, `mimic_ext_cxr_qba_validate`, `mimic_ext_cxr_qba_test` |
| Task and outputs | Visual question answering: the question (`question`) and the answer (`answer`). `question_type` with `output_question_type=True`, and `answer_struct` with `output_struct=True` |
| Images in the table | validate 1,641 images (7,620 questions), test 2,850 images (12,569 questions). The counts are for the `A_frontal/metadata/q1M` export |
| Modality | Chest X-ray, 2-D, JPEG from [MIMIC-CXR-JPG](../cxr-labels/mimic_cxr_jpg.md) with `image_ext=".jpg"`, or DICOM from [MIMIC-CXR](../cxr-labels/mimic_cxr.md) with the default `image_ext=".dcm"` |
| License | PhysioNet credentialed access (CITI training and a signed data use agreement), for the questions and for the MIMIC-CXR images. The license is the PhysioNet Credentialed Health Data License 1.5.0 |
| Download | [Get the data: other datasets](../../../how-to/get-the-data.md#other-datasets). The images come from [MIMIC-CXR](../../../how-to/get-the-data.md#mimic-cxr-dicom) or [MIMIC-CXR-JPG](../../../how-to/get-the-data.md#mimic-cxr-jpg) |

MIMIC-Ext-CXR-QBA is a visual question answering dataset on the studies of MIMIC-CXR
([PhysioNet](https://physionet.org/content/mimic-ext-cxr-qba/1.0.0/)). The questions come from templates. The
answers come from the report sentences and the scene graph of each study. Each answer has tags and boxes on
the image. The source has parquet files that list the questions, and `qa.zip`, with one `qa.json` file of
text for each study.

The export folders of the source hold subsets of the questions with a given quality. RadHarmony reads one
export folder at a time. The RadHarmony tests use `exports/A_frontal/metadata/q1M`: a random set of 1 million
questions with the quality A, A+ or A++, on frontal images only.

## Quick use

`metadata_dir` is an export folder with the parquet files. `qa_zip_path` is `qa.zip`. `base_image_dir` is
the `files/` folder of the images. The default `image_ext=".dcm"` reads the DICOM files of MIMIC-CXR. With
`image_ext=".jpg"`, use the `files/` folder of MIMIC-CXR-JPG 2.0.0.

```
mimic-ext-cxr-qba/
  qa.zip                                   <- qa_zip_path
  exports/A_frontal/metadata/q1M/          <- metadata_dir
    question_metadata.parquet
    question_image_metadata.parquet
    image_metadata.parquet
    patient_metadata.parquet
mimic-cxr-jpg/2.0.0/files/                 <- base_image_dir
  p17/p17838301/s50037760/0788829b-5419d8e4-5ce8eb81-87a77c03-98c15a1a.jpg
```

```python
import torch
from radharmony.dataset import MIMICExtCXRQBATestDataset, MIMICExtCXRQBATrainDataset

kw = dict(
    base_image_dir="/data/mimic-cxr-jpg/2.0.0/files/",
    metadata_dir="/data/mimic-ext-cxr-qba/exports/A_frontal/metadata/q1M/",
    qa_zip_path="/data/mimic-ext-cxr-qba/qa.zip",
    image_ext=".jpg",
    output_question_type=True,
    output_struct=True,
    dtype=torch.float32,
)
test_ds = MIMICExtCXRQBATestDataset(**kw).get_datasets()
sample = test_ds[0]
# sample keys: img, question, answer, patient_id, study_id, question_id, answer_struct, question_type
# sample["img"]: Tensor(1, 224, 224)
# sample["question"]: 'Describe the lung opacity.'
# sample["answer"]: 'There is linear opacification in the left mid lung.'

# The full train split reads about 200,000 studies in qa.zip and is slow. Try max_studies first.
train = MIMICExtCXRQBATrainDataset(**kw, max_studies=1000)
train_ds, val_ds = train.get_datasets(n_splits=5)  # 1/5 of the patients go to val
```

To batch `answer_struct`, pass `collate_fn=bbox_collate` (`from radharmony.dataset import bbox_collate`) to
the `DataLoader`. The lists of answer dicts have different lengths, so the default collate fails.

## How it is harmonized

- Source: `question_metadata.parquet` lists the questions, one row for each question. One row of the
  harmonized table is one question, not one image (D-03). `qa.zip` holds the text.
- One image for each question: `question_image_metadata.parquet` can list more than 1 image for one
  question. The harmonizer keeps the first `image_id` in sort order. In the q1M export, each study then has
  one image.
- IDs: `p17838301` gives `patient_id` `17838301` and `s50037760` gives `study_id` `50037760`, the MIMIC IDs as
  digits. `image_path` is `p17/p17838301/s50037760/0788829b-5419d8e4-5ce8eb81-87a77c03-98c15a1a.dcm`, relative
  to `base_image_dir` (D-04). `image_ext` sets the extension. `question_id` is the study and the source
  `question_id`, for example `50037760_D01_describe_finding_015`, because the source ID is a template name
  that repeats in each study.
- Text: `question` comes from the `qa.json` file of the study. `answer` joins the texts of the `main_answer`
  dicts with a space (D-51). The `details` and `related_information` texts stay only in `answer_struct`.
  Without `qa_zip_path`, `question` and `answer` are empty and `answer_struct` is `None`.
- Split: `split` is the split of the patient in `patient_metadata.parquet`. A split class first keeps the
  patients of its split. Then it applies `max_studies` and reads `qa.zip`.
- View: `view_position` is `img.view_position` of the question's image in `image_metadata.parquet`, with
  the shared names of D-01. The `A_frontal` exports have frontal images only, so it is `AP` or `PA`. Use it to
  keep only PA images, for example `df[df["view_position"] == "PA"]`. It is in the table, not in the sample.
  A question with several images gets the view of its kept image.
  With no `image_metadata.parquet`, the harmonizer warns and the table has no `view_position`.
- Age and sex: the harmonized table has no `age_years` or `sex` (D-01). The source has no age and no sex.
- Dropped rows: the harmonizer drops a question with no image in `question_image_metadata.parquet`. In
  q1M, every question has an image, so it drops no row. A question with no text in `qa.zip` keeps an empty
  `question` and `answer`. The validate and test splits of q1M have no such question.

## Questions and answers

The text columns of the harmonized table:

| Column | Meaning | Source |
|---|---|---|
| `question` | The question text | `question` in `qa.json` |
| `answer` | The `main_answer` texts, joined with a space (D-51) | `answers` in `qa.json` |
| `answer_struct` | The list of the top-level answer dicts, with all their tags. `answer_type` of a dict is `main_answer`, `details` or `related_information` | `answers` in `qa.json` |
| `question_type` | The template of the question, for example `D02_has_finding` | `question.question_type` |
| `quality` | The quality of the question: `4_A`, `5_A+` or `6_A++` | `question.quality.rating` |
| `split` | `train`, `validate` or `test` | `split` in `patient_metadata.parquet` |

Answer types: the source has no closed or open flag. The template tells the kind of answer. Examples from
the test split:

| `question_type` | `question` | `answer` |
|---|---|---|
| `D02_has_finding` | Is there any indication of pneumothorax? | No, there is no indication of pneumothorax. |
| `C04_is_normal_region` | Is the diaphragm normal? | No, the diaphragm is associated with the following abnormal findings: There is obscuration of the lateral aspect of the left hemidiaphragm. |
| `D01_describe_finding` | Describe the lung opacity. | There is linear opacification in the left mid lung. |

A yes or no answer is a full sentence, so compare it as text, not as a yes or no label.

Example row of `mimic_ext_cxr_qba_test`:

| `question_id` | `question` | `answer` | `question_type` | `quality` |
|---|---|---|---|---|
| `50037760_D01_describe_finding_015` | Describe the lung opacity. | There is linear opacification in the left mid lung. | `D01_describe_finding` | `5_A+` |

The sample holds `img`, `question`, `answer`, `patient_id`, `study_id` and `question_id`. With
`output_question_type=True`, it also holds `question_type`. With `output_struct=True`, it also holds
`answer_struct`, the list of answer dicts. `img` is a tensor `(1, 224, 224)`: chest X-rays are gray, so the
class keeps the default `IMG_CHANNELS = 1` (D-42).

Each answer dict has `answer_type`, `text`, `regions`, `localization` and other tags. `localization` is a
dict with one entry for each image of the study, keyed by `image_id`. Each entry holds pixel boxes
`[x1, y1, x2, y2]` on that source image in `bboxes`, for example `[[1408.0, 1100.0, 2722.0, 1888.0]]`. Use
the entry of the image in `image_path`. The class gives no `bbox` key. The transform does not move these
boxes with the image, so D-28 does not apply.

## Splits

The official split of each patient is in `patient_metadata.parquet`. Each split has its own class (D-09).

<!-- claim: id=mimic_ext_cxr_qba.splits.validate.images tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_ext_cxr_qba mimic_ext_cxr_qba_validate.images" expect="1,641" -->
<!-- claim: id=mimic_ext_cxr_qba.splits.validate.rows tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_ext_cxr_qba mimic_ext_cxr_qba_validate.rows" expect="7,620" -->
<!-- claim: id=mimic_ext_cxr_qba.splits.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_ext_cxr_qba mimic_ext_cxr_qba_test.images" expect="2,850" -->
<!-- claim: id=mimic_ext_cxr_qba.splits.test.rows tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_ext_cxr_qba mimic_ext_cxr_qba_test.rows" expect="12,569" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `MIMICExtCXRQBATrainDataset` | `mimic_ext_cxr_qba_train` | | The `train` patients of `patient_metadata.parquet` |
| `MIMICExtCXRQBAValidateDataset` | `mimic_ext_cxr_qba_validate` | 1,641 images (7,620 questions) | The `validate` patients |
| `MIMICExtCXRQBATestDataset` | `mimic_ext_cxr_qba_test` | 2,850 images (12,569 questions) | The `test` patients. Use it as the test split |
| `MIMICExtCXRQBADataset` | `mimic_ext_cxr_qba` | | All questions. It mixes the 3 splits, so do not use it for a benchmark (D-09) |

<!-- claim: id=mimic_ext_cxr_qba.splits.shared tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_ext_cxr_qba patients_in_two_splits" expect="0" -->
The splits share 0 patients (D-34).

<!-- claim: id=mimic_ext_cxr_qba.splits.validate.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_ext_cxr_qba mimic_ext_cxr_qba_validate.patients" expect="481" -->
<!-- claim: id=mimic_ext_cxr_qba.splits.test.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py mimic_ext_cxr_qba mimic_ext_cxr_qba_test.patients" expect="287" -->
The validate split has 481 patients and the test split has 287. The train split has the other questions of
q1M. No test counts them, because one `harmonize()` of the train split takes 30 minutes.

`get_datasets(n_splits=5)` makes a train and val split from the train class. It groups by patient (D-10).

## Arguments

The arguments are the same for the 3 split classes and `MIMICExtCXRQBADataset`.

<!-- args: radharmony.dataset.MIMICExtCXRQBATrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `files/` folder of MIMIC-CXR (DICOM) or MIMIC-CXR-JPG 2.0.0. It holds the `p10/` to `p19/` folders. Not needed with `harmonizer_path`, which holds it |
| `metadata_dir` | `str` | Yes | `None` | An export folder with the parquet files, for example `exports/A_frontal/metadata/q1M/`. Without it, the class raises `ValueError`. Not needed with `harmonized_df`, `harmonizer` or `harmonizer_path` |
| `qa_zip_path` | `str` | No | `None` | `qa.zip`. Without it, `question` and `answer` are empty |
| `image_ext` | `str` | No | `".dcm"` | The extension in `image_path`: `".dcm"` for MIMIC-CXR or `".jpg"` for MIMIC-CXR-JPG |
| `max_studies` | `int` | No | `None` | The most studies to read, for a fast run. In a split class, it counts the studies of that split. `None` reads all |

### Shared arguments (inherited from BaseVQADataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives the VQA steps. They load `img` with 1 channel (D-42), and resize and pad it to 224 x 224 (H x W). They have no `SelectItemsD`, so the text keys stay in the sample |
| `output_struct` | `bool` | No | `False` | Add the `answer_struct` list to the sample. To batch it, use `collate_fn=bbox_collate` |
| `output_question_type` | `bool` | No | `False` | Add `question_type` to the sample |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The text keys stay strings |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer (`.pkl`) of the same split. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import MIMICExtCXRQBATestHarmonizer

h = MIMICExtCXRQBATestHarmonizer(
    metadata_dir="/data/mimic-ext-cxr-qba/exports/A_frontal/metadata/q1M/",
    qa_zip_path="/data/mimic-ext-cxr-qba/qa.zip",
    base_image_dir="/data/mimic-cxr-jpg/2.0.0/files/",
    image_ext=".jpg",
)
df = h.harmonize(study_ids=["50037760"])  # only this study, so the run is fast
print(df.columns.tolist())
# ['patient_id', 'study_id', 'question_id', 'image_path', 'question', 'answer', 'answer_struct',
#  'question_type', 'quality', 'split', 'view_position']
```

`harmonize(study_ids=..., patient_ids=...)` keeps only these studies or patients, before the slow `qa.zip`
step. An ID works with or without its `s` or `p`. It returns the same rows as a filter after `harmonize()`
(D-02). The other split harmonizers and `MIMICExtCXRQBAHarmonizer` take the same arguments.

Speed: the harmonizer opens 1 file in `qa.zip` for each study. The test split took 25 s on 2026-11-04. Use
`max_studies`, `study_ids=` or `patient_ids=` for a fast run.

<!-- args: radharmony.harmonizer.MIMICExtCXRQBATestHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `metadata_dir` | `str` | Yes | `None` | An export folder with the parquet files. Without it, `harmonize()` raises `ValueError`. A split harmonizer with no `patient_metadata.parquet` raises `FileNotFoundError` |
| `qa_zip_path` | `str` | No | `None` | `qa.zip`. Without it, `question` and `answer` are empty |
| `base_image_dir` | `str` | No | `None` | The `files/` folder of the images. `harmonize()` does not read it. `verify_images()` and a saved harmonizer use it |
| `image_ext` | `str` | No | `".dcm"` | The extension in `image_path`: `".dcm"` or `".jpg"` |
| `max_studies` | `int` | No | `None` | The most studies to read. In a split harmonizer, it counts the studies of that split |

## Known data problems

- Empty answer: some questions have an empty `main_answer` text, so their `answer` is "" (D-51). This is
  not a question with no text in `qa.zip`. In q1M, 12 train questions have an empty answer, and the validate
  and test splits have none.
