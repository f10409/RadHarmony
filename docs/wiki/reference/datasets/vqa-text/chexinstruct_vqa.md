# CheXinstruct VQA

<!-- claim: id=chexinstruct_vqa.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chexinstruct_vqa chexinstruct_vqa_train.images" expect="174,218" -->
<!-- claim: id=chexinstruct_vqa.train.rows tier=ci cmd="python scripts/evalkit/dataset_stat.py chexinstruct_vqa chexinstruct_vqa_train.rows" expect="580,166" -->
<!-- claim: id=chexinstruct_vqa.val.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chexinstruct_vqa chexinstruct_vqa_val.images" expect="9,460" -->
<!-- claim: id=chexinstruct_vqa.val.rows tier=ci cmd="python scripts/evalkit/dataset_stat.py chexinstruct_vqa chexinstruct_vqa_val.rows" expect="99,996" -->
<!-- claim: id=chexinstruct_vqa.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chexinstruct_vqa chexinstruct_vqa_test.images" expect="2,659" -->
<!-- claim: id=chexinstruct_vqa.test.rows tier=ci cmd="python scripts/evalkit/dataset_stat.py chexinstruct_vqa chexinstruct_vqa_test.rows" expect="21,296" -->
| | |
|---|---|
| Registry keys | `chexinstruct_vqa_train`, `chexinstruct_vqa_val`, `chexinstruct_vqa_test` |
| Task and outputs | Visual question answering: the question (`question`) and the answer (`answer`). `question_type` with `output_question_type=True`, and `answer_struct` with `output_struct=True` |
| Images in the table | train 174,218 images (580,166 questions), val 9,460 images (99,996 questions), test 2,659 images (21,296 questions) |
| Modality | Chest X-ray, 2-D, JPEG from [MIMIC-CXR-JPG](../cxr-labels/mimic_cxr_jpg.md) |
| License | PhysioNet credentialed access (CITI training and a signed data use agreement) for MIMIC-CXR-JPG, MIMIC-CXR-VQA and MIMIC-Diff-VQA. The CheXagent processor code is CC BY-NC-ND 4.0 |
| Download | [Get the data: other datasets](../../../how-to/get-the-data.md#other-datasets), and the compile step in [Quick use](#quick-use). The images come from [MIMIC-CXR-JPG](../../../how-to/get-the-data.md#mimic-cxr-jpg) |

<!-- claim: id=chexinstruct_vqa.source.questions tier=nas cmd="python scripts/evalkit/dataset_stat.py --source chexinstruct_vqa json_path rows" expect="701,458" -->
CheXinstruct is an instruction-tuning collection from about 28 public chest X-ray datasets
([Chen et al., arXiv 2024](https://arxiv.org/abs/2401.12208)). For each source, the "processor" code of
[Stanford-AIMI/CheXagent](https://github.com/Stanford-AIMI/CheXagent) makes records in one common format.
This page covers 2 VQA sources only: MIMIC-CXR-VQA (open-ended and close-ended questions) and
MIMIC-Diff-VQA (Difference VQA, named Medical-Diff-VQA on PhysioNet). All their images are MIMIC-CXR-JPG
images. The compiled JSON has 701,458 questions.

The other VQA sources of CheXinstruct (VQA-RAD, SLAKE, MedVQA-2019, PMC-VQA and Rad-Restruct) need other
image folders, so the class does not read them. VQA-RAD has its own page: [VQA-RAD](vqa_rad.md).

## Quick use

The class reads `chexinstruct_vqa.json`. You make this file once, from 3 downloads:

1. The images: [MIMIC-CXR-JPG](../cxr-labels/mimic_cxr_jpg.md).
2. The MIMIC-CXR-VQA questions:
   [PhysioNet: MIMIC-Ext-MIMIC-CXR-VQA](https://physionet.org/content/mimic-ext-mimic-cxr-vqa/1.0.0/).
3. The MIMIC-Diff-VQA questions:
   [PhysioNet: Medical-Diff-VQA](https://physionet.org/content/medical-diff-vqa/1.0.1/).

Then copy 5 files of [CheXagent](https://github.com/Stanford-AIMI/CheXagent) (commit `e4f31e6`, folder
`data_chexinstruct/dataset_processors/`) to `vendor/dataset_processors/`, with no change: `base_processor.py`,
`templates.py`, `utils.py`, `mimic_cxr_vqa.py` and `mimic_diff_vqa.py`. Do not copy its `__init__.py`. It imports
all 28 processors, so the import fails when only these 5 files are there. The CheXagent license allows no
changed copies, so RadHarmony does not include these files.

Put the 3 downloads in `data/`, and make the link `vendor/data -> ../data` (see the folder tree below). Then
make the JSON file:

<!-- block: run=skip reason="needs the CheXagent processors and the 3 downloads" -->
```bash
python scripts/compile_chexinstruct_vqa.py --vendor-dir /data/CheXinstruct/vendor
```

The script runs the processors in `vendor/` and writes `compiled/chexinstruct_vqa.json` next to `vendor/`.
`--out` gives another path.

```
CheXinstruct/
  vendor/
    data -> ../data        # a link: the processors read data/... from vendor/
    dataset_processors/    # the CheXagent files, with no change
  data/
    mimic-cxr/             # the MIMIC-CXR-JPG folder (files/ and the CSV files), or a link to it
    mimic-cxr-vqa/mimiccxrvqa/dataset/train.json, valid.json, test.json
    mimic-diff-vqa/mimic_all.csv, code/temp/mimic_pair_questions_temp.csv
  compiled/chexinstruct_vqa.json
```

`base_image_dir` is the `files/` folder of MIMIC-CXR-JPG. Pass `json_path`. Without it, the class looks for
the JSON file only in `base_image_dir` and in the folder above it. It raises `FileNotFoundError` when the
file is not there (D-35).

```python
import torch
from radharmony.dataset import (
    ChexInstructVQATestDataset,
    ChexInstructVQATrainDataset,
    ChexInstructVQAValDataset,
)

kw = dict(
    base_image_dir="/data/mimic-cxr-jpg/2.0.0/files/",
    json_path="/data/CheXinstruct/compiled/chexinstruct_vqa.json",
    output_struct=True,
    output_question_type=True,
    dtype=torch.float32,
)
train_ds = ChexInstructVQATrainDataset(**kw).get_datasets()
val_ds = ChexInstructVQAValDataset(**kw).get_datasets()
test_ds = ChexInstructVQATestDataset(**kw).get_datasets()
sample = test_ds[0]
# sample keys: img, question, answer, patient_id, study_id, question_id, answer_struct, question_type
# sample["img"]: Tensor(1, 224, 224)
# sample["question"]: 'Is any technical assessments present within the mediastinum?'
# sample["answer"]: 'Yes'
```

To batch `answer_struct`, pass `collate_fn=bbox_collate` (`from radharmony.dataset import bbox_collate`) to
the `DataLoader`. `answer_struct` is a dict for a close-ended question and NaN for the others, so the
default `DataLoader` collate fails.

## How it is harmonized

- Source: the compiled `chexinstruct_vqa.json`, a list of records. One record is one question. One row of
  the harmonized table is one question, not one image (D-03). A split class keeps the records with its
  `split` value.
- IDs: the record `study_id` `[mimic-cxr] [56535659]` gives `study_id` `56535659`. The records have no
  patient ID, so `patient_id` comes from the `p<subject_id>` folder of the image path, for example
  `10449873`. `question_id` is `unique_id`, for example `[Open-Ended VQA] [MIMIC-CXR-VQA] [0]`.
- `image_path`: the source path with the `data/mimic-cxr/files/` at the start removed, for example
  `p10/p10449873/s56535659/0942432e-6d4664df-1389611d-a70f5cbc-8734b936.jpg`. It is relative to
  `base_image_dir` (D-04). A MIMIC-Diff-VQA record has a list of 1 path, and the harmonizer takes that path.
- Text: `question` comes from `text` and `answer` from `target_text`, with the spaces at the ends removed.
  `question_type` comes from `task_name`, and `answer_struct` from `options`. See
  [Questions and answers](#questions-and-answers).
- Extra columns: `data_source`, which is `MIMIC-CXR-VQA` or `MIMIC-Diff-VQA`.
- View, age and sex: the source has none of them, so the harmonized table has no `view_position`,
  `age_years` or `sex` (D-01).
- Dropped rows: the harmonizer drops no row of its split. `question_types` and `data_sources` keep only the
  types and sources that you name.

## Questions and answers

The text columns of the harmonized table:

| Column | Meaning | Source |
|---|---|---|
| `question` | The question text | `text` |
| `answer` | The answer text. It is empty for a close-ended question | `target_text` |
| `answer_struct` | `{"yes": bool, "no": bool}` for a close-ended question, NaN for the others | `options` |
| `question_type` | `Open-Ended VQA`, `Close-Ended VQA` or `Difference VQA` | `task_name` |
| `data_source` | `MIMIC-CXR-VQA` or `MIMIC-Diff-VQA` | `data_source` |
| `split` | `train`, `val` or `test` | `split`, which `compile_chexinstruct_vqa.py` adds |

Answer types:

- Open-Ended VQA (MIMIC-CXR-VQA): a text answer, for example `Mediastinum`, `Yes` or `No`. An answer with
  several items is one text with commas, for example `Left lung, right lung`.
- Close-Ended VQA (MIMIC-CXR-VQA): the same yes or no questions as the Open-Ended VQA type. `answer` is
  empty, and `answer_struct` holds the answer. So each yes or no question is 2 times in the harmonized
  table, once for each type.
- Difference VQA (MIMIC-Diff-VQA): the CheXagent processor keeps only the location, level and type questions
  of MIMIC-Diff-VQA. Each question is about 1 image, not about a change between 2 studies.

Example rows of `chexinstruct_vqa_test` (one image):

| `question` | `answer` | `answer_struct` | `question_type` |
|---|---|---|---|
| Is any technical assessments present within the mediastinum? | `Yes` | NaN | `Open-Ended VQA` |
| Is any technical assessments present within the mediastinum? | (empty) | `{"yes": True, "no": False}` | `Close-Ended VQA` |

A Difference VQA row: "Where in the image is the pleural effusion located?" with the answer
`Left upper lung area`.

The sample holds `img`, `question`, `answer`, `patient_id`, `study_id` and `question_id`. With
`output_question_type=True`, it also holds `question_type`. With `output_struct=True`, it also holds
`answer_struct`. `img` is `(1, 224, 224)`: chest X-rays are gray, so the class keeps the default
`IMG_CHANNELS = 1` (D-42).

## Splits

<!-- claim: id=chexinstruct_vqa.splits.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chexinstruct_vqa chexinstruct_vqa_train.images" expect="174,218" -->
<!-- claim: id=chexinstruct_vqa.splits.train.rows tier=ci cmd="python scripts/evalkit/dataset_stat.py chexinstruct_vqa chexinstruct_vqa_train.rows" expect="580,166" -->
<!-- claim: id=chexinstruct_vqa.splits.val.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chexinstruct_vqa chexinstruct_vqa_val.images" expect="9,460" -->
<!-- claim: id=chexinstruct_vqa.splits.val.rows tier=ci cmd="python scripts/evalkit/dataset_stat.py chexinstruct_vqa chexinstruct_vqa_val.rows" expect="99,996" -->
<!-- claim: id=chexinstruct_vqa.splits.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py chexinstruct_vqa chexinstruct_vqa_test.images" expect="2,659" -->
<!-- claim: id=chexinstruct_vqa.splits.test.rows tier=ci cmd="python scripts/evalkit/dataset_stat.py chexinstruct_vqa chexinstruct_vqa_test.rows" expect="21,296" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `ChexInstructVQATrainDataset` | `chexinstruct_vqa_train` | 174,218 images (580,166 questions) | The records with `split` `train` |
| `ChexInstructVQAValDataset` | `chexinstruct_vqa_val` | 9,460 images (99,996 questions) | The records with `split` `val` |
| `ChexInstructVQATestDataset` | `chexinstruct_vqa_test` | 2,659 images (21,296 questions) | The records with `split` `test`. Use it as the test split |
| `ChexInstructVQADataset` | `chexinstruct_vqa` | | All questions. It mixes the 3 splits, so do not use it for a benchmark (D-09) |

<!-- claim: id=chexinstruct_vqa.splits.train.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py chexinstruct_vqa chexinstruct_vqa_train.patients" expect="51,530" -->
<!-- claim: id=chexinstruct_vqa.splits.val.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py chexinstruct_vqa chexinstruct_vqa_val.patients" expect="3,585" -->
<!-- claim: id=chexinstruct_vqa.splits.shared tier=ci cmd="python scripts/evalkit/dataset_stat.py chexinstruct_vqa patients_in_two_splits" expect="2,007" -->
The train split has 51,530 patients, the val split 3,585 and the test split 734. The splits share 2,007
patients (D-34). 497 of the 734 test patients are also in train.

The 2 sources made their splits on their own. The MIMIC-CXR-VQA split alone puts 663 patients in 2 splits.
The MIMIC-Diff-VQA split follows the official MIMIC-CXR split, and alone it shares no patient. For a test
with no shared patient, use `data_sources=["MIMIC-Diff-VQA"]`, or remove the shared patients in your own
code.

`get_datasets(n_splits=5)` makes a train and val split from the train class. It groups by patient (D-10).

## Arguments

The arguments are the same for the 3 split classes and `ChexInstructVQADataset`.

<!-- args: radharmony.dataset.ChexInstructVQATrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `files/` folder of MIMIC-CXR-JPG, the image folder of both sources. `source_image_dirs` can take its place. Not needed with `harmonizer` or `harmonizer_path`, which hold it |
| `source_image_dirs` | `dict[str, str]` | No | `None` | One image folder for each source, for example `{"MIMIC-CXR-VQA": "...", "MIMIC-Diff-VQA": "..."}`. It takes the place of `base_image_dir`. Both sources use one folder, so `base_image_dir` is enough |
| `json_path` | `str` | No | `None` | The compiled `chexinstruct_vqa.json`. Without it, the class looks in `base_image_dir` and in the folder above it, and raises `FileNotFoundError` when the file is not there (D-35). Not needed with `harmonizer_path` |
| `question_types` | `list[str]` | No | `None` | Keep only these types: any of `"Open-Ended VQA"`, `"Close-Ended VQA"` and `"Difference VQA"`. `None` keeps all. An unknown value raises `ValueError`. It also filters a given table (D-02) |
| `data_sources` | `list[str]` | No | `None` | Keep only these sources: `"MIMIC-CXR-VQA"`, `"MIMIC-Diff-VQA"` or both. `None` keeps all. An unknown value raises `ValueError`. It also filters a given table (D-02) |

### Shared arguments (inherited from BaseVQADataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives the VQA steps. They load `img` with 1 channel (D-42), and resize and pad it to 224 x 224 (H x W). They have no `SelectItemsD`, so the text keys stay in the sample |
| `output_struct` | `bool` | No | `False` | Add the `answer_struct` dict to the sample (NaN for a question that is not close-ended). To batch it, use `collate_fn=bbox_collate` |
| `output_question_type` | `bool` | No | `False` | Add `question_type` to the sample |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The text keys stay strings |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer (`.pkl`) of the same split. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import ChexInstructVQATestHarmonizer

h = ChexInstructVQATestHarmonizer(
    json_path="/data/CheXinstruct/compiled/chexinstruct_vqa.json",
    base_image_dir="/data/mimic-cxr-jpg/2.0.0/files/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'question_id', 'image_path', 'question', 'answer', 'answer_struct',
#  'question_type', 'split', 'data_source']
```

`ChexInstructVQATrainHarmonizer`, `ChexInstructVQAValHarmonizer` and `ChexInstructVQAHarmonizer` (the 3
splits together) take the same arguments.

<!-- args: radharmony.harmonizer.ChexInstructVQATestHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `json_path` | `str` | Yes | | The compiled `chexinstruct_vqa.json` |
| `base_image_dir` | `str` | No | `None` | The `files/` folder of MIMIC-CXR-JPG. `harmonize()` does not read it. `verify_images()` and a saved harmonizer use it |
| `source_image_dirs` | `dict[str, str]` | No | `None` | One image folder for each source. It takes the place of `base_image_dir`. An unknown source raises `ValueError` |
| `question_types` | `list[str]` | No | `None` | Keep only these question types. An unknown value raises `ValueError` |
| `data_sources` | `list[str]` | No | `None` | Keep only these sources. An unknown value raises `ValueError` |

## Known data problems

- `unique_id` is unique only in one split. The source starts the count again in each split. So in
  `chexinstruct_vqa` (the 3 splits together), a `question_id` can repeat.
- The MIMIC-CXR-VQA processor shuffles the train questions before it gives `unique_id`, and the processors
  set no seed. So `compile_chexinstruct_vqa.py` seeds `random` (`--seed`, default 0). 2 runs with the same
  seed give the same file. A file from a run with no seed has other `question_id` values for the MIMIC-CXR-VQA
  train rows, in another order. All other ids, and the questions, answers and splits, are the same.
