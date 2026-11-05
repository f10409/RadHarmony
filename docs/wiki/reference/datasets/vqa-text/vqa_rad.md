# VQA-RAD

<!-- claim: id=vqa_rad.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py vqa_rad vqa_rad_train.images" expect="313" -->
<!-- claim: id=vqa_rad.train.rows tier=ci cmd="python scripts/evalkit/dataset_stat.py vqa_rad vqa_rad_train.rows" expect="1,797" -->
<!-- claim: id=vqa_rad.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py vqa_rad vqa_rad_test.images" expect="203" -->
<!-- claim: id=vqa_rad.test.rows tier=ci cmd="python scripts/evalkit/dataset_stat.py vqa_rad vqa_rad_test.rows" expect="451" -->
| | |
|---|---|
| Registry keys | `vqa_rad_train`, `vqa_rad_test` |
| Task and outputs | Visual question answering: the question (`question`) and the answer (`answer`). `question_type` with `output_question_type=True` |
| Images in the table | train 313 images (1,797 questions), test 203 images (451 questions) |
| Modality | X-ray, CT and MRI of the head, chest and abdomen, 2-D, JPEG |
| License | CC0 1.0 |
| Download | [Get the data: other datasets](../../../how-to/get-the-data.md#other-datasets) |

<!-- claim: id=vqa_rad.source.questions tier=nas cmd="python scripts/evalkit/dataset_stat.py --source vqa_rad_train json_path rows" expect="2,248" -->
VQA-RAD is a visual question answering dataset of radiology images from MedPix
([Lau et al., Scientific Data 2018](https://doi.org/10.1038/sdata.2018.251)). Clinicians wrote questions
about the images and gave the answers. `VQA_RAD Dataset Public.json` has 2,248 questions.

## Quick use

`base_image_dir` is the image folder. The class finds `VQA_RAD Dataset Public.json` in this folder or in the
folder above it.

```
VQA-RAD/
  VQA_RAD Dataset Public.json
  VQA_RAD Image Folder/synpic54610.jpg
```

```python
import torch
from radharmony.dataset import VQARadTestDataset, VQARadTrainDataset

train = VQARadTrainDataset(
    base_image_dir="/data/VQA-RAD/VQA_RAD Image Folder/",
    output_question_type=True,
    dtype=torch.float32,
)
train_ds, val_ds = train.get_datasets(n_splits=5)  # 1/5 of the images go to val

test_ds = VQARadTestDataset(
    base_image_dir="/data/VQA-RAD/VQA_RAD Image Folder/",
    output_question_type=True,
    dtype=torch.float32,
).get_datasets()
sample = test_ds[0]
# sample keys: img, question, answer, patient_id, study_id, question_id, question_type
# sample["img"]: Tensor(3, 224, 224)
# sample["question"]: 'Is there evidence of an aortic aneurysm?'
# sample["answer"]: 'yes'
```

## How it is harmonized

- Source: `VQA_RAD Dataset Public.json`. One record is one question. One row of the harmonized table is one
  question, not one image (D-03).
- IDs: the source has no patient or study ID. The `image_name` `synpic54610.jpg` gives `patient_id`
  `synpic54610`, `study_id` `synpic54610` and `image_path` `synpic54610.jpg`. So one image is one patient.
  `image_path` is relative to `base_image_dir` (D-04). `question_id` is the source `qid`, for example `0`.
- Text: `question` and `answer` come from the keys of the same name. The harmonizer removes the spaces at
  both ends. 5 answers are numbers in the source, for example `4` for "How many gallstones are
  identified?". The harmonizer makes them text.
- Split: `phrase_type` `test_freeform` or `test_para` gives the `split` value `test`. `freeform` or
  `para` gives `train`.
- Extra columns: `answer_type` and `image_organ`. The harmonizer removes the space at the end of 2
  `answer_type` values (`"CLOSED "`).
- View, age and sex: the source has none of them, so the harmonized table has no `view_position`, `age_years`
  or `sex` (D-01).
- Dropped rows: the harmonizer drops no row. The image `synpic35356.jpg` has no question, so the harmonized
  table has 314 of the 315 images.

## Questions and answers

The text columns of the harmonized table:

| Column | Meaning | Source |
|---|---|---|
| `question` | The question text | `question` |
| `answer` | The answer text | `answer` |
| `question_type` | The topic of the question: `PRES` (presence), `POS` (position), `ABN` (abnormality), `MODALITY`, `SIZE`, `PLANE`, `ATTRIB` (attribute), `ORGAN`, `COLOR`, `COUNT` or `OTHER`. Some rows have 2 types, for example `PRES, POS` | `question_type` |
| `answer_type` | `CLOSED` or `OPEN` | `answer_type` |
| `image_organ` | `HEAD`, `CHEST` or `ABD` | `image_organ` |
| `split` | `train` or `test` | `phrase_type` |

Answer types: a closed question has a short answer, mostly yes or no. Some closed questions give a choice,
for example "Is the abnormality focal or diffuse?" with the answer "Diffuse". An open question has a free
text answer, for example "Where is the abnormality?" with the answer "left temporal lobe". In the test
split, 272 questions are closed and 179 are open.

Example row of `vqa_rad_test`:

| `question_id` | `image_path` | `question` | `answer` | `question_type` | `answer_type` |
|---|---|---|---|---|---|
| `10` | `synpic42202.jpg` | Is there evidence of an aortic aneurysm? | yes | `PRES` | `CLOSED` |

The sample holds `img`, `question`, `answer`, `patient_id`, `study_id` and `question_id`. With
`output_question_type=True`, it also holds `question_type`. `img` is a tensor `(3, 224, 224)`: MedPix has
color figures, so the class sets `IMG_CHANNELS = 3` (D-42). The VQA steps repeat a gray image to 3 channels.

## Splits

<!-- claim: id=vqa_rad.splits.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py vqa_rad vqa_rad_train.images" expect="313" -->
<!-- claim: id=vqa_rad.splits.train.rows tier=ci cmd="python scripts/evalkit/dataset_stat.py vqa_rad vqa_rad_train.rows" expect="1,797" -->
<!-- claim: id=vqa_rad.splits.test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py vqa_rad vqa_rad_test.images" expect="203" -->
<!-- claim: id=vqa_rad.splits.test.rows tier=ci cmd="python scripts/evalkit/dataset_stat.py vqa_rad vqa_rad_test.rows" expect="451" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `VQARadTrainDataset` | `vqa_rad_train` | 313 images (1,797 questions) | `phrase_type` `freeform` and `para` |
| `VQARadTestDataset` | `vqa_rad_test` | 203 images (451 questions) | `phrase_type` `test_freeform` and `test_para`. Use it as the test split |
| `VQARadDataset` | `vqa_rad` | | All questions. It mixes the 2 splits, so do not use it for a benchmark (D-09) |

<!-- claim: id=vqa_rad.splits.shared tier=ci cmd="python scripts/evalkit/dataset_stat.py vqa_rad patients_in_two_splits" expect="202" -->
The splits share 202 patients (D-34). The paper made the test questions on the same images as the train
questions, so 202 of the 203 test images also have train questions. The split is by question, not by image.

`get_datasets(n_splits=5)` makes a train and val split from the train class. It groups by patient (D-10). The
source has no patient ID, so one image is one patient, and nobody can check that a patient is in only 1 split.

## Arguments

The arguments are the same for `VQARadTrainDataset`, `VQARadTestDataset` and `VQARadDataset`.

<!-- args: radharmony.dataset.VQARadTrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `VQA_RAD Image Folder/` folder. Not needed with `harmonizer_path`, which holds it |
| `json_path` | `str` | No | `None` | `VQA_RAD Dataset Public.json`. Without it, the class looks in `base_image_dir` and in the folder above it, and raises `FileNotFoundError` if the file is not there |

### Shared arguments (inherited from BaseVQADataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives the VQA steps. They load `img` with 3 channels (D-42), and resize and pad it to 224 x 224 (H x W). They have no `SelectItemsD`, so the text keys stay in the sample |
| `output_struct` | `bool` | No | `False` | Add `answer_struct` to the sample. The harmonized table has no `answer_struct` column, so the class warns and adds no key (D-11) |
| `output_question_type` | `bool` | No | `False` | Add `question_type` to the sample |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The text keys stay strings |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer (`.pkl`) of the same split. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import VQARadTrainHarmonizer

h = VQARadTrainHarmonizer(
    json_path="/data/VQA-RAD/VQA_RAD Dataset Public.json",
    base_image_dir="/data/VQA-RAD/VQA_RAD Image Folder/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'question_id', 'image_path', 'question', 'answer',
#  'question_type', 'split', 'answer_type', 'image_organ']
```

`VQARadTestHarmonizer` takes the same arguments.

<!-- args: radharmony.harmonizer.VQARadTrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `json_path` | `str` | Yes | | `VQA_RAD Dataset Public.json` |
| `base_image_dir` | `str` | No | `None` | The `VQA_RAD Image Folder/` folder. `harmonize()` does not read it. `verify_images()` and a saved harmonizer use it |

## Known data problems

- Answer case: the closed answers have both `Yes` and `yes`, and both `No` and `no`. The harmonized table
  keeps the case of the source. Make the case the same before you compare answers.
- Question types: a few values have a typo: `ATRIB` (1 row), `PRSE` (1 row) and `Other` (2 rows, where
  the other rows use `OTHER`).
