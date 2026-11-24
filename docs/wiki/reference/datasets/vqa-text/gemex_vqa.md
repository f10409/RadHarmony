# GEMeX-VQA

<!-- claim: id=gemex_vqa.images tier=ci cmd="python scripts/evalkit/dataset_stat.py gemex_vqa gemex_vqa.images" expect="11" -->
<!-- claim: id=gemex_vqa.rows tier=ci cmd="python scripts/evalkit/dataset_stat.py gemex_vqa gemex_vqa.rows" expect="80" -->
| | |
|---|---|
| Registry keys | `gemex_vqa` |
| Task and outputs | Visual question answering: the question (`question`) and the answer (`answer`). `question_type` with `output_question_type=True`, and `answer_struct` with `output_struct=True` |
| Images in the table | 11 images (80 questions). These are the subset files on Hugging Face |
| Modality | Chest X-ray, 2-D, JPEG from [MIMIC-CXR-JPG](../cxr-labels/mimic_cxr_jpg.md) |
| License | The Hugging Face repo states no license. The images need PhysioNet credentialed access, as for MIMIC-CXR-JPG |
| Download | [Get the data: other datasets](../../../how-to/get-the-data.md#other-datasets). The images come from [MIMIC-CXR-JPG](../../../how-to/get-the-data.md#mimic-cxr-jpg) |

<!-- claim: id=gemex_vqa.paper.questions source="Liu et al., arXiv 2024" expect="1,605,575" -->
<!-- claim: id=gemex_vqa.paper.images source="Liu et al., arXiv 2024" expect="151,025" -->
GEMeX is a chest X-ray visual question answering benchmark
([Liu et al., arXiv 2024](https://arxiv.org/abs/2411.16778)). Each question has an answer, a reason, the
report of the study, and the names and boxes of the image regions that the answer uses. The paper has
1,605,575 questions on 151,025 images. The Hugging Face repo `BoKelvin/GEMeX-VQA` has only a small subset:
4 JSONL files of 20 questions each, one file for each question subtype.

## Quick use

GEMeX has no images of its own. `data_dir` is the folder of the 4 JSONL files. `base_image_dir` is the
`files/` folder of MIMIC-CXR-JPG 2.0.0.

```
gemex-vqa/                                 <- data_dir
  open_ended_question.jsonl
  closed_ended_question.jsonl
  single_choice_question.jsonl
  multi_choice_question.jsonl
mimic-cxr-jpg/2.0.0/files/                 <- base_image_dir
  p10/p10046166/s50051329/abea5eb9-b7c32823-3a14c5ca-77868030-69c83139.jpg
```

```python
import torch
from radharmony.dataset import GEMeXVQADataset

gemex = GEMeXVQADataset(
    data_dir="/data/gemex-vqa/",
    base_image_dir="/data/mimic-cxr-jpg/2.0.0/files/",
    output_question_type=True,
    output_struct=True,
    dtype=torch.float32,
)
ds = gemex.get_datasets()  # all 80 questions. get_datasets(n_splits=5) makes train and val
sample = ds[0]
# sample keys: img, question, answer, patient_id, study_id, question_id, answer_struct, question_type
# sample["img"]: Tensor(1, 224, 224)
# sample["question"]: 'What condition is indicated by the low lung volume in this CXR?'
# sample["answer"]: 'Restrictive lung disease.'
```

To batch `answer_struct`, pass `collate_fn=bbox_collate` (`from radharmony.dataset import bbox_collate`) to
the `DataLoader`. The `answer_struct` dicts hold `None` and lists of different lengths, so the default
`DataLoader` collate fails.

## How it is harmonized

- Source: the 4 JSONL files. One record is one question. One row of the harmonized table is one question,
  not one image (D-03). `question_subtypes` picks the files to read.
- IDs: the `image_path` `p10/p10046166/s50051329/abea5eb9-b7c32823-3a14c5ca-77868030-69c83139.jpg` gives
  `patient_id` `10046166` and `study_id` `50051329`, the MIMIC IDs as digits. `image_path` stays as in the
  source, relative to `base_image_dir` (D-04). `question_id` is the question subtype and `q_id`, for example
  `single_choice_0`, because each file starts `q_id` at 0.
- Answer: the harmonizer joins the list answer of a multi-choice question with commas. `["A", "B", "D"]`
  gives `"A,B,D"`. The other answers stay as in the source.
- Question type: `question_type` comes from `type`. `question_subtype` is the question subtype of the file.
- Extra columns: `reason`, `ori_report`, `visual_regions`, `visual_locations` and `choices`, as in the
  source. `answer_struct` holds the same 5 values in one dict.
- View, age and sex: the source has none of them, so the harmonized table has no `view_position`, `age_years`
  or `sex` (D-01).
- Dropped rows: the harmonizer drops no row.

## Questions and answers

The text columns of the harmonized table:

| Column | Meaning | Source |
|---|---|---|
| `question` | The question text | `question` |
| `answer` | The answer: text, `Yes.` or `No.`, a letter, or letters joined with commas | `answer` |
| `question_type` | `abnormality`, `disease`, `location`, `severity`, `finding` or `size` | `type` |
| `question_subtype` | `open_ended`, `closed_ended`, `single_choice` or `multi_choice` | The file name |
| `reason` | Why the answer is right | `reason` |
| `ori_report` | The report of the study | `ori_report` |
| `visual_regions` | The names of the image regions that the answer uses, for example `['right hemidiaphragm']` | `visual_regions` |
| `visual_locations` | One box for each region. See [Boxes](#boxes) | `visual_locations` |
| `choices` | The options of a choice question, for example `A: Left lung`. `None` for the open and closed questions | `choices` |
| `answer_struct` | A dict of `reason`, `visual_regions`, `visual_locations`, `ori_report` and `choices` | The same keys |

Answer types: each file has 20 questions.

| `question_subtype` | Answer | Example |
|---|---|---|
| `open_ended` | Free text | `Restrictive lung disease.` |
| `closed_ended` | `Yes.` or `No.` (7 and 13 questions) | `No.` |
| `single_choice` | One letter of `choices` | `C` |
| `multi_choice` | Letters of `choices`, joined with commas | `A,B,D` |

Example row of `gemex_vqa`:

| `question_id` | `question` | `answer` | `question_type` | `choices` |
|---|---|---|---|---|
| `single_choice_0` | What is the size of the calcified granuloma observed? | `C` | `size` | `['A: 0.5 cm', 'B: 1.0 cm', 'C: 1.2 cm', 'D: 1.5 cm']` |

The sample holds `img`, `question`, `answer`, `patient_id`, `study_id` and `question_id`. With
`output_question_type=True`, it also holds `question_type`. With `output_struct=True`, it also holds
`answer_struct`. `img` is a tensor `(1, 224, 224)`: chest X-rays are gray, so the class keeps the default
`IMG_CHANNELS = 1` (D-42).

## Boxes

- Source: `visual_locations` holds one box of 4 whole numbers for each name in `visual_regions`, in the same
  order. The 80 questions have 104 boxes, and each question has 1 to 3.
- Format: the values are `[x1, y1, x2, y2]` in a 224 x 224 (H x W) image, not in the MIMIC-CXR-JPG image.
  They are not `[x, y, w, h]`, because `x + w` would be up to 304.
  Most boxes (81 of 104) are the region boxes of [Chest ImaGenome](../cxr-boxes-masks/chest_imagenome.md)
  silver, minus 1 on each value. Example: `abea5eb9-...` has the `right hemidiaphragm` box
  `[43, 126, 129, 150]`, and its Chest ImaGenome scene graph gives `[44, 127, 130, 151]`. `cardiac region`
  matches `cardiac silhouette`. A region of both sides, for example `bilateral lung`, has the box around the
  boxes of the 2 sides. The other 23 boxes differ more. Example: a `spine` box is `[93, 19, 135, 205]`, and
  Chest ImaGenome gives `[94, 0, 136, 223]`.
- The harmonizer copies the boxes with no change to `visual_locations` and `answer_struct`. It does not make
  D-07 boxes, so the harmonized table has no `bbox` or `bbox_labels`.
- Readers (D-36): the source has one box for each region, so the harmonizer does not group boxes.
- The sample: it has no `bbox` key. With `output_struct=True`, `answer_struct["visual_locations"]` holds the
  source boxes, for example `[[43, 35, 212, 156]]` for `bilateral lung`. The transform does not move these
  boxes with the image, so D-28 does not apply.
- Dropped boxes: the harmonizer does not clip or drop a box. All source boxes are inside the 224 x 224
  (H x W) image.

## Splits

<!-- claim: id=gemex_vqa.splits.images tier=ci cmd="python scripts/evalkit/dataset_stat.py gemex_vqa gemex_vqa.images" expect="11" -->
<!-- claim: id=gemex_vqa.splits.rows tier=ci cmd="python scripts/evalkit/dataset_stat.py gemex_vqa gemex_vqa.rows" expect="80" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `GEMeXVQADataset` | `gemex_vqa` | 11 images (80 questions) | The 4 JSONL files |

<!-- claim: id=gemex_vqa.splits.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py gemex_vqa gemex_vqa.patients" expect="7" -->
The 11 images are of 7 patients. GEMeX-VQA has no official split in the subset files.
`get_datasets(n_splits=5)` makes a train and val split. It groups by patient (D-10).

## Arguments

<!-- args: radharmony.dataset.GEMeXVQADataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `data_dir` | `str` | Yes | `None` | The folder of the 4 JSONL files. Without it, the class raises `ValueError`. Not needed with `harmonized_df`, `harmonizer` or `harmonizer_path` |
| `base_image_dir` | `str` | Yes | `None` | The `files/` folder of MIMIC-CXR-JPG 2.0.0. Not needed with `harmonizer_path`, which holds it |
| `question_subtypes` | `list[str]` | No | `None` | The question subtypes to read: any of `"open_ended"`, `"closed_ended"`, `"single_choice"` and `"multi_choice"`. `None` reads all 4. Another value raises `ValueError` |

### Shared arguments (inherited from BaseVQADataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives the VQA steps. They load `img` with 1 channel (D-42), and resize and pad it to 224 x 224 (H x W). They have no `SelectItemsD`, so the text keys stay in the sample |
| `output_struct` | `bool` | No | `False` | Add the `answer_struct` dict to the sample. To batch it, use `collate_fn=bbox_collate` |
| `output_question_type` | `bool` | No | `False` | Add `question_type` to the sample |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The text keys stay strings |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import GEMeXVQAHarmonizer

h = GEMeXVQAHarmonizer(
    data_dir="/data/gemex-vqa/",
    base_image_dir="/data/mimic-cxr-jpg/2.0.0/files/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'question_id', 'image_path', 'question', 'answer', 'answer_struct',
#  'question_type', 'question_subtype', 'reason', 'ori_report', 'visual_regions', 'visual_locations', 'choices']
```

<!-- args: radharmony.harmonizer.GEMeXVQAHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `data_dir` | `str` | Yes | `None` | The folder of the 4 JSONL files. Without it, `harmonize()` raises `ValueError`. A missing file raises `FileNotFoundError` |
| `base_image_dir` | `str` | No | `None` | The `files/` folder of MIMIC-CXR-JPG 2.0.0. `harmonize()` does not read it. `verify_images()` and a saved harmonizer use it |
| `question_subtypes` | `list[str]` | No | `None` | The question subtypes to read. `None` reads all 4. Another value raises `ValueError` |

## Known data problems

- Subset only: the Hugging Face repo has 80 questions, not the full benchmark of the paper. Download the
  files again when the full set is on Hugging Face.
- Box format: the source files and the Hugging Face page do not say what the 4 box numbers mean. The
  format in [Boxes](#boxes) comes from the values and from Chest ImaGenome.
