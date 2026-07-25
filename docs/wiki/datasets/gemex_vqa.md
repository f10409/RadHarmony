# GEMeX-VQA

**Modality:** CXR (over MIMIC-CXR-JPG) | **Format:** JPEG | **Dim:** 2D | **Task:** Visual Question Answering

## Overview

GEMeX-VQA is a chest-X-ray visual-question-answering benchmark released by BoKelvin on HuggingFace ([`BoKelvin/GEMeX-VQA`](https://huggingface.co/datasets/BoKelvin/GEMeX-VQA)). Questions are authored over MIMIC-CXR-JPG images; each question carries a rationale, references to anatomical regions, and (for choice questions) an option list. Four question subtypes are distributed as four JSONL files:

| Subtype | Answer form | Choices | Notes |
|---------|-------------|---------|-------|
| `open_ended` | free-text | none | free-text answer |
| `closed_ended` | `"Yes."` / `"No."` | none | binary decision |
| `single_choice` | single letter (e.g. `"C"`) | list of `"A: ..."` strings | one correct option |
| `multi_choice` | comma-joined letters (e.g. `"A,B,D"`) | list of `"A: ..."` strings | multiple correct options |

GEMeX-VQA has no independent image tree (images live in MIMIC-CXR-JPG), so `base_image_dir` must point at the MIMIC-CXR-JPG `files/` directory. This dataset uses `BaseVQADataset` rather than `BaseRadiologicalDataset` (see [VQA-RAD](vqa_rad.md) for the same pattern).

> The JSONL files currently distributed on the HuggingFace hub are a small sample split (80 questions total, 20 per subtype). Re-download whenever the upstream repo publishes the full benchmark split.

## Download

```bash
huggingface-cli download BoKelvin/GEMeX-VQA --repo-type dataset \
    --local-dir /data/gemex-vqa
```

MIMIC-CXR-JPG is separate (obtain it via [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/), credentialed access).

## Expected layout

```
<data_dir>/                        ← GEMeX-VQA JSONL files
  open_ended_question.jsonl
  closed_ended_question.jsonl
  single_choice_question.jsonl
  multi_choice_question.jsonl

<base_image_dir>/                  ← MIMIC-CXR-JPG files/ root
  p10/p10046166/s50051329/abea5eb9-....jpg
  p11/.../..jpg
  ...
```

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `data_dir` | `str` | Yes* | `None` | Directory containing the 4 JSONL files |
| `base_image_dir` | `str` | Yes* | `None` | MIMIC-CXR-JPG `files/` root |
| `question_subtypes` | `list[str]` | No | all 4 | Any subset of `["open_ended", "closed_ended", "single_choice", "multi_choice"]` |

### Shared arguments (inherited from `BaseVQADataset`)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `output_struct` | `bool` | No | `False` | Include `"answer_struct"` (dict with `reason`, `visual_regions`, `visual_locations`, `ori_report`, `choices`) in data dict |
| `output_question_type` | `bool` | No | `False` | Include `"question_type"` string (e.g. `disease`, `abnormality`, `finding`, `size`, `location`, `severity`) in data dict |
| `transform` | MONAI Compose | No | 2D VQA pipeline | Custom MONAI transform (must not drop `question` / `answer` / `question_id`) |
| `cache_dir` | `str` | No | `"./cache"` | MONAI PersistentDataset cache; `None` disables caching |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype for `"img"` |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonizer state |

\* `data_dir` and `base_image_dir` can be omitted when `harmonizer_path`, `harmonizer`, or `harmonized_df` is provided.

## Usage

```python
from radharmony.dataset import GEMeXVQADataset

ds = GEMeXVQADataset(
    data_dir="/data/gemex-vqa",
    base_image_dir="/data/mimic-cxr-jpg/2.0.0/files",
    output_question_type=True,
)
dataset = ds.get_datasets()
sample = dataset[0]
# sample["img"]           -> Tensor(3, 224, 224)
# sample["question"]      -> "Is there evidence of pneumothorax in the right lung?"
# sample["answer"]        -> "No."
# sample["question_type"] -> "abnormality"
```

Filter to a single subtype (e.g. closed-ended only):

```python
ds = GEMeXVQADataset(
    data_dir="/data/gemex-vqa",
    base_image_dir="/data/mimic-cxr-jpg/2.0.0/files",
    question_subtypes=["closed_ended"],
)
```

Enable structured-answer output (choices, rationale, visual references):

```python
ds = GEMeXVQADataset(
    data_dir="/data/gemex-vqa",
    base_image_dir="/data/mimic-cxr-jpg/2.0.0/files",
    output_struct=True,
)
sample = ds.get_datasets()[0]
# sample["answer_struct"] -> {"reason": ..., "visual_regions": [...],
#                             "visual_locations": [[x, y, w, h], ...],
#                             "ori_report": ..., "choices": [...]}
```

## Harmonizer

```python
from radharmony.harmonizer import GEMeXVQAHarmonizer

h = GEMeXVQAHarmonizer(
    data_dir="/data/gemex-vqa",
    base_image_dir="/data/mimic-cxr-jpg/2.0.0/files",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'question_id', 'image_path', 'question',
#  'answer', 'question_type', 'question_subtype', 'reason',
#  'ori_report', 'visual_regions', 'visual_locations', 'choices']
```

## Harmonizer notes

- **Image paths follow the MIMIC-CXR-JPG layout** (`p{prefix}/p{patient_id}/s{study_id}/{dicom_id}.jpg`); `patient_id` and `study_id` are parsed directly from that path.
- **`question_id` is namespaced by subtype** as `{subtype}_{q_id}` so identifiers stay globally unique across the four JSONL files.
- **Multi-choice answers are joined with commas** (e.g. source `["A", "B", "D"]` becomes `"A,B,D"`) so the schema stays `str`; the original list is preserved inside `answer_struct`.
- **`answer_struct`** carries the rationale (`reason`), the referenced anatomical regions, the referenced bounding boxes (`[x, y, w, h]`), the original report text, and the choices list (or `None` for open/closed-ended).
- **`question_type`** is the source's `type` field (six values: `disease`, `abnormality`, `finding`, `size`, `location`, `severity`).

## VQA data dict

| Key | Type | Notes |
|-----|------|-------|
| `"img"` | `Tensor (3, 224, 224)` | Loaded and normalised image (RGB) |
| `"question"` | `str` | Question text (always populated across all four subtypes) |
| `"answer"` | `str` | Answer text (comma-joined letters for multi-choice) |
| `"patient_id"` | `str` | `p{patient_id}` from the MIMIC-CXR-JPG path |
| `"study_id"` | `str` | `s{study_id}` from the MIMIC-CXR-JPG path |
| `"question_id"` | `str` | `{subtype}_{q_id}` |
| `"question_type"` | `str` | Only when `output_question_type=True` |
| `"answer_struct"` | `dict` | Only when `output_struct=True` |

## Notes on BaseVQADataset

GEMeX-VQA inherits `BaseVQADataset` (like VQA-RAD and MIMIC-Ext-CXR-QBA): the standard 2-D transform's `SelectItemsD` would drop `question` / `answer` / `question_id`, and the `output_cls` / `LABEL_COLS` concept does not apply to free-text QA. `app.py` renders GEMeX-VQA samples with a dedicated per-subtype Q&A summary panel (grouped Closed / Open / Single / Multi sections with option ticks).
