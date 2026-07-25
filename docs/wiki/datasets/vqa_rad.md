# VQA-RAD

**Modality:** CXR / CT / MRI (mixed) | **Format:** JPEG | **Dim:** 2D | **Task:** Visual Question Answering

## Overview

VQA-RAD (Visual Question Answering in Radiology) is a benchmark of clinically generated question-answer pairs over 315 radiology images sampled from MedPix (chest X-ray, CT, and MRI cases across HEAD/CHEST/ABD). It contains 2,248 questions authored or verified by clinicians, split into closed-ended (yes/no, choice) and open-ended (free-text) answer types. Published in Lau et al., *Scientific Data* (2018), released under CC0 1.0.

VQA-RAD has no patient/study concept — every image is its own independent case, so `patient_id` and `study_id` are both derived from the image filename.

This dataset uses `BaseVQADataset` rather than `BaseRadiologicalDataset`. The data dict contains mixed types (Tensor + strings), so the standard `output_cls` / `output_bbox` mechanism does not apply.

## Download

Available from [OSF: Visual Question Answering in Radiology (VQA-RAD)](https://osf.io/89kps/) — the dataset's official, public home. No registration required; CC0 1.0 license.

The OSF project exposes 5 items at the top level: 3 equivalent metadata files (JSON/XLSX/XML — the harmonizer reads the JSON), a `Readme.docx`, and an image folder. The image folder can be fetched as a zip via OSF's folder-zip endpoint:

```bash
# metadata + readme
curl -sL -o "VQA_RAD Dataset Public.json" "https://osf.io/download/6qdas/"
curl -sL -o "Readme.docx" "https://osf.io/download/bd96f/"

# images (315 JPEGs, ~15MB zip)
curl -sL -o "VQA_RAD Image Folder.zip" \
  "https://files.osf.io/v1/resources/89kps/providers/osfstorage/5b21453986d8510011c277bc/?zip="
unzip -q "VQA_RAD Image Folder.zip" -d "VQA_RAD Image Folder"
```

## Expected layout

```
VQA-RAD/
  VQA_RAD Dataset Public.json    # 2,248 QA records (harmonizer reads this)
  VQA_RAD Dataset Public.xlsx    # same data, spreadsheet form (unused)
  VQA_RAD Dataset Public.xml     # same data, XML form (unused)
  Readme.docx
  VQA_RAD Image Folder/          # flat, 315 JPEGs, no subdirs
    synpic100132.jpg
    synpic100176.jpg
    ...
```

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_image_dir` | `str` | Yes* | `None` | `VQA_RAD Image Folder/` directory (flat, direct parent of the 315 JPEGs) |
| `json_path` | `str` | No | auto | Path to `VQA_RAD Dataset Public.json`; auto-discovered in the parent of `base_image_dir` when omitted |

### Shared arguments (inherited from `BaseVQADataset`)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `output_struct` | `bool` | No | `False` | Include `"answer_struct"` in data dict — unset for VQA-RAD, whose answers are plain strings, not structured (no `answer_struct` column is populated) |
| `output_question_type` | `bool` | No | `False` | Include `"question_type"` string (e.g. `PRES`, `ABN`, `MODALITY`, …) in data dict |
| `transform` | MONAI Compose | No | 2D VQA pipeline | Custom MONAI transform (must not use `SelectItemsD`, or `question`/`answer` get dropped) |
| `cache_dir` | `str` | No | `"./cache"` | MONAI PersistentDataset cache; `None` disables caching |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype for `"img"`; use `torch.float32` on CPU |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonized CSV |

*`base_image_dir` can be omitted when `harmonizer_path`, `harmonizer`, or `harmonized_df` is provided.

## Dataset constructor

```python
from radharmony.dataset import VQARadDataset

ds = VQARadDataset(
    base_image_dir="/data/VQA-RAD/VQA_RAD Image Folder",
    output_question_type=True,
)
dataset = ds.get_datasets()
sample = dataset[0]
# sample["img"]           -> Tensor(3, 224, 224)
# sample["question"]      -> "Are regions of the brain infarcted?"
# sample["answer"]        -> "Yes"
# sample["question_type"] -> "PRES"
```

Splitting by the dataset's own official train/test split (rather than a random patient-level split) requires filtering the harmonized DataFrame on `split` before constructing the dataset:

```python
from radharmony.harmonizer import VQARadHarmonizer

h = VQARadHarmonizer(
    json_path="/data/VQA-RAD/VQA_RAD Dataset Public.json",
    base_image_dir="/data/VQA-RAD/VQA_RAD Image Folder",
)
df = h.harmonize()
train_ds = VQARadDataset(
    base_image_dir="/data/VQA-RAD/VQA_RAD Image Folder",
    harmonized_df=df[df["split"] == "train"],
)
test_ds = VQARadDataset(
    base_image_dir="/data/VQA-RAD/VQA_RAD Image Folder",
    harmonized_df=df[df["split"] == "test"],
)
```

## Harmonizer

```python
from radharmony.harmonizer import VQARadHarmonizer

h = VQARadHarmonizer(
    json_path="/data/VQA-RAD/VQA_RAD Dataset Public.json",
    base_image_dir="/data/VQA-RAD/VQA_RAD Image Folder",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'question_id', 'image_path', 'question',
#  'answer', 'question_type', 'split', 'answer_type', 'image_organ']
df.to_csv("vqa_rad_harmonized.csv", index=False)
```

## Load from saved harmonized CSV

```python
import pandas as pd
from radharmony.dataset import VQARadDataset

ds = VQARadDataset(
    base_image_dir="/data/VQA-RAD/VQA_RAD Image Folder",
    harmonized_df=pd.read_csv("vqa_rad_harmonized.csv"),
)
```

## Harmonizer notes

- Reads `VQA_RAD Dataset Public.json` (a flat list of 2,248 records) directly — no separate label/annotation CSV
- No `Patient ID` or study-id field exists in the source; `patient_id` and `study_id` are both the image filename stem (e.g. `synpic54610.jpg` -> `synpic54610`) since every image is an independent case
- `qid` is globally unique across all 2,248 questions and is used directly as `question_id`
- `phrase_type` in the source (`freeform` / `para` / `test_freeform` / `test_para`) marks the dataset's official train/test split — the `test_` prefixed variants are the paper's held-out 451-question test set — carried through as the `split` column (1,797 train / 451 test)
- `answer_type` (`OPEN`/`CLOSED`) and `image_organ` (`HEAD`/`CHEST`/`ABD`) are carried through via `EXTRA_OUTPUT_COLS` as bonus metadata, not part of the core VQA schema
- A handful of source rows have integer answers (e.g. "How many gallstones are identified?" -> `4`) and a couple of `answer_type` values have a trailing space (`"CLOSED "`) — both are normalized (cast to `str`, `.strip()`'d) during harmonization
- `answer_struct` is not populated — VQA-RAD's answers are plain strings, not structured/nested (unlike e.g. RAD-ReStruct)

## VQA data dict

Each sample contains:

| Key | Type | Notes |
|-----|------|-------|
| `"img"` | `Tensor (3, 224, 224)` | Loaded and normalised image (RGB — MedPix JPEGs are not all single-channel) |
| `"question"` | `str` | Question text |
| `"answer"` | `str` | Answer text |
| `"patient_id"` | `str` | Image filename stem (no real patient concept) |
| `"study_id"` | `str` | Same as `patient_id` (no real study concept) |
| `"question_id"` | `str` | `qid` from the source JSON |
| `"question_type"` | `str` | Question category (only when `output_question_type=True`) |

## Notes on BaseVQADataset

VQA-RAD inherits `BaseVQADataset` instead of `BaseRadiologicalDataset` for the same reasons as MIMIC-Ext-CXR-QBA (see [mimic_ext_cxr_qba.md](mimic_ext_cxr_qba.md)): the standard 2-D transform's `SelectItemsD` would drop `question`/`answer`/`question_id`, and the `output_cls`/`LABEL_COLS` concept doesn't apply to free-text QA.

`radharmony.integrity.check_dataset()` (used by `scripts/check_datasets.py` and the integrity notebook) works unmodified against `VQARadDataset` — it already validates image shape/dtype/range/NaN generically and collates arbitrary sample keys, so no VQA-specific stats were needed for a meaningful pass/fail check. It does not currently render question/answer text in the Gradio app's viewer — `app.py` has no VQA-shaped UI (no `question`/`answer` display path); that remains unimplemented for this and every other VQA dataset.
