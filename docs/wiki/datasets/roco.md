# ROCO (Radiology Objects in COntext)

**Modality:** Multimodal (radiology + non-radiology) | **Format:** JPEG | **Dim:** 2D | **Task:** Image captioning

## Overview

ROCO is a large-scale multimodal captioning dataset built from PubMed Central OpenAccess figures. Each entry pairs an image with the figure caption; the `radiology/` subset (~81k train / ~9k valid / ~9k test) covers a broad range of radiographic modalities (CXR, CT, MRI, ultrasound, and more), and the optional `non-radiology/` subset covers everything else. UMLS keywords, CUIs, and semantic-type files ship alongside the captions.

ROCO is exposed through `BaseVQADataset` as a captioning dataset: `question` is always the empty string and `answer` is the caption text. This preserves compatibility with the report-generation evaluator (which scores `answer` against a generated hypothesis).

## Download

Clone the upstream repo and run its fetch script (the raw dataset is not distributed as a single archive; images are fetched from PubMed):

```bash
git clone https://github.com/razorx89/roco-dataset.git
cd roco-dataset
python scripts/fetch.py                # downloads all 3 splits × 2 subsets
```

The `scripts/fetch.py --subdir <name>` argument controls the image subfolder name (default `images`); pass the same name to `image_subdir=` on the dataset constructor.

## Expected layout

```
<base_dir>/                          ← the roco-dataset repo root
  data/
    train/
      radiology/
        captions.txt
        keywords.txt
        cuis.txt
        semtypes.txt
        images/
          ROCO_00001.jpg
          ...
      non-radiology/
        (same structure)
    validation/
      (same structure)
    test/
      (same structure)
```

## File formats

- `captions.txt` (tab-separated): `ROCO_XXXXX\t<caption text>`
- `keywords.txt` (tab-separated): `ROCO_XXXXX\t\t<kw1>\t<kw2>\t...` (index 1 is blank; harmonizer joins remaining keywords with tabs)

## Constructor arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `base_dir` | `str` | Yes* | `None` | Root of the roco-dataset tree (the directory containing `data/`) |
| `splits` | `list[str]` | No | all 3 | Any subset of `["train", "validation", "test"]` |
| `radiology_only` | `bool` | No | `True` | When `False`, also include `non-radiology/` samples |
| `image_subdir` | `str` | No | `"images"` | Image subdirectory name; must match the `--subdir` value passed to `scripts/fetch.py` |

### Shared arguments (inherited from `BaseVQADataset`)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `output_struct` | `bool` | No | `False` | Include `"answer_struct"` in data dict (unused for ROCO; kept for schema parity) |
| `output_question_type` | `bool` | No | `False` | Include `"question_type"` (unused for ROCO) |
| `transform` | MONAI Compose | No | 2D VQA pipeline | Custom MONAI transform |
| `cache_dir` | `str` | No | `"./cache"` | MONAI PersistentDataset cache; `None` disables caching |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | Output tensor dtype for `"img"` |
| `harmonized_df` | `pd.DataFrame` | No | `None` | Pre-built harmonized DataFrame |
| `harmonizer` | harmonizer | No | `None` | Pre-instantiated harmonizer |
| `harmonizer_path` | `str` | No | `None` | Path to saved harmonizer state |

\* `base_dir` can be omitted when `harmonizer_path`, `harmonizer`, or `harmonized_df` is provided.

## Usage

```python
from radharmony.dataset import ROCODataset

ds = ROCODataset(base_dir="/data/roco-dataset")
dataset = ds.get_datasets()
sample = dataset[0]
# sample["img"]      -> Tensor(3, 224, 224)
# sample["question"] -> ""
# sample["answer"]   -> "Chest radiograph demonstrating left lower lobe consolidation ..."
```

Load a specific split:

```python
from radharmony.harmonizer import ROCOHarmonizer

h = ROCOHarmonizer(base_dir="/data/roco-dataset")
df = h.harmonize()
train_df = df[df["split"] == "train"].reset_index(drop=True)

ds = ROCODataset(base_dir="/data/roco-dataset", harmonized_df=train_df)
```

Include the non-radiology subset:

```python
ds = ROCODataset(base_dir="/data/roco-dataset", radiology_only=False)
```

## Harmonizer

```python
from radharmony.harmonizer import ROCOHarmonizer

h = ROCOHarmonizer(base_dir="/data/roco-dataset")
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'question_id', 'image_path',
#  'question', 'answer', 'keywords', 'split', 'subset']
```

## Harmonizer notes

- **No patient / study concept** (each figure is independent): `patient_id`, `study_id`, and `question_id` are all the ROCO ID (`ROCO_XXXXX`).
- **`image_path` is relative to `base_dir`** and takes the form `data/{split}/{subset}/{image_subdir}/{ROCO_XXXXX}.jpg`.
- **Captions with embedded tabs** are preserved: the harmonizer joins fields 1..N with `"\t"` before writing to `answer`.
- **Keywords are tab-joined** (as delivered by ROCO's TSV format) into a single string; split with `.split("\t")` if you need the list.
- **CUIs and semantic types** are not currently carried through; harmonization uses `captions.txt` + `keywords.txt` only.
- **Missing files are skipped silently**: if a caption or keywords file is absent for a given split/subset, that folder is dropped rather than raising.

## VQA data dict

| Key | Type | Notes |
|-----|------|-------|
| `"img"` | `Tensor (3, 224, 224)` | Loaded and normalised image (RGB) |
| `"question"` | `str` | Always `""` (captioning task) |
| `"answer"` | `str` | Caption text |
| `"patient_id"` | `str` | ROCO ID (e.g. `"ROCO_00001"`) |
| `"study_id"` | `str` | Same as `patient_id` |
| `"question_id"` | `str` | Same as `patient_id` |

Pass ROCO to [`ReportGenerationEvaluator`](../evaluator/index.md) as a captioning benchmark: the evaluator scores `answer` against a generator's hypothesis under the RadEval metric suite.
