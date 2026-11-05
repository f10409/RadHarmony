# MedPMC

<!-- claim: id=medpmc.images tier=ci cmd="python scripts/evalkit/dataset_stat.py medpmc medpmc.images" expect="11,123,087" -->
| | |
|---|---|
| Registry keys | `medpmc` |
| Task and outputs | Image captioning: the caption (`answer`). `question` is empty |
| Images in the table | 11,123,087 images, with no filter. One image has one caption |
| Modality | Medical figures from PubMed Central articles: radiology, pathology, microscopy, clinical photos and more. 2-D, JPEG |
| License | CC BY-NC-SA 4.0 (the Hugging Face card). Each record also has its own source and license metadata |
| Download | [Get the data: other datasets](../../../how-to/get-the-data.md#other-datasets) |

MedPMC ([Kim et al., arXiv 2026](https://arxiv.org/abs/2607.07673)) pairs medical figures from PubMed Central
articles with their captions. It has 3 subsets. `multipanel` has whole figures with many panels, and
`singlepanel` has figures with one panel. `subfigure` has the panels cut out of the multipanel figures, with
their own captions. The class gives the caption as `answer`, not as `report`. `ReportGenerationEvaluator`
reads `report`, so it cannot score this class.

RadHarmony makes its own tag tables of specialty and modality, so you can filter, for example, to radiology
X-ray images. These tags are not labels of the MedPMC authors.

## Quick use

The source is a Hugging Face snapshot of `Yale-BIDS-Chen/medpmc-11m-dataset_jun24_baseline`. The images stay
in webdataset tar files. Before you use the class:

1. Build the caption tag table with `scripts/build_medpmc_tags.py --base-dir /data/MedPMC/`. It reads the
   caption of each image from the tar files. The `answer` of the class comes from this table.
2. For the CLIP filters (`modalities`, `require_tagger_agreement`, `min_clip_prob`), run
   `scripts/build_medpmc_tags_clip.py --subsets singlepanel subfigure`. It writes parts to `tags/_clip_parts/`.
   Then run `scripts/merge_clip_tags.py --base-dir /data/MedPMC/` to make the CLIP tag table.
3. Extract the images of your subset with `scripts/materialize_medpmc_subset.py`. Do not extract all images
   to a network drive. Example: `--base-dir /data/MedPMC/ --high-precision-radiology --modalities xray`.

```
MedPMC/
  webdataset/
    metadata/medpmc_index.parquet       # one row for each image
    multipanel/train-000000.tar ...
    singlepanel/train-000000.tar ...
    subfigure/train-000000.tar ...
  tags/medpmc_tags.parquet              # made by scripts/build_medpmc_tags.py
  tags/medpmc_tags_clip.parquet         # made by scripts/merge_clip_tags.py
  images/singlepanel/singlepanel_000000000038.jpg   # made by scripts/materialize_medpmc_subset.py
```

```python
import torch
from radharmony.dataset import MedPMCDataset

ds = MedPMCDataset(
    base_dir="/data/MedPMC/",
    subsets=["singlepanel", "subfigure"],
    radiology_only=True,
    modalities=["xray"],
    require_tagger_agreement=True,
    min_clip_prob=0.9,
    dtype=torch.float32,
)
ds.verify_images()  # drop the rows with no extracted image file
train_ds, val_ds = ds.get_datasets(n_splits=5)  # 1/5 of the articles go to val
sample = train_ds[0]
# sample keys: img, question, answer, patient_id, study_id, question_id
# sample["img"]: Tensor(3, 224, 224)
# sample["answer"]: 'Abdomen X-ray of the patient, showed a large urinary calculus in the bladder.'
```

## How it is harmonized

- Source: `webdataset/metadata/medpmc_index.parquet`, with the caption tag table joined on `image_id`. One
  row of the harmonized table is one image and its caption (D-03). `subsets` keeps only those subsets.
- IDs: the source has no patient ID. `patient_id` is the article (`pmcid`), because the figures of one article
  are not independent. `study_id` is the figure: `parent_image_id` for a subfigure, and `image_id` for the
  other subsets. `question_id` is `image_id`. Example: `patient_id` `PMC4770620`, `study_id`
  `PMC4770620_EUS-5-35-g002` and `question_id` `PMC4770620_EUS-5-35-g002_2`.
- `image_path`: `<source_type>/<key><image_ext>`, for example `subfigure/subfigure_000000000085.jpg`. It is
  relative to `base_image_dir`, which is `<base_dir>/images` by default (D-04).
- Caption: `answer` is `caption` of the caption tag table. With no tag table, `answer` is None in every row,
  and the harmonizer warns. `question` is `""`.
- Tags: the caption tag table gives `specialty`, `modality`, `tag_confidence`, `tag_source` and `tag_method`.
  The CLIP tag table gives `clip_specialty`, `clip_modality` and `clip_prob`, only when a CLIP filter
  (`modalities`, `require_tagger_agreement` or `min_clip_prob`) needs it.
- View, age and sex: the source has none of them, so the harmonized table has no `view_position`,
  `age_years` or `sex` (D-01).
- Dropped rows: by default, the harmonizer drops no row. The filter arguments drop rows (see
  [Arguments](#arguments)). With no caption tag table, `radiology_only`, `specialties`, `modalities`,
  `min_tag_confidence`, `drop_untagged` and `require_tagger_agreement` raise `FileNotFoundError`. A CLIP filter with no CLIP tag table
  also raises `FileNotFoundError`.

## Captions

The text columns of the harmonized table:

| Column | Meaning | Source |
|---|---|---|
| `question` | Always `""`, because this is a captioning set | None |
| `answer` | The caption of the image. A subfigure has its own short caption | The caption tag table |
| `source_type` | `multipanel`, `singlepanel` or `subfigure` | The index |
| `pmcid` | The PubMed Central article | The index |
| `image_id`, `parent_image_id` | The image, and the multipanel figure of a subfigure | The index |
| `subfigure_index`, `figure_label` | The panel number of a subfigure, and the figure name in the article (for example `Fig. 1`) | The index |
| `shard`, `key` | The tar file and the name of the image in it | The index |
| `specialty` | One of 10 values, for example `radiology`, `pathology` or `unknown` | The caption tag table |
| `modality` | The radiology modality: `ct`, `mri`, `fmri`, `xray`, `ultrasound`, `pet_spect`, `angiography`, `mammography` or `radiology_other` | The caption tag table |
| `tag_confidence` | `high` (a specific term matched), `low` (a general term, or a tag from the parent caption) or None | The caption tag table |
| `tag_source` | `own` (the caption of the image) or `parent` (the caption of the whole figure). None with no tag | The caption tag table |
| `tag_method` | The method that made the tag, for example `caption-lexicon-v2` | The caption tag table |
| `clip_specialty`, `clip_modality`, `clip_prob` | The tag of a zero-shot MedPMC-CLIP model on the pixels, and its probability. Only with a CLIP filter | The CLIP tag table |

The caption tag table uses word lists on the caption (`radharmony/harmonizer/medpmc/tagging.py`). The CLIP
tag table has rows only for `singlepanel` and `subfigure`. On 2026-08-22, a clinician checked 151 images where
the 2 taggers disagreed (the docstring of `MedPMCHarmonizer`). CLIP was right for 72.8% of them, and the
caption word lists for 10.6%. So `modalities` uses the CLIP modality, and the caption modality where CLIP has
none.

Example row of `medpmc`, with `radiology_only=True` and `modalities=["xray"]`:

| Column | Value |
|---|---|
| `question_id` | `PMC10009441_gr1` |
| `answer` | `Abdomen X-ray of the patient, showed a large urinary calculus in the bladder.` |
| `source_type` | `singlepanel` |
| `specialty`, `modality` | `radiology`, `xray` |
| `tag_confidence`, `tag_source` | `high`, `own` |
| `clip_modality`, `clip_prob` | `xray`, `0.964` |

The sample holds `img`, `question`, `answer`, `patient_id`, `study_id` and `question_id`. `img` is
`(3, 224, 224)`: many figures are in color, so the class sets `IMG_CHANNELS = 3` (D-42). A gray file gives 3
equal channels. An `answer` of None gives `""` in the sample. The tag columns are only in the
harmonized table, not in the sample.

## Splits

<!-- claim: id=medpmc.splits.images tier=ci cmd="python scripts/evalkit/dataset_stat.py medpmc medpmc.images" expect="11,123,087" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `MedPMCDataset` | `medpmc` | 11,123,087 | `webdataset/metadata/medpmc_index.parquet`, all 3 subsets, with no filter |

<!-- claim: id=medpmc.studies tier=ci cmd="python scripts/evalkit/dataset_stat.py medpmc medpmc.studies" expect="3,799,545" -->
<!-- claim: id=medpmc.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py medpmc medpmc.patients" expect="1,274,778" -->
With no filter, the 11,123,087 images come from 3,799,545 figures (`study_id`) and 1,274,778 articles
(`patient_id`).

MedPMC has no official split. `get_datasets(n_splits=5)` makes a train and val split. It groups by patient
(D-10). Here the patient is the article, so all figures of one article go to one split. For example, a
subfigure and its multipanel figure stay in one split.

## Arguments

<!-- args: radharmony.dataset.MedPMCDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_dir` | `str` | Yes | `None` | The MedPMC folder, with `webdataset/` and `tags/` in it. Not needed with `harmonizer_path`, which holds it |
| `base_image_dir` | `str` | No | `None` | The folder of the extracted images. With `harmonizer_path`, `None` gives the `base_image_dir` of the saved harmonizer. Else, or when the saved harmonizer has none, it gives `<base_dir>/images`. With `harmonizer_path` and no `base_dir`, the class takes `base_dir` from the saved harmonizer |
| `subsets` | `list[str]` | No | `None` | Any of `multipanel`, `singlepanel` and `subfigure`. `None` gives all 3. An unknown name raises `ValueError`. It also filters a given table (D-02) |
| `radiology_only` | `bool` | No | `False` | With `specialties=None`, it gives `specialties=["radiology"]`. With `specialties`, it does nothing. It also filters a given table (D-02) |
| `specialties` | `list[str]` | No | `None` | Keep only these values of `specialty`. A multipanel row has the tag of its whole-figure caption. An unknown name raises `ValueError`. It also filters a given table (D-02) |
| `modalities` | `list[str]` | No | `None` | Keep only these modalities, for example `["ct", "mri"]`. It uses the CLIP modality, and the caption modality where CLIP has none. It needs the CLIP tag table. An unknown name raises `ValueError`. It also filters a given table (D-02) |
| `min_tag_confidence` | `str` | No | `None` | `"high"` keeps only `tag_confidence` `high`. `"low"` keeps `low` and `high`. `None` keeps all rows. Another value raises `ValueError`. It also filters a given table (D-02) |
| `drop_untagged` | `bool` | No | `False` | Drop the rows with `specialty` `unknown` or None. It also filters a given table (D-02) |
| `require_tagger_agreement` | `bool` | No | `False` | Keep only the rows where the 2 taggers give the same specialty, and the same modality when both give one. A row with no CLIP tag (every multipanel row) goes. It needs the caption tag table and the CLIP tag table. Without the caption tag table, `harmonize()` raises `FileNotFoundError`. It also filters a given table (D-02) |
| `min_clip_prob` | `float` | No | `None` | Keep only the rows with `clip_prob` at this value or more. The class drops a row with no CLIP tag. It needs the CLIP tag table. It also filters a given table (D-02) |
| `modality_source` | `str` | No | `"clip"` | The tagger of `modalities`: `"clip"`, `"caption"` or `"either"`, as in the harmonizer table below. Another value raises `ValueError` |
| `include_multipanel` | `bool` | No | `True` | With `False`, a specialty filter drops the multipanel rows. Without a specialty filter, it does nothing |
| `clip_tags_path` | `str` | No | `None` | The CLIP tag table. `None` gives `<base_dir>/tags/medpmc_tags_clip.parquet` |
| `tags_path` | `str` | No | `None` | The caption tag table. `None` gives `<base_dir>/tags/medpmc_tags.parquet` |

With a given table, a filter that needs a column the table does not have raises `ValueError`. Example:
`min_clip_prob` on a table that `harmonize()` made with no CLIP filter, so it has no `clip_prob` column.
`modalities` with `modality_source="clip"` (the default) or `"either"` needs the `clip_modality` column.

### Shared arguments (inherited from BaseVQADataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives the VQA steps. They load `img` with 3 channels (D-42), and resize and pad it to 224 x 224 (H x W). They have no `SelectItemsD`, so the text keys stay in the sample |
| `output_struct` | `bool` | No | `False` | Add `answer_struct` to the sample. The harmonized table has no `answer_struct` column, so the class warns and adds no key (D-11) |
| `output_question_type` | `bool` | No | `False` | Add `question_type` to the sample. The harmonized table has no `question_type` column, so the class warns and adds no key (D-11) |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The text keys stay strings |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer (`.pkl`). A file of another class raises `ValueError`. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import MedPMCHarmonizer

h = MedPMCHarmonizer(
    base_dir="/data/MedPMC/",
    base_image_dir="/data/MedPMC/images/",
    subsets=["subfigure"],
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'question_id', 'image_path', 'question', 'answer', 'source_type', 'pmcid',
#  'image_id', 'parent_image_id', 'subfigure_index', 'figure_label', 'shard', 'key', 'specialty',
#  'modality', 'tag_confidence', 'tag_source', 'tag_method']
```

The harmonizer takes the same filter arguments as the dataset class. Only the dataset class sets
`base_image_dir` to `<base_dir>/images` when it is `None`. The harmonizer keeps `None`.

<!-- args: radharmony.harmonizer.MedPMCHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_dir` | `str` | Yes | `None` | The MedPMC folder. Without it, `harmonize()` raises `ValueError`. With no index file, it raises `FileNotFoundError` |
| `base_image_dir` | `str` | No | `None` | The folder of the extracted images. `harmonize()` does not read it. `verify_images()` and a saved harmonizer use it |
| `subsets` | `list[str]` | No | `None` | Any of `multipanel`, `singlepanel` and `subfigure`. `None` gives all 3. An unknown name raises `ValueError` |
| `radiology_only` | `bool` | No | `False` | With `specialties=None`, it gives `specialties=["radiology"]`. With `specialties`, it does nothing |
| `specialties` | `list[str]` | No | `None` | Keep only these values of `specialty`. An unknown name raises `ValueError` |
| `modalities` | `list[str]` | No | `None` | Keep only these modalities. `modality_source` picks the tagger. An unknown name raises `ValueError` |
| `min_tag_confidence` | `str` | No | `None` | `"high"`, `"low"` or `None`. Another value raises `ValueError` |
| `drop_untagged` | `bool` | No | `False` | Drop the rows with `specialty` `unknown` or None |
| `require_tagger_agreement` | `bool` | No | `False` | Keep only the rows where the 2 taggers agree. Every multipanel row goes, because the CLIP tag table has no multipanel row. Without the caption tag table, `harmonize()` raises `FileNotFoundError` |
| `min_clip_prob` | `float` | No | `None` | Keep only the rows with `clip_prob` at this value or more |
| `modality_source` | `str` | No | `"clip"` | The tagger of `modalities`. `"clip"` takes the CLIP modality, and the caption modality where CLIP has none. `"caption"` takes the caption modality. `"either"` keeps a row when one of the 2 matches. Another value raises `ValueError` |
| `clip_tags_path` | `str` | No | `None` | The CLIP tag table. `None` gives `<base_dir>/tags/medpmc_tags_clip.parquet` |
| `include_multipanel` | `bool` | No | `True` | With `False`, a specialty filter drops the multipanel rows, because their tag comes from the caption of the whole figure. Without a specialty filter, it does nothing |
| `tags_path` | `str` | No | `None` | The caption tag table. `None` gives `<base_dir>/tags/medpmc_tags.parquet` |

## Known data problems

- No specialty or modality: the source has no such field. The paper gives the specialty shares from a check
  by hand of a sample of images, not a label for each image.
- Benchmark images: PubMed Central figures include images of MIMIC-CXR, CheXpert, VinDr and other benchmark
  datasets. If you train on MedPMC and test on these benchmarks, remove these images from the training data
  first.
- Same pixels twice: each subfigure is also a panel of its multipanel figure. To count each figure once,
  count the unique `study_id` values, or leave out the `multipanel` subset.
