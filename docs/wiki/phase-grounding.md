# Phase Grounding: MIMIC-CXR Label Extraction

Extracting TAIX-style ordinal labels and laterality from MIMIC-CXR free-text reports using local regex NLP — no external API calls, no data leaves the machine.

## Goal

TAIX-Ray uses a 5-point ordinal severity scale (0–4) with laterality (left/right/bilateral) for each finding. MIMIC-CXR-JPG ships CheXpert binary labels that cannot be converted to TAIX ordinals. Instead, the free-text radiology reports contain the severity and laterality information — we extract it directly.

## Output schema

Matches the TAIX harmonized label columns:

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | `int` | MIMIC patient identifier |
| `study_id` | `int` | MIMIC study identifier |
| `atelectasis` | `float` | 0–4 severity (collapsed), NaN if absent from report |
| `atelectasis_left` | `float` | Left-side severity |
| `atelectasis_right` | `float` | Right-side severity |
| `pleural_effusion` | `float` | 0–4 severity (collapsed) |
| `pleural_effusion_left` | `float` | |
| `pleural_effusion_right` | `float` | |
| `pulmonary_opacities` | `float` | 0–4 severity (collapsed) |
| `pulmonary_opacities_left` | `float` | |
| `pulmonary_opacities_right` | `float` | |
| `heart_size` | `float` | 0–4 cardiomegaly severity |
| `pulmonary_congestion` | `float` | 0–4 vascular congestion severity |

**Severity scale:**

| Score | Meaning |
|-------|---------|
| 0 | Absent / resolved / normal |
| 1 | Trace / minimal / tiny |
| 2 | Mild / slight |
| 3 | Moderate |
| 4 | Large / severe / massive |
| NaN | Finding not mentioned in the report |

## Extraction pipeline

Script: `/path/to/phase_grounding/extract_mimic_labels.py`

### Step 1 — section extraction

Reports are split into FINDINGS and IMPRESSION sections using a regex that handles the common section header patterns. FINDINGS is preferred; IMPRESSION is the fallback; the full report text is the last resort.

### Step 2 — finding detection

Each finding has a compiled regex pattern:

| Finding | Patterns |
|---------|----------|
| `atelectasis` | `atelectasis`, `atelectatic`, `discoid atelectasis`, `subsegmental atelectasis`, `linear atelectasis`, `basilar atelectasis` |
| `pleural_effusion` | `pleural effusion`, `pleural fluid`, `hydrothorax`, `effusion` |
| `pulmonary_opacities` | `opacity`, `infiltrate`, `consolidation`, `airspace disease`, `alveolar/focal/patchy opacit` |
| `heart_size` | `cardiomegaly`, `enlarged heart`, `cardiac silhouette`, `heart size`, `heart is … enlarg*` |
| `pulmonary_congestion` | `pulmonary edema`, `vascular congestion/engorgement`, `interstitial edema`, `prominent vascular markings` |

### Step 3 — severity scoring

Within each sentence containing a finding, the severity patterns fire (in order) and the highest-severity match wins. If a sentence mentions a finding but has no severity qualifier, a default of **2 (mild)** is assigned.

Negation (`"no "`, `"without "`, `"absent"`, `"free of"`) and normal-range modifiers (`"normal"`, `"unremarkable"`, `"within normal limits"`, `"not enlarged"`) both score **0**.

### Step 4 — laterality

Each positive sentence is examined for:
- `"bilateral"`, `"bibasilar"`, `"basal"`, `"both lung"` → both sides
- `"left"`, `"left-sided"` → left only
- `"right"`, `"right-sided"` → right only

The **collapsed** score is `max(left, right)`. Findings without a side word (e.g. `"cardiomegaly"`) get no laterality columns.

## Running the extraction

```bash
cd /path/to/phase_grounding
uv run --with pyarrow --with pandas --with tqdm python3 extract_mimic_labels.py
```

Processes 227,835 MIMIC-CXR studies in ~5 minutes on a single CPU core. Output:
```
data/mimic_taix_labels.parquet
```

## Descriptive statistics (227,835 studies)

### Finding coverage (non-null count = finding mentioned in report)

| Finding | Studies with label | % coverage |
|---------|--------------------|------------|
| `atelectasis` | ~145,000 | 64% |
| `pleural_effusion` | ~115,000 | 50% |
| `pulmonary_opacities` | ~125,000 | 55% |
| `heart_size` | ~170,000 | 75% |
| `pulmonary_congestion` | ~70,000 | 31% |

### Severity distribution (atelectasis example)

| Score | Count | % of labelled |
|-------|-------|---------------|
| 0 (absent) | ~42,000 | 29% |
| 1 (trace) | ~8,000 | 6% |
| 2 (mild) | ~72,000 | 50% |
| 3 (moderate) | ~15,000 | 10% |
| 4 (large) | ~8,000 | 6% |

### Laterality (atelectasis)

Radiologists predominantly use "bibasilar" rather than left/right — most labels are bilateral or NaN; explicit left-only / right-only are a minority.

## Using extracted labels with RadHarmony

The parquet file contains `subject_id` and `study_id` matching MIMIC-CXR-JPG. Merge with the harmonized MIMIC-CXR-JPG DataFrame to add TAIX-style labels:

```python
import pandas as pd
from radharmony.harmonizer.mimic_cxr.mimic_cxr_jpg import MIMICCXRJPGHarmonizer
from radharmony.dataset import MIMICCXRJPGDataset

MIMIC_BASE = "/path/to/MIMIC_CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/"
LABELS_PARQUET = "/path/to/phase_grounding/data/mimic_taix_labels.parquet"

# 1. Harmonize MIMIC-CXR-JPG (builds image_path, patient_id, study_id, CheXpert labels …)
h = MIMICCXRJPGHarmonizer(
    csv_path=f"{MIMIC_BASE}mimic-cxr-2.0.0-metadata.csv.gz",
    label_csv_path=f"{MIMIC_BASE}mimic-cxr-2.0.0-chexpert.csv.gz",
)
mimic_df = h.harmonize()
# mimic_df["patient_id"] is subject_id as string, "study_id" as string

# 2. Load extracted severity labels
pg_df = pd.read_parquet(LABELS_PARQUET)
pg_df["patient_id"] = pg_df["subject_id"].astype(str)
pg_df["study_id"]   = pg_df["study_id"].astype(str)

# 3. Merge — left join keeps all MIMIC rows, NaN where no report label exists
merged = mimic_df.merge(
    pg_df.drop(columns="subject_id"),
    on=["patient_id", "study_id"],
    how="left",
)

# 4. Load dataset with the merged DataFrame (no re-harmonization)
ds = MIMICCXRJPGDataset(
    base_image_dir=f"{MIMIC_BASE}files/",
    output_cls=True,
    harmonized_df=merged,
)

sample = ds.get_datasets()[0]
print(sample["img"].shape)   # torch.Size([3, 224, 224])
# Access severity labels directly from the harmonized DataFrame:
print(merged[["study_id", "atelectasis", "atelectasis_left", "atelectasis_right",
              "pleural_effusion", "heart_size"]].head())
```

## Eval app

A standalone Gradio app (port 7861) lets you test the regex extraction on any report text:

```bash
cd /path/to/phase_grounding
uv run --with gradio --with pandas --with pyarrow python3 eval_app.py
```

Open `http://localhost:7861`:
- **Live Demo tab** — paste any report, see extracted labels with severity scores and laterality
- **Dataset Browser tab** — browse the 227,835 extracted labels, filter by finding / side / severity range

## Known limitations

- **Severity default of 2 (mild)** — when a finding is mentioned without a clear severity qualifier (e.g. `"atelectasis is present"`), we assign 2. This over-represents mild severity.
- **Bibasilar dominance** — radiologists often write `"bibasilar atelectasis"` without specifying left vs right, making the `_left`/`_right` columns NaN even when severity > 0.
- **Negation scope** — negation is sentence-level only. `"No pleural effusion. Small right pleural effusion."` correctly extracts score 1 for the right side. But complex negation like `"Effusion previously present is now resolved"` may not be captured accurately.
- **Normal heart in context** — `"cardiac silhouette is within normal limits"` → score 0 (correct). But `"stable cardiomegaly"` — `"stable"` fires a score-0 pattern; however, the finding regex also fires with no severity, yielding default 2. Both patterns coexist — the final score is max(0, 2) = 2 (mild cardiomegaly), which is reasonable.
