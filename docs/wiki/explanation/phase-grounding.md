# MIMIC-CXR label extraction

TAIX-Ray gives each finding a grade from 0 to 4 (0 to 3 for heart size). Some findings also have a grade for
each side (left and right). MIMIC-CXR-JPG gives only the CheXpert codes (1, 0, -1 or blank), with no grade and
no side. These codes cannot become TAIX-Ray grades.

But the free-text MIMIC-CXR reports often say how large a
finding is and on which side. So a script reads the reports and finds the grade and the side with regular
expressions. It runs on the local computer, with no external API, so no report leaves the computer.

The script is `extract_mimic_labels.py` in the separate `phase_grounding` project. It is not in this repo.
This page explains what it does and where it can be wrong. The numbers on this page come from its output
file `mimic_taix_labels.parquet`, read on 2026-11-04.

## Output

The output has one row for each MIMIC-CXR study (version 2.1.0 reports). The script skips a study with no
report file. It has 13 columns:

| Column | Values |
|---|---|
| `subject_id`, `study_id` | The MIMIC-CXR IDs, as integers |
| `atelectasis`, `pleural_effusion`, `pulmonary_opacities` | The grade of the finding: the larger of the left and right grades |
| `atelectasis_left`, `atelectasis_right`, `pleural_effusion_left`, `pleural_effusion_right`, `pulmonary_opacities_left`, `pulmonary_opacities_right` | The grade on one side |
| `heart_size`, `pulmonary_congestion` | The grade of the finding. These findings have no side |

The grades:

| Grade | Words in the sentence |
|---|---|
| 0 | no, without, absent, resolved, clear, free of, no evidence of, normal, unremarkable, within normal, stable, unchanged, not enlarged |
| 1 | trace, minimal, tiny, subtle, subcentimeter, micro, small |
| 2 | mild, slight, early, low-grade, minor. Also the default when the sentence has no grade word |
| 3 | moderate, moderate-sized |
| 4 | large, severe, massive, extensive, significant, marked, considerable |
| NaN | The report does not mention the finding |

The TAIX-Ray `heart_size` has the grades 0 to 3 ([TAIX-Ray](../reference/datasets/cxr-labels/taix_ray.md)).
The script gives `heart_size` the grades 0 to 4, so grade 4 has no TAIX-Ray match.

## How a report becomes grades

The script uses the FINDINGS section of the report. With no FINDINGS section, it uses the IMPRESSION
section. With neither, it uses the full text. It splits the text into sentences at each `.`, `;` and new
line.

A sentence mentions a finding when it has one of the words of that finding:

| Finding | Words (and their forms) |
|---|---|
| `atelectasis` | atelectasis, atelectatic |
| `pleural_effusion` | pleural effusion, pleural fluid, hydrothorax, effusion |
| `pulmonary_opacities` | opacity, infiltrate, consolidation, airspace disease |
| `heart_size` | cardiomegaly, enlarged heart, cardiac enlargement, enlargement of the heart, heart size, cardiac silhouette, "heart is ... enlarged" |
| `pulmonary_congestion` | pulmonary edema, interstitial edema, vascular congestion, vascular engorgement, vascular plethora, "vascular markings ... prominent" |

For each sentence that mentions the finding, the script finds every grade word of the sentence. When the
sentence has a grade word above 0, the highest such grade wins. A word of grade 0 counts only when the
sentence has no grade word above 0. A sentence with no grade word gives 2. Over all sentences of the
report, the highest grade wins.

The side of a lateral finding (atelectasis, pleural effusion, opacities) comes from the same sentence:

- "bilateral", "bibasilar", "bibasal", "basal", "both lung" or "both side" give both sides. The plurals
  "both lungs" and "both sides" do not match.
- "left" and "right" in one sentence give both sides.
- Only "left" gives the left side. Only "right" gives the right side.
- A sentence with no side word gives both sides.

Examples, from a run of the script on 2026-11-04:

| Sentence | Output |
|---|---|
| "Small bilateral pleural effusions." | `pleural_effusion` 1, left 1, right 1 |
| "IMPRESSION: Moderate pulmonary edema." | `pulmonary_congestion` 3 |
| "Heart size is normal." | `heart_size` 0 |
| "No pleural effusion. Small right pleural effusion." | `pleural_effusion` 1, left 0, right 1 |

## What the output looks like

The percent of all studies that mention each finding:

| Finding | Studies with a grade |
|---|---|
| `pleural_effusion` | 67% |
| `pulmonary_opacities` | 46% |
| `heart_size` | 37% |
| `atelectasis` | 25% |
| `pulmonary_congestion` | 24% |

Of the studies that mention `atelectasis`, 7% have grade 0, 9% grade 1, 78% grade 2, 3% grade 3 and 3%
grade 4. In 73% of these studies, the left and right grades are the same. 15% have a grade on the left
side only, and 10% on the right side only.

## Where the output can be wrong

- Grade 2 is too common. A finding with no grade word gets 2, for example "Atelectasis is present". So 78%
  of the atelectasis grades are 2.
- The side is often a guess. A sentence with no side word gives the grade to both sides. A sentence with
  both "left" and "right" gives each finding both sides. Example: "Left lower lobe opacity and right
  pleural effusion." gives both sides to the opacity and to the effusion. "Basal" gives both sides, so "Mild
  basal atelectasis on the left." gives both sides. "Opacities in both lungs, worse on the left." gives the
  left side only.
- A negation covers the full sentence, not one finding. A grade word above 0 wins over a negation. Example:
  "No pleural effusion but mild atelectasis." gives `pleural_effusion` 2, not 0, because "mild" (grade 2) is
  in the same sentence.
- A FINDINGS section ends at the first header word, also inside a sentence and in lower case. The header words
  are FINDINGS, IMPRESSION, TECHNIQUE, COMPARISON, INDICATION, EXAMINATION, HISTORY and PROCEDURE. Example: the
  report "FINDINGS: In comparison with the study of ___, there is a new large left pleural effusion." gives
  the section "In". So the report gets no grade.
- The word "effusion" alone counts as a pleural effusion. Example: "Small pericardial effusion." gives
  `pleural_effusion` 1 on both sides.
- "Prominent vascular markings." gives no grade, because the pattern needs "vascular marking" followed by
  "prominent", or the singular "prominent vascular marking".
- Grade 1 has another meaning than in TAIX-Ray. In TAIX-Ray, grade 1 is "Questionable". Here, grade 1 is a
  small finding, for example "small effusion".
- "Stable" gives 0. Example: "Stable cardiomegaly." gives `heart_size` 0. The project decision D-44 says
  that a stable finding is present, with grade 2. The script does not follow D-44 yet.

## Use the grades with RadHarmony

`subject_id` and `study_id` match the `patient_id` and `study_id` of the MIMIC-CXR-JPG harmonized table, after
you change the IDs from integers to text. This block adds the grades to the table of the official train split.
The prefix `taix_` keeps the grade columns apart from the CheXpert label columns with the same names, for
example `pleural_effusion`.

<!-- block: run=skip reason="needs mimic_taix_labels.parquet of the phase_grounding project, which is not in configs/data_paths.json" -->
```python
import pandas as pd
from radharmony.dataset import MIMICCXRJPGTrainDataset

ds = MIMICCXRJPGTrainDataset(base_image_dir="/data/mimic-cxr-jpg/2.0.0/files/", output_cls=True)
df = ds.get_harmonized_df()

grades = pd.read_parquet("/data/phase_grounding/mimic_taix_labels.parquet")
grades = grades.astype({"subject_id": str, "study_id": str}).rename(columns={"subject_id": "patient_id"})
grades = grades.set_index(["patient_id", "study_id"]).add_prefix("taix_").reset_index()
merged = df.merge(grades, on=["patient_id", "study_id"], how="left")
print(len(merged) == len(df), merged["taix_pleural_effusion"].notna().mean().round(2))  # True 0.7
```

The grades are extra columns of the table. A dataset class made with `harmonized_df=merged` keeps the 14
CheXpert labels in `cls`, because it takes its label columns from its harmonizer.

The `phase_grounding` project also has a Gradio app, `eval_app.py`, on port 7861. Its tabs run the script
on a report that you paste, and show the rows of the output file.

## See also

- [Label policy](label-policy.md): the label codes of RadHarmony.
- [MIMIC-CXR-JPG](../reference/datasets/cxr-labels/mimic_cxr_jpg.md) and
  [TAIX-Ray](../reference/datasets/cxr-labels/taix_ray.md).
