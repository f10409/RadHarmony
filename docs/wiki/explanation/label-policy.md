# Label policy

Each source writes its labels in its own way. One dataset writes `1.0`, `0.0`, `-1.0` and blanks. Another
writes a list of finding names, or one grade from 0 to 4. RadHarmony puts the labels of every dataset in
one format. So a probe can use the label columns of any dataset with no code for that dataset (D-05).

This page explains the rules and why we chose them. The page of each dataset gives its own label rules
([Dataset catalog](../reference/datasets/index.md)).

## The 4 codes

| Code | Meaning | What the probes do |
|---|---|---|
| `1` | Positive: the finding is there | Use it |
| `0` | Negative: the finding is not there | Use it |
| `-1` | Uncertain: the source says "maybe" | Skip it |
| NaN | Not mentioned, or not known | Skip it |

These codes are the CheXpert codes, which many papers use. The probes fit and score each label only on the
rows where it is 0 or 1. So a NaN label is not a negative. The project keeps `-1` in `cls`
(D-05). Before, `-1` became NaN, so `cls` lost the difference between "uncertain" and "not mentioned".

## Uncertain labels

The dataset classes of CheXpert, CheXpert Plus, CheXphoto, MIMIC-CXR and MIMIC-CXR-JPG take the argument
`drop_uncertain`. With `True`, the class drops each row with `-1` in any label. With `False`, the row
stays, and `cls` keeps `-1`. The default is `True` in most classes. The labeled MIMIC-CXR-JPG test class
keeps `-1` by default. The page of each dataset gives its default.

PadChest also has `drop_uncertain`, but
there it drops the rows labeled `exclude` or `suboptimal_study`.

## Blank labels

A CheXpert-style source leaves a label blank when the report does not mention the finding. The argument
`not_mentioned` sets what a blank gives:

- `"negative"` (the default) gives 0, as most CheXpert papers do.
- `"nan"` keeps NaN, so the probes skip it.

Why 0 is the default: 73% of the CheXpert train label cells are blank. With NaN, each probe trains only on
the few rows that mention its finding. For example, Pleural Other is blank in 97% of the CheXpert train
rows.

## A row with no label at all

Some rows have no label at all. Examples: a PadChest row whose label list is empty, or a CheXpert-style
row where every label is blank. Such a row gets NaN in every label (D-05). This is also true with
`not_mentioned="negative"`, because the label is unknown, not negative. Example:
777 PadChest rows have the raw `Labels` value `['']`, and their reports name findings such as scoliosis. As
negatives, they would teach the probe wrong labels.

## Other rules

- A class that a source does not annotate gets NaN, not 0. Example: the MedSeg dataset2 masks have no
  pleural effusion class, so `pleural_effusion` is NaN in these rows.
- A blank source value that a harmonizer splits into one-hot columns gives NaN in all of these columns, not
  0. Example: a blank RSNA 2024 lumbar severity gives NaN in `normal_mild`, `moderate` and `severe`.
- A trigger word sets only its own finding. Example: the OpenI term `Lung/hypoinflation` does not set
  `atelectasis`.
- A harmonizer can keep raw grades, for example the TAIX-Ray grades 0 to 4. Its dataset class maps them to 0
  and 1 by default (`label_mode="binary"`), and `"ordinal"` keeps the grades
  ([TAIX-Ray](../reference/datasets/cxr-labels/taix_ray.md)).
- A regression target, for example the bone age, goes in its own columns (`REG_COLS`) and in `reg`, not in
  `cls`.

## Label names

Label columns have snake case names: `pleural_effusion`, not `Pleural Effusion`. The harmonizer sorts them
in alphabetical order, and `cls` uses the same order (D-05). So the names and the order of `cls` do not
depend on the source. Box labels (`bbox_labels`) use snake case too (D-07).

## Labels from report text

Some labels come from the free text of a report, not from a label file. Then a rule must decide what a
phrase means. Example: "stable cardiomegaly" means that the heart is still large, so the finding is present
(D-44). [MIMIC-CXR label extraction](phase-grounding.md) uses such rules.
