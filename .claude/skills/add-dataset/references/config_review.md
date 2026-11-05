# Brief: config review of a new dataset

You review the config JSON of a new RadHarmony dataset before anybody writes its code. You have no earlier
context. That is on purpose. Check each fact yourself.

You get: the config path `configs/datasets/<name>.json`, the registry keys and the data folders.

## Rules

- Report only. Do not change the config, the code or the docs. Do not change any file in the repo. Put your
  scratch scripts in your scratchpad folder, outside the repo.
- Every finding needs proof: a command and its output. A finding with no command is not a finding.
- Read the source files with pandas or plain Python. Do not import `radharmony`. The code does not exist
  yet, and a number from the code under test proves nothing (D-23).
- Use the data folders that the main agent gives you in your task. You cannot ask the user. If a folder is
  missing or not readable, skip the checks that need it, and list them in your report. Do not copy a path
  into your report.
- Do not install packages. Do not write in the data folders.

## What to check

1. **Each field against the source files.** Open the files and confirm each field. Start with the file
   names and their variants, the ID columns and the `image_path` form. Then check the label columns, the
   view column, the masks, the boxes and the reports.
2. **`base_image_dir`** (D-04). It is the deepest folder that holds every image. For 3 random rows,
   `base_image_dir` + `image_path` is a real file (or a DICOM series folder). Each official split has its own
   folder when the authors give one (D-09).
3. **One row per image** (D-03). Look for an image with more than one row in the main table. Look for a join
   key with more than one line in a second table.
4. **The design entries in `docs/wiki/design.md`.** For each one, say if the config follows it:
   - D-05 labels: the codes 1, 0, -1 and NaN, the blank labels, the rows with no label at all, the one-hot
     columns.
   - D-06 age in years, with the unit of the source. D-26 sex: the source values that map to `M`, `F` or NaN.
   - D-07 boxes: the axis order and the units of the source. Check the order on a few images, not only in
     the docs.
   - D-09 and D-25 splits: a class for each official split, and a separate test class for images with no
     labels. D-34: the patients that the splits share.
   - D-33 masks: an image with no mask is "not annotated", and an annotated image with no finding has an
     all-zero mask.
   - D-35 file names: the class finds a file only in `base_image_dir` and its parent folder.
5. **The expected stats.** Compute each number of `expected_stats` again from the source files, with your own
   code. Compare with the paper or the website. List each number that differs, and each difference with no
   written reason in `_comment`.
6. **A behavior that no design entry covers.** List it as an open question, with an example from the data.

## Output

Return your findings as text in your final message. A sub-agent cannot write report files. The main agent
saves the text as `findings.md`. One section for each finding:

```markdown
### CR-<n> <short title>
- Field: <config field, or the stat and split>
- What the config says: <quote>
- What the source says: <fact, with a number or an example row>
- Proof: <the command>
  <its output, cut to the important lines>
- Design entry: <D-xx, or "none: open question">
- Suggestion: <the change, or the question for the user>
```

End with a table of all findings: ID, field, design entry, suggestion. If you find nothing for a check,
write the check and "no finding", with the command that you ran.
