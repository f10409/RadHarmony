# Investigation — Lumbar Spine "more than one DICOM series" warnings

**Status:** ⏳ **awaiting Frank's input** — investigation done, harmonization options A–D drafted, no option chosen yet, no implementation work started
**Authors:** Judy + Claude
**Date:** 2026-04-25
**For:** Frank, who shipped the per-series harmonizer refactor (`ef8f089`).
**Pertains to:** `radharmony/dataset/rsna_2024_lumbar_spine.py`, `radharmony/harmonizer/rsna_2024_lumbar_spine.py`, optionally `transforms.py` (loader configuration).
**Decision needed:** see "Open questions for Frank" near the bottom — the load-bearing question is whether the multi-stack axial coverage actually matters for the level-specific labels.

---

## What surfaced this

Running the integrity-check cell on RSNA 2024 Lumbar Spine after Frank's series-id refactor:

```
DONE  (706.2s)
  success: 6,294
  failed:  0
```

All 6,294 per-series rows loaded without exception — the schema change is working. **But 36 ITK warnings appeared during the iteration:**

```
the directory: .../train_images/<study>/<series> contains more than one DICOM series.
```

Looked benign at first ("ITK saw something weird, used a default, kept going"). On investigation it turns out **the 36 affected volumes are silently being loaded with only ~50% of their actual slices** — a real data-completeness issue.

---

## Investigation findings

### The scope is narrow but the impact per affected sample is large

| Fact | Value |
|---|---|
| Affected dirs | **36** out of 6,294 series (0.6%) |
| All affected are | **Axial T2** sequences (no Sagittal T1, no Sagittal T2/STIR triggers it) |
| Affected fraction within Axial T2 | **36 / 2,340 = 1.5%** of all axial T2 series |
| Slices loaded vs slices on disk for affected dirs | **~24–54% (median ~49%)** |
| Slices loaded vs slices on disk for clean Axial T2 controls | **100%** |

So: a small slice of the dataset (1.5% of axial T2), but for those samples we're losing roughly half the data without anything in the harmonized df indicating it.

### What ITK is actually complaining about

Per-file pydicom inspection of all 36 affected dirs:

- **Each affected directory contains exactly one `SeriesInstanceUID`.** The "more than one DICOM series" message is misleading by DICOM-standard reckoning — there really is only one series.
- All 36 dirs have **2 distinct `ImageOrientationPatient` values** within the single series. So do many *clean* axial T2 dirs. Orientation alone is not the splitter.
- Slice-position analysis on a representative pair:

| Dir | Files | Distinct orientations | Slice-step median (mm) | Largest gap (mm) | Gap / median |
|---|---|---|---|---|---|
| `114899184/1364910156` (affected) | 50 | 2 | 4.36 | 60.94 | **14×** |
| `4003253/2448190387` (clean) | 43 | 2 | 4.40 | 28.46 | 6.5× |

ITK's GDCM-based reader appears to split a series into "loadable sub-series" when slice-position discontinuity exceeds some internal threshold. Both affected and clean have multi-stack geometry (typical for axial lumbar acquisition where each disc level L1–S1 gets its own stack), but the affected ones have a *bigger* inter-stack gap that crosses ITK's threshold. The threshold is not documented in our code — we'd need to confirm by reading GDCM source if precise replication matters.

### What ITKReader does after splitting

When ITK splits one DICOM directory into multiple loadable sub-series, `LoadImageD(reader="ITKReader")` silently picks **one** of them (in our testing, the first stack by file order) and ignores the others. Confirmed empirically:

```
AFFECTED  114899184/1364910156   50 files   loaded shape (608, 608, 24)   24/50 = 48%
AFFECTED  1258848546/178314290   49 files   loaded shape (608, 608, 23)   23/49 = 47%
AFFECTED  2019892566/2390719503  25 files   loaded shape (384, 384,  6)    6/25 = 24%
CLEAN     4003253/2448190387     43 files   loaded shape (320, 320, 43)  43/43 = 100%
CLEAN     4646740/3201256954     54 files   loaded shape (384, 384, 54)  54/54 = 100%
```

For lumbar-spine axial T2, each stack typically corresponds to one disc level. So "load only the first stack" means **the loaded volume covers one disc level instead of all five (L1–S1)**. For models trained on level-specific labels (e.g. `subarticular_stenosis_l4_l5`), that's a meaningful reduction in input signal.

---

## Specific examples Frank can reproduce

### Reproduction recipe

```python
from monai.transforms import LoadImage
import os

reader = LoadImage(reader="ITKReader", ensure_channel_first=False)
BASE = "/path/to/rsna-2024-lumbar-spine-degenerative-classification/train_images"

for relpath in [
    "114899184/1364910156",   # affected
    "1258848546/178314290",   # affected
    "2019892566/2390719503",  # affected — most degenerate (24%)
    "4003253/2448190387",     # clean control
]:
    n_files = sum(1 for f in os.listdir(f"{BASE}/{relpath}") if f.endswith(".dcm"))
    vol = reader(f"{BASE}/{relpath}")
    if isinstance(vol, tuple):
        vol = vol[0]
    print(f"{relpath:30}  files={n_files:3d}  loaded={tuple(vol.shape)}  "
          f"frac={min(vol.shape) / n_files:.0%}")
```

Expected output (verified 2026-04-25):

```
114899184/1364910156          files= 50  loaded=(608, 608, 24)  frac=48%
1258848546/178314290          files= 49  loaded=(608, 608, 23)  frac=47%
2019892566/2390719503         files= 25  loaded=(384, 384,  6)  frac=24%
4003253/2448190387            files= 43  loaded=(320, 320, 43)  frac=100%
```

### Full list of 36 affected directories

```
114899184/1364910156
1258848546/178314290
1271819130/396937199
1537608176/2603019387
1538136131/533252904
1603568458/3451074679
1973833645/1328374636
1992037544/2051789894
1995123254/1133656256
2019892566/2390719503
2083466060/2440822236
2334206006/4191635045
2470505035/1297079442
2568819355/3366910731
2755347468/3708423406
2780132468/4151611107
2795583238/3365980706
3339741647/4195187002
335455502/790180412
3542237003/458336097
3651144029/1415702790
3707028884/3238050001
3884015124/1519537115
3885334932/4045988081
3930841971/2216414062
3956571539/4190692765
4024872715/2479444365
4165566893/4033318924
4165566893/807064932
4193490688/549563660
582364168/3242313646
594735110/4075111689
74782131/3401861580
766494595/519827232
809072026/2594153114
979209761/3101981332
```

(All paths relative to `/path/to/datasets/external/rsna-2024-lumbar-spine-degenerative-classification/train_images/`.)

---

## Proposed harmonization options

Four approaches, ordered easiest → most invasive. Each has different ergonomic and signal trade-offs.

### A. Document and ignore  *(safest, cheapest)*

Add a one-line note to `docs/dataset_guide.md` under the lumbar spine entry: *"For ~1.5% of Axial T2 series, ITK loads only the first stack of the multi-stack acquisition (~50% of slices). If your model is sensitive to multi-level coverage, see `docs/proposals/lumbar_spine_multi_series_dirs.md`."* No code change. Acknowledges the issue, lets users decide whether it matters for their use case.

**When this is right:** if the downstream task is fine with one-disc-level coverage (e.g. patient-level predictions), or if 0.6% of total samples being half-coverage doesn't move the needle.

### B. Drop the affected rows from the harmonized df  *(small diff, defensible)*

Detect at harmonize time which directories ITK would split (by counting DICOMs and comparing to a quick `LoadImage(...).shape` probe, or by reading positions and looking for big jumps). Drop those rows from the per-series df. Result: harmonized count drops from 6,294 → 6,258.

```python
# pseudocode in RSNA2024LumbarSpineHarmonizer._build_image_path or post-pass
for idx, relpath in df["image_path"].items():
    n_files = ...
    loaded = ...  # quick probe
    if loaded.shape[-1] < n_files * 0.9:  # >10% slice loss
        df.drop(idx, inplace=True)
```

**When this is right:** if you want a clean dataset where every loaded volume is "complete" by definition, even at the cost of dropping 36 samples.
**Cost:** ~1.5% of axial T2 silently disappears from training; users who need those need to know to look elsewhere.

### C. Switch the reader to `pydicom` + manual slice assembly  *(medium)*

Stop relying on ITK's series-splitting heuristic. Build the volume ourselves: read every `.dcm` in the directory with pydicom, sort by `InstanceNumber` (or `ImagePositionPatient[2]`), stack into a `numpy` array, wrap in `MetaTensor`. Bypasses the GDCM splitter entirely.

**When this is right:** if maximum coverage matters and we're willing to maintain our own slice-stacking code path. Reduces dependency on ITK's opaque internal logic.
**Cost:** more code to maintain; need to handle the multi-orientation case ourselves (some affected dirs really *do* have 2 different scan planes mixed in one series — a custom loader has to decide whether to stack them or split them); loses ITK's robustness on edge cases (corrupt DICOMs, etc.).

### D. Split each affected dir into multiple harmonized rows  *(most invasive, highest fidelity)*

Detect ITK's split points at harmonize time, emit *N* rows per affected directory (one per detected stack). Each row gets its own synthetic `series_id` (e.g. `<series_id>_stack0`, `<series_id>_stack1`, ...) so they're uniquely addressable. Result: harmonized count grows from 6,294 → ~6,360.

The level-specific labels in `train.csv` (`*_l1_l2_*`, `*_l2_l3_*`, etc.) could potentially be routed to the *correct* stack rather than blindly attached to all of them — but that requires knowing which stack covers which disc level, which would mean inspecting `ImagePositionPatient` and matching against vertebra-level annotations. Significant work.

**When this is right:** if multi-disc-level granularity is a first-class need (e.g. training one model per disc level, or per-stack attention maps).
**Cost:** Substantial. Needs a level-detection heuristic, possibly per-stack label routing, definitely careful versioning since it changes the harmonized df shape.

### My (Claude's) recommendation

**A or B**, with **A as the default**. The 36 affected studies are 1.5% of axial T2 / 0.6% of all series, and ITK is loading a coherent (if smaller) sub-volume — not corrupt data, just incomplete. For most uses that's tolerable, and a doc note lets affected users opt out. Move to D only if a specific downstream task hits the limitation.

---

## Open questions for Frank

1. **Is multi-level axial coverage actually needed for the labels we care about?** All level-specific labels (`*_l1_l2_*` through `*_l5_s1_*`) are study-level — not series-level — in the CSV. Frank, do you know whether those labels were annotated by reading specific stacks, or globally for the whole spine? If the latter, the dropped stacks may not lose label-relevant signal.
2. **Did you intentionally trust ITK's splitter, or was it just the path of least resistance?** If the latter, options C/D become more attractive; if the former, A/B is the consistent choice.
3. **Acceptable to ship the harmonizer as-is** with a doc note (option A), or do you want to land option B before this dataset is used in any serious training?
4. **Stack-to-disc-level mapping for option D** — is there a published RSNA-side ground-truth mapping anywhere, or would we need to derive it from slice positions?

---

## Out of scope

- Investigating GDCM's split threshold internals — interesting but not needed to choose a harmonization approach.
- Other datasets — this investigation is lumbar-spine-specific. Other multi-series datasets (e.g. RSNA PE, where each "series" has thousands of slices) don't show this pattern in our integrity runs.
- Whether the missing stacks contain any unique label signal — would need a separate analysis cross-referencing slice positions with the annotated vertebra levels in `train_label_coordinates.csv`.

---

## Reproducibility scripts

Investigation scripts used to produce the numbers above are in `/tmp/` for now (one-off analysis):

- `/tmp/investigate_multi_series.py` — per-dir SeriesInstanceUID + SeriesDescription tally
- `/tmp/compare_dicom_geometry.py` — slice-position diff between an affected and a clean control
- `/tmp/measure_impact.py` — load-vs-disk slice-count comparison

Happy to upstream any of these into `notebooks/` if useful for downstream debugging.
