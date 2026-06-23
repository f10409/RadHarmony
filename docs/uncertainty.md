# Label Uncertainty in RadHarmony

A discussion document for shaping the project-wide strategy on uncertain
labels. Drafted while integrating BRAX (which surfaced inconsistencies with
how CheXpert / MIMIC-CXR handle the same `-1` "uncertainty" code) — should
be reviewed with Frank before any backport / breaking change.

## What "uncertainty" actually means in our datasets

Every dataset that derives labels from radiology reports via the **CheXpert
NLP labeller** ships cells in a small, well-defined alphabet:

| Cell value | Meaning |
|---|---|
| `1` | Mention of the finding extracted as **positive** |
| `0` | Mention extracted as **negation** |
| `-1` | Mention extracted but the labeller could **not disambiguate** positive vs negative (uncertainty) |
| blank / NaN | The labeller did **not extract a mention** of this finding |

This applies to:
- **CheXpert** (Stanford, English reports, original labeller)
- **MIMIC-CXR / MIMIC-CXR-JPG** (BIDMC, English reports, CheXpert labeller)
- **CheXpert-Plus** (Stanford follow-up, same encoding)
- **BRAX** (Brazilian, Portuguese reports, NegEx-Portuguese front-end + CheXpert back-end)

The four datasets share a label vocabulary of 14 findings and the same
`{1, 0, -1, NaN}` cell semantics. Anything we decide here applies to all
of them.

## Why the choice matters

The CheXpert paper (Irvin et al., AAAI 2019) compared four mappings —
**U-Zeros**, **U-Ones**, **U-Ignore**, **U-MultiClass** — and showed
**there is no universal best**:

- **Atelectasis, Edema** — U-Ones consistently wins (uncertain mentions of
  these almost always mean the finding *is* present in some borderline form)
- **Cardiomegaly, Consolidation** — U-Zeros wins (uncertain mentions tend
  to be ruled out on closer reading)
- **Pleural Effusion** — gains from U-Ignore + per-pathology calibration
- Best overall AUC on their CheXpert validation came from a **per-finding
  mix**, not a single global strategy

So the "right" strategy depends on the pathology, the loss function, and
the downstream task. RadHarmony cannot pick one for the user — but it
**can** make the choices visible and uniform across datasets.

## Where RadHarmony stands today (as of `integration-brax`)

| Dataset | Default behaviour | API |
|---|---|---|
| CheXpert | drop any row with a `-1` | `drop_uncertain: bool = True` |
| MIMIC-CXR / MIMIC-CXR-JPG | drop any row with a `-1` | `drop_uncertain: bool = True` |
| CheXpert-Plus | (TBD — verify in source) | TBD |
| BRAX (this branch) | preserve cells exactly | `uncertain_strategy: str = "raw"` |

This is **inconsistent**. Three problems:

1. **Default destruction.** `CheXpert(drop_uncertain=True)` silently throws
   away ~30% of the studies. Most users never realise. BRAX took the other
   stance (preserve) which is faithful but yields `-1` in `cls` tensors,
   which destroys most loss functions if not handled.
2. **No middle ground.** The legacy `drop_uncertain=False` flag keeps the
   row but writes `NaN` into the cls tensor — which also kills BCE/CE
   gradients silently. There is no built-in U-Zeros / U-Ones / U-Ignore
   mapping for CheXpert / MIMIC-CXR.
3. **Different vocabulary.** "drop_uncertain" is a binary; "uncertain_strategy"
   is a string with five options. Code that wraps both has to special-case.

## Proposed strategy options (already implemented in `BRAXHarmonizer`)

Five mappings, applied per cell:

| Strategy | NaN → | -1 → | Row dropped? | Use case |
|---|---|---|---|---|
| `"raw"` | NaN (preserved) | -1 (preserved) | no | faithful representation; uncertainty diagnostics; custom loss |
| `"u_zeros"` | 0 | 0 | no | conservative; baseline most papers report |
| `"u_ones"` | 0 | 1 | no | aggressive; matches Atelectasis / Edema sensitivity findings |
| `"u_ignore"` | 0 | NaN | no | Bernoulli-with-mask training; loss must mask `NaN` |
| `"drop"` | 0 | NaN, then `dropna(subset=labels)` | yes | matches today's `CheXpert(drop_uncertain=True)` |

`"raw"` is BRAX's default. `"u_zeros"` is the safest training-ready
default. Both have legitimate claims to being the project default.

## Open questions for Frank

1. **What should the project default be?** Faithful preservation (`"raw"`)
   or training-ready (`"u_zeros"`)? Trade-off: faithfulness vs. footgun.
   - Argument for `"raw"`: BRAX default; matches the "harmonize, don't
     mutate" principle of harmonizers. Researchers can opt into a strategy
     once, instead of having a default strategy applied silently.
   - Argument for `"u_zeros"`: minimum surprise for users running
     `output_cls=True` and a stock BCE loss. Aligns with the paper's
     baseline.
2. **Backport to CheXpert / MIMIC-CXR / MIMIC-CXR-JPG / CheXpert-Plus?**
   - Pro: one consistent API, retire `drop_uncertain` as a deprecation
     alias for `uncertain_strategy="drop"`.
   - Con: behavioural change to four heavily-used datasets; needs a release
     note and version bump; downstream notebooks pinning the old default
     may break silently.
   - Suggested path: ship the new API alongside the old one for one minor
     version, emit a `DeprecationWarning` from `drop_uncertain`, then
     remove.
3. **Do we want a per-finding strategy?** The CheXpert paper's best result
   was a mix (U-Ones for some classes, U-Zeros for others). API sketch:
   ```python
   uncertain_strategy = {
       "atelectasis": "u_ones",
       "cardiomegaly": "u_zeros",
       ...,
       "default": "u_zeros",
   }
   ```
   Worth doing now, or defer until someone actually asks?
4. **Should there be a paired uncertainty mask in the sample dict?**
   When `output_cls=True`, optionally also yield `cls_certain` — a
   `{0, 1}` tensor that's `0` where the original cell was `-1` or `NaN`.
   Lets a loss multiply by the mask without re-deriving it from `-1`
   sentinels. Adds one tensor key; needs a flag.
5. **Smoothed / probabilistic targets?** Mapping `-1 → 0.5` is a fifth
   strategy used by some papers (treat uncertain as a soft label). Worth
   adding as `"smooth"` or out of scope?
6. **Uncertainty across labellers.** BRAX uses NegEx-Portuguese on top of
   CheXpert. Are the `-1`s in BRAX comparable to the `-1`s in CheXpert? If
   the calibration is different we may want a per-source post-processing
   step. Probably moot for our use cases, but flag it.

## Suggested phased plan (proposal, not committed)

**Phase 0** — current state: BRAX has the new five-strategy API; legacy
datasets keep `drop_uncertain`.

**Phase 1** — write `radharmony/utils/uncertainty.py` with a single
`apply_strategy(df, label_cols, strategy)` helper that all four
CheXpert-labeller harmonizers call. No behaviour change yet — just
deduplicate the logic.

**Phase 2** — add `uncertain_strategy` to CheXpert / MIMIC-CXR /
MIMIC-CXR-JPG / CheXpert-Plus. Map old `drop_uncertain=True` → new
`uncertain_strategy="drop"`. Keep both for one minor version;
`drop_uncertain` raises `DeprecationWarning`.

**Phase 3** — pick a project default (resolve question 1 above). Apply
uniformly. Bump major version if changing today's CheXpert default.

**Phase 4** — optional: per-finding dict strategy (question 3),
uncertainty mask (question 4), smoothed targets (question 5). Nice-to-haves
that ride on the Phase 1 helper.

## What would unblock a decision

- A short experiment: pick three pathologies (Atelectasis, Cardiomegaly,
  Pleural Effusion), train a stock CheXpert baseline four times — once per
  strategy — and report per-class AUC on a held-out test set. Reproduces
  the CheXpert paper finding on RadHarmony's pipeline and gives us our own
  numbers to point to when defending the default.
- Frank's read on the deprecation cost — how many internal notebooks and
  downstream consumers rely on `drop_uncertain=True` being the silent
  default?
- Confirmation of CheXpert-Plus's current behaviour (this doc has a TBD).

## References

- Irvin et al., *CheXpert: A Large Chest Radiograph Dataset with
  Uncertainty Labels and Expert Comparison.* AAAI 2019. — defines the
  four uncertain-mapping strategies and reports per-pathology results.
- Reis et al., *BRAX, Brazilian labeled chest X-ray dataset.* Scientific
  Data 2022. — confirms BRAX uses the CheXpert labeller back-end with a
  Portuguese NegEx front-end.
- `radharmony/harmonizer/brax.py` — current implementation of the
  five-strategy API.
- `radharmony/harmonizer/chexpert.py` / `mimic_cxr.py` / `mimic_cxr_jpg.py`
  — current `drop_uncertain` implementations to be migrated.
