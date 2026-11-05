# Repo parts for the whole-repo review (`/evaluate --full`)

Each repo part gets 2 bug hunters (opus and sonnet) and 1 checker (haiku).
The intent file lists the design decisions that apply to the part, and the test steps that `docs/wiki/design.md` does not list.

| ID | Repo part | Intent | Code | GPU |
|---|---|---|---|---|
| H1 | Harmonizers 1 | `intents/harmonizers.md` | `radharmony/harmonizer/` brax, chest_imagenome, chestxdet10, chestxray14, chexinstruct_vqa, chexlocalize, chexpert (with chexpert_plus), chexphoto, ct_rate, emory_cxr, gemex_vqa, medpmc, medseg_covid_ct, mimic_cxr, mimic_ext_cxr_qba | no |
| H2 | Harmonizers 2 and the base harmonizer | `intents/harmonizers.md` | `radharmony/harmonizer/` base.py, base_vqa.py, _age.py, _bbox.py, _dicom_tags.py, _ids.py, _labels.py, _split.py, montgomery_cxr, mrrate, ms_cxr, ms_cxr_t, node21, openi_cxr, padchest, padchest_gr, qata_cov19, radchestct, ranzcr_clip, rexgradient, roco, rocov2 | no |
| H3 | Harmonizers 3 | `intents/harmonizers.md` | `radharmony/harmonizer/` rsna_2022_cervical_spine, rsna_2024_lumbar_spine, rsna_abdominal_trauma_2023, rsna_bone_age, rsna_pe_detection, rsna_pneumonia, shenzhen_cxr, siim_acr_ptx, siim_covid19, taix_ray, vindr_cxr, vindr_pcxr, vqa_rad | no |
| D | Dataset classes and transforms | `intents/datasets.md` | `radharmony/dataset/` (with base.py, base_vqa.py, transforms.py) | no |
| P | Preprocessors and utils | `intents/preproc.md` | `radharmony/preprocessor/`, `radharmony/utils/`, `radharmony/evaluator/transforms.py`, `radharmony/evaluator/wrappers.py` | no |
| A | App, scripts, registry, integrity checker | `intents/app.md` | `app.py`, `radharmony_internal/` (if the repo has it), `radharmony/registry.py`, `radharmony/evaluator/_registry.py`, `radharmony/integrity.py`, `scripts/` (not `scripts/legacy/`), `pyproject.toml` | no |
| B | Backbones | `intents/backbones.md` | `radharmony/evaluator/backbones/` | yes |
| E | Evaluators and metrics | `intents/evaluators.md` | `radharmony/evaluator/` (not backbones), with `evaluator/base.py`, `classification/base.py`, `segmentation/base.py` | if needed |

For a branch review (no `--full`), the changed files are one repo part, named `branch`. Give the hunters the intent files of the parts that the diff touches.

## Data and environments

- Real data. The data folders come from the data paths file: `configs/data_paths.json`, or the file that
  `RADHARMONY_DATA_PATHS` names (`docs/wiki/get-started/install.md`, "Data paths for the checks").
  Read them with `import sys; sys.path.insert(0, "scripts"); from dataset_paths import paths, entry, path_state`.
  Then `paths("vindr_cxr_train")` gives the constructor arguments. Read `entry(key).get("note")` too.
  A key with the status `alias` has no paths. Use a key from its `use` list.
- The main agent can also give data folders in your task. Use them for keys with no entry.
- If the file or the key is missing, `paths()` raises `MissingPathsError` with a hint. A sub-agent cannot ask the
  user: skip the real-data check, and report what you skipped. Mark the item BLOCKED.
- A dataset is BLOCKED when you cannot read its files: `paths(key)` raises, or `path_state(path)` is not `"ok"`
  for one of its paths. Test its logic on small fake inputs only, and say that real data was not used.
- The dev venv: `.venv/bin/python`. It can lack `transformers`, `timm` or `open_clip`.
- The model venv: `$RADHARMONY_MODEL_PYTHON`, the Python of a venv with the backbone packages (for example
  `/path/to/model/.venv/bin/python`). Run it as `PYTHONPATH=<repo root> $RADHARMONY_MODEL_PYTHON`. Without
  `PYTHONPATH`, that venv can import an installed copy of the code, not the repo copy. If the variable is not
  set, the main agent asks the user for the path and gives it in your task.
- GPU: pick the GPU with the most free memory (`nvidia-smi`). Give each GPU agent its own GPU.
