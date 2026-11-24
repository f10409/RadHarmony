"""Tests for configs/data_paths.json, the one source of NAS paths.

The CI tests check the config against the code: every registry key has an
entry, and every path argument exists in the class signature. The NAS test
runs only where NAS3 and NAS4 are mounted, and checks that every path exists.
It cannot see inside a folder that you cannot read, so a typo below a
"restricted" folder passes.

Run with::

    pytest tests/test_data_paths.py -v
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import radharmony.dataset  # noqa: E402,F401  (registers the datasets)
from radharmony.registry import list_datasets, resolve_dataset  # noqa: E402

from dataset_paths import is_complete, load, path_state, paths  # noqa: E402
from fixtures.init_params import init_params  # noqa: E402

# The dev repo commits a complete file. A user file with some entries (or no file) skips this module:
# tests/test_data_paths_public.py checks that case and the example file.
if not is_complete():
    pytest.skip("the data paths file of this clone is not complete", allow_module_level=True)
import gen_nas_paths_page  # noqa: E402

DATASETS = load()
STATUSES = {"ok", "restricted", "not_on_nas", "alias"}


def _all_paths(e: dict) -> dict[str, str]:
    out = {f"paths.{a}": p for a, p in e.get("paths", {}).items()}
    for name, v in e.get("variants", {}).items():
        out |= {f"variants.{name}.{a}": p for a, p in v.items()}
    return out


def test_every_registry_key_has_an_entry():
    missing = sorted(set(list_datasets()) - set(DATASETS))
    assert not missing, f"add these keys to configs/data_paths.json: {missing}"


def test_no_entry_without_a_registry_key():
    extra = sorted(set(DATASETS) - set(list_datasets()))
    assert not extra, f"these config keys are not registry keys: {extra}"


@pytest.mark.parametrize("key", sorted(DATASETS))
def test_entry_shape(key):
    e = DATASETS[key]
    assert e["status"] in STATUSES
    if e["status"] == "alias":
        assert e.get("use"), "an alias entry needs a 'use' list"
        for k in e["use"]:
            assert k in DATASETS and DATASETS[k]["status"] in ("ok", "restricted"), k
    elif e["status"] in ("ok", "restricted"):
        assert _all_paths(e), "an 'ok' or 'restricted' entry needs paths"
        assert e.get("verified"), "record the date the paths were checked"
    for arg in e.get("env", {}):
        assert arg in e.get("paths", {}), f"env override for unknown argument {arg!r}"


@pytest.mark.parametrize("key", sorted(DATASETS))
def test_path_arguments_exist_in_the_signature(key):
    params = init_params(resolve_dataset(key))
    bad = [k for k in _all_paths(DATASETS[key]) if k.rsplit(".", 1)[1] not in params]
    assert not bad, f"{resolve_dataset(key).__name__} has no argument for {bad}"


def test_paths_rejects_entries_without_paths():
    alias = next(k for k, e in DATASETS.items() if e["status"] == "alias")
    with pytest.raises(ValueError, match="Use one of"):
        paths(alias)


def test_env_override(monkeypatch):
    key, e = next((k, e) for k, e in DATASETS.items() if e.get("env"))
    arg, var = next(iter(e["env"].items()))
    monkeypatch.setenv(var, "/tmp/override")
    assert paths(key)[arg] == "/tmp/override"


def test_nas_paths_page_is_up_to_date():
    assert gen_nas_paths_page.main(["--check"]) == 0


# The mount roots of the config paths, for example the NAS3 and NAS4 dataset folders.
_ROOTS = {"/".join(p.split("/")[:4]) for e in DATASETS.values() for p in _all_paths(e).values()}


@pytest.mark.skipif(not all(os.path.isdir(r) for r in _ROOTS), reason="NAS not mounted")
@pytest.mark.parametrize("key", sorted(DATASETS))
def test_paths_exist_on_nas(key):
    e = DATASETS[key]
    states = {k: path_state(p) for k, p in _all_paths(e).items()}
    assert "missing" not in states.values(), states
    if e["status"] == "ok":
        assert set(states.values()) <= {"ok"}, f"mark {key!r} as restricted: {states}"


def test_ms_cxr_t_image_path_resolves(tmp_path):
    """V-H2-8: the config base_image_dir plus image_path is a file.

    The test writes a 1-row CSV in the MS-CXR-T format, with a real MIMIC-CXR-JPG
    image from the MS-CXR CSV. So it needs no MS-CXR-T file.
    """
    import pandas as pd

    from radharmony.harmonizer.ms_cxr_t import MSCXRTHarmonizer

    ms_cxr_csv = paths("ms_cxr")["csv_path"]
    if not os.access(ms_cxr_csv, os.R_OK):
        pytest.skip("MS-CXR CSV not readable")
    did = pd.read_csv(ms_cxr_csv)["path"].iloc[0].removeprefix("files/").removesuffix(".jpg")
    pid, sid = did.split("/")[1][1:], did.split("/")[2][1:]
    row = {"dicom_id": did, "previous_dicom_id": did, "subject_id": pid,
           "study_id": sid, "previous_study_id": sid}
    for f in ("consolidation", "edema", "pleural_effusion", "pneumonia", "pneumothorax"):
        row[f + "_progression"] = "stable"
        row[f + "_label_quality"] = "one_expert"
    csv = tmp_path / "ms_cxr_t.csv"
    pd.DataFrame([row]).to_csv(csv, index=False)

    base = paths("ms_cxr_t")["base_image_dir"]
    df = MSCXRTHarmonizer(csv_path=str(csv), mimic_base_dir=base).harmonize()
    for col in ("image_path", "previous_image_path"):
        assert os.path.isfile(os.path.join(base, df[col].iloc[0])), (base, df[col].iloc[0])
