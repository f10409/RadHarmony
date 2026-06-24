"""Dataset config-file validation — no data dependency.

Every ``configs/datasets/*.json`` must parse, carry the required keys, declare
a known modality, point at a registry key that actually resolves, and contain
no real data paths (configs ship with ``/path/to/`` placeholders only).

Run with::

    .venv/bin/python -m pytest tests/test_configs.py -v
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

# Importing the packages populates the registry so registry_key can resolve.
import radharmony.dataset  # noqa: F401
import radharmony.harmonizer  # noqa: F401
from radharmony.registry import resolve_dataset

_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = sorted((_ROOT / "configs" / "datasets").glob("*.json"))
REQUIRED_KEYS = {"dataset_name", "registry_key", "modality"}
VALID_MODALITIES = {"2D", "3D"}
_REAL_PATH = re.compile(r"/mnt/NAS|NAS\d+/datasets")


def test_configs_present():
    assert CONFIGS, "no dataset configs found under configs/datasets/"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_well_formed(path):
    d = json.loads(path.read_text())
    missing = REQUIRED_KEYS - d.keys()
    assert not missing, f"{path.name} missing required keys: {missing}"
    assert d["modality"] in VALID_MODALITIES, (
        f"{path.name}: modality {d['modality']!r} not in {VALID_MODALITIES}"
    )


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_registry_key_resolves(path):
    key = json.loads(path.read_text())["registry_key"]
    resolve_dataset(key)  # raises KeyError if the key isn't registered


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_has_no_real_paths(path):
    assert not _REAL_PATH.search(path.read_text()), (
        f"{path.name} contains a real data path — configs must use placeholders"
    )
