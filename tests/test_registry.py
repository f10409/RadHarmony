"""Registry integrity tests — no data dependency.

Importing ``radharmony.dataset`` / ``radharmony.harmonizer`` triggers every
built-in dataset's ``@register_dataset`` decorator, so these checks guard
against import breakage, duplicate keys, and registry/class drift.

Run with::

    .venv/bin/python -m pytest tests/test_registry.py -v
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Importing the packages populates the registry via @register_dataset.
import radharmony.dataset  # noqa: F401
import radharmony.harmonizer  # noqa: F401
from radharmony.registry import list_datasets, register_dataset, resolve_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_builtins_registered():
    names = list_datasets()
    assert len(names) > 20, f"expected many built-in datasets, got {len(names)}"


def test_list_is_sorted_and_unique():
    names = list_datasets()
    assert names == sorted(names)
    assert len(names) == len(set(names))


def test_every_key_resolves_to_a_class():
    for name in list_datasets():
        cls = resolve_dataset(name)
        assert isinstance(cls, type), f"{name!r} resolved to non-class {cls!r}"


def test_resolve_unknown_raises():
    with pytest.raises(KeyError):
        resolve_dataset("definitely_not_a_registered_dataset_xyz")


def test_duplicate_registration_raises():
    existing = list_datasets()[0]
    with pytest.raises(ValueError):

        @register_dataset(existing)
        class _Dup:  # noqa: D401 — must not reach the registry
            pass


def _fresh(code: str) -> str:
    """Run ``code`` in a new interpreter (empty registry) and return stdout."""
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=ROOT,
                       env={**os.environ, "PYTHONPATH": str(ROOT)},
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_plain_import_lists_the_builtin_datasets():
    """V-A-1: the registry.py docstring example works after ``import radharmony``."""
    out = _fresh("import radharmony\n"
                 "from radharmony.registry import resolve_dataset, list_datasets\n"
                 "print(len(list_datasets()), resolve_dataset('mimic_cxr').__name__)")
    n, name = out.split()
    assert int(n) > 20
    assert name == "MIMICCXRDataset"


def test_evaluator_docstring_lists_the_builtin_evaluators():
    out = _fresh("from radharmony.evaluator import list_evaluators\n"
                 "print(' '.join(list_evaluators()))")
    assert {"conv_probe_seg", "linear_probe", "knn_probe", "zero_shot",
            "finetune"} <= set(out.split())
