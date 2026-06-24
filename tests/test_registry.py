"""Registry integrity tests — no data dependency.

Importing ``radharmony.dataset`` / ``radharmony.harmonizer`` triggers every
built-in dataset's ``@register_dataset`` decorator, so these checks guard
against import breakage, duplicate keys, and registry/class drift.

Run with::

    .venv/bin/python -m pytest tests/test_registry.py -v
"""
from __future__ import annotations

import pytest

# Importing the packages populates the registry via @register_dataset.
import radharmony.dataset  # noqa: F401
import radharmony.harmonizer  # noqa: F401
from radharmony.registry import list_datasets, register_dataset, resolve_dataset


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
