"""Overlay registration tests — INTERNAL repo only.

Auto-skips where ``radharmony_internal`` isn't installed (e.g. the public
framework), so it's harmless in any checkout. This file is internal-only and
must not be copied to the public repo.

Run with::

    .venv/bin/python -m pytest tests/test_internal_overlay.py -v
"""
from __future__ import annotations

import pytest

pytest.importorskip("radharmony_internal")

import radharmony_internal  # noqa: E402,F401  — registers the private datasets
from radharmony.registry import list_datasets, resolve_dataset  # noqa: E402

# None currently — emory_cxr / ms_cxr / ms_cxr_t were promoted into the public
# ``radharmony`` package. Populate this list when a private dataset is added
# back to the overlay to re-enable the registration check.
PRIVATE_DATASETS: list[str] = []


@pytest.mark.skipif(not PRIVATE_DATASETS, reason="overlay currently ships no private datasets")
@pytest.mark.parametrize("name", PRIVATE_DATASETS or ["__none__"])
def test_private_dataset_registered(name):
    assert name in list_datasets()
    assert isinstance(resolve_dataset(name), type)


def test_app_plugin_callable():
    from radharmony_internal.app import register_app_datasets

    assert callable(register_app_datasets)
