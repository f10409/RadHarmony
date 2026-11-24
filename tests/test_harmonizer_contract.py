"""L-008 and L-014: the harmonizer contract on fake data. No data needed.

The checks are in ``tests/fixtures/harmonizer_contract.py``. The fake data is in
``tests/fixtures/fake_datasets.py``. The full run checks the same contract on
real data in ``tests/test_dataset_stats.py``.

Run with::

    .venv/bin/python -m pytest tests/test_harmonizer_contract.py -v
"""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

from fixtures.fake_datasets import FAKE
from fixtures.harmonizer_contract import contract_failures
from radharmony.harmonizer.base import BaseHarmonizer

#: (dataset, check) -> reason. A known failure is a strict xfail: when it passes, remove it here.
KNOWN: dict[tuple[str, str], str] = {}


@pytest.mark.parametrize("name", list(FAKE))
def test_harmonizer_contract(name, tmp_path):
    h = FAKE[name](tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = h.harmonize()
        failures = contract_failures(h, df, tmp_path)
    known = {c: r for (n, c), r in KNOWN.items() if n == name}
    assert {c: p for c, p in failures.items() if c not in known} == {}
    fixed = sorted(set(known) - set(failures))
    assert not fixed, f"{fixed} pass now: remove them from KNOWN"
    if known:
        pytest.xfail("; ".join(sorted(set(known.values()))))


def test_contract_finds_an_absolute_image_path(tmp_path):
    """D-04: the contract fails for a harmonizer that writes an absolute ``image_path``."""
    class _Absolute(BaseHarmonizer):
        def _build_patient_id(self):
            self.df["patient_id"] = self.df["pid"]

        def _build_study_id(self):
            self.df["study_id"] = self.df["sid"]

        def _build_image_path(self):
            self.df["image_path"] = "/data/images/" + self.df["path"]

    csv = tmp_path / "abs.csv"
    pd.DataFrame({"pid": ["p0"], "sid": ["s0"], "path": ["a.png"]}).to_csv(csv, index=False)
    h = _Absolute(csv_path=str(csv))
    failures = contract_failures(h, h.harmonize(), tmp_path, reharmonize=False)
    assert set(failures) == {"image_path_relative"}
    assert "/data/images/a.png" in failures["image_path_relative"]
