"""CheXpert demographics wiring — the columns exist on disk and must reach the output."""
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from radharmony.harmonizer.chexpert.chexpert import CheXpertHarmonizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dataset_paths import paths  # noqa: E402

# No entry (a clone with its own paths file): the empty path skips the tests below.
CSV = paths("chexpert_train", missing_ok=True).get("csv_path", "")
XLSX = str(Path(CSV).parent / "Chexpert race labels.xlsx")
pytestmark = pytest.mark.skipif(
    not (os.path.isfile(CSV) and os.path.isfile(XLSX)), reason="CheXpert not mounted"
)


def _small(tmp_path, n=4000):
    d = pd.read_csv(CSV, low_memory=False).head(n)
    p = tmp_path / "train_small.csv"
    d.to_csv(p, index=False)
    return str(p)


def test_without_path_behaviour_is_unchanged(tmp_path):
    """Omitting demographics_path must not change anything for existing callers."""
    h = CheXpertHarmonizer(csv_path=_small(tmp_path))
    out = h.harmonize()
    assert "race" not in out.columns
    assert len(out) > 0


def test_demographics_reach_the_output(tmp_path):
    h = CheXpertHarmonizer(csv_path=_small(tmp_path), demographics_path=XLSX)
    out = h.harmonize()
    for c in ("sex", "age_years", "race", "ethnicity"):
        assert c in out.columns, f"{c} missing from harmonized output"
    assert out["race"].notna().mean() > 0.8, "join matched too few rows"


def test_age_is_the_age_at_each_image(tmp_path):
    """V-H1-40 (D-06): the workbook has one AGE_AT_CXR per patient, so it is not an age column.

    age_years comes from the Age of each row of the CSV.
    """
    csv = _small(tmp_path)
    out = CheXpertHarmonizer(csv_path=csv, demographics_path=XLSX).harmonize()
    assert "age" not in out.columns
    src = pd.read_csv(csv, low_memory=False)
    # image_path drops the first 2 parts of Path ("CheXpert-v1.0/train/").
    ages = dict(zip(src["Path"].str.split("/", n=2).str[2], src["Age"], strict=True))
    got = out.set_index("image_path")["age_years"]
    keys = [k for k in got.index[:200] if k in ages]
    assert len(keys) == 200
    assert all(got[k] == ages[k] for k in keys)


def test_race_is_not_silently_collapsed(tmp_path):
    """The harmonizer must pass PRIMARY_RACE through, not bucket it."""
    h = CheXpertHarmonizer(csv_path=_small(tmp_path), demographics_path=XLSX)
    out = h.harmonize()
    vals = set(out["race"].dropna().astype(str))
    assert len(vals) > 2, f"race looks collapsed: {vals}"


def test_missing_workbook_raises_rather_than_silently_skipping(tmp_path):
    h = CheXpertHarmonizer(csv_path=_small(tmp_path),
                           demographics_path=str(tmp_path / "nope.xlsx"))
    with pytest.raises(FileNotFoundError):
        h.harmonize()


def test_join_does_not_duplicate_rows(tmp_path):
    a = CheXpertHarmonizer(csv_path=_small(tmp_path)).harmonize()
    b = CheXpertHarmonizer(csv_path=_small(tmp_path), demographics_path=XLSX).harmonize()
    assert len(a) == len(b), "demographics join changed the row count"
