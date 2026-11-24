"""Age unit normalisation: conversions, and that every harmonizer declares one.

The bug this guards: RadHarmony unified the age column NAME across datasets but
not its VALUE, so ``age`` meant years on Emory, days on TAIX and a DICOM AS
string on ReXGradient. Consumers reading the harmonized name silently got
nonsense.
"""
import numpy as np
import pandas as pd
import pytest

from radharmony.harmonizer._age import plausible_mask, to_years


def test_years_passthrough():
    out = to_years(pd.Series([0, 45, 90]), "years")
    assert out.tolist() == [0.0, 45.0, 90.0]


def test_days_to_years():
    # TAIX: median 24,923 days must land at ~68.2 years, not be read as an age.
    out = to_years(pd.Series([24923, 7473]), "days")
    assert out.iloc[0] == pytest.approx(68.2, abs=0.1)
    assert out.iloc[1] == pytest.approx(20.5, abs=0.1)


def test_months_to_years():
    assert to_years(pd.Series([144]), "months").iloc[0] == pytest.approx(12.0)


def test_dicom_as_all_units():
    out = to_years(pd.Series(["002D", "010W", "018M", "029Y"]), "dicom_as")
    assert out.iloc[0] == pytest.approx(2 / 365.25)
    assert out.iloc[1] == pytest.approx(70 / 365.25)
    assert out.iloc[2] == pytest.approx(1.5)
    assert out.iloc[3] == pytest.approx(29.0)


def test_dicom_as_neonate_is_not_read_as_years():
    """'002D' is two DAYS old. Stripping the suffix would give 2 years."""
    assert to_years(pd.Series(["002D"]), "dicom_as").iloc[0] < 0.01


def test_unparseable_becomes_nan():
    out = to_years(pd.Series(["", "abc", None, np.nan]), "dicom_as")
    assert out.isna().all()


def test_implausible_rejected():
    # 40,414 days is ~110y and fine; 40,414 taken as YEARS must be dropped.
    assert not np.isnan(to_years(pd.Series([40414]), "days").iloc[0])
    assert np.isnan(to_years(pd.Series([40414]), "years").iloc[0])


def test_unknown_unit_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="unknown age unit"):
        to_years(pd.Series([1]), "fortnights")


def test_plausible_mask_is_nan_safe():
    m = plausible_mask(pd.Series([np.nan, 0.0, 45.0, 200.0]))
    assert m.tolist() == [False, True, True, False]


def test_declared_units_are_valid():
    """Every harmonizer that declares a unit must declare a supported one."""
    import importlib
    import pkgutil

    import radharmony.harmonizer as H

    valid = {"years", "days", "months", "dicom_as"}
    seen = 0
    for mod in pkgutil.walk_packages(H.__path__, H.__name__ + "."):
        try:
            m = importlib.import_module(mod.name)
        except Exception:
            continue
        for obj in vars(m).values():
            unit = getattr(obj, "AGE_SOURCE_UNIT", None)
            if unit is None:
                continue
            seen += 1
            assert unit in valid, f"{obj.__name__} declares bad unit {unit!r}"
            assert getattr(obj, "AGE_SOURCE_COL", None), f"{obj.__name__} has unit but no column"
    assert seen >= 6, f"expected >=6 harmonizers declaring an age unit, found {seen}"


def test_padchest_age_is_study_year_minus_birth_year(tmp_path):
    """D-06: PadChest has only the birth year, so age_years is the study year minus the birth year.

    A missing birth year gives NaN. A birth year after the study gives a negative age, so NaN too.
    """
    from radharmony.harmonizer import PadChestHarmonizer

    pd.DataFrame({
        "ImageID": ["a.png", "b.png", "c.png", "d.png"], "ImageDir": [0, 0, 0, 0],
        "PatientID": ["p1", "p2", "p3", "p4"], "StudyID": ["s1", "s2", "s3", "s4"],
        "Labels": ["['normal']"] * 4, "Projection": ["PA"] * 4, "PatientSex_DICOM": ["M", "F", "M", "F"],
        "PatientBirth": [1930.0, 2014.0, np.nan, 2020.0],
        "StudyDate_DICOM": [20140915, 20150101, 20150101, 20150101],
    }).to_csv(tmp_path / "padchest.csv", index=False)
    df = PadChestHarmonizer(csv_path=str(tmp_path / "padchest.csv")).harmonize()
    ages = df.set_index("patient_id")["age_years"]
    assert ages["p1"] == 84.0 and ages["p2"] == 1.0
    assert np.isnan(ages["p3"]) and np.isnan(ages["p4"])
