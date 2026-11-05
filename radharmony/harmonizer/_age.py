"""Convert the age of each dataset to one column, ``age_years`` (float).

Each dataset stores age in its own unit. Examples:

    ================  ==========================================
    dataset           what the age column holds
    ================  ==========================================
    Emory, BRAX       years
    Shenzhen          years, months or days (``"male 16month"``)
    Montgomery        years (parsed from ``"Patient's Age: 027Y"``)
    TAIX-Ray          **days**, median 24,923 (68.2 years)
    ReXGradient       **DICOM AS string**, for example ``002D`` or ``029Y``
    RSNA Bone Age     **months**
    ================  ==========================================

Read as years, a TAIX age of 24,923 days is wrong. A ReXGradient ``002D`` (two days old)
becomes 2 if you cut the suffix.

Each harmonizer declares its source unit. The code never guesses it from the size of the
numbers, because that fails for children. An unknown unit raises ``ValueError``.
"""
from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd

AgeUnit = Literal["years", "days", "months", "dicom_as"]

#: DICOM Age String (VR = AS): exactly 4 chars, ``nnn`` + one of D/W/M/Y.
_AS_RE = re.compile(r"^\s*(\d{1,3})\s*([DWMY])\s*$", re.IGNORECASE)

_AS_TO_YEARS = {"D": 1.0 / 365.25, "W": 7.0 / 365.25, "M": 1.0 / 12.0, "Y": 1.0}

#: An age above this value becomes NaN. An age of 0 is valid (newborn) but is also a common
#: "missing" value. It stays, and the caller can filter it with ``plausible_mask``.
_MAX_PLAUSIBLE_YEARS = 130.0


def _parse_dicom_as(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    m = _AS_RE.match(str(value))
    if not m:
        return np.nan
    return int(m.group(1)) * _AS_TO_YEARS[m.group(2).upper()]


def to_years(values: pd.Series, unit: AgeUnit) -> pd.Series:
    """Convert an age column to float years.

    Args:
        values: The raw age column.
        unit: The source unit, declared by the harmonizer. An unknown unit raises ``ValueError``.

    Returns:
        float Series of years, NaN where unparseable or implausible.
    """
    if unit == "dicom_as":
        out = values.map(_parse_dicom_as).astype(float)
    elif unit in ("years", "days", "months"):
        num = pd.to_numeric(values, errors="coerce").astype(float)
        div = {"years": 1.0, "days": 365.25, "months": 12.0}[unit]
        out = num / div
    else:
        raise ValueError(
            f"unknown age unit {unit!r}; declare one of "
            "'years', 'days', 'months', 'dicom_as'"
        )
    return out.where((out >= 0) & (out <= _MAX_PLAUSIBLE_YEARS))


def plausible_mask(age_years: pd.Series, lo: float = 0.0, hi: float = 120.0) -> pd.Series:
    """Boolean mask of ages inside a plausible range, NaN-safe."""
    return age_years.notna() & age_years.between(lo, hi)
