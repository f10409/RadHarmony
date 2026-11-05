"""Read tags from DICOM headers, for the harmonizers.

Some datasets have the sex and the age of the patient only in the DICOM
headers, not in a CSV of the release. D-06 says: read the tags ``PatientSex``
and ``PatientAge``. A missing or empty tag gives NaN.
"""
from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

#: The tags that hold the sex and the age of the patient (D-06, D-26).
PATIENT_TAGS = ("PatientSex", "PatientAge")


def read_dicom_tags(base_dir: str | None, image_paths: list[str], tags: list[str],
                    num_workers: int = 8) -> dict[str, dict]:
    """Read some tags from the header of each DICOM file, in parallel threads.

    The read stops before the pixel data, so it is fast.

    Args:
        base_dir: Folder that ``image_paths`` are relative to. ``None``: the
            paths are absolute.
        image_paths: Paths of the DICOM files.
        tags: DICOM keywords, for example ``["Rows", "PatientSex"]``.
        num_workers: Number of threads. 1 or less reads one file at a time.

    Returns:
        ``{image_path: {tag: value}}``. A file that cannot be read is left out,
        and a warning gives their number. A missing tag gives ``None``.
    """
    import pydicom

    def _read_one(path: str) -> tuple[str, dict | None]:
        full = os.path.join(base_dir, path) if base_dir else path
        try:
            ds = pydicom.dcmread(full, stop_before_pixels=True, specific_tags=list(tags))
            return path, {t: getattr(ds, t, None) for t in tags}
        except Exception:
            return path, None

    if num_workers and num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            results = list(ex.map(_read_one, image_paths))
    else:
        results = [_read_one(p) for p in image_paths]
    failed = [p for p, vals in results if vals is None]
    if failed:
        warnings.warn(
            f"{len(failed)} of {len(results)} DICOM files could not be read under {base_dir}, "
            f"for example {failed[0]!r}. Their tags are NaN.",
            UserWarning,
            stacklevel=2,
        )
    return {p: vals for p, vals in results if vals is not None}


def add_patient_tags(df: pd.DataFrame, base_dir: str | None, num_workers: int = 8,
                     path_col: str = "image_path", extra_tags: tuple = ()) -> pd.DataFrame:
    """Add the columns ``PatientSex`` and ``PatientAge`` from the DICOM headers.

    A missing or empty tag, or a file that cannot be read, gives NaN (D-06).
    The values stay as in the header, for example "M" and "045Y". The base
    steps map them to ``sex`` and ``age_years``.

    Args:
        df: Table with one DICOM path per row.
        base_dir: Folder that the paths are relative to.
        num_workers: Number of threads for the header reads.
        path_col: Column that holds the paths.
        extra_tags: More tags to add in the same header read, for example
            ``("ViewPosition",)`` (D-01, Q-18).

    Returns:
        A copy of ``df`` with the 2 new columns, and one column for each extra tag.
    """
    tags = list(PATIENT_TAGS) + list(extra_tags)
    headers = read_dicom_tags(base_dir, df[path_col].dropna().unique().tolist(), tags, num_workers)
    df = df.copy()
    for tag in tags:
        df[tag] = tag_values(headers, df[path_col], tag)
    return df


def tag_values(headers: dict[str, dict], paths, tag: str) -> list:
    """Return the text of one tag for each path, from a :func:`read_dicom_tags` result.

    A missing or empty tag, or a file that was not read, gives NaN.
    """
    out = []
    for path in paths:
        v = (headers.get(path) or {}).get(tag)
        text = "" if v is None else str(v).strip()
        out.append(text or np.nan)
    return out
