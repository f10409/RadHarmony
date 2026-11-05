"""Compare harmonize() with ``expected_stats`` in ``configs/datasets/*.json``.

The expected numbers come from the dataset paper or website and from the raw
files read with pandas, not from RadHarmony code (D-23). The format::

    "expected_stats": {
        "_comment": "the sources, and why a number differs from the paper",
        "patients_in_two_splits": 0,
        "<registry key of one split>": {
            "images": 100, "studies": 90, "patients": 80,
            "positives": {"<label>": 10}, "uncertain": {"<label>": 2},
            "rows_with_bbox": 0, "boxes": 0, "rows_with_mask": 0, "rows_with_report": 0,
            "sex": {"F": 40, "M": 60}, "age_years": [18, 90],
            "rows": 100,
            "harmonizer": "<class in radharmony.harmonizer>",
            "harmonizer_args": {"<harmonizer argument>": "<data_paths.json argument>"},
            "variant": "<variant of the data_paths.json entry>",
            "known_gaps": {"<stat>": "<design entry and finding ID>"}
        }
    }

``positives`` and ``uncertain`` list every label column. ``rows`` is for a
table with one row per study (D-03). The default is ``images``. ``harmonizer``
is for a dataset class with no ``_HARMONIZER_CLS``. ``harmonizer_args`` maps
a harmonizer argument to a path in ``configs/data_paths.json`` with another
name. ``variant`` picks a variant of an entry with ``variants``. A
``slice_output_dir`` from the config is replaced by a temporary folder, so
the test writes no file on the NAS. A stat in ``known_gaps`` is an expected failure while the table has no
column for it. When the column exists, the stat must match and the entry must
go. The tests on the data run only where the data is readable.

Run with::

    .venv/bin/python -m pytest tests/test_dataset_stats.py -v
"""
from __future__ import annotations

import inspect
import itertools
import json
import sys
import tempfile
import warnings
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

import radharmony.dataset  # noqa: F401  (fills the registry)
import radharmony.harmonizer
from radharmony.dataset.base import BaseRadiologicalDataset
from radharmony.registry import resolve_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from dataset_paths import MissingPathsError, entry, is_complete, path_state, paths  # noqa: E402
from fixtures.harmonizer_contract import contract_failures  # noqa: E402
from fixtures.init_params import init_params  # noqa: E402

STAT_KEYS = ("images", "studies", "patients", "positives", "uncertain", "rows_with_bbox", "boxes",
             "rows_with_mask", "rows_with_report", "sex", "age_years")
OTHER_KEYS = ("harmonizer", "harmonizer_args", "known_gaps", "rows", "variant")


def _load_expected() -> dict[str, dict]:
    """Return {config name: expected_stats} for the configs that have them."""
    configs = sorted((ROOT / "configs" / "datasets").glob("*.json"))
    blocks = {cfg.stem: json.loads(cfg.read_text()).get("expected_stats") for cfg in configs}
    return {name: block for name, block in blocks.items() if block}


EXPECTED = _load_expected()
SPLITS = {key: exp for block in EXPECTED.values() for key, exp in block.items() if isinstance(exp, dict)}


@lru_cache(maxsize=None)
def _paths(key: str) -> dict:
    """Return the config paths of a key, with its ``variant``. A slice folder goes to a temporary folder."""
    out = paths(key, SPLITS[key].get("variant"))
    if "slice_output_dir" in out:
        out["slice_output_dir"] = tempfile.mkdtemp(prefix=f"stats_{key}_")
    return out


@lru_cache(maxsize=None)
def _harmonized(key: str):
    """Return (harmonizer, harmonize() result) for a registry key, or skip the test."""
    try:
        kwargs = dict(_paths(key))
    except (ValueError, MissingPathsError) as e:  # status not_on_nas or alias, or no entry in this clone
        pytest.skip(str(e))
    blocked = [p for a, p in kwargs.items() if a != "slice_output_dir" and path_state(p) != "ok"]
    if blocked:
        pytest.skip(f"{key}: data not readable: {blocked[0]}")
    name = SPLITS[key].get("harmonizer")
    cls = getattr(radharmony.harmonizer, name) if name else resolve_dataset(key)._HARMONIZER_CLS
    kwargs.update({arg: kwargs[src] for arg, src in SPLITS[key].get("harmonizer_args", {}).items()})
    accepted = init_params(cls)  # a split harmonizer passes **kwargs to its parent
    h = cls(**{k: v for k, v in kwargs.items() if k in accepted})
    return h, h.harmonize()


def _filled(v) -> bool:
    """True if a cell holds a value: not NaN, None, "" or an empty list."""
    if v is None or (pd.api.types.is_scalar(v) and pd.isna(v)):
        return False
    if isinstance(v, str):
        return v.strip() not in ("", "[]")
    try:
        return len(v) > 0
    except TypeError:
        return True


def _n_boxes(v) -> int:
    """Count the boxes in one ``bbox`` cell: a list of boxes, one flat box or a JSON string."""
    if isinstance(v, str):
        v = json.loads(v) if v.strip() else []
    if not _filled(v):
        return 0
    return len(v) if isinstance(v[0], (list, tuple)) else 1


def _stat(h, df: pd.DataFrame, stat: str):
    """Compute one stat of the table ``df`` that harmonizer ``h`` returned."""
    # A VQA harmonizer has no get_label_cols(): its table has no label columns.
    labels = [c.lower().replace(" ", "_") for c in getattr(h, "get_label_cols", lambda: [])()]
    if stat == "images":
        return df["image_path"].nunique()
    if stat == "studies":
        return df["study_id"].nunique()
    if stat == "patients":
        return df["patient_id"].nunique()
    if stat == "positives":
        return {c: int((df[c] == 1).sum()) if c in df else None for c in labels}
    if stat == "uncertain":
        return {c: int(n) for c in labels if c in df and (n := df[c].isna().sum())}
    if stat == "boxes":
        return int(df["bbox"].map(_n_boxes).sum()) if "bbox" in df else 0
    if stat.startswith("rows_with_"):
        col = {"bbox": "bbox", "mask": "mask_path", "report": "report"}[stat[len("rows_with_"):]]
        return int(df[col].map(_filled).sum()) if col in df else 0
    if stat == "sex":
        return {str(k): int(n) for k, n in df["sex"].value_counts().items()} if "sex" in df else None
    if stat == "age_years":
        return [df["age_years"].min(), df["age_years"].max()] if "age_years" in df else None
    raise KeyError(stat)


@pytest.mark.parametrize(
    "key,stat", [(k, s) for k, exp in SPLITS.items() for s in STAT_KEYS if s in exp],
    ids=lambda v: v)
def test_stat_matches_expected(key, stat):
    """Each stat of harmonize() equals the expected value."""
    exp = SPLITS[key]
    got = _stat(*_harmonized(key), stat)
    gap = exp.get("known_gaps", {}).get(stat)
    if gap and got is None:
        pytest.xfail(gap)
    assert got == exp[stat]
    assert not gap, f"{stat} matches now: remove it from known_gaps"


@pytest.mark.parametrize("key", list(SPLITS))
def test_expected_stats_entry_is_valid(key):
    """Each split is a data_paths.json key, and each name in its entry is known (no data needed)."""
    exp = SPLITS[key]
    if is_complete():  # a clone with some entries (or no file) checks only the format
        entry(key)
    assert set(exp) <= set(STAT_KEYS + OTHER_KEYS), f"unknown keys: {set(exp) - set(STAT_KEYS + OTHER_KEYS)}"
    assert set(exp.get("known_gaps", {})) <= set(exp)
    assert "harmonizer" not in exp or hasattr(radharmony.harmonizer, exp["harmonizer"])


@pytest.mark.parametrize("key", list(SPLITS))
def test_one_row_per_image(key):
    """D-03: the row count equals the image count, unless the config gives ``rows``."""
    _, df = _harmonized(key)
    assert len(df) == SPLITS[key].get("rows", SPLITS[key]["images"])


@pytest.mark.parametrize("key", list(SPLITS))
def test_saved_table_matches_harmonize(key, tmp_path):
    """D-24 and D-08: harmonized_df, and the table after save and load, equal harmonize()."""
    h, df = _harmonized(key)
    pd.testing.assert_frame_equal(h.harmonized_df, df)
    h.save(str(tmp_path / "h.pkl"))
    loaded = type(h).load_from_saved(str(tmp_path / "h.pkl"))
    pd.testing.assert_frame_equal(loaded.harmonized_df, df)


#: (registry key, check) -> (reason, strict). A strict known failure must go when it passes.
CONTRACT_KNOWN: dict[tuple[str, str], tuple[str, bool]] = {}


@pytest.mark.parametrize("key", list(SPLITS))
def test_harmonizer_contract(key, tmp_path):
    """L-008 and L-014 on real data: the checks in ``tests/fixtures/harmonizer_contract.py``."""
    h, df = _harmonized(key)
    ds_cls = resolve_dataset(key)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # load_from_saved raises for a file of another class (radharmony/harmonizer/base.py), so
        # the harmonizer_path check skips a config that names another harmonizer.
        failures = contract_failures(h, df, tmp_path, reharmonize=False,  # CI checks the reset step
                                     dataset_cls=ds_cls if type(h) is ds_cls._HARMONIZER_CLS else None)
    known = {c: rs for (k, c), rs in CONTRACT_KNOWN.items() if k == key}
    assert {c: p for c, p in failures.items() if c not in known} == {}
    fixed = sorted(c for c, (_, strict) in known.items() if strict and c not in failures)
    assert not fixed, f"{fixed} pass now: remove them from CONTRACT_KNOWN"
    if set(known) & set(failures):
        pytest.xfail("; ".join(sorted({known[c][0] for c in set(known) & set(failures)})))


FLAGS = {"output_cls": "cls", "output_mask": "mask", "output_bbox": "bbox", "output_report": "report"}
FLAG_COLUMN = {"mask": "mask_path", "bbox": "bbox", "report": "report"}


@pytest.mark.parametrize("key,flag", [(k, f) for k in SPLITS for f in FLAGS], ids=lambda v: v)
def test_output_flag_on_a_real_sample(key, flag):
    """D-11 (L-010): a supported flag puts its key in a real sample. Another flag warns and gives no key."""
    cls = resolve_dataset(key)
    if not issubclass(cls, BaseRadiologicalDataset):
        pytest.skip("a VQA dataset has no output flags")
    out = FLAGS[flag]
    supported = out in cls.SUPPORTED_OUTPUTS
    if supported and out == "mask" and "mask_output_dir" in inspect.signature(cls).parameters:
        pytest.skip("the masks need mask_output_dir: a run that writes every mask")
    _, df = _harmonized(key)
    col = FLAG_COLUMN.get(out)
    filled = df[df[col].map(_filled)] if col in df else df.iloc[:0]
    if supported and col and not len(filled):
        # D-33: a row with no mask or report is not a sample. Example: the stats
        # table of mimic_cxr_train reads no report file, because that is too slow.
        # Its validate and test keys check the report flag.
        pytest.skip(f"{key}: no row of the stats table has a {col!r} value")
    one_row = (filled if len(filled) else df).head(1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = cls(**_paths(key), harmonized_df=one_row, cache_dir=None, **{flag: True})
    sample = ds.get_datasets()[0]
    # An instance can support less than its class. Example: ReXGradient with no label CSV has no cls.
    supported = out in ds.SUPPORTED_OUTPUTS
    if supported:
        assert out in sample, f"{key}: {flag}=True, but the sample has no {out!r} key"
    else:
        assert [w for w in caught if issubclass(w.category, UserWarning)], f"{key}: {flag}=True gives no warning"
        assert out not in sample


@pytest.mark.parametrize("name", [n for n, b in EXPECTED.items() if "patients_in_two_splits" in b])
def test_patients_in_two_splits(name):
    """D-09, D-10: the official splits share the given number of patients (usually 0)."""
    block = EXPECTED[name]
    patients = [set(_harmonized(k)[1]["patient_id"]) for k, v in block.items() if isinstance(v, dict)]
    shared = set().union(*(a & b for a, b in itertools.combinations(patients, 2)))
    assert len(shared) == block["patients_in_two_splits"]
