"""A clone with no ``configs/data_paths.json``, or with only some entries (the public repo).

Decided by Frank Li (2026-11-04): a user fills their own ``configs/data_paths.json`` for the datasets that
they want to check, from ``configs/data_paths.example.json``. A missing file or key gives a clear message,
and the tests that need real data skip. ``RADHARMONY_DATA_PATHS`` points to another file. No real data.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs" / "data_paths.example.json"


def _run(code: str, data_paths: Path, *args: str) -> subprocess.CompletedProcess:
    """Run Python code (or a script, if ``code`` ends with .py) with ``RADHARMONY_DATA_PATHS`` set."""
    env = {**os.environ, "RADHARMONY_DATA_PATHS": str(data_paths), "PYTHONPATH": str(ROOT)}
    cmd = [sys.executable, "-W", "ignore"] + ([code] if code.endswith(".py") else ["-c", code]) + list(args)
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)


_PATHS = "import sys; sys.path.insert(0, 'scripts'); import dataset_paths as d\n"


def test_no_file_gives_the_hint(tmp_path):
    r = _run(_PATHS + "d.paths('montgomery_cxr')", tmp_path / "none.json")
    assert r.returncode != 0 and "data_paths.example.json" in r.stderr, r.stderr
    r = _run(_PATHS + "print(d.paths('montgomery_cxr', missing_ok=True))", tmp_path / "none.json")
    assert "__missing_paths__" in r.stdout, r.stderr


def test_a_file_with_some_entries_gives_the_hint_for_the_others(tmp_path):
    f = tmp_path / "mine.json"
    f.write_text(json.dumps({"schema": 1, "datasets": {
        "montgomery_cxr": {"paths": {"base_image_dir": "/path/to/MontgomerySet/CXR_png/"}, "status": "ok"}}}))
    r = _run(_PATHS + "print(d.paths('montgomery_cxr'))", f)
    assert "/path/to/MontgomerySet/CXR_png/" in r.stdout, r.stderr
    r = _run(_PATHS + "d.paths('chexpert_train')", f)
    assert r.returncode != 0 and "No path for 'chexpert_train'" in r.stderr, r.stderr


def test_check_script_runs_only_the_datasets_with_paths(tmp_path):
    r = _run("scripts/check_datasets.py", tmp_path / "none.json", "--datasets", "montgomery_cxr")
    assert r.returncode != 0 and "data_paths.example.json" in r.stdout + r.stderr, r.stdout + r.stderr
    r = _run("scripts/check_datasets.py", tmp_path / "none.json", "--max-samples", "1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "skipped" in r.stdout and "no path" in r.stdout, r.stdout


@pytest.mark.parametrize("test_file", ["test_chexpert_demographics.py", "test_data_paths.py", "test_node21.py",
                                       "test_rexgradient_labels.py", "test_scripts.py", "test_dataset_stats.py"])
@pytest.mark.parametrize("paths_file", ["none", "partial"])
def test_the_tests_that_need_the_file_skip_without_it(tmp_path, test_file, paths_file):
    """No file, or a user file with only some entries (the example file): no test fails."""
    data_paths = tmp_path / "none.json" if paths_file == "none" else EXAMPLE
    env = {**os.environ, "RADHARMONY_DATA_PATHS": str(data_paths), "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"tests/{test_file}"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode in (0, 5), r.stdout[-2000:]  # 5: every test skipped or none collected
    assert " failed" not in r.stdout and " error" not in r.stdout.lower().split("warnings")[0], r.stdout[-2000:]


def test_the_example_file_is_valid_and_has_no_real_path():
    sys.path.insert(0, str(ROOT / "scripts"))
    import radharmony.dataset  # noqa: F401  (fills the registry)
    from dataset_paths import load
    from radharmony.registry import list_datasets

    entries = load(EXAMPLE)
    assert entries and set(entries) <= set(list_datasets())
    for key, e in entries.items():
        assert e["status"] == "ok", key
        assert all(p.startswith("/path/to/") for p in e["paths"].values()), key


def test_a_nas_claim_with_no_path_is_skipped_not_an_error(tmp_path, monkeypatch):
    """``dataset_stat.py --source`` exits 3 with the hint, and ``claims.py`` counts it as skipped."""
    r = _run("scripts/evalkit/dataset_stat.py", tmp_path / "none.json", "--source", "chexpert_train", "csv_path",
             "rows")
    assert r.returncode == 3 and "data_paths.example.json" in r.stderr, r.stderr
    sys.path.insert(0, str(ROOT / "scripts" / "evalkit"))
    import claims

    monkeypatch.setenv("RADHARMONY_DATA_PATHS", str(tmp_path / "none.json"))
    note = {"id": "x", "tier": "nas", "expect": "1",
            "cmd": "python scripts/evalkit/dataset_stat.py --source chexpert_train csv_path rows"}
    assert claims.check_claim("docs/wiki/x.md", 1, note, "1") == []
