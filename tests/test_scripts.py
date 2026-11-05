"""Tests for the command-line scripts in ``scripts/``.

The tests use small fake folders in ``tmp_path``. They need no NAS and no
network. The expected values come from the fake data that each test writes.

Run with::

    pytest tests/test_scripts.py -v
"""
from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import check_datasets  # noqa: E402
from dataset_paths import MISSING, is_complete  # noqa: E402

# These tests pick real keys of a complete data paths file (the dev repo). Another clone runs the other tests.
needs_paths_file = pytest.mark.skipif(not is_complete(), reason="the data paths file is not complete")


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    """Run a script with the test interpreter, offline, from the repo root."""
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, "-W", "ignore", str(SCRIPTS / script), *args],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)


# --------------------------------------------------------------------------
# check_datasets.py
# --------------------------------------------------------------------------
@needs_paths_file
def test_check_datasets_skip_rejects_unknown_key(capsys):
    """V-A-15: a typo in --skip is an error, as in --datasets."""
    with pytest.raises(SystemExit) as e:
        check_datasets.main(argv=["--datasets", "rsna_bone_age", "--skip", "rsna_boneage"])
    assert e.value.code == 2
    assert "rsna_boneage" in capsys.readouterr().err


# A-6 (D-39): the datasets that README.md and docs/wiki/reference/datasets/index.md list. Every dataset
# whose class is in the public package is in scripts/check_datasets.py. Only EmoryCXR
# (internal data) is in scripts/check_datasets_internal.py.
@pytest.mark.parametrize("cls_name", [
    "CheXphotoTrainDataset", "ChestXDet10TrainDataset", "Node21Dataset",
    "VinDrPCXRDataset", "GEMeXVQADataset", "ROCODataset", "TAIXRayDataset",
    "MSCXRTrainDataset", "MSCXRValDataset", "MSCXRTestDataset", "MSCXRDataset", "MSCXRTDataset",
    "MIMICExtCXRQBADataset",
])
def test_check_datasets_covers_dataset(cls_name):
    assert cls_name in {s.cls.__name__ for s in check_datasets.get_specs()}


# Keys whose specs are in scripts/check_datasets_internal.py (internal data, D-39).
_INTERNAL_KEYS = {"emory_cxr", "emory_cxr_dicom"}


def _all_specs() -> list:
    """The public specs, plus the internal specs when the internal overlay is installed."""
    specs = list(check_datasets.get_specs())
    try:
        import check_datasets_internal
    except ImportError:
        return specs
    return specs + list(check_datasets_internal.INTERNAL_SPECS)


def _data_keys() -> dict[str, str]:
    """Registry key -> status in configs/data_paths.json, without the "alias" keys (no data)."""
    import radharmony.dataset  # noqa: F401  (fills the registry)
    from dataset_paths import entry, is_complete
    from radharmony.registry import list_datasets
    if not is_complete():
        pytest.skip("the data paths file of this clone is not complete")
    status = {k: entry(k)["status"] for k in list_datasets()}
    return {k: s for k, s in status.items() if s != "alias"}


def test_check_datasets_covers_every_registry_key_with_data():
    """Each registry key that is not an "alias" has a spec with its class."""
    from radharmony.registry import resolve_dataset
    specs = _all_specs()
    covered = {s.cls for s in specs}
    has_internal = any(s.key in _INTERNAL_KEYS for s in specs)
    missing = [k for k in _data_keys() if resolve_dataset(k) not in covered
               and (has_internal or k not in _INTERNAL_KEYS)]
    assert not missing, f"no check_datasets spec for {missing}"


def test_check_datasets_restricted_data_is_not_in_the_default_run():
    """A default run must pass for every user, so restricted or missing data runs only with --datasets."""
    from radharmony.registry import resolve_dataset
    status = _data_keys()
    status_of_cls = {resolve_dataset(k): s for k, s in status.items()}
    wrong = [s.key for s in _all_specs()
             if status_of_cls.get(s.cls) in ("restricted", "not_on_nas") and s.default]
    assert not wrong, f"these specs run by default but not every user can read the data: {wrong}"


def _init_params(cls) -> set[str]:
    """Argument names of the constructor, with the parents when it passes **kwargs on."""
    names = set()
    for c in cls.__mro__:
        if "__init__" not in vars(c):
            continue
        params = inspect.signature(c.__init__).parameters
        names |= set(params)
        if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
            break
    return names


@needs_paths_file
def test_check_datasets_default_run_skips_data_that_is_not_on_the_nas(monkeypatch):
    """ROCO (v1) is not on the NAS. A default run that always fails gates nothing."""
    ran = []
    monkeypatch.setattr(check_datasets, "run_one", lambda spec, *a: ran.append(spec.key) or
                        check_datasets.IntegrityReport(dataset=spec.key))
    monkeypatch.setattr(check_datasets, "print_report", lambda *a: None)
    assert check_datasets.main(argv=["--skip", ",".join(
        s.key for s in check_datasets.get_specs() if s.key not in ("roco", "rsna_bone_age"))]) == 0
    assert ran == ["rsna_bone_age"]
    ran.clear()
    check_datasets.main(argv=["--datasets", "roco"])
    assert ran == ["roco"]


def test_check_datasets_run_one_catches_a_crash(monkeypatch):
    """V-A-9: an error inside check_dataset gives an ERROR row, not a stopped run."""
    def boom(*a, **k):
        raise TypeError("object of type 'float' has no len()")
    spec = next(s for s in check_datasets.get_specs() if s.key == "rsna_bone_age")
    monkeypatch.setattr(check_datasets, "check_dataset", boom)
    monkeypatch.setattr(spec, "cls", lambda **k: object())
    monkeypatch.setattr(check_datasets, "subsample_dataset", lambda *a: (1, 1))
    r = check_datasets.run_one(spec, 4, 0)
    assert r.status == "ERROR"
    assert "has no len" in r.error_message


def test_check_datasets_names_cls_in_sorted_order(monkeypatch, tmp_path):
    """V-D-13: ``cls`` is in sorted label order (D-05), so the report uses that order too."""
    class Unsorted:
        LABEL_COLS = ["no_finding", "edema", "atelectasis"]

    # 4 images, each with only cls[0] positive. In the sorted order, index 0 is "atelectasis".
    report = check_datasets.IntegrityReport(dataset="fake", cls_pos_counts={0: 4})
    monkeypatch.setattr(check_datasets, "run_one", lambda spec, *a: report)
    monkeypatch.setattr(check_datasets, "print_report", lambda *a: None)
    out = tmp_path / "r.json"
    spec = check_datasets.DatasetSpec("fake", Unsorted, default=False)
    check_datasets.main(extra_specs=[spec], argv=["--datasets", "fake", "--json", str(out)])
    import json
    assert json.loads(out.read_text())[0]["cls_positive"] == {"atelectasis": 4}


@pytest.mark.parametrize("spec", check_datasets.get_specs(), ids=lambda s: s.key)
def test_check_datasets_spec_arguments_exist(spec):
    if callable(spec.kwargs):
        pytest.skip("kwargs are built at run time")
    if MISSING in spec.kwargs:
        pytest.skip("no entry in the data paths file of this clone")
    unknown = set(spec.kwargs) - _init_params(spec.cls)
    assert not unknown, f"{spec.key}: {spec.cls.__name__} has no argument {sorted(unknown)}"


# --------------------------------------------------------------------------
# materialize_mrrate.py and materialize_medpmc_subset.py
# --------------------------------------------------------------------------
@pytest.mark.parametrize("script", ["materialize_mrrate.py", "materialize_medpmc_subset.py"])
def test_help_lists_no_none_choice(script):
    """V-A-16: 'None' in the choices cannot be typed."""
    r = _run(script, "--help")
    assert r.returncode == 0, r.stderr
    assert "{None" not in r.stdout


@pytest.fixture
def fake_mrrate(tmp_path):
    """3 test studies. Study s1 is already on disk (it has img/)."""
    (tmp_path / "meta").mkdir()
    (tmp_path / "mri" / "b0" / "s1" / "img").mkdir(parents=True)
    pd.DataFrame({"study_uid": ["s1", "s2", "s3"], "batch_id": "b0", "split": "test"}
                 ).to_csv(tmp_path / "meta" / "splits.csv", index=False)
    return tmp_path


def test_mrrate_no_skip_existing_downloads_again(fake_mrrate):
    """V-A-26: --no-skip-existing turns the skip off. Offline, the download fails."""
    r = _run("materialize_mrrate.py", "--base-dir", str(fake_mrrate), "--studies", "s1")
    assert "1 already present" in r.stdout, r.stdout + r.stderr
    pytest.importorskip("huggingface_hub")  # the download step needs it
    r = _run("materialize_mrrate.py", "--base-dir", str(fake_mrrate), "--studies", "s1",
             "--no-skip-existing")
    assert "0 already present, 1 failed" in r.stdout, r.stdout + r.stderr


def test_mrrate_studies_ignore_limit(fake_mrrate):
    """V-A-27: the help says that --studies overrides --limit."""
    r = _run("materialize_mrrate.py", "--base-dir", str(fake_mrrate),
             "--studies", "s1", "s2", "s3", "--limit", "1")
    assert r.stdout.startswith("3 studies"), r.stdout + r.stderr


# --------------------------------------------------------------------------
# MedPMC scripts
# --------------------------------------------------------------------------
def test_tagger_recall_counts_every_image(tmp_path):
    """V-A-22: a CT image tagged 'radiology_other' or not tagged is a miss."""
    (tmp_path / "hf" / "data").mkdir(parents=True)
    ids = ["i1", "i2", "i3", "i4"]
    pd.DataFrame({"ID": ids, "CUIs": ["C0040405"] * 4}).to_csv(
        tmp_path / "test_concepts_manual.csv", index=False)  # all 4 are CT
    pd.DataFrame({"image_id": ids, "caption": [
        "CT scan of the chest", "Computed tomography of the abdomen",
        "Axial image shows a lesion", "A figure of the patient"]}
    ).to_parquet(tmp_path / "hf" / "data" / "test-0.parquet")
    r = _run("eval_medpmc_tagger_on_rocov2.py", "--roco-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    per_image = pd.read_csv(tmp_path / "tagger_eval_test.csv")
    assert len(per_image) == 4  # the 4 fake images
    recall = (per_image["pred_modality"] == "ct").sum() / len(per_image)
    row = next(line for line in r.stdout.splitlines() if line.startswith("ct "))
    assert float(row.split()[3]) == pytest.approx(recall, abs=1e-3), row


def test_tagger_macro_f1_counts_a_class_with_no_hit(tmp_path):
    """A modality with support but no correct tag has F1 0, and it is in macro-F1."""
    from sklearn.metrics import f1_score

    (tmp_path / "hf" / "data").mkdir(parents=True)
    ids = ["i1", "i2", "i3"]
    # 2 CT images and 1 MRI image. The MRI caption says CT, so MRI has no hit.
    pd.DataFrame({"ID": ids, "CUIs": ["C0040405", "C0040405", "C0024485"]}).to_csv(
        tmp_path / "test_concepts_manual.csv", index=False)
    pd.DataFrame({"image_id": ids, "caption": [
        "CT scan of the chest", "A figure of the patient", "CT scan of the head"]}
    ).to_parquet(tmp_path / "hf" / "data" / "test-0.parquet")
    r = _run("eval_medpmc_tagger_on_rocov2.py", "--roco-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    d = pd.read_csv(tmp_path / "tagger_eval_test.csv")
    assert len(d) == 3
    expected = f1_score(d["true_modality"], d["pred_modality"].fillna("(none)"),
                        labels=["ct", "mri"], average="macro", zero_division=0)
    line = next(x for x in r.stdout.splitlines() if x.startswith("macro-F1"))
    assert float(line.split()[-1]) == pytest.approx(expected, abs=1e-3), line


def test_provenance_uses_figure_rollup(tmp_path):
    """V-A-23: precedence is human > clip > figure_rollup > caption."""
    m, review = tmp_path / "medpmc", tmp_path / "review"
    for d in (m / "webdataset" / "metadata", m / "tags", review / "labels"):
        d.mkdir(parents=True)
    pd.DataFrame({"image_id": ["f1"], "key": ["kf1"], "source_type": ["multipanel"]}
                 ).to_parquet(m / "webdataset" / "metadata" / "medpmc_index.parquet")
    pd.DataFrame({"key": ["other"], "specialty": ["radiology"], "modality": ["ct"]}
                 ).to_parquet(m / "tags" / "medpmc_tags_clip.parquet")
    pd.DataFrame({"image_id": ["f1"], "specialty": ["radiology"], "modality": ["mri"]}
                 ).to_parquet(m / "tags" / "medpmc_tags.parquet")
    pd.DataFrame({"image_id": ["f1"], "top_modality": ["ct"]}
                 ).to_parquet(m / "tags" / "medpmc_figure_rollup.parquet")
    out = tmp_path / "resolved.parquet"
    r = _run("build_medpmc_label_provenance.py", "--base-dir", str(m),
             "--review-dir", str(review), "--out", str(out))
    assert r.returncode == 0, r.stderr
    row = pd.read_parquet(out).iloc[0]
    assert (row.modality_final, row.modality_source) == ("ct", "figure_rollup")


def test_annotation_frame_with_an_empty_stratum(tmp_path):
    """V-A-28: --n 3 gives 0 rows for stratum D. The frame is still written."""
    (tmp_path / "webdataset" / "metadata").mkdir(parents=True)
    (tmp_path / "tags").mkdir()
    ids = [f"i{i}" for i in range(6)]
    pd.DataFrame({"image_id": ids, "key": ids, "source_type": "singlepanel", "shard": "s0.tar"}
                 ).to_parquet(tmp_path / "webdataset" / "metadata" / "medpmc_index.parquet")
    pd.DataFrame({"key": ids, "specialty": "radiology", "modality": "ct",
                  "clip_prob": 0.95, "clip_margin": 0.5}
                 ).to_parquet(tmp_path / "tags" / "medpmc_tags_clip.parquet")
    pd.DataFrame({"image_id": ids,
                  "specialty": ["radiology", "radiology", "pathology", "unknown",
                                "radiology", "radiology"],
                  "modality": ["ct", "mri", None, None, "ct", "ct"], "tag_confidence": "high"}
                 ).to_parquet(tmp_path / "tags" / "medpmc_tags.parquet")
    r = _run("build_radiology_annotation_frame.py", "--base-dir", str(tmp_path), "--n", "3")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "tags" / "annotation_frame_radiology.parquet").exists()


def test_check_datasets_empty_key_list_is_an_error(capsys):
    """V-A-49: --datasets ',' names no dataset, so it is an error, not a pass."""
    with pytest.raises(SystemExit) as e:
        check_datasets.main(argv=["--datasets", ","])
    assert e.value.code == 2


def test_check_datasets_negative_max_samples_is_an_error(capsys):
    """V-A-43: a negative --max-samples is an error with exit code 2."""
    with pytest.raises(SystemExit) as e:
        check_datasets.main(argv=["--datasets", "rsna_bone_age", "--max-samples", "-1"])
    assert e.value.code == 2


@needs_paths_file
def test_check_datasets_zero_samples_exits_1(monkeypatch):
    """Q-1 (D-39): a dataset with 0 samples is an ERROR, so the script exits 1."""
    from radharmony.integrity import IntegrityReport

    def fake_run_one(spec, max_samples, default_workers):
        return IntegrityReport(dataset="X", status="ERROR", error_message="Dataset has 0 samples.")
    monkeypatch.setattr(check_datasets, "run_one", fake_run_one)
    assert check_datasets.main(argv=["--datasets", "rsna_bone_age"]) == 1


# --------------------------------------------------------------------------
# compile_chexinstruct_vqa.py
# --------------------------------------------------------------------------
_FAKE_PROCESSOR = '''
import json
import random


class {cls}:
    def __init__(self):
        # The real processors read data/... relative paths, so the script must run in the vendor folder.
        self.sets = json.load(open("data/{name}.json"))
        # The real MIMICCXRVQAProcessor shuffles its train list in __init__, with no seed.
        for s in self.sets.values():
            random.shuffle(s["train"])

{methods}
'''


def _fake_vendor(tmp_path, n_train=1):
    """Write the fake vendor folder: 2 processors and their data, with ``n_train`` train rows per task set."""
    import json

    vendor = tmp_path / "vendor"
    (vendor / "dataset_processors").mkdir(parents=True)
    (vendor / "data").mkdir()
    for cls, name, methods in (
        ("MIMICCXRVQAProcessor", "mimic_cxr_vqa", ("create_open_ended_vqa", "create_close_ended_vqa")),
        ("MIMICDiffVQAProcessor", "mimic_diff_vqa", ("create_difference_vqa",)),
    ):
        body = "\n".join(f"    def {m}(self):\n        return self.sets['{m}']\n" for m in methods)
        (vendor / "dataset_processors" / f"{name}.py").write_text(
            _FAKE_PROCESSOR.format(cls=cls, name=name, methods=body))
        def train(m):
            return [{"q": f"{m}-t"}] if n_train == 1 else [{"q": f"{m}-t{i}"} for i in range(n_train)]
        sets = {m: {"train": train(m), "val": [], "test": [{"q": f"{m}-x"}]} for m in methods}
        (vendor / "data" / f"{name}.json").write_text(json.dumps(sets))
    return vendor


def test_compile_chexinstruct_vqa_flattens_the_3_task_sets(tmp_path):
    """The script runs the 2 processors in the vendor folder and writes one row per sample, with ``split``."""
    import json

    vendor = _fake_vendor(tmp_path)
    out = tmp_path / "compiled" / "chexinstruct_vqa.json"

    r = _run("compile_chexinstruct_vqa.py", "--vendor-dir", str(vendor), "--out", str(out))
    assert r.returncode == 0, r.stderr
    rows = json.loads(out.read_text())
    assert [(row["q"], row["split"]) for row in rows] == [
        ("create_open_ended_vqa-t", "train"), ("create_open_ended_vqa-x", "test"),
        ("create_close_ended_vqa-t", "train"), ("create_close_ended_vqa-x", "test"),
        ("create_difference_vqa-t", "train"), ("create_difference_vqa-x", "test")]
    assert "Wrote 6 samples" in r.stdout


def test_compile_chexinstruct_vqa_seed_gives_the_same_file(tmp_path):
    """Decided by Frank Li (2026-11-04): the script seeds ``random``, so 2 runs give the same train order and ids."""
    vendor = _fake_vendor(tmp_path, n_train=30)
    outs = [tmp_path / f"run{i}.json" for i in range(3)]
    for out, seed in zip(outs, ("0", "0", "1"), strict=True):
        r = _run("compile_chexinstruct_vqa.py", "--vendor-dir", str(vendor), "--out", str(out), "--seed", seed)
        assert r.returncode == 0, r.stderr
    assert outs[0].read_text() == outs[1].read_text()
    assert outs[0].read_text() != outs[2].read_text()


def test_each_check_name_is_a_registry_key_or_a_variant_of_one():
    """Decided by Frank Li (2026-11-04): a check is named by its registry key, or '<registry key>_<variant>'."""
    import radharmony.dataset  # noqa: F401  (fills the registry)
    from radharmony.registry import list_datasets, resolve_dataset

    keys = set(list_datasets())
    for s in check_datasets.get_specs():
        assert s.key in keys or any(s.key.startswith(k + "_") and resolve_dataset(k) is s.cls for k in keys), s.key


def test_an_old_check_name_still_runs_with_a_warning(capsys):
    """The old short names (for example 'montgomery') map to the registry key for one release."""
    assert check_datasets.resolve_names(["montgomery", "chexpert_train"]) == ["montgomery_cxr", "chexpert_train"]
    assert "montgomery is an old name" in capsys.readouterr().err
