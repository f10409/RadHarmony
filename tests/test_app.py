"""Tests for the Gradio app functions in ``app.py``.

The tests call the functions directly. They do not start a server. The
samples are small fake image files (PNG, NIfTI, DICOM) in ``tmp_path``.

Run with::

    pytest tests/test_app.py -v
"""
from __future__ import annotations

import inspect
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("gradio")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app  # noqa: E402

# flip, flip_axis, jitter, shift, scale, noise, std, smooth, min, max, elastic, sigma,
# mag min, mag max, affine, translate, rotate, scale, shear, transpose, indices, override
_NO_AUG = [False, 1, False, 0.1, 0.1, False, 0.05, False, 0.5, 1.5, False, 5, 1, 2,
           False, 10, 0.1, 0.1, 0.1, False, "0,2,1", False]


def _aug(flip=False):
    a = list(_NO_AUG)
    a[0] = flip
    return a


@pytest.fixture
def png(tmp_path):
    from PIL import Image
    p = tmp_path / "img.png"
    Image.fromarray((np.random.RandomState(0).rand(100, 200) * 255).astype(np.uint8)).save(p)
    return str(p)


def _box_state(sample):
    return app._make_state([sample], [], [], {"img", "bbox", "bbox_labels"},
                           is_3d=False, dataset_name="fake")


# --------------------------------------------------------------------------
# load_dataset
# --------------------------------------------------------------------------
class _Adapter:
    """A dataset stand-in with a fixed table and fixed verify_images() result."""

    LABEL_COLS: list = []
    REG_COLS: list = []
    _cols: dict = {}

    def __init__(self, df, dropped):
        self._df, self._dropped = df, dropped

    def get_harmonized_df(self):
        return self._df

    def verify_images(self, drop_missing=True):
        return self._dropped


def _load_fake(monkeypatch, tmp_path, df, dropped):
    cfg = app.DatasetConfig(False, "", lambda *a, **k: (_Adapter(df, dropped), None),
                            modality="CXR")
    monkeypatch.setitem(app.DATASET_REGISTRY, "Fake", cfg)
    return app.load_dataset("Fake", str(tmp_path), "", "", "", None, "",
                            True, False, False, False, False, progress=lambda *a, **k: None)


def test_load_with_no_sample_left_is_an_error(monkeypatch, tmp_path):
    """V-A-29: 0 samples gives an error message and no state."""
    empty = pd.DataFrame({"image_path": []})
    dropped = pd.DataFrame({"image_path": ["a.png", "b.png", "c.png"]})
    state, msg, _ = _load_fake(monkeypatch, tmp_path, empty, dropped)
    assert state is None
    assert "0 samples" in msg or "No sample" in msg, msg


def test_load_message_names_the_drop_reason(monkeypatch, tmp_path, png):
    """V-A-18: rows dropped as unreadable are not called missing."""
    df = pd.DataFrame({"image_path": [png]})
    dropped = pd.DataFrame({"image_path": ["x.png"] * 5, "verify_reason": ["unreadable"] * 5})
    state, msg, _ = _load_fake(monkeypatch, tmp_path, df, dropped)
    assert state is not None
    assert "unreadable" in msg and "missing" not in msg, msg


def test_labels_box_names_cls_in_sorted_order(monkeypatch, tmp_path, png):
    """V-A-4: ``cls`` is in sorted label order (D-05). The Labels box must use that order."""
    class Unsorted(_Adapter):
        LABEL_COLS = ["no_finding", "edema", "atelectasis"]
        _cols = {"cls": sorted(LABEL_COLS)}  # as BaseRadiologicalDataset._cols

    # The image has only atelectasis.
    df = pd.DataFrame({"image_path": [os.path.basename(png)], "no_finding": [0.0], "edema": [0.0],
                       "atelectasis": [1.0]})
    cfg = app.DatasetConfig(False, "", lambda *a, **k: (Unsorted(df, pd.DataFrame()), None), modality="CXR")
    monkeypatch.setitem(app.DATASET_REGISTRY, "Fake", cfg)
    state, msg, _ = app.load_dataset("Fake", str(tmp_path), "", "", "", None, "",
                                     True, False, False, False, False, progress=lambda *a, **k: None)
    assert state is not None, msg
    state["data_dicts"][0]["cls"] = torch.tensor(state["data_dicts"][0]["cls"])
    res = app._run(state, 64, False, False, False, *_aug())
    assert res[3] == "atelectasis"


@pytest.mark.parametrize("entry, subset, folder", [
    ("MedSeg COVID-19 CT", "dataset1", "dataset1_hu-1000_400"),
    ("MedSeg COVID-19 CT (val_im, no masks)", "", "dataset1_val_hu-1000_400"),
])
def test_medseg_sample_path_is_in_the_slice_folder(tmp_path, entry, subset, folder):
    """V-A-30, V-H1-42: the app reads a MedSeg slice from ``<slice_output_dir>/<subset>_hu<low>_<high>/images/``."""
    import SimpleITK as sitk

    rel, slices = tmp_path / "release", tmp_path / "slices"
    rel.mkdir()
    vol = np.random.RandomState(0).randint(-1000, 400, size=(3, 64, 64)).astype(np.int16)
    for name, arr in (("tr_im", vol), ("tr_mask", np.zeros_like(vol, np.uint8)), ("val_im", vol)):
        sitk.WriteImage(sitk.GetImageFromArray(arr), str(rel / f"{name}.nii.gz"))
    # The dataset1 slice map: SIRM case, slice index, age and sex (D-10).
    (rel / "Test-Images-Clinical-Details.csv").write_text(
        "SIRM Case,Slice position in .nii.gz,Patient age,Patient Sex (Male = 0 Female = 1)\n1,0,50,0\n1,1,,\n2,2,60,1\n")

    state, msg, _ = app.load_dataset(entry, str(rel), "", str(slices), subset, None, "",
                                     True, False, False, False, False, progress=lambda *a, **k: None)
    assert state is not None, msg
    volume = "val" if "val" in folder else "tr"
    assert state["data_dicts"][0]["img"] == str(slices / folder / "images" / f"dataset1_{volume}_0000.png")
    random.seed(0)
    res = app.run_2d(state, 64, True, True, True, [], *_aug())
    assert "Error" not in str(res[1]), res[1]


def _write_series(folder, positions, series=None):
    """Write a fake CT series: PixelSpacing (row 0.5, column 0.6), SliceThickness 1.0."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    study, series = generate_uid(), series or generate_uid()
    for pos in positions:
        meta = FileMetaDataset()
        meta.TransferSyntaxUID, meta.MediaStorageSOPClassUID = ExplicitVRLittleEndian, CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.Modality, ds.PatientID = study, series, "CT", "p1"
        ds.ImageOrientationPatient, ds.ImagePositionPatient = [1, 0, 0, 0, 1, 0], [float(v) for v in pos]
        ds.PixelSpacing, ds.SliceThickness = [0.5, 0.6], 1.0
        ds.Rows = ds.Columns = 8
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.PixelData = np.zeros((8, 8), np.uint16).tobytes()
        ds.save_as(str(folder / f"{meta.MediaStorageSOPInstanceUID}.dcm"), enforce_file_format=True)


def test_native_spacing_of_a_dicom_series_is_the_spacing_of_the_3d_load(tmp_path):
    """V-A-31: z is the step between slice positions, not SliceThickness. x and y are in ITK order.

    The fake slices are 0.8 mm apart, and SliceThickness is 1.0. Some RSNA series also have a
    thickness that is not the slice step.
    """
    _write_series(tmp_path, [(0, 0, z) for z in (3.2, 0.0, 1.6, 0.8, 2.4)])  # file order is not slice order
    np.testing.assert_allclose(app._probe_native_spacing(str(tmp_path)), (0.6, 0.5, 0.8))


@pytest.mark.parametrize("case", ["plain", "gantry tilt", "two series"])
def test_native_spacing_equals_the_itk_spacing(tmp_path, case):
    """V-A-31: the probe gives the spacing of the 3-D load (ITKReader).

    This is also true for a tilted series, as in 2 RSNA series. It is true for a folder with
    2 series, where ITK reads only one.
    """
    import monai as mn

    if case == "gantry tilt":
        _write_series(tmp_path, [(0, 0.4 * k, 0.8 * k) for k in range(5)])
    else:
        _write_series(tmp_path, [(0, 0, 0.8 * k) for k in range(5)])
    if case == "two series":
        _write_series(tmp_path, [(0, 0, 2.0 * k + 0.1) for k in range(5)])
    loaded = mn.transforms.LoadImage(reader="ITKReader", image_only=True)(str(tmp_path))  # as _base_load_3d
    np.testing.assert_allclose(app._probe_native_spacing(str(tmp_path)), [float(v) for v in loaded.pixdim])


# --------------------------------------------------------------------------
# _run
# --------------------------------------------------------------------------
def test_labels_box_with_no_positive_label(png):
    """V-A-14: 'No finding' is a label name. A sample with every label 0 is not that."""
    sample = {"img": png, "cls": torch.zeros(3)}
    state = app._make_state([sample], ["atelectasis", "no_finding", "edema"], [],
                            {"img", "cls"}, is_3d=False, dataset_name="fake")
    res = app._run(state, 64, False, False, False, *_aug())
    assert res[3] != "No finding"
    assert "positive" in res[3].lower(), res[3]


@pytest.mark.parametrize("flip", [False, True])
def test_transform_snippet_runs_on_a_box_sample(png, flip):
    """V-A-20: the snippet runs, and the run keeps the state keys."""
    sample = {"img": png, "bbox": [[0.10, 0.20, 0.30, 0.60]], "bbox_labels": ["nodule"]}
    state = _box_state(sample)
    random.seed(0)
    res = app._run(state, 64, True, False, True, *_aug(flip))
    assert state["available_keys"] == {"img", "bbox", "bbox_labels"}
    code = res[6]
    assert "with_bbox_as_mask()" in code
    ns: dict = {}
    exec(code, ns)
    out = ns["transform"](dict(sample))
    assert "bbox" in out and "bbox_mask" not in out


def test_the_color_figure_entries_preview_3_channels(png):
    """V-D-27 (D-42): the preview and the snippet of ROCO, ROCOv2, MedPMC and VQA-RAD use 3 channels."""
    names = list(app.DATASET_REGISTRY)
    for prefix in app._COLOR_FIGURE_ENTRIES:  # a renamed entry fails here
        assert any(n.startswith(prefix) for n in names), prefix
    assert app._channels("ROCO (radiology captioning, Train)") == 3
    assert app._channels("CheXinstruct (Test)") == 1
    sample = {"img": png}
    state = app._make_state([sample], [], [], {"img"}, is_3d=False, dataset_name="VQA-RAD (Train)")
    code = app._run(state, 64, False, False, False, *_aug())[6]
    assert "channels=3" in code
    ns: dict = {}
    exec(code, ns)
    assert ns["transform"](dict(sample))["img"].shape[0] == 3


def test_bbox_class_filter_does_not_change_the_stored_sample(png):
    """V-A-21: 'Run on same sample' starts from the full sample, not the filtered one."""
    sample = {"img": png, "bbox": "[[0.1, 0.3, 0.1, 0.3], [0.6, 0.9, 0.6, 0.9]]",
              "bbox_labels": '["nodule", "effusion"]'}
    state = _box_state(sample)
    res = app.run_2d(state, 64, False, False, True, ["nodule"], *_aug())
    last = res[7]
    assert last["bbox_labels"] == sample["bbox_labels"]
    res = app.run_2d_same(state, last, 64, False, False, True, ["effusion"], *_aug())
    shown = res[8][0][1]["bbox_labels"]
    shown = app._parse_str_list(shown)
    assert list(shown) == ["effusion"], shown


# --------------------------------------------------------------------------
# UI wiring
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", [
    "MR-RATE (brain/spine MRI + reports)", "MedPMC (PMC, OUR derived tags)",
])
def test_extra_field_is_visible_when_the_build_reads_it(name):
    """V-A-24: these entries read extra_field, and they set a placeholder."""
    cfg = app.DATASET_REGISTRY[name]
    src = inspect.getsource(cfg.build)
    body = src.split(":\n", 1)[1]  # skip the signature, which every build has
    assert "extra_field" in body
    extra_field_update = app._update_extra_fields(name)[2]
    assert extra_field_update["visible"] is True


def _box_builders() -> set[str]:
    """The app build functions whose own default is output_bbox=True."""
    return {name for name, fn in vars(app).items()
            if name.startswith("_build_") and callable(fn) and 'flags.get("output_bbox", True)' in inspect.getsource(fn)}


@pytest.mark.parametrize("name", [n for n, c in app.DATASET_REGISTRY.items() if not c.is_group_header])
def test_bbox_checkbox_follows_the_build_default(name):
    """V-A-13 (D-11): picking a dataset sets the BBox checkbox to the output_bbox default of its build function."""
    cfg = app.DATASET_REGISTRY[name]
    src = inspect.getsource(cfg.build)
    expected = any(f"{b}(" in src or cfg.build.__name__ == b for b in _box_builders())
    assert app._update_extra_fields(name)[5]["value"] is expected


def test_chest_imagenome_gold_loads_boxes_with_the_default_checkbox():
    """V-A-13: the box datasets are on, for example Chest ImaGenome gold. A label-only dataset is off."""
    assert app._update_extra_fields("Chest ImaGenome (Gold)")[5]["value"] is True
    assert app._update_extra_fields("BRAX (DICOM)")[5]["value"] is False


@pytest.mark.parametrize("with_header, loaded", [(False, False), (True, True)])
def test_plugin_entry_with_no_family_header_does_not_stop_the_app(tmp_path, with_header, loaded):
    """V-A-42: "a bad plugin shouldn't kill the app". An entry with no family header is not loaded."""
    import subprocess

    header = 'registry["── My family ──"] = _header("Other")\n    ' if with_header else ""
    (tmp_path / "my_app_plugin.py").write_text(
        "def register_app_datasets(registry, DatasetConfig, _header, app_dtype):\n"
        f"    {header}registry['My Private Dataset'] = DatasetConfig(\n"
        "        False, 'Extra', lambda *a, **k: (None, 'demo'), modality='Other')\n")
    env = dict(os.environ, RADHARMONY_APP_PLUGINS="my_app_plugin", PYTHONPATH=f"{tmp_path}:{ROOT}")
    proc = subprocess.run([sys.executable, "-c", "import app; print('My Private Dataset' in app.DATASET_REGISTRY)"],
                          cwd=ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-500:]
    assert proc.stdout.strip().splitlines()[-1] == str(loaded)
    assert ("not loaded" in proc.stdout) is not loaded


def test_css_reaches_the_page():
    """V-A-17: Gradio 6 takes css in launch(), not in Blocks()."""
    src = (ROOT / "app.py").read_text()
    launch = src.split("demo.launch(")[-1].split(")")[0]
    assert "css=_CSS" in launch, launch


# --------------------------------------------------------------------------
# Q-25 (D-09): the app lists the split keys only, and every dataset
# --------------------------------------------------------------------------
def _inventory_old_keys() -> set[str]:
    """The old keys of the inventory (D-09). They mix the splits, or a file picks the split.

    An old key is in a row that says "not for benchmarks", or in the same row as its split keys.
    Example: `chexpert` next to `chexpert_train` and `chexpert_valid`.
    """
    import re

    old = set()
    for r in (ROOT / "docs" / "wiki" / "reference" / "datasets" / "index.md").read_text().splitlines():
        if not r.startswith("|") or r.count("|") < 8:
            continue
        keys = re.findall(r"`([a-z0-9_]+)`", r.split("|")[6])
        split_of = re.compile(r"_(train|val|valid|validate|validation|test)(_|$)")
        old |= {k for k in keys
                if "not for benchmarks" in r or any(o.startswith(k) and split_of.match(o[len(k):]) for o in keys)}
    return old


def _app_keys() -> dict[str, list[str]]:
    """Registry key -> the app entries whose build function names its dataset or harmonizer class."""
    import re

    import radharmony.dataset  # noqa: F401  (fills the registry)
    from radharmony.registry import list_datasets, resolve_dataset

    by_class = {resolve_dataset(k).__name__: k for k in list_datasets()}
    used: dict[str, list[str]] = {}
    for name, cfg in app.DATASET_REGISTRY.items():
        if getattr(cfg, "build", None) is None:
            continue
        for c in set(re.findall(r"\b([A-Z][A-Za-z0-9]+(?:Dataset|Harmonizer))\b", inspect.getsource(cfg.build))):
            key = by_class.get(c) or by_class.get(c.replace("Harmonizer", "Dataset"))
            if key:
                used.setdefault(key, []).append(name)
    return used


def test_app_lists_the_split_keys_and_every_dataset():
    """Q-25 (D-09): no app entry builds an old key, and every other key has an entry.

    Emory CXR DICOM stays out of the app (Frank Li, 2026-11-01).
    """
    from radharmony.registry import list_datasets

    old, used = _inventory_old_keys(), _app_keys()
    assert len(old) >= 30, old
    assert {k: used[k] for k in old if k in used} == {}
    assert sorted(set(list_datasets()) - set(used) - old - {"emory_cxr_dicom"}) == []


# --------------------------------------------------------------------------
# A-5: datasets that the docs list must be in the app
# --------------------------------------------------------------------------
@pytest.mark.parametrize("entry, cls_name", [
    ("VQA-RAD (Test)", "VQARadTestDataset"),
    ("CheXinstruct (VQA, Test)", "ChexInstructVQATestDataset"),
    ("MIMIC-Ext-CXR-QBA (Test)", "MIMICExtCXRQBATestDataset"),
])
def test_vqa_datasets_are_in_the_app(entry, cls_name):
    cfg = app.DATASET_REGISTRY[entry]
    assert cfg.modality == "VQA"
    assert cls_name in inspect.getsource(cfg.build)


# D-09: each official split key of these datasets has its own app entry.
@pytest.mark.parametrize("entry, cls_name", [
    ("ROCO (radiology captioning, Train)", "ROCOTrainHarmonizer"),
    ("ROCO (radiology captioning, Validation)", "ROCOValidationHarmonizer"),
    ("ROCO (radiology captioning, Test)", "ROCOTestHarmonizer"),
    ("ROCOv2 (radiology, curated UMLS modality, Train)", "ROCOv2TrainHarmonizer"),
    ("ROCOv2 (radiology, curated UMLS modality, Validation)", "ROCOv2ValidationHarmonizer"),
    ("ROCOv2 (radiology, curated UMLS modality, Test)", "ROCOv2TestHarmonizer"),
    ("VQA-RAD (Train)", "VQARadTrainDataset"),
    ("VQA-RAD (Test)", "VQARadTestDataset"),
    ("CheXinstruct (VQA, Train)", "ChexInstructVQATrainDataset"),
    ("CheXinstruct (VQA, Val)", "ChexInstructVQAValDataset"),
    ("CheXinstruct (VQA, Test)", "ChexInstructVQATestDataset"),
    ("MIMIC-Ext-CXR-QBA (Train)", "MIMICExtCXRQBATrainDataset"),
    ("MIMIC-Ext-CXR-QBA (Validate)", "MIMICExtCXRQBAValidateDataset"),
    ("MIMIC-Ext-CXR-QBA (Test)", "MIMICExtCXRQBATestDataset"),
])
def test_official_split_entries_are_in_the_app(entry, cls_name):
    cfg = app.DATASET_REGISTRY[entry]
    assert cfg.modality == "VQA"
    assert cls_name in inspect.getsource(cfg.build)


def _fake_roco(root: Path) -> None:
    """Write one radiology caption for each ROCO split. The image ID has the split name (for example ROCO_train)."""
    for split in ("train", "validation", "test"):
        folder = root / "data" / split / "radiology"
        folder.mkdir(parents=True)
        (folder / "captions.txt").write_text(f"ROCO_{split}\tA {split} caption\n")
        (folder / "keywords.txt").write_text(f"ROCO_{split}\t\tchest\n")


@pytest.mark.parametrize("entry, split", [
    ("ROCO (radiology captioning, Train)", "train"),
    ("ROCO (radiology captioning, Validation)", "validation"),
    ("ROCO (radiology captioning, Test)", "test"),
])
def test_roco_split_entry_gives_only_its_split(tmp_path, entry, split):
    _fake_roco(tmp_path)
    ds, err = app.DATASET_REGISTRY[entry].build(str(tmp_path), "", "", "", None)
    assert err is None, err
    df = ds.get_harmonized_df()
    assert df["question_id"].tolist() == [f"ROCO_{split}"]


def test_vqa_rad_split_entries_load_on_the_nas():
    """VQA-RAD has 1,797 train and 451 test questions (Lau et al. 2018, the source JSON)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from dataset_paths import is_complete, paths
    base = paths("vqa_rad", missing_ok=not is_complete()).get("base_image_dir", "")
    if not os.access(base, os.R_OK):
        pytest.skip("VQA-RAD not readable")
    for entry, n_questions in (("VQA-RAD (Train)", 1797), ("VQA-RAD (Test)", 451)):
        ds, err = app.DATASET_REGISTRY[entry].build(base, "", "", "", None)
        assert err is None, err
        qa = ds.get_harmonized_df()["qa_text"]
        assert qa.str.count(r"^Q \(|\n\nQ \(").sum() == n_questions, entry


def test_gemex_row_shows_only_its_own_question(monkeypatch, tmp_path):
    """Q-3 (D-11): a GEMeX VQA row in the app shows its own question and answer, not those of its image."""
    import radharmony.harmonizer as H

    table = pd.DataFrame({
        "image_path": ["a.jpg", "a.jpg"], "question": ["Is the heart normal?", "Which side is the effusion?"],
        "answer": ["No", "A"], "question_type": ["open", "choice"],
        "question_subtype": ["open_ended", "single_choice"], "choices": [None, ["A: left", "B: right"]],
        "visual_locations": [[], []], "visual_regions": [[], []], "ori_report": ["r", "r"]})

    class _Fake:
        def __init__(self, **kwargs):
            pass

        def harmonize(self):
            return table.copy()

    monkeypatch.setattr(H, "GEMeXVQAHarmonizer", _Fake)
    adapter, err = app._build_gemex_vqa(str(tmp_path), "", str(tmp_path), "", None)
    assert err is None, err
    qa = adapter.get_harmonized_df()["qa_text"].tolist()
    assert "Is the heart normal?" in qa[0] and "effusion" not in qa[0]
    assert "effusion" in qa[1] and "heart" not in qa[1] and "A: left ✓" in qa[1]


def test_vqa_adapter_shows_a_close_ended_answer_from_answer_struct(tmp_path, png):
    """V-A-45: a CheXinstruct close-ended row has an empty ``answer``. Its answer is the true option of
    ``answer_struct``, so the Q&A box shows "yes", not an empty answer."""
    df = pd.DataFrame({"image_path": [png, png], "question": ["Is a tube present?", "Where is it?"],
                       "answer": ["", "left"], "question_type": ["Close-Ended VQA", "Open-Ended VQA"],
                       "answer_struct": [{"yes": True, "no": False}, None]})
    out = app._vqa_adapter(df, str(tmp_path), "fake").get_harmonized_df()
    assert out["qa_text"].tolist() == ["Q (Close-Ended VQA): Is a tube present?\n   → yes",
                                       "Q (Open-Ended VQA): Where is it?\n   → left"]


def test_gemex_box_undoes_the_224_letterbox():
    """V-A-44 (D-07): a GEMeX box is in the 224 x 224 Chest ImaGenome space (long side 224, then padded).

    Expected: the Chest ImaGenome original pixel box of the same region, divided by the image size. For the
    trachea of a 3056 x 2544 image, this is [0.150, 0.381, 0.317, 0.433]. The old code divided by 256.
    """
    box = app._gemex_box([70, 46, 96, 89], width=3056, height=2544)
    assert box == pytest.approx([0.150, 0.381, 0.317, 0.433], abs=0.01)
    assert app._gemex_box([8, 85, 168, 205], width=3056, height=2544)[1] == pytest.approx(1.0, abs=0.01)


def test_vqa_adapter_gives_one_row_per_question(tmp_path, png):
    """Q-3 (D-11): an app VQA row shows only its own question and answer, as the VQA dataset class does."""
    df = pd.DataFrame({"image_path": [png, png], "question": ["Is it normal?", "Which side?"],
                       "answer": ["No", "Left"], "question_type": ["ABN", "POS"]})
    ad = app._vqa_adapter(df, str(tmp_path), "fake")
    out = ad.get_harmonized_df()
    assert out["qa_text"].tolist() == ["Q (ABN): Is it normal?\n   → No", "Q (POS): Which side?\n   → Left"]


def test_vqa_adapter_makes_absolute_paths_relative(tmp_path, png):
    """A table with an absolute image_path (an old CheXinstruct table). load_dataset puts base_dir in front."""
    df = pd.DataFrame({"image_path": [png], "question": ["Q"], "answer": ["A"]})
    ad = app._vqa_adapter(df, str(tmp_path), "fake")
    rel = ad.get_harmonized_df()["image_path"].iloc[0]
    assert not os.path.isabs(rel)
    assert os.path.samefile(str(tmp_path) + "/" + rel, png)
    assert ad.verify_images().empty


def test_vqa_rad_loads_on_the_nas():
    sys.path.insert(0, str(ROOT / "scripts"))
    from dataset_paths import is_complete, paths
    base = paths("vqa_rad_test", missing_ok=not is_complete()).get("base_image_dir", "")
    if not os.access(base, os.R_OK):
        pytest.skip("VQA-RAD not readable")
    state, msg, _ = app.load_dataset("VQA-RAD (Test)", base, "", "", "", None, "",
                                     True, False, False, False, False,
                                     progress=lambda *a, **k: None)
    assert state is not None, msg
    assert "qa_text" in state["available_keys"]


@pytest.mark.parametrize("entry, key", [
    ("VinDr-CXR (Train)", "vindr_cxr_train"), ("VinDr-CXR (Test)", "vindr_cxr_test"),
    ("RAD-ChestCT (Train)", "radchestct_train"),
    ("Emory CXR v2", "emory_cxr"), ("ReXGradient-160K (Train)", "rexgradient_train"),
    ("CheXlocalize", "chexlocalize"),
])
def test_app_finds_the_files_in_sub_folders_on_the_nas(entry, key, monkeypatch):
    """D-35: infer_path looks only in exact places, so the app names the sub folders as the dataset classes do.

    The dataset __init__ stops at once: the test checks only that the app found the source files.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from dataset_paths import is_complete, paths
    from radharmony.dataset.base import BaseRadiologicalDataset

    base = paths(key, missing_ok=not is_complete()).get("base_image_dir", "")
    if not os.access(base, os.R_OK):
        pytest.skip(f"{key} not readable")

    class _Built(Exception):
        pass

    def stop(self, *args, **kwargs):
        raise _Built

    monkeypatch.setattr(BaseRadiologicalDataset, "__init__", stop)
    try:
        _, err = app.DATASET_REGISTRY[entry].build(base, "", "", "", None, output_cls=True)
    except _Built:
        err = None
    assert err is None, err


# --------------------------------------------------------------------------
# Review of 2026-11-02, group e
# --------------------------------------------------------------------------
def _load(entry, base, csv="", extra="", extra2="", dd=None, cls=True, **kw):
    return app.load_dataset(entry, base, csv, extra, extra2, dd, "", cls, False, False, False, False,
                            progress=lambda *a, **k: None)


def test_roco_caption_is_shown(tmp_path):
    """V-A-32: the caption of a ROCO sample is in the Q&A text box."""
    from PIL import Image
    folder = tmp_path / "data" / "test" / "radiology"
    (folder / "images").mkdir(parents=True)
    (folder / "captions.txt").write_text("ROCO_t\tCT of the chest shows a nodule\n")
    (folder / "keywords.txt").write_text("ROCO_t\t\tchest\tnodule\n")
    Image.fromarray((np.random.RandomState(0).rand(64, 64) * 255).astype(np.uint8)).save(
        folder / "images" / "ROCO_t.jpg")
    state, msg, _ = _load("ROCO (radiology captioning, Test)", str(tmp_path))
    assert state is not None, msg
    res = app._run(state, 64, False, False, False, *_aug())
    assert "CT of the chest shows a nodule" in res[5]


def test_tilde_base_dir_works(tmp_path, monkeypatch):
    """V-A-33: a base folder typed with '~' gives the same result as the full path."""
    from PIL import Image
    monkeypatch.setenv("HOME", str(tmp_path))
    img = tmp_path / "d" / "valid" / "patient00001" / "study1"
    img.mkdir(parents=True)
    Image.fromarray(np.zeros((32, 32), np.uint8)).save(img / "view1_frontal.jpg")
    cols = ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA", "No Finding", "Enlarged Cardiomediastinum",
            "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
            "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
            "Support Devices"]
    row = ["CheXpert-v1.0/valid/patient00001/study1/view1_frontal.jpg", "Male", 50, "Frontal", "AP",
           1.0] + [0.0] * 13
    pd.DataFrame([row], columns=cols).to_csv(tmp_path / "d" / "valid.csv", index=False)
    for base in (str(tmp_path / "d" / "valid") + "/", "~/d/valid/"):
        state, msg, _ = _load("CheXpert (Valid)", base)
        assert state is not None, (base, msg)


def test_chexpert_valid_without_valid_csv_is_an_error(tmp_path):
    """V-A-34: a missing valid.csv must not load train.csv."""
    (tmp_path / "CheXpert-v1.0" / "valid").mkdir(parents=True)
    pd.DataFrame({"Path": ["x"]}).to_csv(tmp_path / "CheXpert-v1.0" / "train.csv", index=False)
    from radharmony.dataset import CheXpertValidDataset
    with pytest.raises(FileNotFoundError, match="valid.csv"):
        CheXpertValidDataset(base_image_dir=str(tmp_path / "CheXpert-v1.0" / "valid") + "/",
                             cache_dir=None)
    state, msg, _ = _load("CheXpert (Valid)", str(tmp_path / "CheXpert-v1.0" / "valid") + "/")
    assert state is None and "valid.csv" in msg


def test_medpmc_modality_pt_nm_is_accepted():
    """V-A-35: PT/NM is in the placeholder and the error message, so it must work."""
    from radharmony.harmonizer.medpmc import vocab
    assert vocab.normalize("PT/NM") == "pet_spect"
    assert vocab.normalize(vocab.MODALITY_VOCAB["pet_spect"]["display"]) == "pet_spect"


@pytest.mark.parametrize("entry", ["RSNA Abdominal Trauma (Test)", "RSNA 2024 Lumbar Spine (Test)"])
def test_rsna_test_entries_show_no_unused_csv_box(entry):
    """V-A-36: the build ignores the extra field, so the app hides it."""
    cfg = app.DATASET_REGISTRY[entry]
    upd = app._update_extra_fields(entry)
    assert cfg.extra_field_label == ""
    assert upd[2]["visible"] is False  # the first extra textbox


def test_mimic_cxr_error_names_a_box_the_user_can_fill(tmp_path):
    """V-A-37: with no split file near the CSVs, the message names the Label CSV box."""
    (tmp_path / "cxr-record-list.csv").write_text("subject_id\n1\n")
    state, msg, _ = _load("MIMIC-CXR (Test)", str(tmp_path))
    assert state is None
    assert "split_csv_path" not in msg and "Label CSV" in msg, msg


def test_match_native_values_stay_in_the_slider_range():
    """V-A-48: 'Match native' gives values that the sliders accept."""
    import gradio as gr
    slider = gr.Slider(app._SPACING_MIN, app._SPACING_MAX, value=1.0, step=0.05)
    got = app._match_native_values((0.234, 0.234, 8.0))
    assert got == (0.25, 0.25, 5.0)
    for v in got:
        slider.preprocess(v)
