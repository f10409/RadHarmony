"""``verify_images()`` with no argument uses the image folder of the harmonizer (D-04, V-H3-36).

Shenzhen and VinDr-PCXR keep the folder as ``base_dir``, and SIIM-ACR PTX keeps it as
``dicom_dir``. Each test gives a table of 2 rows: one file exists in ``tmp_path`` and one does not.
"""
from __future__ import annotations

import pandas as pd
import pytest


def _shenzhen(folder):
    from radharmony.harmonizer.shenzhen_cxr import ShenzhenCXRHarmonizer
    return ShenzhenCXRHarmonizer(base_dir=folder)


def _vindr_pcxr(folder):
    from radharmony.harmonizer.vindr_pcxr.vindr_pcxr_split import VinDrPCXRTestHarmonizer
    return VinDrPCXRTestHarmonizer(base_dir=folder)


def _siim_acr_ptx(folder):
    from radharmony.harmonizer.siim_acr_ptx import SIIMACRPTXTestHarmonizer
    return SIIMACRPTXTestHarmonizer(dicom_dir=folder)


@pytest.mark.parametrize("make, present, missing", [
    (_shenzhen, "CHNCXR_0001_0.png", "CHNCXR_0002_0.png"),
    (_vindr_pcxr, "test/a.dicom", "test/b.dicom"),
    (_siim_acr_ptx, "p1/s1/a.dcm", "p2/s2/b.dcm"),
])
def test_verify_images_with_no_argument_uses_the_image_folder(make, present, missing, tmp_path):
    """V-H3-36: verify_images() with no argument checks the image folder of the harmonizer."""
    (tmp_path / present).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / present).write_bytes(b"x")
    h = make(str(tmp_path))
    h.set_harmonized_df(pd.DataFrame({"patient_id": ["1", "2"], "study_id": ["1", "2"],
                                      "image_path": [present, missing]}))
    gone = h.verify_images(drop_missing=False)
    assert gone["image_path"].tolist() == [missing]


def _pe_dataset(df, folder):
    from radharmony.dataset import RSNAPEDetectionDataset
    return RSNAPEDetectionDataset(harmonized_df=df, base_image_dir=folder, cache_dir=None)


def _abdominal_dataset(df, folder):
    from radharmony.dataset import RSNAAbdominalTrauma2023Dataset
    return RSNAAbdominalTrauma2023Dataset(harmonized_df=df, base_image_dir=folder, cache_dir=None)


@pytest.mark.parametrize("make", [_pe_dataset, _abdominal_dataset])
def test_series_dataset_verify_images_needs_the_image_folder(make, tmp_path):
    """D-04: ``image_path`` is a series folder relative to ``base_image_dir``.

    With no folder, verify_images() raises. Before, it dropped every row.
    With the folder, it drops only the missing series.
    """
    (tmp_path / "1" / "10").mkdir(parents=True)
    df = pd.DataFrame({"patient_id": ["1", "2"], "study_id": ["1", "2"], "image_path": ["1/10", "2/20"]})
    ds = make(df, None)
    with pytest.raises(ValueError, match="D-04"):
        ds.verify_images()
    assert len(ds._get_harmonized_df()) == 2

    ds = make(df, str(tmp_path))
    gone = ds.verify_images()
    assert gone["image_path"].tolist() == ["2/20"]
    assert ds._get_harmonized_df()["image_path"].tolist() == ["1/10"]


def _vqa_dataset(df):
    from radharmony.dataset.base_vqa import BaseVQADataset

    class _VQA(BaseVQADataset):
        def _get_harmonized_df(self):
            return self._preset_harmonized_df

    return _VQA(harmonized_df=df, cache_dir=None)


def _named(name):
    def make(df):
        import radharmony.dataset as D
        return getattr(D, name)(harmonized_df=df, cache_dir=None)
    return make


_NO_FOLDER = [_named(n) for n in (
    "CheXpertDataset", "ChestXray14Dataset", "ChestXray14BboxDataset", "RSNA2024LumbarSpineDataset",
    "RSNA2022CervicalSpineDataset", "RSNA2022CervicalSpineBboxDataset")] + [_vqa_dataset]


@pytest.mark.parametrize("make", _NO_FOLDER)
def test_verify_images_with_relative_paths_and_no_folder_raises(make):
    """D-04: with relative paths and no ``base_image_dir``, verify_images() raises and keeps every row.

    Before, each path was read from the working folder, so every row looked missing and was dropped.
    """
    df = pd.DataFrame({"patient_id": ["1", "2"], "study_id": ["1", "2"], "image_path": ["a/1.png", "b/2.png"],
                       "question": ["q", "q"], "answer": ["a", "a"]})
    ds = make(df)
    with pytest.raises(ValueError, match="D-04"):
        ds.verify_images()
    assert len(ds._get_harmonized_df()) == 2


def test_verify_images_with_absolute_paths_needs_no_folder(tmp_path):
    """D-04: an absolute ``image_path`` needs no ``base_image_dir``. Only the missing file is dropped."""
    (tmp_path / "1.png").write_bytes(b"x")
    df = pd.DataFrame({"patient_id": ["1", "2"], "study_id": ["1", "2"],
                       "image_path": [str(tmp_path / "1.png"), str(tmp_path / "2.png")]})
    ds = _named("CheXpertDataset")(df)
    assert ds.verify_images()["image_path"].tolist() == [str(tmp_path / "2.png")]


def test_chexinstruct_verify_images_uses_the_folder_of_each_source(tmp_path):
    """D-04: with 2 source folders, the class has no ``base_image_dir``. verify_images() joins each source folder.

    Before 2026-11-04, every row was dropped. Then the shared check raised. Now only the missing file is dropped.
    """
    import json

    from radharmony.dataset import ChexInstructVQADataset

    def rec(i, source):
        return {"unique_id": i, "study_id": f"[mimic-cxr] [5{i}]", "data_source": source,
                "task_name": "Open-Ended VQA", "image_path": f"data/mimic-cxr/files/p1/p1{i}/s5{i}/d.jpg",
                "text": "q", "target_text": "a", "qa_pair": None, "split": "train"}

    (tmp_path / "c.json").write_text(json.dumps([rec(1, "MIMIC-CXR-VQA"), rec(2, "MIMIC-Diff-VQA")]))
    (tmp_path / "a" / "p1/p11/s51").mkdir(parents=True)
    (tmp_path / "a" / "p1/p11/s51/d.jpg").write_bytes(b"x")  # source b has no file
    ds = ChexInstructVQADataset(source_image_dirs={"MIMIC-CXR-VQA": str(tmp_path / "a"),
                                                   "MIMIC-Diff-VQA": str(tmp_path / "b")},
                                json_path=str(tmp_path / "c.json"), cache_dir=None)
    assert ds.base_image_dir is None
    gone = ds.verify_images()
    assert gone["data_source"].tolist() == ["MIMIC-Diff-VQA"]
    assert ds._get_harmonized_df()["data_source"].tolist() == ["MIMIC-CXR-VQA"]
