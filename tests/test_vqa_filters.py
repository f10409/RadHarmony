"""Filters and saved folders of the ROCO, ROCOv2, MedPMC and CheXinstruct VQA dataset classes. No real data.

D-02: a filter argument of a dataset class also filters a given table (``harmonized_df=``,
``harmonizer=`` or ``harmonizer_path=``). The expected rows come from the fake source files.
"""
from __future__ import annotations

import pandas as pd
import pytest
from test_roco import _rocov2_tree


def _ids(ds):
    return sorted(ds._get_harmonized_df()["question_id"])


# ---------------------------------------------------------------------------
# ROCOv2: IMG_A is ultrasound, IMG_B is MRI (test split)
# ---------------------------------------------------------------------------
def test_rocov2_filters_apply_to_a_given_table(tmp_path):
    """D-02: ``modalities`` and the split of a split class also filter a given table."""
    from radharmony.dataset import ROCOv2Dataset, ROCOv2TestDataset, ROCOv2TrainDataset
    from radharmony.harmonizer.rocov2 import ROCOv2Harmonizer

    _rocov2_tree(tmp_path)
    full = ROCOv2Harmonizer(base_dir=str(tmp_path), splits=["test"]).harmonize()
    assert _ids(ROCOv2Dataset(harmonized_df=full, modalities=["mri"], cache_dir=None)) == ["IMG_B"]
    assert _ids(ROCOv2Dataset(harmonized_df=full, cache_dir=None)) == ["IMG_A", "IMG_B"]
    assert _ids(ROCOv2TestDataset(harmonized_df=full, cache_dir=None)) == ["IMG_A", "IMG_B"]
    assert _ids(ROCOv2TrainDataset(harmonized_df=full, cache_dir=None)) == []


def test_rocov2_harmonizer_path_gives_the_saved_image_folder(tmp_path):
    """With ``harmonizer_path``, ``base_image_dir`` is the folder of the saved harmonizer, not ``<base_dir>/images``."""
    from radharmony.dataset import ROCOv2Dataset
    from radharmony.harmonizer.rocov2 import ROCOv2Harmonizer

    _rocov2_tree(tmp_path)
    h = ROCOv2Harmonizer(base_dir=str(tmp_path), base_image_dir=str(tmp_path / "pngs"), splits=["test"])
    h.harmonize()
    h.save(str(tmp_path / "h.pkl"))
    assert ROCOv2Dataset(harmonizer_path=str(tmp_path / "h.pkl"), cache_dir=None).base_image_dir == str(tmp_path / "pngs")


# ---------------------------------------------------------------------------
# ROCO: one radiology and one non-radiology image
# ---------------------------------------------------------------------------
def _roco_table():
    return pd.DataFrame({"patient_id": ["r", "n"], "study_id": ["r", "n"], "question_id": ["r", "n"],
                         "image_path": ["a.jpg", "b.jpg"], "question": ["", ""], "answer": ["x", "y"],
                         "split": ["test", "test"], "subset": ["radiology", "non-radiology"]})


@pytest.mark.parametrize("radiology_only, expected", [(None, ["n", "r"]), (True, ["r"]), (False, ["n", "r"])])
def test_roco_radiology_only_on_a_given_table(radiology_only, expected):
    """Decided by Frank Li (2026-11-04): on a given table, ``True`` keeps the radiology rows. ``None`` and ``False`` keep all."""
    from radharmony.dataset import ROCODataset

    ds = ROCODataset(harmonized_df=_roco_table(), radiology_only=radiology_only, cache_dir=None)
    assert _ids(ds) == expected


def test_roco_radiology_only_on_a_table_with_no_subset_column_raises():
    """``radiology_only=True`` needs the ``subset`` column. A table without it raises ``ValueError``."""
    from radharmony.dataset import ROCODataset

    ds = ROCODataset(harmonized_df=_roco_table().drop(columns="subset"), radiology_only=True, cache_dir=None)
    with pytest.raises(ValueError, match="subset"):
        ds._get_harmonized_df()


def test_roco_default_still_harmonizes_radiology_only(tmp_path):
    """The default ``None`` gives ``radiology_only=True`` to the harmonizer, as before."""
    from radharmony.dataset import ROCODataset

    ds = ROCODataset(base_dir=str(tmp_path), cache_dir=None)
    assert ds._harmonizer.radiology_only is True


# ---------------------------------------------------------------------------
# MedPMC: s1 is single-panel, m1 is multi-panel. Both are radiology CT with a high-confidence tag.
# ---------------------------------------------------------------------------
@pytest.fixture()
def medpmc_root(tmp_path):
    (tmp_path / "webdataset" / "metadata").mkdir(parents=True)
    pd.DataFrame({"source_type": ["singlepanel", "multipanel"], "pmcid": ["PMC1", "PMC2"],
                  "image_id": ["s1", "m1"], "parent_image_id": [None, None],
                  "subfigure_index": [None, None], "figure_label": ["1", "2"],
                  "shard": ["a", "b"], "key": ["k1", "k2"], "image_ext": [".jpg", ".jpg"]},
                 ).to_parquet(tmp_path / "webdataset" / "metadata" / "medpmc_index.parquet")
    (tmp_path / "tags").mkdir()
    pd.DataFrame({"key": ["k1", "k2"], "specialty": ["radiology", "radiology"], "modality": ["ct", "mri"],
                  "clip_prob": [0.99, 0.5]}).to_parquet(tmp_path / "tags" / "medpmc_tags_clip.parquet")
    return tmp_path


def _medpmc_tags(root):
    pd.DataFrame({"image_id": ["s1", "m1"], "caption": ["CT of chest", "A: chest CT. B: brain MRI"],
                  "specialty": ["radiology", "radiology"], "modality": ["ct", "ct"],
                  "tag_confidence": ["high", "high"], "tag_source": ["caption", "caption"],
                  "tag_method": ["lex", "lex"]}).to_parquet(root / "tags" / "medpmc_tags.parquet")


def test_medpmc_filters_apply_to_a_given_table(medpmc_root):
    """D-02: the subset, specialty and CLIP filters of the dataset class also filter a given table."""
    from radharmony.dataset import MedPMCDataset
    from radharmony.harmonizer import MedPMCHarmonizer

    _medpmc_tags(medpmc_root)
    full = MedPMCHarmonizer(base_dir=str(medpmc_root), min_clip_prob=0.0).harmonize()
    assert sorted(full["question_id"]) == ["m1", "s1"]
    assert _ids(MedPMCDataset(harmonized_df=full, subsets=["singlepanel"], cache_dir=None)) == ["s1"]
    assert _ids(MedPMCDataset(harmonized_df=full, min_clip_prob=0.9, cache_dir=None)) == ["s1"]
    assert _ids(MedPMCDataset(harmonized_df=full, radiology_only=True, include_multipanel=False,
                              cache_dir=None)) == ["s1"]
    # The default modality_source="clip" reads the CLIP modality: m1 is MRI there.
    assert _ids(MedPMCDataset(harmonized_df=full, modalities=["mri"], cache_dir=None)) == ["m1"]
    assert _ids(MedPMCDataset(harmonized_df=full, modalities=["mri"], modality_source="caption",
                              cache_dir=None)) == []


def test_medpmc_filter_on_a_table_with_no_such_column_raises(medpmc_root):
    """A filter that needs a column the given table does not have raises. It does not drop every row."""
    from radharmony.dataset import MedPMCDataset
    from radharmony.harmonizer import MedPMCHarmonizer

    _medpmc_tags(medpmc_root)
    full = MedPMCHarmonizer(base_dir=str(medpmc_root)).harmonize()  # no CLIP filter, so no clip_prob column
    with pytest.raises(ValueError, match="clip_prob"):
        MedPMCDataset(harmonized_df=full, min_clip_prob=0.9, cache_dir=None)._get_harmonized_df()
    # The default modality_source="clip" needs clip_modality. Before, it used the caption tag with no error,
    # so it gave other rows than harmonize() with the same arguments.
    with pytest.raises(ValueError, match="clip_modality"):
        MedPMCDataset(harmonized_df=full, modalities=["ct"], cache_dir=None)._get_harmonized_df()
    assert _ids(MedPMCDataset(harmonized_df=full, modalities=["ct"], modality_source="caption",
                              cache_dir=None)) == ["m1", "s1"]


def test_medpmc_tagger_agreement_without_a_tag_table_raises(medpmc_root):
    """Agreement compares the caption tag with the CLIP tag. With no caption tag table, it raises.

    Before, every caption tag was None, so the filter kept 0 rows with no message.
    """
    from radharmony.harmonizer import MedPMCHarmonizer

    with pytest.raises(FileNotFoundError, match="medpmc_tags.parquet"):
        MedPMCHarmonizer(base_dir=str(medpmc_root), require_tagger_agreement=True).harmonize()


def test_medpmc_harmonizer_path_gives_the_saved_image_folder(medpmc_root):
    """With ``harmonizer_path``, ``base_image_dir`` is the folder of the saved harmonizer."""
    from radharmony.dataset import MedPMCDataset
    from radharmony.harmonizer import MedPMCHarmonizer

    _medpmc_tags(medpmc_root)
    h = MedPMCHarmonizer(base_dir=str(medpmc_root), base_image_dir=str(medpmc_root / "pngs"))
    h.harmonize()
    h.save(str(medpmc_root / "h.pkl"))
    ds = MedPMCDataset(harmonizer_path=str(medpmc_root / "h.pkl"), cache_dir=None)
    assert ds.base_image_dir == str(medpmc_root / "pngs")


def test_medpmc_dataset_passes_modality_source_and_include_multipanel(medpmc_root):
    """The dataset class takes the 2 harmonizer arguments and gives them to its harmonizer."""
    from radharmony.dataset import MedPMCDataset

    _medpmc_tags(medpmc_root)
    ds = MedPMCDataset(base_dir=str(medpmc_root), radiology_only=True, include_multipanel=False,
                       modality_source="caption", cache_dir=None)
    assert ds._harmonizer.modality_source == "caption" and ds._harmonizer.include_multipanel is False
    assert _ids(ds) == ["s1"]


# ---------------------------------------------------------------------------
# CheXinstruct VQA
# ---------------------------------------------------------------------------
def test_chexinstruct_filters_apply_to_a_given_table():
    """D-02: ``question_types`` and ``data_sources`` also filter a given table."""
    from radharmony.dataset import ChexInstructVQADataset

    df = pd.DataFrame({"patient_id": ["1", "2", "3"], "study_id": ["1", "2", "3"], "question_id": ["a", "b", "c"],
                       "image_path": ["x.jpg", "y.jpg", "z.jpg"], "question": ["q"] * 3, "answer": ["a"] * 3,
                       "question_type": ["Open-Ended VQA", "Close-Ended VQA", "Open-Ended VQA"],
                       "data_source": ["MIMIC-CXR-VQA", "MIMIC-CXR-VQA", "MIMIC-Diff-VQA"]})
    ds = ChexInstructVQADataset(harmonized_df=df, question_types=["Open-Ended VQA"],
                                data_sources=["MIMIC-CXR-VQA"], cache_dir=None)
    assert _ids(ds) == ["a"]
    assert _ids(ChexInstructVQADataset(harmonized_df=df, cache_dir=None)) == ["a", "b", "c"]
