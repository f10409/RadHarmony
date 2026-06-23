"""Tests for BaseVQADataset / BaseVQAHarmonizer using synthetic data.

No NAS / qa.zip dependency — fixtures build a tiny in-memory harmonized
DataFrame plus on-disk PNG "images" inside ``tmp_path`` so the full
harmonize → save/load → split → __getitem__ path is exercised.

Run with::

    .venv/bin/python -m pytest tests/test_vqa_base.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from radharmony.dataset.base_vqa import BaseVQADataset
from radharmony.harmonizer.base_vqa import BaseVQAHarmonizer


def _make_image(path) -> None:
    arr = (np.random.rand(32, 32) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


@pytest.fixture
def synth_dataset(tmp_path):
    """Materialize a 6-row synthetic VQA dataset (3 patients × 2 questions)."""
    rows = []
    for pi in range(3):
        patient_id = f"p{pi:08d}"
        study_id = f"s{pi:09d}"
        img_path = tmp_path / f"{patient_id}.png"
        _make_image(img_path)
        for qi in range(2):
            rows.append(
                {
                    "patient_id": patient_id,
                    "study_id": study_id,
                    "question_id": f"q{qi:03d}",
                    "image_path": str(img_path),
                    "question": f"What is in image {pi}, question {qi}?",
                    "answer": f"answer-{pi}-{qi}",
                    "answer_struct": [{"text": f"ans-{qi}", "regions": ["lung"]}],
                    "question_type": "describe" if qi == 0 else "is_abnormal",
                    "quality": "4_A",
                }
            )
    return pd.DataFrame(rows)


class _SyntheticHarmonizer(BaseVQAHarmonizer):
    """Trivial in-memory harmonizer for testing."""

    def __init__(self, df: pd.DataFrame, base_image_dir: str = None):
        super().__init__(base_image_dir=base_image_dir)
        self._stash = df

    def _harmonizer_init_snapshot(self) -> dict:
        # df can't round-trip via init args; load_from_saved relies on the
        # stored harmonized_df instead.
        snap = super()._harmonizer_init_snapshot()
        snap["df"] = self._stash
        return snap

    def harmonize(self) -> pd.DataFrame:
        self.df = self._stash.copy()
        return self._select_harmonized_columns(self.df)


class _SyntheticVQADataset(BaseVQADataset):
    _HARMONIZER_CLS = _SyntheticHarmonizer

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        return self._harmonizer.harmonize()


def test_harmonizer_schema(synth_dataset):
    h = _SyntheticHarmonizer(synth_dataset)
    df = h.harmonize()
    for col in ("patient_id", "study_id", "question_id", "image_path", "question", "answer"):
        assert col in df.columns
    assert len(df) == 6


def test_harmonizer_rejects_missing_required_cols(synth_dataset):
    bad = synth_dataset.drop(columns=["answer"])
    h = _SyntheticHarmonizer(bad)
    with pytest.raises(ValueError, match="missing required columns"):
        h.harmonize()


def test_harmonizer_save_load_roundtrip(synth_dataset, tmp_path):
    h = _SyntheticHarmonizer(synth_dataset)
    h.harmonize()
    save_path = tmp_path / "harm.pkl"
    h.save(str(save_path))

    loaded = _SyntheticHarmonizer.load_from_saved(str(save_path))
    pd.testing.assert_frame_equal(
        loaded.harmonized_df.reset_index(drop=True),
        h.harmonized_df.reset_index(drop=True),
    )


def test_dataset_sample_keys(synth_dataset, tmp_path):
    """Each sample dict carries text fields plus an image tensor."""
    ds_obj = _SyntheticVQADataset(
        base_image_dir=None,
        cache_dir=str(tmp_path / "cache"),
        harmonized_df=synth_dataset,
        output_struct=True,
        output_question_type=True,
    )
    ds = ds_obj.get_datasets()
    sample = ds[0]
    assert "img" in sample
    assert isinstance(sample["question"], str) and sample["question"]
    assert isinstance(sample["answer"], str) and sample["answer"]
    assert sample["patient_id"].startswith("p")
    assert sample["study_id"].startswith("s")
    assert sample["question_id"].startswith("q")
    assert sample["question_type"] in {"describe", "is_abnormal"}
    assert isinstance(sample["answer_struct"], list)
    assert sample["answer_struct"][0]["regions"] == ["lung"]


def test_dataset_optional_outputs_off_by_default(synth_dataset, tmp_path):
    ds_obj = _SyntheticVQADataset(
        base_image_dir=None,
        cache_dir=str(tmp_path / "cache"),
        harmonized_df=synth_dataset,
    )
    ds = ds_obj.get_datasets()
    sample = ds[0]
    assert "answer_struct" not in sample
    assert "question_type" not in sample


def test_patient_level_split(synth_dataset, tmp_path):
    """Train/val split must hold all of a patient's questions on one side."""
    ds_obj = _SyntheticVQADataset(
        base_image_dir=None,
        cache_dir=str(tmp_path / "cache"),
        harmonized_df=synth_dataset,
    )
    train_ds, val_ds = ds_obj.get_datasets(n_splits=3)

    train_patients = {d["patient_id"] for d in train_ds.data}
    val_patients = {d["patient_id"] for d in val_ds.data}
    assert train_patients.isdisjoint(val_patients), "patients leaked across train/val"
    assert len(train_patients) + len(val_patients) == 3
