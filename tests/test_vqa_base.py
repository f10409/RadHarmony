"""Tests for BaseVQADataset / BaseVQAHarmonizer using synthetic data.

No NAS / qa.zip dependency — fixtures build a tiny in-memory harmonized
DataFrame plus on-disk PNG "images" inside ``tmp_path`` so the full
harmonize → save/load → split → __getitem__ path is exercised.

Run with::

    .venv/bin/python -m pytest tests/test_vqa_base.py -v
"""
from __future__ import annotations

import os

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
    # Only the documented keys. MONAI ToTensorD adds a trace record, img_transforms.
    assert set(sample) == {"img", "question", "answer", "patient_id", "study_id", "question_id",
                           "question_type", "answer_struct"}


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


@pytest.mark.parametrize("flag, col", [("output_struct", "answer_struct"),
                                       ("output_question_type", "question_type")])
def test_output_flag_with_no_column_warns_and_adds_no_key(synth_dataset, tmp_path, flag, col):
    """D-11: a flag that asks for a column the table does not have warns once, and the sample has no key."""
    ds_obj = _SyntheticVQADataset(base_image_dir=None, cache_dir=None,
                                  harmonized_df=synth_dataset.drop(columns=col), **{flag: True})
    with pytest.warns(UserWarning, match=f"{flag}=True, but the table has no {col} column") as rec:
        ds = ds_obj.get_datasets()
    assert col not in ds[0]
    assert sum(col in str(w.message) for w in rec) == 1
    with pytest.warns(UserWarning, match=f"no {col} column"):
        next(ds_obj.get_folds(n_splits=3))


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


def test_the_default_transform_gives_img_channels(synth_dataset, tmp_path):
    """V-D-27 (D-42): the default transform gives ``IMG_CHANNELS`` channels, here 3 for a gray file."""

    class _Color(_SyntheticVQADataset):
        IMG_CHANNELS = 3

    ds = _Color(base_image_dir=None, cache_dir=None, harmonized_df=synth_dataset).get_datasets()
    assert ds[0]["img"].shape[0] == 3
    assert _SyntheticVQADataset(base_image_dir=None, cache_dir=None,
                                harmonized_df=synth_dataset).get_datasets()[0]["img"].shape[0] == 1


def test_verify_images_needs_the_image_folder_for_relative_paths(synth_dataset, tmp_path):
    """V-H2-28 (D-04): with relative paths and no folder, verify_images() raises. It does not drop every row."""
    rel = synth_dataset.assign(image_path=synth_dataset["image_path"].map(os.path.basename))
    h = _SyntheticHarmonizer(rel)
    h.harmonize()
    with pytest.raises(ValueError, match="relative to base_image_dir"):
        h.verify_images()
    assert len(h.harmonized_df) == len(rel)
    assert h.verify_images(base_image_dir=str(tmp_path)).empty and len(h.harmonized_df) == len(rel)
    h.base_image_dir = str(tmp_path)  # a harmonizer that stores its folder needs no argument
    assert h.verify_images().empty


@pytest.mark.parametrize("split", ["get_datasets", "get_folds"])
def test_a_val_split_with_a_random_step_warns(synth_dataset, tmp_path, split):
    """D-49: as in the image datasets, a random step and no val_transform give a warning. The default does not."""
    import warnings

    import monai.transforms as mt

    from radharmony.dataset.base_vqa import _default_vqa_2d_transform

    def split_of(ds):
        return ds.get_datasets(n_splits=3) if split == "get_datasets" else next(ds.get_folds(n_splits=3))

    flip = mt.Compose([_default_vqa_2d_transform(), mt.RandFlipd(keys=["img"], prob=0.5)])
    ds = _SyntheticVQADataset(base_image_dir=None, cache_dir=None, harmonized_df=synth_dataset, transform=flip)
    with pytest.warns(UserWarning, match="val_transform"):
        split_of(ds)
    ds = _SyntheticVQADataset(base_image_dir=None, cache_dir=None, harmonized_df=synth_dataset)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        split_of(ds)
