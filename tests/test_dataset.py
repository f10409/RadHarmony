"""Dataset + loader tests for the core 2-D pipeline — no real data.

Builds a tiny synthetic harmonizer/dataset (mirroring the real subclasses) over
throwaway PNGs in ``tmp_path``, then exercises the full
harmonize → build data dicts → load image → __getitem__ path, the optional
``cls`` output, the loader length, and patient-level splitting.

Run with::

    .venv/bin/python -m pytest tests/test_dataset.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from radharmony.dataset.base import BaseRadiologicalDataset
from radharmony.dataset.transforms import RadiologyTransform2D
from radharmony.harmonizer.base import BaseHarmonizer
from radharmony.utils.data_utils import split_data


# --------------------------------------------------------------------------
# Minimal synthetic harmonizer + dataset (stand-ins for real subclasses)
# --------------------------------------------------------------------------

class _SynthHarmonizer(BaseHarmonizer):
    LABEL_COLS = ["tb"]

    def __init__(self, df: pd.DataFrame, base_image_dir: str = None):
        super().__init__(csv_path=None)
        self._stash = df
        self.base_dir = base_image_dir

    def harmonize(self) -> pd.DataFrame:
        self.df = self._stash.copy()
        return self._select_harmonized_columns(self.df)


class _SynthDataset(BaseRadiologicalDataset):
    LABEL_COLS = ["tb"]
    _HARMONIZER_CLS = _SynthHarmonizer

    def __init__(self, harmonized_df, cache_dir=None, output_cls=False,
                 dtype=torch.float32):
        keys = {"img"} | ({"cls"} if output_cls else set())
        transform = RadiologyTransform2D(
            img_size=32, output_keys=keys, dtype=dtype
        ).get_transform()
        super().__init__(
            base_image_dir=None, transform=transform, cache_dir=cache_dir,
            output_cls=output_cls, harmonized_df=harmonized_df,
        )

    def _get_harmonized_df(self):
        return self._try_resolve_preset_harmonized()


@pytest.fixture
def synth_df(tmp_path):
    """4 patients × 1 image each, with a binary 'tb' label, on disk as PNGs."""
    rows = []
    for pi in range(4):
        pid = f"p{pi:03d}"
        img = tmp_path / f"{pid}.png"
        Image.fromarray((np.random.rand(40, 30) * 255).astype(np.uint8)).save(img)
        rows.append({
            "patient_id": pid,
            "study_id": f"s{pi:03d}",
            "image_path": str(img),
            "tb": pi % 2,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Harmonizer
# --------------------------------------------------------------------------

def test_harmonizer_schema(synth_df):
    out = _SynthHarmonizer(synth_df).harmonize()
    for col in ("patient_id", "study_id", "image_path", "tb"):
        assert col in out.columns
    assert len(out) == 4


def test_harmonizer_missing_required_column_raises(synth_df):
    bad = synth_df.drop(columns=["image_path"])
    with pytest.raises(KeyError):
        _SynthHarmonizer(bad).harmonize()


# --------------------------------------------------------------------------
# Dataset + loader
# --------------------------------------------------------------------------

def test_sample_has_image_tensor(synth_df):
    ds = _SynthDataset(harmonized_df=synth_df).get_datasets()
    sample = ds[0]
    assert "img" in sample
    assert tuple(sample["img"].shape) == (1, 32, 32)
    assert sample["img"].dtype == torch.float32


def test_output_cls_adds_label_vector(synth_df):
    ds = _SynthDataset(harmonized_df=synth_df, output_cls=True).get_datasets()
    sample = ds[0]
    assert "cls" in sample
    assert len(sample["cls"]) == 1            # one label column ("tb")


def test_cls_absent_by_default(synth_df):
    ds = _SynthDataset(harmonized_df=synth_df).get_datasets()
    assert "cls" not in ds[0]


def test_loader_length_matches_rows(synth_df):
    ds = _SynthDataset(harmonized_df=synth_df).get_datasets()
    assert len(ds) == len(synth_df) == 4


def test_split_partitions_all_samples(synth_df):
    train_ds, val_ds = _SynthDataset(harmonized_df=synth_df).get_datasets(n_splits=2)
    assert len(train_ds) > 0 and len(val_ds) > 0
    assert len(train_ds) + len(val_ds) == 4


# --------------------------------------------------------------------------
# Patient-level split guarantee (no leakage across train/val)
# --------------------------------------------------------------------------

def test_patient_level_split_no_leakage():
    # 6 patients, 3 images each — split must keep a patient wholly on one side.
    df = pd.DataFrame([
        {"patient_id": f"p{p}", "image_path": f"{p}_{i}.png"}
        for p in range(6) for i in range(3)
    ])
    train, val = split_data(df, group_column="patient_id", n_splits=2)
    train_p, val_p = set(train["patient_id"]), set(val["patient_id"])
    assert train_p.isdisjoint(val_p), "a patient appeared in both train and val"
    assert train_p | val_p == set(df["patient_id"])
    assert len(train) + len(val) == len(df)
