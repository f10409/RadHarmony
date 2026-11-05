"""Dataset + loader tests for the core 2-D pipeline — no real data.

Builds a tiny synthetic harmonizer/dataset (mirroring the real subclasses) over
throwaway PNGs in ``tmp_path``, then exercises the full
harmonize → build data dicts → load image → __getitem__ path, the optional
``cls`` output, the loader length, and patient-level splitting.

Run with::

    .venv/bin/python -m pytest tests/test_dataset.py -v
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from radharmony.dataset.base import BaseRadiologicalDataset, _transform_key
from radharmony.dataset.transforms import RadiologyTransform2D
from radharmony.harmonizer.base import BaseHarmonizer
from radharmony.utils.data_utils import split_data


# --------------------------------------------------------------------------
# Minimal synthetic harmonizer + dataset (stand-ins for real subclasses)
# --------------------------------------------------------------------------

class _SynthHarmonizer(BaseHarmonizer):
    LABEL_COLS = ["tuberculosis"]

    def __init__(self, df: pd.DataFrame, base_image_dir: str = None):
        super().__init__(csv_path=None)
        self._stash = df
        self.base_dir = base_image_dir

    def harmonize(self) -> pd.DataFrame:
        self.df = self._stash.copy()
        return self._select_harmonized_columns(self.df)


class _SynthDataset(BaseRadiologicalDataset):
    LABEL_COLS = ["tuberculosis"]
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
    """4 patients × 1 image each, with a binary 'tuberculosis' label, on disk as PNGs."""
    rows = []
    for pi in range(4):
        pid = f"p{pi:03d}"
        img = tmp_path / f"{pid}.png"
        Image.fromarray((np.random.rand(40, 30) * 255).astype(np.uint8)).save(img)
        rows.append({
            "patient_id": pid,
            "study_id": f"s{pi:03d}",
            "image_path": str(img),
            "tuberculosis": pi % 2,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Harmonizer
# --------------------------------------------------------------------------

def test_harmonizer_schema(synth_df):
    out = _SynthHarmonizer(synth_df).harmonize()
    for col in ("patient_id", "study_id", "image_path", "tuberculosis"):
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
    assert len(sample["cls"]) == 1            # one label column ("tuberculosis")


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


@pytest.mark.parametrize("n_patients, n_splits, empty", [(4, 1, "train"), (3, 10, "validation")])
def test_an_empty_split_raises(n_patients, n_splits, empty):
    """Q-21 (D-10): a split with no patient raises ValueError. The message names n_splits and the patients."""
    df = pd.DataFrame({"patient_id": [f"p{p}" for p in range(n_patients)],
                       "image_path": [f"{p}.png" for p in range(n_patients)]})
    with pytest.raises(ValueError, match=f"n_splits={n_splits} .*{n_patients} patients.*{empty}"):
        split_data(df, group_column="patient_id", n_splits=n_splits)


@pytest.mark.parametrize("n_splits", [1, 4])
def test_a_fold_with_no_patient_raises(n_splits):
    """Q-21 (D-10): get_folds gives no empty fold. 3 patients cannot make 1 or 4 folds."""
    from radharmony.utils.data_utils import kfold_splits

    df = pd.DataFrame({"patient_id": ["a", "b", "c"], "image_path": ["a.png", "b.png", "c.png"]})
    with pytest.raises(ValueError, match=f"n_splits={n_splits} .*3 patients"):
        list(kfold_splits(df, "patient_id", n_splits))


# --------------------------------------------------------------------------
# visualize_samples with boxes (V-D-22)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_boxes", [1, 2])
@pytest.mark.parametrize("per_label", [False, True])
def test_visualize_samples_draws_every_box(per_label, n_boxes):
    """V-D-22: a sample with N boxes in an (N, 4) tensor gives N rectangles.

    Since D-28, also one box comes as a (1, 4) tensor. Before the fix, no box was drawn.
    """
    import matplotlib

    matplotlib.use("Agg")
    boxes = [[0.1, 0.4, 0.2, 0.5], [0.5, 0.9, 0.5, 0.9]][:n_boxes]
    sample = {"img": torch.zeros(1, 32, 32), "cls": torch.tensor([1.0]), "bbox": torch.tensor(boxes)}

    class _Samples(list):
        data = [{"cls": [1.0]}]   # visualize_samples(per_label=True) scans this

    ds = _SynthDataset.__new__(_SynthDataset)
    fig = next(ds.visualize_samples(_Samples([sample]), indices=None if per_label else [0],
                                    per_label=per_label))
    assert len(fig.axes[0].patches) == n_boxes


def test_visualize_samples_shows_only_positive_labels():
    """V-D-36 (D-05): a NaN (unknown) or -1 (uncertain) label draws a figure and is not in the title."""
    import matplotlib

    matplotlib.use("Agg")
    sample = {"img": torch.zeros(1, 32, 32), "cls": torch.tensor([float("nan"), 1.0, -1.0, 2.0])}
    ds = _SynthDataset.__new__(_SynthDataset)
    ds.LABEL_COLS = ["atelectasis", "edema", "effusion", "stenosis"]
    title = next(ds.visualize_samples([sample], indices=[0])).axes[0].get_title()
    assert title == "[0]  edema, stenosis (2)"


# --------------------------------------------------------------------------
# The cache key has the transform settings (D-15, V-D-8)
# --------------------------------------------------------------------------

def _builder(size, flip=0.0):
    b = RadiologyTransform2D(img_size=size, output_keys={"img"}, dtype=torch.float32)
    return b.with_flip(spatial_axis=1, prob=flip) if flip else b


def _split_ds(synth_df, flip):
    """The synthetic dataset, with a random flip in its transform when ``flip`` is True."""
    ds = _SynthDataset(harmonized_df=synth_df)
    ds.transform = _builder(32, flip=0.5 if flip else 0.0).get_transform()
    return ds


@pytest.mark.parametrize("split", ["get_datasets", "get_folds"])
def test_a_val_split_with_a_random_step_warns(synth_df, split):
    """Q-4 (D-49): no val_transform and a random flip in the dataset transform give a warning.

    The val split still uses the dataset transform, as reference/api-datasets.md says.
    """
    ds = _split_ds(synth_df, flip=True)
    with pytest.warns(UserWarning, match="val_transform"):
        _, val = ds.get_datasets(n_splits=2) if split == "get_datasets" else next(ds.get_folds(n_splits=2))
    assert val.transform is ds.transform


@pytest.mark.parametrize("split, case", [
    ("get_datasets", "val_transform"), ("get_datasets", "no_random_step"), ("get_datasets", "no_split"),
    ("get_folds", "val_transform"), ("get_folds", "no_random_step"),
])
def test_a_val_split_with_no_random_step_does_not_warn(synth_df, split, case):
    """Q-4 (D-49): no warning with a val_transform, with no random step, or with no split (n_splits=None)."""
    import warnings

    ds = _split_ds(synth_df, flip=case != "no_random_step")
    kw = {"val_transform": _builder(32).get_transform()} if case == "val_transform" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if case == "no_split":
            ds.get_datasets()
        elif split == "get_datasets":
            ds.get_datasets(n_splits=2, **kw)
        else:
            next(ds.get_folds(n_splits=2, **kw))


def test_a_random_step_is_found_inside_lists_and_compose():
    """Q-4 (D-49): a random step counts in a list or a nested Compose. OneOf is random, a plain Compose subclass is not."""
    import monai as mn

    from radharmony.dataset.base import _has_random_step

    class _MyCompose(mn.transforms.Compose):
        pass

    flip, same = mn.transforms.RandFlipd(keys=["img"], prob=0.5), mn.transforms.Identityd(keys=["img"])
    assert _has_random_step([mn.transforms.Compose([mn.transforms.Compose([flip])])])
    assert _has_random_step(mn.transforms.OneOf([same]))
    assert not _has_random_step(_MyCompose([same]))
    assert not _has_random_step(None)


def _cached_ds(tmp_path, transform):
    if not (tmp_path / "a.png").exists():
        Image.fromarray(np.full((60, 40), 100, np.uint8)).save(tmp_path / "a.png")
    ds = _SynthDataset.__new__(_SynthDataset)
    ds.cache_dir, ds.transform = str(tmp_path / "cache"), transform
    return ds._make_dataset([{"img": str(tmp_path / "a.png")}], cache_subdir="train")


def test_a_new_image_size_does_not_read_the_old_cache(tmp_path):
    """V-D-8: run 1 caches 64 x 64. Run 2 asks for 128 x 128 with the same cache_dir and gets it."""
    assert tuple(_cached_ds(tmp_path, _builder(64).get_transform())[0]["img"].shape) == (1, 64, 64)
    assert tuple(_cached_ds(tmp_path, _builder(128).get_transform())[0]["img"].shape) == (1, 128, 128)
    assert len(os.listdir(tmp_path / "cache" / "train")) == 2


def test_the_same_settings_reuse_the_cache_and_augmentation_is_not_in_the_key(tmp_path):
    """The key covers the steps before the first random step, as the MONAI cache does.

    A flip of 0.5 or 0.7 caches the same steps, so the cache is shared.
    """
    for t in (_builder(64, flip=0.5).get_transform(), _builder(64, flip=0.5).get_transform(),
              _builder(64, flip=0.7).get_transform()):
        _cached_ds(tmp_path, t)[0]
    assert len(os.listdir(tmp_path / "cache" / "train")) == 1
    no_pad = RadiologyTransform2D(img_size=64, pad=False, output_keys={"img"}, dtype=torch.float32)
    assert _transform_key(no_pad.get_transform()) != _transform_key(_builder(64).get_transform())


def test_the_channel_count_is_in_the_transform_key():
    """V-D-27 (D-42, D-15): 1 and 3 channels give 2 cache keys."""
    one, three = (RadiologyTransform2D(img_size=64, channels=c, output_keys={"img"}).get_transform() for c in (1, 3))
    assert _transform_key(one) != _transform_key(three)


def test_the_transform_key_is_the_same_in_a_new_process():
    """No memory address or random state goes into the key."""
    import subprocess
    import sys

    code = ("import torch; from radharmony.dataset.base import _transform_key; "
            "from radharmony.dataset.transforms import RadiologyTransform2D as T; "
            "print(_transform_key(T(img_size=64, output_keys={'img'}, dtype=torch.float32).get_transform()))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()[-1]
    assert out == _transform_key(_builder(64).get_transform())


def test_the_transform_key_has_the_code_of_a_function_step():
    """V-D-37 (D-15): 2 lambdas with one name and other code give 2 keys."""
    import monai as mn

    times2 = [mn.transforms.Lambdad(keys=["img"], func=lambda x: x * 2)]
    times3 = [mn.transforms.Lambdad(keys=["img"], func=lambda x: x * 3)]
    assert _transform_key(times2) != _transform_key(times3)


def test_the_transform_key_has_the_code_of_a_radharmony_step(monkeypatch):
    """V-D-37 (D-15): a new ``__call__`` of a RadHarmony step gives a new key."""
    from radharmony.dataset.transforms import _SetChannelsD

    steps = [_SetChannelsD(1)]
    before = _transform_key(steps)
    monkeypatch.setattr(_SetChannelsD, "__call__", lambda self, data: dict(data))
    assert _transform_key(steps) != before


class _Box:
    """Hold a value one level deeper in the key."""

    def __init__(self, inner):
        self.inner = inner


def test_the_transform_key_has_the_code_of_a_deep_step_and_not_its_docstring():
    """V-D-37 (D-15): a function 7 levels deep still adds its code. A docstring edit keeps the key.

    The reader steps of ``RadiologyEncoderTransform`` are this deep.
    """
    import monai as mn

    def one(x):
        """Old text."""
        return x * 2

    def same(x):
        """New text."""
        return x * 2

    def other(x):
        """Old text."""
        return x * 3

    same.__qualname__ = other.__qualname__ = one.__qualname__
    keys = [_transform_key([mn.transforms.Lambdad(keys=["img"], func=_Box(_Box(_Box(_Box(f)))))])
            for f in (one, same, other)]
    assert keys[0] == keys[1] != keys[2]


class _Scale:
    """A step with a bound method whose object holds the setting."""

    def __init__(self, k):
        self.k = k

    def f(self, x):
        return x * self.k


def test_the_transform_key_sees_partials_defaults_builtins_methods_and_staticmethods():
    """D-15 gap (critic of V-D-37): each pair of 2 other steps gives 2 keys."""
    import functools

    import monai as mn

    def mul(x, k):
        return x * k

    pairs = {
        "partial": (functools.partial(mul, k=2), functools.partial(mul, k=3)),
        "defaults": (lambda x, k=2: x * k, lambda x, k=3: x * k),
        "torch builtin": (torch.abs, torch.neg),
        "numpy ufunc": (np.abs, np.sqrt),
        "bound method": (_Scale(2).f, _Scale(3).f),
        "staticmethod": (staticmethod(lambda x: x * 2), staticmethod(lambda x: x * 3)),
    }
    for name, (a, b) in pairs.items():
        ka, kb = (_transform_key([mn.transforms.Lambdad(keys=["img"], func=f)]) for f in (a, b))
        assert ka != kb, name


def test_the_code_in_the_transform_key_is_the_same_for_any_hash_seed():
    """V-D-37 (D-15): a set of strings in the code of a step gives one key in every process."""
    import subprocess
    import sys

    code = ("import monai as mn; from radharmony.dataset.base import _transform_key; "
            "print(_transform_key([mn.transforms.Lambdad(keys=['img'], "
            "func=lambda x: x if x.dtype in {'a', 'b', 'c', 'd'} else x)]))")
    keys = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": seed}).stdout.split()[-1] for seed in ("1", "2", "3")}
    assert len(keys) == 1


def test_the_transform_key_has_the_reader_and_takes_a_list():
    """A new reader, or a new size in a list of steps, gives a new key. MONAI caches both."""
    import monai as mn

    def load(reader):
        return mn.transforms.Compose([mn.transforms.LoadImaged(keys="img", reader=reader)])

    def steps(size):
        return [mn.transforms.LoadImaged(keys="img"), mn.transforms.Resized(keys="img", spatial_size=(size, size))]

    assert _transform_key(load("ITKReader")) != _transform_key(load("PILReader"))
    assert _transform_key(steps(224)) != _transform_key(steps(512))


@pytest.mark.parametrize("bbox", [False, True])
def test_the_3d_cache_holds_the_preprocessed_volume(tmp_path, bbox):
    """V-D-28 (D-15): the 3-D cache file has the final shape, as the 2-D cache has, not the raw volume."""
    import glob

    sitk = pytest.importorskip("SimpleITK")
    from radharmony.dataset.transforms import RadiologyTransform3D
    sitk.WriteImage(sitk.GetImageFromArray(np.random.RandomState(0).rand(40, 30, 20).astype(np.float32)),
                    str(tmp_path / "v.nii.gz"))
    keys = {"img", "bbox"} if bbox else {"img"}
    t = RadiologyTransform3D(img_size=16, output_keys=keys, dtype=torch.float32).get_transform()
    sample = {"img": str(tmp_path / "v.nii.gz")}
    if bbox:
        sample["bbox"] = [[0.2, 0.6, 0.3, 0.7, 0.1, 0.5]]
    ds = _SynthDataset.__new__(_SynthDataset)
    ds.cache_dir, ds.transform = str(tmp_path / "cache"), t
    out = ds._make_dataset([sample], cache_subdir="train")[0]
    (pt,) = glob.glob(str(tmp_path / "cache" / "**" / "*.pt"), recursive=True)
    cached = torch.load(pt, weights_only=False)
    assert tuple(cached["img"].shape) == tuple(out["img"].shape) == (1, 16, 16, 16)
    if bbox:
        assert torch.equal(torch.as_tensor(cached["bbox"]), torch.as_tensor(out["bbox"]))


def test_the_transform_key_ignores_the_train_mode_and_takes_any_tensor():
    """A model in a cached step can switch to eval(). A bfloat16 tensor or a tensor with requires_grad gets a key."""
    import monai as mn

    def make(model, mean):
        return mn.transforms.Compose([mn.transforms.Lambdad(keys="img", func=lambda x: model(x) - mean)])

    model = torch.nn.Linear(2, 2)
    t = make(model, torch.ones(2))
    before = _transform_key(t)
    model.eval()
    assert _transform_key(t) == before
    for mean in (torch.ones(2, dtype=torch.bfloat16), torch.ones(2, requires_grad=True)):
        assert len(_transform_key(make(model, mean))) == 12


# --------------------------------------------------------------------------
# DICOM pixel values in the dataset load steps (D-13)
# --------------------------------------------------------------------------

def _dataset_load(path):
    """Run the dataset load steps of ``_base_load_2d`` up to the window, as min-max [-1, 1]."""
    import monai as mn

    from radharmony.dataset.transforms import _base_load_2d

    steps = [s for s in _base_load_2d(["img"]) if not isinstance(s, mn.transforms.ScaleIntensityRangePercentilesD)]
    img = mn.transforms.Compose(steps)({"img": str(path)})["img"].numpy()[0].astype(np.float64)
    return 2 * (img - img.min()) / (img.max() - img.min()) - 1


def test_dataset_load_of_signed_monochrome1_dicom_with_window(tmp_path):
    """D-13: a signed MONOCHROME1 DICOM looks as in a viewer. GDCM inverts it as -1 - raw."""
    from synthetic_images import dicom_display_values, to_unit_range, write_dicom

    stored = (np.add.outer(np.arange(30) * 20, np.arange(40) * 40) - 1500).astype(np.int16)
    path = write_dicom(tmp_path / "signed.dcm", stored, photometric="MONOCHROME1", window=(0, 1000), signed=True)
    np.testing.assert_allclose(_dataset_load(path), to_unit_range(dicom_display_values(path)), atol=1e-4)


@pytest.mark.parametrize("name", ["a.dcm", "a.dicom", "1.2.840.113619.2.55.3.604688119"])
@pytest.mark.parametrize("photometric", ["MONOCHROME1", "MONOCHROME2"])
def test_dataset_load_uses_the_window_for_any_dicom_file_name(tmp_path, name, photometric):
    """D-13: the window and the HighBit step run for a DICOM file with any name, also with no ending."""
    from synthetic_images import dicom_display_values, to_unit_range, write_dicom

    stored = np.tile(np.arange(0, 3000, 300, dtype=np.uint16), (4, 1))
    path = write_dicom(tmp_path / name, stored, photometric=photometric, window=(1500, 1000))
    np.testing.assert_allclose(_dataset_load(path), to_unit_range(dicom_display_values(path)), atol=1e-4)


def test_dataset_load_does_not_read_an_mha_copy_with_pydicom(tmp_path, monkeypatch):
    """D-13: an .mha copy of a DICOM file keeps the tag 0028|0004, but it is not a DICOM file."""
    import SimpleITK as sitk
    from synthetic_images import write_dicom

    import radharmony.dataset.base as base

    stored = np.tile(np.arange(0, 3000, 300, dtype=np.uint16), (4, 1))
    dcm = write_dicom(tmp_path / "a.dcm", stored)
    mha = str(tmp_path / "a.mha")
    sitk.WriteImage(sitk.ReadImage(str(dcm)), mha)
    expected = _dataset_load(dcm)

    def no_read(*args, **kwargs):
        raise AssertionError("pydicom read an .mha file")

    monkeypatch.setattr(base.pydicom, "dcmread", no_read)
    np.testing.assert_allclose(_dataset_load(mha), expected)


def test_highbit_step_skips_a_color_dicom(tmp_path):
    """D-13: an RGB DICOM with no HighBit tag keeps the pixels that it loaded. Before, the step crashed."""
    import monai.transforms as mt
    import pydicom
    from synthetic_images import write_dicom

    from radharmony.dataset.base import _fix_missing_highbit

    rgb = np.random.default_rng(0).integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
    path = write_dicom(tmp_path / "rgb.dcm", rgb)
    ds = pydicom.dcmread(path)
    del ds.HighBit
    ds.save_as(path)
    x = mt.LoadImage(image_only=True, ensure_channel_first=True)(path)
    before = x.clone()
    np.testing.assert_array_equal(_fix_missing_highbit(x).numpy(), before.numpy())
