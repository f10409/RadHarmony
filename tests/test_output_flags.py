"""D-11 output flags (L-010). No data needed.

Every image dataset class takes the 5 output flags. A flag that the class does
not support (not in ``SUPPORTED_OUTPUTS``) gives a warning, not ``TypeError``.
The full run checks 1 real sample for each flag in ``tests/test_dataset_stats.py``.

Run with::

    .venv/bin/python -m pytest tests/test_output_flags.py -v
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

import radharmony.dataset  # noqa: F401  (fills the registry)
from radharmony.dataset.base import BaseRadiologicalDataset
from radharmony.registry import list_datasets, resolve_dataset

FLAGS = {"output_cls": "cls", "output_mask": "mask", "output_bbox": "bbox", "output_report": "report",
         "output_reg": "reg"}
IMAGE_KEYS = [k for k in sorted(list_datasets()) if issubclass(resolve_dataset(k), BaseRadiologicalDataset)]


def _init_params(cls) -> set[str]:
    """The constructor arguments, with the ones that ``**kwargs`` passes to a parent."""
    names = set()
    for kls in cls.__mro__:
        if "__init__" not in vars(kls):
            continue
        params = inspect.signature(kls.__init__).parameters
        names |= set(params)
        if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
            break
    return names


@pytest.mark.parametrize("key", IMAGE_KEYS)
def test_every_output_flag_is_an_argument(key):
    """A-1 (L-010): the same code can ask every dataset for the same outputs."""
    missing = sorted(set(FLAGS) - _init_params(resolve_dataset(key)))
    assert not missing, f"{key} raises TypeError for {missing}"


TABLE = pd.DataFrame({"patient_id": ["p1"], "study_id": ["s1"], "image_path": ["a.png"]})


def _required_keys(transform) -> set[str]:
    """The keys that a step of the transform needs (no ``allow_missing_keys``)."""
    keys = set()
    for step in getattr(transform, "transforms", [transform]):
        if hasattr(step, "transforms"):
            keys |= _required_keys(step)
        elif getattr(step, "keys", None) and not getattr(step, "allow_missing_keys", False):
            keys |= set(step.keys)
    return keys


@pytest.mark.parametrize("key,flag", [(k, f) for k in IMAGE_KEYS for f, out in FLAGS.items()
                                      if out not in resolve_dataset(k).SUPPORTED_OUTPUTS])
def test_an_unsupported_flag_warns(key, flag, tmp_path):
    """D-11, V-D-25: a flag that the class does not support gives a warning.

    The default transform must not need its key. No sample has the key, so the
    transform would raise ``KeyError`` (CT-RATE and SIIM-ACR did, no ID).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = resolve_dataset(key)(base_image_dir=str(tmp_path), harmonized_df=TABLE, cache_dir=None,
                                  **{flag: True})
    assert [w for w in caught if issubclass(w.category, UserWarning)], f"{key}: {flag}=True gives no warning"
    assert FLAGS[flag] not in _required_keys(ds.transform)


@pytest.mark.parametrize("with_labels", [False, True])
def test_rexgradient_preset_table_decides_cls(with_labels, tmp_path):
    """V-D-39 (D-11): with a preset table, ReXGradient gives cls only when the table has its labels.

    A table that the class made with no labels_csv_path has no label columns. Then output_cls=True
    warns, and the transform does not need 'cls' (before, every sample raised KeyError).
    """
    from radharmony.dataset import ReXGradientTrainDataset

    labels = ReXGradientTrainDataset._CHEXPERT14
    df = TABLE.assign(**({c: [0.0] for c in labels} if with_labels else {}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = ReXGradientTrainDataset(base_image_dir=str(tmp_path), harmonized_df=df, cache_dir=None,
                                     output_cls=True)
    warned = any("output_cls" in str(w.message) for w in caught)
    assert warned is not with_labels
    assert ("cls" in _required_keys(ds.transform)) is with_labels


@pytest.mark.parametrize("key", ["montgomery_cxr", "shenzhen_cxr"])
def test_report_reaches_the_sample(key, tmp_path):
    """V-A-6, V-D-6: the class supports ``report``, so the default transform keeps it."""
    Image.fromarray(np.zeros((32, 32), np.uint8)).save(tmp_path / "a.png")
    df = TABLE.assign(report=["normal chest"])
    ds = resolve_dataset(key)(base_image_dir=str(tmp_path), harmonized_df=df, cache_dir=None,
                              output_report=True).get_datasets()
    assert ds[0]["report"] == "normal chest"


@pytest.mark.parametrize("cache", [False, True])
def test_ms_cxr_t_previous_image_reaches_the_sample(cache, tmp_path):
    """V-D-31 (D-11): output_previous=True gives the prior-visit image under ``previous_img``.

    It has the steps of ``img``, so it equals the ``img`` of a row whose image is the prior file.
    """
    from radharmony.dataset import MSCXRTDataset

    rng = np.random.RandomState(0)
    for name in ("cur.png", "prev.png"):
        Image.fromarray(rng.randint(0, 255, (64, 48)).astype(np.uint8)).save(tmp_path / name)
    df = pd.DataFrame({"patient_id": ["p"], "study_id": ["s2"], "image_path": ["cur.png"],
                       "previous_image_path": ["prev.png"], "previous_study_id": ["s1"]})
    kw = dict(base_image_dir=str(tmp_path), cache_dir=str(tmp_path / "cache") if cache else None)

    pair = MSCXRTDataset(harmonized_df=df, output_previous=True, **kw).get_datasets()
    # With no cache, so that the expected image does not come from the cache file of the pair.
    no_cache = dict(kw, cache_dir=None)
    alone = MSCXRTDataset(harmonized_df=df.assign(image_path=["prev.png"]), **no_cache).get_datasets()
    cur = MSCXRTDataset(harmonized_df=df, **no_cache).get_datasets()
    assert len(pair) == 1 and pair.data[0]["img"].endswith("cur.png")
    assert torch.equal(pair[0]["previous_img"], alone[0]["img"])
    assert torch.equal(pair[0]["img"], cur[0]["img"])
    assert "previous_img" not in cur[0]
    assert len(pair[0:1]) == 1 and "previous_img" in pair[[0]][0]  # slice and list access, as a MONAI Dataset


def _ms_cxr_t_table(n):
    return pd.DataFrame({"patient_id": [f"p{i}" for i in range(n)], "study_id": [f"s{i}" for i in range(n)],
                         "image_path": ["a.png"] * n, "previous_image_path": ["a.png"] * n,
                         "previous_study_id": [f"q{i}" for i in range(n)]})


def test_ms_cxr_t_previous_image_gets_its_own_random_draw_in_a_worker(tmp_path):
    """V-D-31 (D-11): a random step draws again for ``previous_img``, also for the first sample of a worker.

    Both images are the same file, so a shared draw gives the same tensor.
    """
    import monai.transforms as mt

    from radharmony.dataset import MSCXRTDataset

    Image.fromarray(np.full((16, 16), 100, np.uint8)).save(tmp_path / "a.png")
    t = mt.Compose([mt.LoadImaged(keys=["img"], image_only=True, ensure_channel_first=True),
                    mt.RandGaussianNoised(keys=["img"], prob=1.0, std=10.0)])
    ds = MSCXRTDataset(base_image_dir=str(tmp_path), harmonized_df=_ms_cxr_t_table(4), output_previous=True,
                       cache_dir=None, transform=t).get_datasets()
    batches = list(torch.utils.data.DataLoader(ds, batch_size=1, num_workers=2))
    assert not any(torch.equal(b["img"], b["previous_img"]) for b in batches)


def test_ms_cxr_t_previous_image_needs_its_column(tmp_path):
    """V-D-31: a table with no ``previous_image_path`` column names the column."""
    from radharmony.dataset import MSCXRTDataset

    df = _ms_cxr_t_table(1).drop(columns=["previous_image_path"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = MSCXRTDataset(base_image_dir=str(tmp_path), harmonized_df=df, output_previous=True, cache_dir=None)
        with pytest.raises(ValueError, match="previous_image_path"):
            ds.get_datasets()


def _selected_keys(transform) -> set[str]:
    """The keys that the ``SelectItemsD`` step of the transform keeps."""
    import monai.transforms as mt

    keys = set()
    for step in getattr(transform, "transforms", [transform]):
        if hasattr(step, "transforms"):
            keys |= _selected_keys(step)
        elif isinstance(step, mt.SelectItemsd):
            keys |= set(step.keys)
    return keys


@pytest.mark.parametrize("key", [k for k in IMAGE_KEYS if "bbox" in resolve_dataset(k).SUPPORTED_OUTPUTS])
def test_output_bbox_keeps_every_key_of_the_flag(key, tmp_path):
    """L-013 (D-11): output_bbox=True gives 'bbox' and 'bbox_labels', and 'bbox_findings' where supported.

    The default transform keeps these keys. A table with these columns then gives them in each
    sample. RSNA Pneumonia, SIIM-COVID19, RSNA 2024 lumbar and VinDr-PCXR dropped 'bbox_labels' (V-D-29).
    """
    cls = resolve_dataset(key)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = cls(base_image_dir=str(tmp_path), harmonized_df=TABLE, cache_dir=None, output_bbox=True)
    expected = {"bbox", "bbox_labels"} | ({"bbox_findings"} if cls.SUPPORTS_BBOX_FINDINGS else set())
    missing = expected - _selected_keys(ds.transform)
    assert not missing, f"{key}: the default transform drops {sorted(missing)}"


@pytest.mark.parametrize("key", IMAGE_KEYS)
def test_a_class_that_supports_reg_names_its_reg_columns(key):
    """D-11: a class with "reg" in SUPPORTED_OUTPUTS has REG_COLS. Else output_reg=True gives an empty reg."""
    cls = resolve_dataset(key)
    if "reg" in cls.SUPPORTED_OUTPUTS:
        assert cls.REG_COLS, f"{key}: 'reg' is in SUPPORTED_OUTPUTS, but REG_COLS is empty"


def _inventory_rows():
    """Yield (row name, outputs text, registry keys) for each row of the docs/wiki/reference/datasets/index.md inventory."""
    import pathlib
    import re

    text = (pathlib.Path(__file__).resolve().parents[1] / "docs/wiki/reference/datasets/index.md").read_text()
    table = text.split("## Inventory")[1].split("\n## ")[0]
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6 or cells[0] in ("Dataset", "") or set(cells[0]) <= {"-"}:
            continue
        yield cells[0], cells[4], re.findall(r"`([a-z0-9_]+)`", cells[5])


@pytest.mark.parametrize("name, outputs, keys", list(_inventory_rows()), ids=lambda v: v if isinstance(v, str) else "")
def test_inventory_outputs_match_the_classes(name, outputs, keys):
    """A-13 (D-11): the "Outputs available" column lists the outputs that the class of each key supports.

    A row with a test key can say "(test key: none)", because a test split has no labels (D-25).
    """
    import re

    main, _, test_part = outputs.partition("(test key:")
    for key in [k for k in keys if k in IMAGE_KEYS]:  # a VQA class has no output flags (D-11)
        part = test_part if key.endswith("_test") and test_part else main
        listed = set(re.findall(r"\b(cls|mask|bbox|report|reg)\b", part))
        supported = set(resolve_dataset(key).SUPPORTED_OUTPUTS) & set(FLAGS.values())
        assert listed == supported, f"{name} / {key}: listed {sorted(listed)}, class {sorted(supported)}"


def test_a_transform_that_drops_a_requested_key_warns_at_once(tmp_path):
    """Decided by Frank Li (2026-11-04), D-11: a given transform whose last step drops a flag's key warns."""
    from radharmony.dataset import MontgomeryCXRDataset
    from radharmony.dataset.transforms import RadiologyTransform2D

    df = TABLE.assign(report=["normal chest"])
    with pytest.warns(UserWarning, match=r"output_report=True, but the transform keeps only \['cls', 'img'\]"):
        MontgomeryCXRDataset(base_image_dir=str(tmp_path), harmonized_df=df, cache_dir=None, output_cls=True,
                             output_report=True, transform=RadiologyTransform2D().get_transform())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        keep = RadiologyTransform2D(output_keys={"img", "cls", "report"}).get_transform()
        MontgomeryCXRDataset(base_image_dir=str(tmp_path), harmonized_df=df, cache_dir=None, output_cls=True,
                             output_report=True, transform=keep)
        MontgomeryCXRDataset(base_image_dir=str(tmp_path), harmonized_df=df, cache_dir=None, output_cls=True,
                             output_report=True)
