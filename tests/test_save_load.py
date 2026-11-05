"""D-08: save() then load_from_saved() rebuilds the harmonizer. No data needed.

Each test builds a harmonizer with fake paths, gives it a small table, saves
it and loads it again. The loaded harmonizer must have the same class, the same
constructor arguments and the same table (D-08).

Run with::

    .venv/bin/python -m pytest tests/test_save_load.py -v
"""
from __future__ import annotations

import inspect
import pickle

import pandas as pd
import pytest

import radharmony.dataset  # noqa: F401  (fills the registry)
import radharmony.harmonizer as H
import radharmony.harmonizer.mrrate as M
from radharmony.dataset.base import BaseRadiologicalDataset
from radharmony.harmonizer.base import BaseHarmonizer
from radharmony.harmonizer.base_vqa import BaseVQAHarmonizer
from radharmony.registry import list_datasets, resolve_dataset

HARMONIZERS = sorted(
    {c for c in [*vars(H).values(), *vars(M).values()]
     if isinstance(c, type) and issubclass(c, (BaseHarmonizer, BaseVQAHarmonizer))
     and c not in (BaseHarmonizer, BaseVQAHarmonizer)},
    key=lambda c: c.__name__)

TABLE = pd.DataFrame({"patient_id": ["p1"], "study_id": ["s1"], "image_path": ["a.png"]})


def _kwargs(cls, d) -> dict:
    """Constructor arguments with fake paths under ``d``. Some are not the default, to see if load keeps them."""
    special = {
        "Node21Harmonizer": dict(base_image_dir=d, variant="original", keep_negatives=False),
        "ChestXDet10Harmonizer": dict(csv_path=f"{d}/test.json", base_image_dir=d, split="test",
                                      keep_negatives=False),
        "ChestXDet10TrainHarmonizer": dict(csv_path=f"{d}/train.json", keep_negatives=False),
        "ChestXDet10TestHarmonizer": dict(csv_path=f"{d}/test.json", keep_negatives=False),
        "ROCOHarmonizer": dict(base_dir=d, radiology_only=False),
        "ROCOv2Harmonizer": dict(base_dir=d, base_image_dir=f"{d}/images", require_modality=True),
        "MRRATEHarmonizer": dict(base_dir=d, modalities=["T1w"], center_only=True, with_masks=False,
                                 report_section="impression"),
        "QaTaCOV19TrainHarmonizer": dict(base_dir=f"{d}/Train Set/Images"),
        "QaTaCOV19TestHarmonizer": dict(base_dir=f"{d}/Test Set/Images"),
        "EmoryCXRDicomHarmonizer": dict(csv_path=f"{d}/m.csv", mapping_csv_path=f"{d}/map.csv"),
        "MedSegCOVIDCTHarmonizer": dict(base_dir=d, subset="dataset2"),
        "PadChestGRHarmonizer": dict(base_image_dir=d, split="test", report_language="es"),
        "PadChestGRTrainHarmonizer": dict(base_image_dir=d),
        "PadChestGRValidationHarmonizer": dict(base_image_dir=d),
        "PadChestGRTestHarmonizer": dict(base_image_dir=d),
        "RSNABoneAgeHarmonizer": dict(base_image_dir=d, csv_path=f"{d}/train.csv"),
        "RSNABoneAgeTrainHarmonizer": dict(base_image_dir=d, csv_path=f"{d}/train.csv"),
        "RSNABoneAgeValHarmonizer": dict(base_image_dir=d, csv_path=f"{d}/val.csv"),
    }
    if cls.__name__ in special:
        return special[cls.__name__]
    for kls in cls.__mro__:  # a split class can take only (*args, **kwargs): then use its parent
        params = list(inspect.signature(kls.__init__).parameters.values())[1:]
        required = [p.name for p in params if p.default is p.empty and p.kind is p.POSITIONAL_OR_KEYWORD]
        if required or not any(p.kind is p.VAR_KEYWORD for p in params):
            return {name: d for name in required}
    return {}


def _init_attrs(h) -> dict:
    """The public plain attributes that the constructor sets."""
    plain = (str, int, float, bool, tuple, list, type(None))
    return {k: v for k, v in vars(h).items()
            if not k.startswith("_") and k not in ("df", "LABEL_COLS") and isinstance(v, plain)}


@pytest.mark.parametrize("cls", HARMONIZERS, ids=lambda c: c.__name__)
def test_load_rebuilds_the_harmonizer(cls, tmp_path):
    """V-H2-1, V-H1-5, V-H2-22, V-H2-10, V-H2-16: load keeps the class, the arguments and the table."""
    h = cls(**_kwargs(cls, str(tmp_path)))
    h.set_harmonized_df(TABLE)
    h.save(str(tmp_path / "h.pkl"))
    loaded = cls.load_from_saved(str(tmp_path / "h.pkl"))
    assert type(loaded) is cls
    assert _init_attrs(loaded) == _init_attrs(h)
    pd.testing.assert_frame_equal(loaded.harmonized_df, TABLE)


def _optional_args(cls):
    """(name, a value that is not the default) for each optional path, folder or bool argument."""
    for kls in cls.__mro__:
        if "__init__" not in vars(kls):
            continue
        params = list(inspect.signature(kls.__init__).parameters.values())[1:]
        named = [p for p in params if p.kind is p.POSITIONAL_OR_KEYWORD]
        if named or not any(p.kind is p.VAR_KEYWORD for p in params):
            break
    out = []
    for p in named:
        if isinstance(p.default, bool):
            out.append((p.name, not p.default))
        elif p.default is None and ("path" in p.name or "dir" in p.name):
            out.append((p.name, f"/x/{p.name}"))
    return out


@pytest.mark.parametrize("cls,name,value", [(c, n, v) for c in HARMONIZERS for n, v in _optional_args(c)],
                         ids=lambda v: v.__name__ if isinstance(v, type) else str(v))
def test_load_keeps_each_optional_argument(cls, name, value, tmp_path):
    """D-08: an argument that is not the default is the same after load (for example CheXpert demographics_path)."""
    try:
        h = cls(**{**_kwargs(cls, str(tmp_path)), name: value})
    except (ValueError, FileNotFoundError):
        pytest.skip(f"{name}={value!r} is not valid alone for {cls.__name__}")
    h.set_harmonized_df(TABLE)
    h.save(str(tmp_path / "h.pkl"))
    assert _init_attrs(cls.load_from_saved(str(tmp_path / "h.pkl"))) == _init_attrs(h)


def test_load_keeps_the_chexphoto_subset_and_the_medpmc_modality_source(tmp_path):
    """D-08: 2 string arguments with a fixed list of values."""
    for cls, kwargs in ((H.CheXphotoHarmonizer, dict(csv_path="/x/train.csv", subset="natural")),
                        (H.MedPMCHarmonizer, dict(base_dir="/x", modality_source="caption"))):
        h = cls(**kwargs)
        h.set_harmonized_df(TABLE)
        h.save(str(tmp_path / "h.pkl"))
        assert _init_attrs(cls.load_from_saved(str(tmp_path / "h.pkl"))) == _init_attrs(h)


# --- Dataset(harmonizer_path=...) ------------------------------------------------

def _save(hcls, d, table=TABLE, **kwargs) -> str:
    h = hcls(**(kwargs or _kwargs(hcls, d)))
    h.set_harmonized_df(table)
    path = f"{d}/{hcls.__name__}.pkl"
    h.save(path)
    return path


def _keys_with_harmonizer():
    return [k for k in sorted(list_datasets()) if getattr(resolve_dataset(k), "_HARMONIZER_CLS", None)]


@pytest.mark.parametrize("key", _keys_with_harmonizer())
def test_dataset_loads_a_file_of_its_harmonizer(key, tmp_path):
    """A dataset with no ``base_image_dir`` loads a file of its ``_HARMONIZER_CLS`` (D-08)."""
    cls = resolve_dataset(key)
    ds = cls(harmonizer_path=_save(cls._HARMONIZER_CLS, str(tmp_path)), cache_dir=None)
    pd.testing.assert_frame_equal(ds.get_harmonized_df()[list(TABLE.columns)], TABLE)


@pytest.mark.parametrize("key", [k for k in _keys_with_harmonizer()
                                 if issubclass(resolve_dataset(k), BaseRadiologicalDataset)])
def test_dataset_takes_only_the_declared_labels(key, tmp_path):
    """L-014 (D-08): after ``harmonizer_path=`` and ``set_harmonized_df()``, the labels are the declared ones."""
    cls = resolve_dataset(key)
    hcls = cls._HARMONIZER_CLS
    h = hcls(**_kwargs(hcls, str(tmp_path)))
    labels = sorted(c.lower().replace(" ", "_") for c in h.get_label_cols())[:2]
    # 2 extra output columns of real tables: CheXpert Plus race, TAIX-Ray physician_id (text).
    table = TABLE.assign(**{c: [1.0] for c in labels}, race=["white"], physician_id=["dr a"])
    h.set_harmonized_df(table)
    h.save(str(tmp_path / "h.pkl"))
    ds = cls(harmonizer_path=str(tmp_path / "h.pkl"), cache_dir=None)
    ds.get_harmonized_df()  # this load sets LABEL_COLS
    assert {"race", "physician_id"}.isdisjoint(ds.LABEL_COLS)
    assert set(labels) <= set(ds.LABEL_COLS)
    # As verify_images(drop_missing=True) does. It sets labels only for a class with empty LABEL_COLS.
    ds.set_harmonized_df(table)
    assert {"race", "physician_id"}.isdisjoint(ds.LABEL_COLS)


# V-H3-17: each split dataset loads a file of its own split harmonizer.
SPLIT_HARMONIZERS = [
    ("siim_acr_ptx_train", "SIIMACRPTXTrainHarmonizer"),
    ("siim_acr_ptx_test", "SIIMACRPTXTestHarmonizer"),
    ("rsna_pneumonia_kaggle_train", "RSNAPneumoniaKaggleTrainHarmonizer"),
    ("rsna_pneumonia_kaggle_test", "RSNAPneumoniaKaggleTestHarmonizer"),
    ("siim_covid19_train", "SIIMCOVID19TrainHarmonizer"),
    ("siim_covid19_test", "SIIMCOVID19TestHarmonizer"),
    ("chestxray14_train", "ChestXray14TrainHarmonizer"),
    ("chestxray14_test", "ChestXray14TestHarmonizer"),
    # V-H3-21: the 3-D test classes.
    ("rsna_2022_cervical_spine_test", "RSNA2022CervicalSpineTestHarmonizer"),
    ("rsna_2024_lumbar_spine_test", "RSNA2024LumbarSpineTestHarmonizer"),
    ("rsna_abdominal_trauma_2023_test", "RSNAAbdominalTrauma2023TestHarmonizer"),
    ("rsna_pe_detection_test", "RSNAPEDetectionTestHarmonizer"),
]


@pytest.mark.parametrize("key,name", SPLIT_HARMONIZERS)
def test_split_dataset_loads_its_split_harmonizer(key, name, tmp_path):
    """V-H3-17: the split dataset loads the file of its split harmonizer, with no ``base_image_dir``."""
    ds = resolve_dataset(key)(harmonizer_path=_save(getattr(H, name), str(tmp_path)), cache_dir=None)
    pd.testing.assert_frame_equal(ds.get_harmonized_df()[list(TABLE.columns)], TABLE)


# D-09: a file that the old class saved for this split still loads.
OLD_FILES = [
    ("chestxdet10_train", "ChestXDet10Harmonizer", dict(split="train")),
    ("chestxdet10_test", "ChestXDet10Harmonizer", dict(split="test")),
    ("siim_acr_ptx_train", "SIIMACRPTXHarmonizer", {}),
    ("rsna_pneumonia_kaggle_train", "RSNAPneumoniaKaggleHarmonizer", {}),
    ("siim_covid19_train", "SIIMCOVID19Harmonizer", {}),
]


@pytest.mark.parametrize("key,name,extra", OLD_FILES, ids=lambda v: v if isinstance(v, str) else "")
def test_split_dataset_loads_a_file_of_the_old_class(key, name, extra, tmp_path):
    hcls = getattr(H, name)
    table = TABLE.assign(split=extra["split"]) if extra else TABLE
    path = _save(hcls, str(tmp_path), table, **{**_kwargs(hcls, str(tmp_path)), **extra})
    ds = resolve_dataset(key)(harmonizer_path=path, cache_dir=None)
    pd.testing.assert_frame_equal(ds.get_harmonized_df()[list(table.columns)], table)


def _old_chestxdet10_file(tmp_path, table) -> str:
    """A file that ChestXDet10Harmonizer saved before 2026-10-28: its saved arguments have no ``split``."""
    path = _save(H.ChestXDet10Harmonizer, str(tmp_path), table)
    with open(path, "rb") as f:
        payload = pickle.load(f)
    payload["init"] = {k: v for k, v in payload["init"].items() if k not in ("split", "keep_negatives")}
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def test_split_dataset_rejects_a_file_of_another_split(tmp_path):
    """D-09: a train class does not give the test rows of an old file."""
    path = _old_chestxdet10_file(tmp_path, TABLE.assign(split="test"))
    with pytest.raises(ValueError, match="has rows of the split"):
        resolve_dataset("chestxdet10_train")(harmonizer_path=path, cache_dir=None).get_harmonized_df()
    ds = resolve_dataset("chestxdet10_test")(harmonizer_path=path, cache_dir=None)
    assert ds.get_harmonized_df()["split"].tolist() == ["test"]


def test_split_dataset_rejects_an_old_file_with_no_split(tmp_path):
    """D-09: with no split in the table or in the saved arguments, the split class cannot check the rows."""
    with pytest.raises(ValueError, match="no split"):
        H.ChestXDet10TrainHarmonizer.load_from_saved(_old_chestxdet10_file(tmp_path, TABLE))


# --- LABEL_COLS after load (base code) ---------------------------------------------

@pytest.mark.parametrize("key", ["chest_imagenome_gold", "rsna_bone_age_train"])
def test_harmonized_df_takes_no_extra_columns_as_labels(key, tmp_path):
    """D-08: with ``harmonized_df=``, a class with no labels takes none. Bone age is a target, not a label."""
    kwargs = {"annotation_dir": str(tmp_path)} if key == "chest_imagenome_gold" else {}
    table = TABLE.assign(bbox_labels=[["left lung"]], image_width=[100], age_months=[120.0])
    ds = resolve_dataset(key)(harmonized_df=table, cache_dir=None, base_image_dir=str(tmp_path), **kwargs)
    ds.get_harmonized_df()  # this load sets LABEL_COLS
    assert ds.LABEL_COLS == []


class _TwoLabels(BaseHarmonizer):
    """The VinDr-CXR label names ``Calcification`` and ``COPD`` in one CSV."""

    LABEL_COLS = ["COPD", "Calcification"]

    def _build_patient_id(self):
        self.df["patient_id"] = self.df["image_id"]

    def _build_study_id(self):
        self.df["study_id"] = self.df["image_id"]

    def _build_image_path(self):
        self.df["image_path"] = self.df["image_id"] + ".png"


def test_label_order_is_the_same_after_load(tmp_path):
    """D-05 and D-08: snake_case order is calcification, copd. Load must keep it."""
    pd.DataFrame({"image_id": ["a"], "COPD": [1], "Calcification": [0]}).to_csv(tmp_path / "x.csv", index=False)
    h = _TwoLabels(csv_path=str(tmp_path / "x.csv"))
    h.harmonize()
    assert h.LABEL_COLS == ["Calcification", "COPD"]
    h.save(str(tmp_path / "h.pkl"))
    assert _TwoLabels.load_from_saved(str(tmp_path / "h.pkl")).LABEL_COLS == ["Calcification", "COPD"]


def test_no_labels_stay_no_labels_after_load(tmp_path):
    """V-H1-15, V-H2-3: a harmonizer with no labels (Chest ImaGenome gold) has none after load."""
    h = H.ChestImaGenomeGoldHarmonizer(annotation_dir=str(tmp_path))
    h.set_harmonized_df(TABLE.assign(bbox_labels=[["lung"]], image_width=[100]))
    h.save(str(tmp_path / "h.pkl"))
    assert H.ChestImaGenomeGoldHarmonizer.load_from_saved(str(tmp_path / "h.pkl")).LABEL_COLS == []


def test_verify_images_needs_the_image_folder_for_relative_paths(tmp_path):
    """V-H2-23 (D-04): with relative paths and no folder, verify_images() raises. It does not drop every row."""
    (tmp_path / "a.png").write_bytes(b"x")
    h = H.ChestImaGenomeGoldHarmonizer(annotation_dir=str(tmp_path))
    h.set_harmonized_df(TABLE.assign(image_path=["a.png"]))
    with pytest.raises(ValueError, match="relative to base_image_dir"):
        h.verify_images()
    assert h.verify_images(base_image_dir=str(tmp_path)).empty
    h.base_image_dir = str(tmp_path)  # a harmonizer that stores its folder needs no argument
    assert h.verify_images().empty and len(h.harmonized_df) == 1


@pytest.mark.parametrize("name", ["RSNA2024LumbarSpineTrainHarmonizer", "RSNAPEDetectionTrainHarmonizer",
                                  "RSNAAbdominalTrauma2023TrainHarmonizer", "RSNA2022CervicalSpineTrainHarmonizer",
                                  "RSNA2022CervicalSpineBboxHarmonizer"])
def test_series_verify_images_follows_d04(name, tmp_path):
    """V-H3-22 (D-04): the 3-D overrides check series folders, with the same folder rule as the base."""
    (tmp_path / "s1").mkdir()
    (tmp_path / "s1" / "1.dcm").write_bytes(b"x")
    hcls = getattr(H, name)
    h = hcls(**_kwargs(hcls, str(tmp_path)))
    h.set_harmonized_df(TABLE.assign(image_path=["s1"]))
    h.base_image_dir = None
    with pytest.raises(ValueError, match="relative to base_image_dir"):
        h.verify_images()
    assert h.verify_images(base_image_dir=str(tmp_path)).empty
    h.base_image_dir = str(tmp_path)
    assert h.verify_images().empty and len(h.harmonized_df) == 1


@pytest.mark.parametrize("key,hcls,kwargs", [
    ("node21", H.Node21Harmonizer, {"variant": "processed"}),
    ("chestxdet10_test", H.ChestXDet10TestHarmonizer, {"csv_path": "test.json"}),
])
def test_dataset_takes_the_image_folder_of_a_saved_harmonizer(tmp_path, key, hcls, kwargs):
    """L-006: with harmonizer_path only, the image folder comes from the file, not from a NAS default."""
    kw = {k: f"{tmp_path}/{v}" if k == "csv_path" else v for k, v in kwargs.items()}
    path = _save(hcls, str(tmp_path), TABLE.assign(split="test"), base_image_dir=str(tmp_path), **kw)
    ds = resolve_dataset(key)(harmonizer_path=path, cache_dir=None)
    assert ds.base_image_dir == str(tmp_path)
