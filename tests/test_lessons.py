"""Lesson tests: each test stops a real mistake from coming back.

Each test has a row in eval/RULES.md with the commit that made the mistake.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dataset_paths import is_complete, paths  # noqa: E402


def test_l001_chexpert_age_years_is_filled(tmp_path):
    """L-001: a declared age source must fill age_years (origin 95bb587)."""
    from radharmony.harmonizer.chexpert.chexpert import CheXpertHarmonizer

    csv = paths("chexpert_train", missing_ok=not is_complete()).get("csv_path", "")
    if not os.path.isfile(csv):
        pytest.skip("CheXpert is not mounted, or the data paths file has no chexpert_train entry")
    small = tmp_path / "train.csv"
    pd.read_csv(csv, nrows=200).to_csv(small, index=False)
    out = CheXpertHarmonizer(csv_path=str(small)).harmonize()
    assert "age_years" in out.columns
    assert out["age_years"].notna().mean() > 0.9


# ---------------------------------------------------------------------------
# L-009: every probe that trains drops NaN train labels, per label (D-18)
# ---------------------------------------------------------------------------
def _l009_data():
    """Two labels with a signal. 30% NaN train labels per label, at different rows."""
    import numpy as np

    rng = np.random.default_rng(0)
    n_tr, n_te = 120, 80
    x = rng.normal(size=(n_tr + n_te, 6)).astype(np.float32)
    y = (rng.random((n_tr + n_te, 2)) < 0.4).astype(np.float32)
    x[:, 0] += 1.5 * y[:, 0]
    x[:, 1] += 1.5 * y[:, 1]
    x[:, 2] += 3.0  # keeps the feature norms away from 0
    y_tr = y[:n_tr].copy()
    for k in range(2):
        y_tr[rng.random(n_tr) < 0.3, k] = np.nan
    return x[:n_tr], y_tr, x[n_tr:], y[n_tr:]


def _l009_probes():
    """The 5 probes that train, with settings where each label's fit uses only its own rows.

    The validation part and the n_train draw depend on the row set, so the check
    uses all rows and no validation part. Fine-tune trains a frozen-feature
    linear head with one batch per epoch.
    """
    import torch

    from radharmony.evaluator import (FinetuneEvaluator, KNNProbeEvaluator, LinearProbeEvaluator,
                                      PrototypeProbeEvaluator, SVMProbeEvaluator)

    common = dict(device="cpu", autocast_dtype=None, num_workers=0,
                  threshold_strategy="fixed:0.5", store_final_model=True, val_fraction=0.0)
    return [
        (LinearProbeEvaluator, torch.nn.Identity, common),
        (SVMProbeEvaluator, torch.nn.Identity, common),
        (KNNProbeEvaluator, torch.nn.Identity, dict(common, k=7)),
        (PrototypeProbeEvaluator, torch.nn.Identity, common),
        (FinetuneEvaluator, lambda: torch.nn.Linear(6, 6),
         dict(common, freeze_backbone=True, epochs=5, lr=1e-2, batch_size=256)),
    ]


def _accepted(cls, kw: dict) -> dict:
    """Keep the arguments that ``cls`` accepts, so the check also runs on old code."""
    import inspect

    names = set()
    for c in cls.__mro__:
        if "__init__" in c.__dict__:
            names |= set(inspect.signature(c.__init__).parameters)
    return {k: v for k, v in kw.items() if k in names}


def _l009_final_scores(ev, name, x):
    """Score ``x`` with the final model of one label (``store_final_model=True``)."""
    import numpy as np
    import torch

    feats = ev._maybe_normalize(np.asarray(x, dtype=np.float32))
    fm = getattr(ev, "final_models_", None)
    if fm is None:  # fine-tune: one head output per label
        k = ev.final_labels_.index(name)
        with torch.no_grad():
            z = ev.final_encoder_(torch.as_tensor(x))
            if ev.l2_normalize:
                z = torch.nn.functional.normalize(z, dim=1)
            return ev.final_head_(z)[:, k].numpy()
    m = fm[name]
    if hasattr(m, "predict_proba"):
        return m.predict_proba(feats)[:, 1]
    if hasattr(m, "decision_function"):
        return m.decision_function(feats)
    if isinstance(m, dict):  # kNN: index and labels of the rows with this label
        idx = m["nn"].kneighbors(feats)[1]
        return m["y_train"][idx].mean(axis=1)
    c_pos, c_neg = m  # prototype
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    return feats @ c_pos - feats @ c_neg


@pytest.mark.parametrize("probe", range(5), ids=["linear", "svm", "knn", "prototype", "finetune"])
def test_l009_nan_train_labels_equal_rows_with_label(probe):
    """L-009: with NaN train labels, each label's result equals the result on the rows with that label.

    This holds for the test metrics and for the final model. Origin: only the
    linear probe dropped NaN labels, and only in its main path (V-E-1, R-code-1).
    """
    import numpy as np
    import torch

    from _toy_eval import ToyClsDS

    cls, make_encoder, kw = _l009_probes()[probe]
    x_tr, y_tr, x_te, y_te = _l009_data()
    names = ["a", "b"]
    train, test = ToyClsDS(x_tr, y_tr, names), ToyClsDS(x_te, y_te, names)

    def run(train_ds):
        torch.manual_seed(0)
        ev = cls(make_encoder(), train_dataset=train_ds, test_dataset=test, **_accepted(cls, kw))
        df = ev.evaluate()
        return ev, df[df["bootstrap"] == -1].set_index("label")

    ev_all, df_all = run(train)
    metrics = [c for c in df_all.columns if c not in ("n_train", "fold", "seed", "bootstrap")]
    for k, name in enumerate(names):
        ev_k, df_k = run(train.subset(~np.isnan(y_tr[:, k])))
        np.testing.assert_allclose(df_all.loc[name, metrics].astype(float),
                                   df_k.loc[name, metrics].astype(float),
                                   rtol=1e-4, atol=1e-5, err_msg=f"{cls.__name__} label {name}")
        np.testing.assert_allclose(_l009_final_scores(ev_all, name, x_te),
                                   _l009_final_scores(ev_k, name, x_te),
                                   rtol=1e-4, atol=1e-5, err_msg=f"{cls.__name__} final model {name}")


# L-011: the box in a sample sits on the white rectangle of the fake image (D-28).
# The image file is black with one white rectangle. Its box, in the format of
# D-07, is the spec. After each spatial step, the box must still match the
# bright pixels of the returned image.
_RECT_2D = (6, 24, 4, 20)                 # rows 6..23, cols 4..19 of a 60 x 40 image
_RECT_3D = (2, 10, 3, 15, 4, 20)          # slices of a 20 x 30 x 40 volume


def _fake_image(tmp_path, ndim):
    """Write the fake image file. Return its path and its box as fractions."""
    np = pytest.importorskip("numpy")
    if ndim == 2:
        from PIL import Image

        arr = np.zeros((60, 40), dtype=np.uint8)
        r = _RECT_2D
        arr[r[0]:r[1], r[2]:r[3]] = 255
        path = tmp_path / "rect.png"
        Image.fromarray(arr).save(path)
    else:
        sitk = pytest.importorskip("SimpleITK")
        arr = np.zeros((20, 30, 40), dtype=np.float32)
        r = _RECT_3D
        arr[r[0]:r[1], r[2]:r[3], r[4]:r[5]] = 100.0
        path = tmp_path / "rect.nii.gz"
        sitk.WriteImage(sitk.GetImageFromArray(arr), str(path))
    box = [r[i] / arr.shape[i // 2] for i in range(2 * ndim)]
    return str(path), box


def _bright_box(img):
    """Return the box of the bright pixels of a (C, *spatial) image, as fractions."""
    import torch

    idx = torch.nonzero(img[0].float() > 0)
    out = []
    for i, n in enumerate(img.shape[1:]):
        out += [idx[:, i].min().item() / n, (idx[:, i].max().item() + 1) / n]
    return out


def _steps(ndim):
    """(name, builder step) pairs. Each step makes one spatial change."""
    flips = [(f"flip{a}", lambda b, a=a: b.with_flip(spatial_axis=a, prob=1.0)) for a in range(ndim)]
    perm = [0, 2, 1] if ndim == 2 else [0, 3, 1, 2]
    affine = (dict(translate_range=(3, 3), rotate_range=(0.3,), scale_range=(0.1, 0.1))
              if ndim == 2 else
              dict(translate_range=(2, 2, 2), rotate_range=(0.2, 0.2, 0.2), scale_range=(0.1,) * 3,
                   shear_range=(0.05,) * 6))
    elastic = (dict(spacing=16, magnitude_range=(1.0, 2.0)) if ndim == 2
               else dict(sigma_range=(2.0, 3.0), magnitude_range=(20.0, 30.0)))
    return [
        ("default", lambda b: b),
        *flips,
        ("transpose", lambda b: b.with_transpose(indices=perm)),
        ("affine", lambda b: b.with_affine(prob=1.0, **affine)),
        ("elastic", lambda b: b.with_elastic_deformation(prob=1.0, **elastic)),
    ]


@pytest.mark.parametrize("pad", [True, False], ids=["pad", "no_pad"])
@pytest.mark.parametrize("ndim,name", [(nd, name) for nd in (2, 3) for name, _ in _steps(nd)])
def test_l011_box_stays_on_the_white_rectangle(tmp_path, ndim, name, pad):
    """L-011: after the default transform and each spatial step, the box sits on the rectangle."""
    import torch

    from radharmony.dataset.transforms import RadiologyTransform2D, RadiologyTransform3D

    add = dict(_steps(ndim))[name]
    path, box = _fake_image(tmp_path, ndim)
    cls = RadiologyTransform2D if ndim == 2 else RadiologyTransform3D
    size = 64 if ndim == 2 else 32
    builder = add(cls(img_size=size, pad=pad, output_keys={"img", "bbox", "bbox_labels"},
                      dtype=torch.float32))
    t = builder.get_transform()
    t.set_random_state(seed=0)
    out = t({"img": path, "bbox": [box], "bbox_labels": ["rect"]})

    got = torch.as_tensor(out["bbox"], dtype=torch.float32).reshape(-1, 2 * ndim)
    assert got.shape == (1, 2 * ndim), (name, out["bbox"])
    expected = _bright_box(out["img"])
    # 1.5 pixels of tolerance: the resize and the warps blur the edge of the rectangle.
    tol = [1.5 / n for n in out["img"].shape[1:] for _ in range(2)]
    err = [abs(g - e) for g, e in zip(got[0].tolist(), expected, strict=True)]
    assert all(e <= t_ for e, t_ in zip(err, tol, strict=True)), (name, got[0].tolist(), expected)


# L-012: a palette PNG reads as the colors it shows, and an RGBA PNG gives 3 channels (D-13).
# The fake palette PNG shows 255 - i for index i, as half of the Shenzhen PNGs do. Its gray
# image is the spec.


def _recipe_image_loaders():
    """Yield (file, loader name, first step) for each image loader of a backbone recipe.

    An image loader is a ``_loader*`` assignment to ``mt.Compose([...])`` whose name has no
    "mask". A mask loader keeps ``mt.LoadImage``, because a palette mask has class numbers
    in the index.
    """
    import ast

    for path in sorted((ROOT / "radharmony/evaluator/backbones").glob("*.py")):
        if path.name.startswith("_"):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)):
                continue
            name = node.targets[0].id
            if not name.startswith("_loader") or "mask" in name or not isinstance(node.value, ast.Call):
                continue
            steps = node.value.args[0] if node.value.args else None
            if isinstance(steps, ast.List) and steps.elts:
                yield path.name, name, ast.unparse(steps.elts[0])


def test_l012_recipe_readers_read_palette_and_alpha_pngs_as_shown(tmp_path):
    """L-012: reader='monai' and every recipe image loader read a palette PNG as shown and drop alpha."""
    import numpy as np
    from PIL import Image

    from radharmony.evaluator.backbones._utils import load_image
    from radharmony.evaluator.transforms import _resolve_reader

    gray = np.add.outer(np.arange(20) * 5, np.arange(30) * 3).astype(np.uint8)
    pal = Image.fromarray(gray, "P")
    pal.putpalette([v for i in range(256) for v in (255 - i,) * 3])  # index i shows 255 - i
    pal.save(tmp_path / "pal.png")
    Image.fromarray(gray).convert("RGBA").save(tmp_path / "rgba.png")

    monai_read = _resolve_reader("monai")
    np.testing.assert_array_equal(monai_read(str(tmp_path / "pal.png"))[0].numpy(), 255 - gray)
    assert tuple(monai_read(str(tmp_path / "rgba.png")).shape) == (3, 20, 30)
    np.testing.assert_array_equal(load_image()(str(tmp_path / "pal.png"))[0].numpy().T, 255 - gray)

    loaders = list(_recipe_image_loaders())
    assert len(loaders) >= 13  # the image loaders on 2026-10-31
    wrong = [(f, n, first) for f, n, first in loaders if first != "load_image()"]
    assert not wrong, f"recipe image loaders that do not start with load_image(): {wrong}"


# ---------------------------------------------------------------------------
# L-016: a source row with no label at all gives NaN in every label (D-05)
# ---------------------------------------------------------------------------
def _l016_chexpert_style(build):
    """Wrap a Q-4 builder of tests/test_harmonizer_data.py. Its 3rd row has every label blank."""
    def make(d):
        h, ids = build(d)
        return h, ids[2]
    return make


def _l016_padchest(d):
    from radharmony.harmonizer import PadChestHarmonizer

    pd.DataFrame({
        "ImageID": ["a.png", "b.png"], "ImageDir": [0, 0], "PatientID": ["p1", "p2"], "StudyID": ["s1", "s2"],
        "Labels": ["['cardiomegaly']", "[]"], "Projection": ["PA", "PA"], "PatientSex_DICOM": ["M", "F"],
        "PatientBirth": [1950.0, 1960.0], "StudyDate_DICOM": [20140915, 20140915],
    }).to_csv(d / "padchest.csv", index=False)
    return PadChestHarmonizer(csv_path=str(d / "padchest.csv")), "p2"


def _l016_openi(d):
    from radharmony.harmonizer.openi_cxr import OpenICXRHarmonizer

    (d / "ecgen-radiology").mkdir()
    for i, term in ((1, "Cardiomegaly"), (2, "No Indexing")):
        (d / "ecgen-radiology" / f"{i}.xml").write_text(
            f'<eCitation><uId id="CXR{i}"/><pmcId id="{i}"/><MeSH><major>{term}</major></MeSH>'
            f'<parentImage id="CXR{i}_1_IM-0001-3001"><figureId>F1</figureId></parentImage></eCitation>')
    return OpenICXRHarmonizer(base_dir=str(d)), "CXR2"


def _l016_builders():
    import test_harmonizer_data as t

    q4 = {"chexpert": t._q4_chexpert, "chexpert_plus": t._q4_chexpert_plus, "mimic_jpg": t._q4_mimic_jpg,
          "mimic_jpg_test": t._q4_mimic_jpg_test, "mimic_dicom": t._q4_mimic_dicom}
    return {**{k: _l016_chexpert_style(b) for k, b in q4.items()}, "padchest": _l016_padchest, "openi": _l016_openi}


@pytest.mark.parametrize("name", ["chexpert", "chexpert_plus", "mimic_jpg", "mimic_jpg_test", "mimic_dicom",
                                  "padchest", "openi"])
def test_l016_a_row_with_no_label_at_all_is_nan(tmp_path, name):
    """L-016: with the default call, a source row with no label at all gets NaN in every label (D-05).

    "No label at all" is a row with every label blank, or a row with only the source placeholder
    (PadChest ``[]``, OpenI "No Indexing"). Origin: V-H1-41 and V-H2-41.
    """
    import warnings

    h, row = _l016_builders()[name](tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = h.harmonize().set_index("patient_id")
    labels = [c for c in (x.lower().replace(" ", "_") for x in h.LABEL_COLS) if c in df.columns]
    assert labels and df.loc[row, labels].isna().all()


# ---------------------------------------------------------------------------
# L-018: each 3-D box step keeps the box on its cube, for 0, 1 and 3 boxes, with
# and without with_spacing (V-D-41, V-D-42). The cubes in the file are the spec.
# ---------------------------------------------------------------------------
# Slices of a 6 x 40 x 40 volume (z, y, x). The cubes do not touch.
_CUBES_3D = [(1, 3, 4, 14, 4, 14), (3, 5, 20, 34, 6, 16), (1, 4, 8, 30, 24, 36)]


def _l018_volume(tmp_path, n):
    """Write a volume with 8 mm slices and 1 mm pixels, with n bright cubes. Return its path and boxes."""
    np = pytest.importorskip("numpy")
    sitk = pytest.importorskip("SimpleITK")
    arr = np.zeros((6, 40, 40), dtype=np.float32)
    cubes = _CUBES_3D[:max(n, 1)]  # 0 boxes: a cube with no box
    for c in cubes:
        arr[c[0]:c[1], c[2]:c[3], c[4]:c[5]] = 100.0
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 8.0))  # (x, y, z): anisotropic, as a thick-slice MRI
    path = tmp_path / f"cubes{n}.nii.gz"
    sitk.WriteImage(img, str(path))
    boxes = [[c[i] / arr.shape[i // 2] for i in range(6)] for c in cubes[:n]]
    return str(path), boxes


def _l018_bright_boxes(img):
    """Return the box of each bright cube of a (1, D, H, W) image, as fractions."""
    ndimage = pytest.importorskip("scipy.ndimage")

    x = img[0].float().numpy()
    found, _ = ndimage.label(x > 0)  # > 0 is half way between the background (-1) and a cube (1)
    return [[s.start / n if k == 0 else s.stop / n for s, n in zip(sl, x.shape, strict=True) for k in (0, 1)]
            for sl in ndimage.find_objects(found)]


@pytest.mark.parametrize("pixdim", [None, (1.0, 1.0, 1.0)], ids=["no_spacing", "with_spacing"])
@pytest.mark.parametrize("n", [0, 1, 3])
def test_l018_3d_box_stays_on_its_cube(tmp_path, n, pixdim):
    """L-018: a 3-D sample gives one box for each box of the row, on its cube, also after with_spacing."""
    import torch

    from radharmony.dataset.transforms import RadiologyTransform3D

    path, boxes = _l018_volume(tmp_path, n)
    labels = ["a", "b", "c"][:n]
    builder = RadiologyTransform3D(img_size=48, output_keys={"img", "bbox", "bbox_labels"}, dtype=torch.float32)
    if pixdim is not None:
        builder = builder.with_spacing(pixdim)
    out = builder.get_transform()({"img": path, "bbox": boxes, "bbox_labels": labels})

    assert tuple(out["bbox"].shape) == (n, 6)
    assert out["bbox_labels"] == labels
    cubes = _l018_bright_boxes(out["img"])
    # 1.5 voxels of tolerance, as in L-011: the resample and the resize blur the cube edge.
    tol = [1.5 / s for s in out["img"].shape[1:] for _ in range(2)]
    for got in out["bbox"].tolist():
        err = min((max(abs(g - e) / t for g, e, t in zip(got, cube, tol, strict=True)) for cube in cubes))
        assert err <= 1.0, (got, cubes)


# ---------------------------------------------------------------------------
# L-019: a backbone extra names every module that the model code imports (V-A-46)
# ---------------------------------------------------------------------------
def _hf_snapshot(repo: str):
    import glob

    found = glob.glob(str(Path.home() / f".cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))
    return Path(found[0]) if found else None


#: (extra, folder of the model code, first module). A folder that is not on this machine skips its case.
_L019_CASES = [
    ("medimageinsights", lambda: _hf_snapshot("lion-ai/MedImageInsights"), "medimageinsightmodel"),
    ("eva_x", lambda: ROOT / "third_party_models" / "EVA-X", "eva_x"),
    ("chexfound", lambda: ROOT / "third_party_models" / "CheXFound", "chexfound.models.vision_transformer"),
]

#: Import names whose package has another name.
_L019_DISTS = {"yaml": "pyyaml", "PIL": "pillow", "sklearn": "scikit-learn", "cv2": "opencv-python-headless",
               "skimage": "scikit-image"}


def _l019_module_level_imports(root: Path, first: str) -> dict:
    """Return {third-party module: file} for the module-level imports, from ``first`` through the local modules."""
    import ast

    def local_file(mod):
        p = root.joinpath(*mod.split("."))
        return next((c for c in (p.with_suffix(".py"), p / "__init__.py") if c.exists()), None)

    seen, third, todo = set(), {}, [first]
    while todo:
        mod = todo.pop()
        f = None if mod in seen else local_file(mod)
        seen.add(mod)
        if f is None:
            continue
        pkg = f.relative_to(root).with_suffix("").parts[:-1]
        for node in ast.parse(f.read_text()).body:
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level:
                names = [".".join(list(pkg[:len(pkg) - node.level + 1]) + ([node.module] if node.module else []))]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module]
            else:
                continue
            for name in names:
                parts = name.split(".")
                local = next((".".join(parts[:i]) for i in range(len(parts), 0, -1)
                              if local_file(".".join(parts[:i]))), None)
                if local:
                    todo.append(local)
                elif parts[0] not in sys.stdlib_module_names:
                    third.setdefault(parts[0], str(f.relative_to(root)))
    return third


@pytest.mark.parametrize("extra,folder,first", _L019_CASES, ids=[c[0] for c in _L019_CASES])
def test_l019_backbone_extra_names_every_module_that_the_model_code_imports(extra, folder, first):
    """L-019: each module that the model code imports at module level is in the extra or the core dependencies.

    The package must be named directly, not only through another package. Origin: V-A-46 (timm,
    transformers, einops and safetensors were missing from the medimageinsights extra).
    """
    import re
    import tomllib

    root = folder()
    if root is None or not root.exists():
        pytest.skip(f"the model code of {extra} is not on this machine")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    name = lambda req: re.split(r"[\[<>=!~ ;@]", req.strip(), maxsplit=1)[0].lower().replace("_", "-")  # noqa: E731
    listed = {name(r) for r in project["dependencies"] + project["optional-dependencies"][extra]}
    imports = _l019_module_level_imports(root, first)
    missing = {m: f for m, f in imports.items() if _L019_DISTS.get(m, m).lower().replace("_", "-") not in listed}
    assert imports and not missing, f"[{extra}] does not name these imported modules: {missing}"


# ---------------------------------------------------------------------------
# L-017: the embedding cache raises when an input that sets the features changes (D-41).
# Two image folders hold the same file names with other pixels, as TAIX-Ray full size and
# TAIX-Ray 512 do. Each case first checks that the changed input gives other features.
# ---------------------------------------------------------------------------
def _l017_run(folder, encoder, transform, autocast_dtype, cache=None):
    """Return the features of the 6 images in ``folder``, from ``LinearProbeEvaluator._load_or_extract``."""
    import numpy as np

    from radharmony.dataset.base import BaseRadiologicalDataset
    from radharmony.evaluator import LinearProbeEvaluator

    class _Real(BaseRadiologicalDataset):
        LABEL_COLS: list = []

        def _get_harmonized_df(self):
            return self._try_resolve_preset_harmonized()

    df = pd.DataFrame({"image_path": [f"p{i}/a.png" for i in range(6)], "patient_id": [f"p{i}" for i in range(6)],
                       "a": np.arange(6) % 2})
    ds = _Real(base_image_dir=str(folder), transform=transform, cache_dir=None, output_cls=True, harmonized_df=df)
    ev = LinearProbeEvaluator(encoder, dataset=ds, device="cpu", num_workers=0, autocast_dtype=autocast_dtype,
                              embedding_cache=cache)
    return ev._load_or_extract(ds)[0]


@pytest.mark.parametrize("change", ["encoder", "transform", "autocast_dtype", "image_folder"])
def test_l017_the_embedding_cache_raises_when_an_input_of_the_features_changes(tmp_path, change):
    """L-017: another encoder, transform, ``autocast_dtype`` or image folder raises ValueError (D-41).

    Origin: V-E-34 (``autocast_dtype`` and the image folder were not in the cache check), and the
    fixed V-E-2 and V-E-25 (encoder and transform).
    """
    import monai.transforms as mt
    import numpy as np
    import torch
    from PIL import Image

    for folder, scale in (("full", 1), ("small", 3)):
        rng = np.random.default_rng(0)
        for i in range(6):
            (tmp_path / folder / f"p{i}").mkdir(parents=True)
            pixels = (rng.integers(0, 80, (8, 8)) * scale).astype(np.uint8)
            Image.fromarray(pixels).save(tmp_path / folder / f"p{i}" / "a.png")

    def transform(extra=()):
        return mt.Compose([mt.LoadImaged(keys="img", ensure_channel_first=True, image_only=True), *extra,
                           mt.ToTensord(keys=["img", "cls"])])

    def encoder(seed):
        torch.manual_seed(seed)
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 8))

    base = dict(folder=tmp_path / "full", encoder=encoder(0), transform=transform(), autocast_dtype=torch.bfloat16)
    other = dict(base, **{"encoder": {"encoder": encoder(1)},
                          "transform": {"transform": transform([mt.ScaleIntensityd(keys="img")])},
                          "autocast_dtype": {"autocast_dtype": None},
                          "image_folder": {"folder": tmp_path / "small"}}[change])
    cache = str(tmp_path / "emb")
    cached = _l017_run(**base, cache=cache)
    assert not np.allclose(_l017_run(**other), cached)  # the change gives other features
    match = {"autocast_dtype": "has autocast_dtype", "image_folder": "has base_image_dir"}.get(change, "comes from")
    with pytest.raises(ValueError, match=match):
        _l017_run(**other, cache=cache)


def _calls_super_harmonize(func) -> bool:
    """Return True if the code of ``func`` calls ``super().harmonize(...)``. Docstrings and comments do not count."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(func))))
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "harmonize"
               and isinstance(n.func.value, ast.Call) and isinstance(n.func.value.func, ast.Name)
               and n.func.value.func.id == "super"
               for n in ast.walk(tree))


def test_l022_every_harmonize_override_calls_the_base_steps():
    """L-022 (D-30): every ``harmonize()`` override in the package calls ``super().harmonize()``.

    A copy of the base steps does not get the fixes of the base. Example: before 2026-11-03,
    ``ReXGradientHarmonizer.harmonize()`` copied the steps and skipped the image copies.
    """
    import importlib
    import inspect
    import pkgutil

    import radharmony.harmonizer as pkg
    from radharmony.harmonizer.base import BaseHarmonizer
    from radharmony.harmonizer.base_vqa import BaseVQAHarmonizer

    bad = []
    for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        module = importlib.import_module(mod.name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls in (BaseHarmonizer, BaseVQAHarmonizer) or cls.__module__ != module.__name__:
                continue
            own = cls.__dict__.get("harmonize")
            if own is not None and not _calls_super_harmonize(own):
                bad.append(f"{cls.__module__}.{cls.__qualname__}")
    assert not bad, f"harmonize() overrides that skip super().harmonize(): {bad}"


# Files that are not a dataset: the package files, the base files and the helpers (a name that starts with "_").
_L023_SHARED = {"__init__.py", "base.py", "base_vqa.py", "transforms.py", "collate.py"}


def test_l023_each_dataset_is_in_its_own_folder():
    """L-023: each dataset and harmonizer module is in its own folder, ``<name>/<name>.py``.

    Example: before 2026-11-05, ``radharmony/dataset/emory_cxr.py`` was a flat file. The public repo had
    ``emory_cxr/emory_cxr.py``, and the merge had to drop one copy.
    """
    bad = []
    for pkg in ("radharmony", "radharmony_internal"):
        for part in ("dataset", "harmonizer"):
            for f in sorted((ROOT / pkg / part).glob("*.py")):
                if f.name not in _L023_SHARED and not f.name.startswith("_"):
                    bad.append(f.relative_to(ROOT).as_posix())
    assert not bad, f"move each file to <name>/<name>.py with an __init__.py: {bad}"
