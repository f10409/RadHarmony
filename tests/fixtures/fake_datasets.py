"""Harmonizers on small fake source files, for the contract test (L-008). No data needed.

Each builder writes the fake files in ``tmp_path`` and returns a harmonizer that
is ready to harmonize. The fake files have the real file names and columns, with
a few rows. The builders are in the dataset test files: the first 6 had fake data
before L-008, and the D-05 test (``test_label_order.py``) adds 5. This module
needs the ``tests`` folder on ``sys.path`` (pytest's default import mode).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the tests folder

import pandas as pd  # noqa: E402

from radharmony.harmonizer import (  # noqa: E402
    ChestXDet10Harmonizer,
    CheXphotoHarmonizer,
    CheXpertPlusHarmonizer,
    EmoryCXRHarmonizer,
    PadChestGRHarmonizer,
)


def chestxdet10(tmp_path):
    from test_chestxdet10 import write_fake
    csv, root = write_fake(tmp_path)
    return ChestXDet10Harmonizer(csv_path=csv, base_image_dir=root)


def chexphoto(tmp_path):
    from test_chexphoto import write_fake
    return CheXphotoHarmonizer(csv_path=write_fake(tmp_path))


def chexpert_plus(tmp_path):
    from test_chexpert_plus import _fake
    csv, base = _fake(tmp_path)
    return CheXpertPlusHarmonizer(csv_path=str(csv), base_image_dir=str(base))


def emory_cxr(tmp_path):
    from test_emory_cxr import _write_labels, _write_metadata
    meta = _write_metadata(tmp_path)
    labels = _write_labels(tmp_path, pd.read_csv(meta)["AccessionNumber_anon"].unique())
    return EmoryCXRHarmonizer(csv_path=str(meta), base_image_dir=str(tmp_path), label_csv_path=str(labels))


def padchest_gr(tmp_path):
    from test_padchest_gr import _standard
    return PadChestGRHarmonizer(base_image_dir=str(_standard(tmp_path)))


def siim_acr_ptx(tmp_path):
    from test_siim_acr_ptx import _fake_release
    return _fake_release(tmp_path)


def _from_label_order(name):
    def build(tmp_path):
        import test_label_order
        return getattr(test_label_order, name)(tmp_path)
    return build


#: name -> builder(tmp_path) -> harmonizer. The first 6 had fake data before L-008.
FAKE = {
    "chestxdet10": chestxdet10,
    "chexphoto": chexphoto,
    "chexpert_plus": chexpert_plus,
    "emory_cxr": emory_cxr,
    "padchest_gr": padchest_gr,
    "siim_acr_ptx": siim_acr_ptx,
    **{n: _from_label_order(n) for n in ("vindr_cxr_train", "vindr_cxr_test", "vindr_pcxr", "openi", "mrrate")},
}
