"""Q-13: one name list for the 2-D ``view_position``, as for ``sex`` (D-26). No data needed.

Run with::

    .venv/bin/python -m pytest tests/test_view_position.py -v
"""
from __future__ import annotations

import pandas as pd
import pytest

import radharmony.harmonizer as H
from radharmony.harmonizer.base import BaseHarmonizer
from radharmony.harmonizer.emory_cxr import EmoryCXRDicomHarmonizer

# Source values from the real tables (BRAX, PadChest, ReXGradient, MIMIC-CXR-JPG, CheXpert, OpenI ...).
RAW = ["PA", " pa ", "POSTERO_ANTERIOR", "AP", "ANTERO_POSTERIOR", "Lateral", "LAT", "L", "LL", "RL", "Unknown",
       "UNK", "N/A", "EXCLUDE", None, "Frontal", "AP_horizontal", "RAO", "XTABLE LATERAL"]
WANT = ["PA", "PA", "PA", "AP", "AP", "LATERAL", "LATERAL", "LATERAL", "LATERAL", "LATERAL", None, None, None,
        None, None, "FRONTAL", "AP_HORIZONTAL", "RAO", "XTABLE LATERAL"]


class _Views(BaseHarmonizer):
    """A 2-D harmonizer whose source has one view column."""

    VIEW_POSITION_SOURCE_COL = "ViewPosition"

    def _build_patient_id(self):
        self.df["patient_id"] = self.df["image_id"]

    def _build_study_id(self):
        self.df["study_id"] = self.df["image_id"]

    def _build_image_path(self):
        self.df["image_path"] = self.df["image_id"] + ".png"


def _views(cls, tmp_path) -> list:
    pd.DataFrame({"image_id": [f"i{n:02d}" for n in range(len(RAW))], "ViewPosition": RAW}).to_csv(
        tmp_path / "v.csv", index=False)
    out = cls(csv_path=str(tmp_path / "v.csv")).harmonize()
    return [None if pd.isna(v) else v for v in out["view_position"]]


def test_view_position_uses_one_name_list(tmp_path):
    """Q-13: PA, AP and LATERAL. Unknown words give NaN. Any other value keeps its raw value in upper case."""
    assert _views(_Views, tmp_path) == WANT


def test_a_class_can_keep_its_own_view_names(tmp_path):
    """Q-13: with VIEW_POSITION_NAMES = False, the source values stay as they are."""

    class _Raw(_Views):
        VIEW_POSITION_NAMES = False

    assert _views(_Raw, tmp_path) == [None if v == "N/A" else v for v in RAW]  # read_csv reads N/A as NaN


@pytest.mark.parametrize("cls", [H.CTRATEHarmonizer, H.RSNA2024LumbarSpineHarmonizer, EmoryCXRDicomHarmonizer],
                         ids=lambda c: c.__name__)
def test_3d_views_and_emory_cxr_dicom_keep_their_names(cls):
    """Q-13, Q-18: a 3-D series name (for example "Sagittal T1") is not a 2-D view. Emory CXR DICOM stays as before."""
    assert cls.VIEW_POSITION_NAMES is False


def test_the_names_are_the_ones_of_the_decision():
    """Q-13: the shared table gives only PA, AP and LATERAL."""
    from radharmony.harmonizer.base import _VIEW_VALUES

    assert set(_VIEW_VALUES.values()) == {"PA", "AP", "LATERAL"}


def test_chexpert_frontal_with_ll_has_no_known_view(tmp_path):
    """Q-13, Q-14: a frontal image with LL in AP/PA gives NaN, as for Emory. A lateral image gives LATERAL."""
    paths = [f"CheXpert-v1.0/train/patient0000{i}/study1/view1_frontal.jpg" for i in (1, 2, 3, 4)]
    labels = {c: [0.0] * 4 for c in H.CheXpertHarmonizer.LABEL_COLS}
    pd.DataFrame({"Path": paths, "Sex": ["Male"] * 4, "Age": [50] * 4,
                  "Frontal/Lateral": ["Frontal", "Frontal", "Frontal", "Lateral"],
                  "AP/PA": ["AP", "PA", "LL", None], **labels}).to_csv(tmp_path / "train.csv", index=False)
    out = H.CheXpertHarmonizer(csv_path=str(tmp_path / "train.csv")).harmonize()
    assert [None if pd.isna(v) else v for v in out["view_position"]] == ["AP", "PA", None, "LATERAL"]


def test_unknown_strings_give_nan_and_an_opt_out_class_keeps_them():
    """Q-13: "N/A", "None" and a blank reach the step as strings from JSON (ReXGradient). Emory CXR DICOM keeps them.

    V-H2-43: the ReXGradient train JSON has 2 views "None" and 1 view "ANTERO_POSTERIOR" (D-01: AP gives AP).
    """
    df = pd.DataFrame({"view_position": ["N/A", "", " Unknown ", "None", "none", "LL", "Lateral",
                                         "ANTERO_POSTERIOR", "antero_posterior"]})
    got = _Views.__new__(_Views)._build_view_position_names(df)["view_position"].tolist()
    assert [None if pd.isna(v) else v for v in got] == [None, None, None, None, None, "LATERAL", "LATERAL",
                                                        "AP", "AP"]
    kept = EmoryCXRDicomHarmonizer.__new__(EmoryCXRDicomHarmonizer)._build_view_position_names(df)
    assert kept["view_position"].tolist() == df["view_position"].tolist()
