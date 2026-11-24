"""MIMIC-Ext-CXR-QBA harmonizer tests. No real data.

The expected values come from the fake files and from D-01 and D-51, not from the code.

Run with::

    .venv/bin/python -m pytest tests/test_mimic_ext_cxr_qba.py -v
"""
from __future__ import annotations

import json
import zipfile

import pandas as pd
import pytest

from radharmony.harmonizer import MIMICExtCXRQBAHarmonizer


def _folder(tmp_path, answers):
    """1 patient, 1 study, 1 question with the answer dicts ``answers``. Return (metadata_dir, qa.zip)."""
    p, s = "p10000001", "s50000001"
    meta = tmp_path / "metadata"
    meta.mkdir()
    pd.DataFrame({"patient_id": [p], "study_id": [s], "question_id": ["q1"],
                  "question.question_type": ["abnormality"], "question.quality.rating": ["A"]}) \
        .set_index(["patient_id", "study_id", "question_id"]).to_parquet(meta / "question_metadata.parquet")
    pd.DataFrame({"patient_id": [p], "study_id": [s], "question_id": ["q1"], "image_id": ["img0"]}) \
        .set_index(["patient_id", "study_id"]).to_parquet(meta / "question_image_metadata.parquet")
    with zipfile.ZipFile(tmp_path / "qa.zip", "w") as z:
        z.writestr(f"qa/{p[:3]}/{p}/{s}.qa.json",
                   json.dumps({"questions": [{"question_id": "q1", "question": "Any finding?", "answers": answers}]}))
    return str(meta), str(tmp_path / "qa.zip")


def test_answer_holds_only_the_main_answer_texts(tmp_path):
    """Q-14 (D-51): ``answer`` joins the main answers. The details and related information stay in answer_struct."""
    answers = [{"answer_type": "main_answer", "text": "Mild pulmonary edema."},
               {"answer_type": "details", "text": "The edema is in both lungs."},
               {"answer_type": "main_answer", "text": "Small left effusion."},
               {"answer_type": "related_information", "text": "No pneumothorax."}]
    meta, qa = _folder(tmp_path, answers)
    df = MIMICExtCXRQBAHarmonizer(metadata_dir=meta, qa_zip_path=qa).harmonize()
    assert df.loc[0, "answer"] == "Mild pulmonary edema. Small left effusion."
    assert df.loc[0, "answer_struct"] == answers


def _image_metadata(meta, views):
    """Write ``image_metadata.parquet`` with one image ``img{i}`` for each view in ``views``. ``img0`` is the image of the question."""
    pd.DataFrame({"patient_id": ["p10000001"] * len(views), "study_id": ["s50000001"] * len(views),
                  "image_id": [f"img{i}" for i in range(len(views))], "img.view_position": views}) \
        .set_index(["patient_id", "study_id", "image_id"]).to_parquet(f"{meta}/image_metadata.parquet")


@pytest.mark.parametrize("source, expected", [("PA", "PA"), ("AP", "AP"), ("LL", "LATERAL")])
def test_view_position_comes_from_the_image_metadata(tmp_path, source, expected):
    """D-01: ``view_position`` is the ``img.view_position`` of the question's image, with the shared names (Q-13)."""
    meta, qa = _folder(tmp_path, [{"text": "No finding."}])
    _image_metadata(meta, [source, "AP"])  # img1 is not the image of the question
    df = MIMICExtCXRQBAHarmonizer(metadata_dir=meta, qa_zip_path=qa).harmonize()
    assert df["view_position"].tolist() == [expected]


def test_no_image_metadata_warns_and_gives_no_view_position(tmp_path):
    """An export with no ``image_metadata.parquet`` still harmonizes, with a warning and no ``view_position``."""
    meta, qa = _folder(tmp_path, [{"text": "No finding."}])
    with pytest.warns(UserWarning, match="image_metadata.parquet"):
        df = MIMICExtCXRQBAHarmonizer(metadata_dir=meta, qa_zip_path=qa).harmonize()
    assert "view_position" not in df.columns and len(df) == 1
