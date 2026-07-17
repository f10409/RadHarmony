"""Tests for VQAEvaluator + VQA metrics using synthetic data.

No model / NAS dependency — a fake answerer stands in for the VLM and a tiny
in-memory harmonized DataFrame drives a BaseVQADataset. Exercises the
MedGemma-protocol scoring (overall tokenized F1, closed yes/no accuracy,
open token-F1/recall) end to end.

Run with::

    .venv/bin/python -m pytest tests/test_vqa_evaluator.py -v
"""
from __future__ import annotations

import pandas as pd
import pytest

from radharmony.dataset.base_vqa import BaseVQADataset
from radharmony.evaluator.language.vqa import VQAEvaluator
from radharmony.evaluator.metrics.vqa import (
    extract_yes_no,
    is_yes_no_answer,
    token_recall,
    tokenized_f1,
)


# ── metric unit tests ──────────────────────────────────────────────────────
def test_tokenized_f1_identical_and_disjoint():
    assert tokenized_f1("cardiomegaly", "cardiomegaly") == 1.0
    assert tokenized_f1("normal", "pneumonia") == 0.0
    # partial overlap: "left pleural effusion" vs "pleural effusion" → 2*P*R/(P+R)
    assert tokenized_f1("left pleural effusion", "pleural effusion") == pytest.approx(0.8)


def test_tokenized_f1_normalizes_case_punct_articles():
    assert tokenized_f1("The Yes.", "yes") == 1.0


def test_token_recall():
    assert token_recall("pleural effusion", "left pleural effusion") == pytest.approx(2 / 3)
    assert token_recall("", "anything") == 0.0


def test_extract_and_is_yes_no():
    assert extract_yes_no("No, there is no effusion.") == "no"
    assert extract_yes_no("Yes.") == "yes"
    assert extract_yes_no("cardiomegaly") is None
    assert is_yes_no_answer("Yes") is True
    assert is_yes_no_answer("cardiomegaly") is False


# ── evaluator integration test ──────────────────────────────────────────────
def _identity(sample: dict) -> dict:
    # Keep img as-is (a path string); the fake answerer ignores the image.
    return sample


class _SynthVQADataset(BaseVQADataset):
    """Minimal concrete VQA dataset that serves a preset harmonized df.

    Mirrors how real subclasses (VQARadDataset) implement _get_harmonized_df:
    resolve the preset first.
    """

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        if preset is None:
            raise AssertionError("expected a preset harmonized_df")
        return preset


@pytest.fixture
def synth_vqa(tmp_path):
    """A 4-row synthetic VQA set: 2 closed (yes/no) + 2 open questions."""
    rows = [
        {"patient_id": "p0", "study_id": "s0", "question_id": "q0",
         "image_path": "p0.png", "question": "Is there cardiomegaly?", "answer": "yes"},
        {"patient_id": "p0", "study_id": "s0", "question_id": "q1",
         "image_path": "p0.png", "question": "Is there a fracture?", "answer": "no"},
        {"patient_id": "p1", "study_id": "s1", "question_id": "q2",
         "image_path": "p1.png", "question": "What is the main finding?",
         "answer": "pleural effusion"},
        {"patient_id": "p1", "study_id": "s1", "question_id": "q3",
         "image_path": "p1.png", "question": "Which organ is imaged?",
         "answer": "chest"},
    ]
    df = pd.DataFrame(rows)
    ds = _SynthVQADataset(
        base_image_dir=str(tmp_path),
        harmonized_df=df,
        transform=_identity,
        cache_dir=None,
    )
    # question -> reference answer, for the perfect answerer below
    refs = {r["question"]: r["answer"] for r in rows}
    return ds, refs


def test_perfect_answerer_scores_one(synth_vqa):
    ds, refs = synth_vqa

    def perfect(imgs, questions):
        return [refs[q] for q in questions]

    ev = VQAEvaluator(perfect, dataset=ds, batch_size=2, num_workers=0)
    df = ev.evaluate()

    assert set(df["label"]) == {"overall", "closed", "open"}
    assert ev.n_samples_ == 4
    overall = df[df["label"] == "overall"].iloc[0]
    closed = df[df["label"] == "closed"].iloc[0]
    assert overall["token_f1"] == pytest.approx(1.0)
    assert closed["accuracy"] == pytest.approx(1.0)
    # point estimate only (no bootstrap) → one row per bucket
    assert (df["bootstrap"] == -1).all()
    assert len(df) == 3


def test_wrong_closed_answer_lowers_accuracy(synth_vqa):
    ds, refs = synth_vqa

    def flip_closed(imgs, questions):
        out = []
        for q in questions:
            ans = refs[q]
            out.append({"yes": "no", "no": "yes"}.get(ans, ans))
        return out

    ev = VQAEvaluator(flip_closed, dataset=ds, batch_size=4, num_workers=0)
    df = ev.evaluate()
    closed = df[df["label"] == "closed"].iloc[0]
    assert closed["accuracy"] == pytest.approx(0.0)


def test_bootstrap_row_count(synth_vqa):
    ds, refs = synth_vqa

    def perfect(imgs, questions):
        return [refs[q] for q in questions]

    ev = VQAEvaluator(perfect, dataset=ds, batch_size=4, num_workers=0, n_bootstrap=5)
    df = ev.evaluate()
    # The "overall" bucket is always present (non-empty pairs): 1 point
    # estimate + 5 bootstrap resamples = 6 rows. closed/open buckets can be
    # absent in a resample that happens to draw only one answer type, so the
    # total is <= 3 * 6 = 18.
    assert (df["label"] == "overall").sum() == 6
    assert sorted(df["bootstrap"].unique()) == [-1, 0, 1, 2, 3, 4]
    assert len(df) <= 18


def test_answerer_ignoring_context_still_works(synth_vqa):
    # A legacy single-arg generator (imgs only) must still run via the base's
    # arity detection.
    ds, refs = synth_vqa

    def one_arg(imgs):
        return ["yes"] * len(imgs)

    ev = VQAEvaluator(one_arg, dataset=ds, batch_size=4, num_workers=0)
    df = ev.evaluate()
    assert (df["label"] == "overall").sum() == 1


# ── report-generation path (no RadEval): generation + indication plumbing ───
class _StubReportDataset:
    """Minimal stand-in exposing get_datasets(n_splits, num_cores)."""

    def __init__(self, samples):
        self._samples = samples

    def get_datasets(self, n_splits=None, num_cores=0):
        return self._samples


def test_report_generate_only_and_indication(tmp_path):
    from radharmony.evaluator.language.report_generation import (
        ReportGenerationEvaluator,
    )

    report = tmp_path / "r0.txt"
    report.write_text(
        "INDICATION: 60F with fever.\n"
        "FINDINGS: Patchy right base opacity.\n"
        "IMPRESSION: Pneumonia."
    )
    ds = _StubReportDataset([{"img": "r0.png", "report": str(report)}])

    seen = {}

    def gen(img_paths, indications=None):
        seen["indications"] = indications
        return ["Right basilar consolidation."] * len(img_paths)

    ev = ReportGenerationEvaluator(
        gen, dataset=ds, ref_section="findings", use_indication=True,
        num_workers=0, output_dir=str(tmp_path),
    )
    out = ev.generate_only(str(tmp_path / "pairs.parquet"))
    df = pd.read_parquet(out)

    assert list(df.columns) == ["sample_id", "reference", "hypothesis"]
    assert df.iloc[0]["reference"] == "Patchy right base opacity."
    assert df.iloc[0]["hypothesis"] == "Right basilar consolidation."
    # The parsed indication was passed to the generator.
    assert seen["indications"] == ["60F with fever."]
    assert ev.n_samples_ == 1


def test_report_no_indication_passes_none(tmp_path):
    from radharmony.evaluator.language.report_generation import (
        ReportGenerationEvaluator,
    )

    report = tmp_path / "r1.txt"
    report.write_text("FINDINGS: Clear lungs.\nIMPRESSION: Normal.")
    ds = _StubReportDataset([{"img": "r1.png", "report": str(report)}])

    seen = {}

    def gen(img_paths, indications=None):
        seen["indications"] = indications
        return ["Normal chest."] * len(img_paths)

    ev = ReportGenerationEvaluator(gen, dataset=ds, num_workers=0)
    ev.generate_only(str(tmp_path / "p.parquet"))
    # use_indication defaults False → contexts all None.
    assert seen["indications"] == [None]


# ── report parser: indication section ───────────────────────────────────────
def test_parse_indication_section():
    from radharmony.evaluator.language._report_parser import parse_report_section

    report = (
        "INDICATION: 55M with cough.\n"
        "TECHNIQUE: PA and lateral.\n"
        "FINDINGS: No acute cardiopulmonary process.\n"
        "IMPRESSION: Normal chest."
    )
    assert parse_report_section(report, "indication") == "55M with cough."
    # No indication header → empty (never falls back to the body).
    assert parse_report_section("FINDINGS: Clear lungs.", "indication") == ""
    # Sanity: findings/impression still parse.
    assert parse_report_section(report, "findings") == "No acute cardiopulmonary process."
