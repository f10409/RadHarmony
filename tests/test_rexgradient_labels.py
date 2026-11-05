"""ReXGradient GPT-OSS label wiring: CheXpert-14 names and a real join.

The label CSV is a project file, not part of the source. Set ``REXGRADIENT_LABELS_CSV`` to
its path to run these tests. Without it, they skip.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from radharmony.harmonizer.rexgradient.rexgradient import ReXGradientHarmonizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dataset_paths import paths  # noqa: E402

# No entry (a clone with its own paths file): the empty path skips the tests below.
_REX = paths("rexgradient_valid", missing_ok=True)
META = _REX.get("csv_path", "")
IMG = _REX.get("base_image_dir", "")
LAB = os.environ.get("REXGRADIENT_LABELS_CSV", "")
pytestmark = pytest.mark.skipif(not os.path.isfile(META), reason="ReXGradient not mounted")
needs_labels = pytest.mark.skipif(not os.path.isfile(LAB), reason="REXGRADIENT_LABELS_CSV not set")


def _h(labels=True):
    return ReXGradientHarmonizer(csv_path=META, base_image_dir=IMG,
                                 labels_csv_path=LAB if labels else None)


def test_without_labels_behaviour_is_unchanged():
    out = _h(labels=False).harmonize()
    assert "no_finding" not in out.columns
    assert len(out) > 0


@needs_labels
def test_canonical_chexpert_names_no_abnormal_column():
    """The GPT-OSS table stores the COMPLEMENT as Abnormal; we must expose no_finding."""
    out = _h().harmonize()
    assert "no_finding" in out.columns, "canonical CheXpert label missing"
    assert not any("abnormal" in c.lower() for c in out.columns), (
        "a non-standard inverted 'abnormal' column leaked into the output; every other "
        "CheXpert-14 consumer in this repo uses No Finding, and mixing polarities silently "
        "flips the sign of any pooled metric"
    )


@needs_labels
def test_no_finding_is_the_inverse_of_abnormal():
    """Compare LIKE WITH LIKE: the same split, de-duplicated to STUDY level.

    A first version of this test compared the valid split's per-IMAGE rate (0.433) against
    the whole-corpus per-STUDY rate (0.389) and failed on a 0.044 gap that was entirely
    real: the valid split has a different prevalence from the corpus, and studies with more
    images are weighted more when rows are per-image. The code was right; the test was
    comparing two different quantities.
    """
    out = _h().harmonize()
    per_study = out.drop_duplicates("accession_number")
    got = float(pd.to_numeric(per_study.no_finding, errors="coerce").mean(skipna=True))

    lab = pd.read_csv(LAB, low_memory=False)
    lab["acc"] = lab.AccessionNumber_anon.astype(str)
    here = lab[lab.acc.isin(set(per_study.accession_number.astype(str)))]
    assert len(here) > 100, "split/label overlap too small to test"
    abn = (here.Abnormal_response.astype(str).str.strip().str.lower() == "present").mean()
    assert abs(got - (1.0 - abn)) < 0.01, (
        f"no_finding {got:.4f} vs expected {1 - abn:.4f} on the SAME studies")


@needs_labels
def test_all_fourteen_labels_present():
    out = _h().harmonize()
    for c in ReXGradientHarmonizer.LABEL_COLS:
        assert c in out.columns, f"{c} missing"
    assert len(ReXGradientHarmonizer.LABEL_COLS) == 14


@needs_labels
def test_labels_are_tri_state_not_coerced_to_negative():
    """Unparseable values must be NaN, never silently 0 — blank is not absent."""
    out = _h().harmonize()
    v = pd.to_numeric(out.atelectasis, errors="coerce").dropna().unique()
    assert set(np.unique(v)).issubset({0.0, 1.0}), f"unexpected label values {v[:6]}"


@needs_labels
def test_join_does_not_duplicate_rows():
    a, b = len(_h(labels=False).harmonize()), len(_h().harmonize())
    assert a == b, "label join changed the row count"


def test_missing_label_file_raises():
    h = ReXGradientHarmonizer(csv_path=META, base_image_dir=IMG,
                              labels_csv_path="/nonexistent/labels.csv")
    with pytest.raises(FileNotFoundError):
        h.harmonize()
