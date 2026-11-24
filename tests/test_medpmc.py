"""MedPMC harmonizer tests. No real data.

The expected values come from the fake index and tag table, not from the code.

Run with::

    .venv/bin/python -m pytest tests/test_medpmc.py -v
"""
from __future__ import annotations

import pandas as pd
import pytest

from radharmony.harmonizer import MedPMCHarmonizer


@pytest.fixture()
def root(tmp_path):
    """One single-panel and one multi-panel figure, both tagged radiology, as in the real tag table.

    The real tag table has no ``contains_specialty`` column. The tag of a multi-panel row comes from the
    caption of the whole figure.
    """
    (tmp_path / "webdataset" / "metadata").mkdir(parents=True)
    pd.DataFrame({"source_type": ["singlepanel", "multipanel"], "pmcid": ["PMC1", "PMC2"],
                  "image_id": ["s1", "m1"], "parent_image_id": [None, None],
                  "subfigure_index": [None, None], "figure_label": ["1", "2"],
                  "shard": ["a", "b"], "key": ["k1", "k2"], "image_ext": [".jpg", ".jpg"]},
                 ).to_parquet(tmp_path / "webdataset" / "metadata" / "medpmc_index.parquet")
    (tmp_path / "tags").mkdir()
    pd.DataFrame({"image_id": ["s1", "m1"], "caption": ["CT of chest", "A: chest CT. B: brain MRI"],
                  "specialty": ["radiology", "radiology"], "modality": ["ct", "ct"],
                  "tag_confidence": ["high", "high"], "tag_source": ["caption", "caption"],
                  "tag_method": ["lex", "lex"]},
                 ).to_parquet(tmp_path / "tags" / "medpmc_tags.parquet")
    return str(tmp_path)


@pytest.mark.parametrize("include,expected", [(True, ["m1", "s1"]), (False, ["s1"])])
def test_include_multipanel_false_gives_a_panel_level_set(root, include, expected):
    """V-H1-47: with a specialty filter, include_multipanel=False drops the multi-panel rows."""
    df = MedPMCHarmonizer(base_dir=root, radiology_only=True, include_multipanel=include).harmonize()
    assert sorted(df["question_id"]) == expected


def test_include_multipanel_false_with_no_specialty_filter_keeps_every_row(root):
    """With no specialty filter, the flag does nothing. ``subsets=`` picks the figure types."""
    df = MedPMCHarmonizer(base_dir=root, include_multipanel=False).harmonize()
    assert sorted(df["question_id"]) == ["m1", "s1"]
