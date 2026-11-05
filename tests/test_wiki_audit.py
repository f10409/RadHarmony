"""Tests for the ``<!-- args: X -->`` line of the wiki audit (scripts/evalkit/wiki_audit.py)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "evalkit"))

import wiki_audit  # noqa: E402

WIKI = ROOT / "docs" / "wiki"

# The CheXpert Plus harmonizer has csv_path, label_json_path and base_image_dir. The
# dataset class has all 3 too, with csv_path=None. So a guess from the names is a tie.
_HARMONIZER_TABLE = """\
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | The CSV |
| `label_json_path` | `str` | No | `None` | The labels |
| `base_image_dir` | `str` | No | `None` | The folder |
"""


@pytest.fixture(scope="module")
def tm():
    return wiki_audit.build_truth(WIKI)


def _audit(tmp_path, tm, text):
    page = tmp_path / "wiki" / "p.md"
    page.parent.mkdir(exist_ok=True)
    page.write_text(text)
    return wiki_audit.audit_page(page, page.parent, tm)


def test_args_line_names_the_class_of_the_table(tmp_path, tm):
    """Without the line, the tie goes to the dataset class, which gets the findings."""
    imports = "```python\nfrom radharmony.dataset import CheXpertPlusDataset\n" \
              "from radharmony.harmonizer import CheXpertPlusHarmonizer\n```\n\n## Harmonizer\n\n"
    guessed = _audit(tmp_path, tm, imports + _HARMONIZER_TABLE)
    assert "missing-default" in {f.type for f in guessed}
    assert {f.symbol for f in guessed} == {"CheXpertPlusDataset"}
    named = _audit(tmp_path, tm, imports + "<!-- args: radharmony.harmonizer.CheXpertPlusHarmonizer -->\n"
                   + _HARMONIZER_TABLE)
    assert named == []


def test_args_line_that_does_not_resolve_is_an_error(tmp_path, tm):
    found = _audit(tmp_path, tm, "## Harmonizer\n\n<!-- args: radharmony.harmonizer.NoSuchHarmonizer -->\n"
                   + _HARMONIZER_TABLE)
    assert [f.type for f in found] == ["args-line-unresolved"]


def test_a_subtable_with_its_own_args_line_is_checked_alone(tmp_path, tm):
    """A ``###`` table under another class's section starts a new section when its line names another class."""
    text = ("## Arguments\n\n<!-- args: radharmony.harmonizer.CheXpertPlusHarmonizer -->\n" + _HARMONIZER_TABLE
            + "\n### Train only\n\n<!-- args: radharmony.harmonizer.CheXpertHarmonizer -->\n"
            "| Argument | Type | Required | Default | Description |\n|---|---|---|---|---|\n"
            "| `demographics_path` | `str` | No | `None` | The workbook |\n")
    found = _audit(tmp_path, tm, text)
    # CheXpertHarmonizer also has csv_path and base_image_dir, which the second table does not list.
    assert sorted(f.symbol for f in found if f.type == "missing-arg-row") == ["CheXpertHarmonizer"] * 2


_SPEC = ("| Argument | Type | Required | Default | Description |\n|---|---|---|---|---|\n"
         "| `csv_path` | `str` | No | `None` | The CSV |\n"
         "| `label_json_path` | `str` | No | `None` | The labels |\n")


def test_an_args_line_that_no_table_uses_is_an_error(tmp_path, tm):
    """A typo or text between the line and the table must not hide the line (F1)."""
    for line in ("<!-- args: radharmony.harmonizer.CheXpertPlusHarmonizer -->\nSome text.\n",
                 "<!-- arg: radharmony.harmonizer.CheXpertPlusHarmonizer -->\n"):
        found = _audit(tmp_path, tm, "## Harmonizer\n\n" + line + "\n" + _HARMONIZER_TABLE)
        assert ("args-line-orphan", 3) in {(f.type, f.line) for f in found}, line


def test_unresolved_args_line_gives_the_line_of_the_comment(tmp_path, tm):
    found = _audit(tmp_path, tm, "## Harmonizer\n\n<!-- args: radharmony.harmonizer.NoSuch -->\n\n"
                   + _HARMONIZER_TABLE)
    assert [(f.type, f.line) for f in found] == [("args-line-unresolved", 3)]


def test_args_line_on_the_shared_table_names_the_whole_section(tmp_path, tm):
    """The specific table and its shared table are one section of the line's class (F2)."""
    base = "## Arguments\n\n" + _SPEC + "\n### Shared arguments (inherited from BaseRadiologicalDataset)\n\n"
    found = _audit(tmp_path, tm, base + "<!-- args: radharmony.dataset.CheXpertPlusTrainDataset -->\n"
                   + _SPEC.replace("csv_path", "cache_dir").replace("label_json_path", "dtype"))
    assert {f.symbol for f in found} == {"CheXpertPlusTrainDataset"}
    # One section: only the rows that neither table lists are missing.
    missing = {f.detail.split("`")[1] for f in found if f.type == "missing-arg-row"}
    assert "csv_path" not in missing and "cache_dir" not in missing and "base_image_dir" in missing


def test_an_args_line_inside_a_code_fence_is_skipped(tmp_path, tm):
    text = "```markdown\n<!-- args: radharmony.dataset.MyDataset -->\n" + _HARMONIZER_TABLE + "```\n"
    assert _audit(tmp_path, tm, text) == []
