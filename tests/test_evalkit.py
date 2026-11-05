"""Tests for the automatic checks in scripts/evalkit/ and the /evaluate merge script."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "evalkit"))
sys.path.insert(0, str(ROOT / ".claude" / "skills" / "evaluate" / "scripts"))

import base_guard  # noqa: E402
import blocks  # noqa: E402
import claims  # noqa: E402
import dataset_stat  # noqa: E402
import merge  # noqa: E402
import problems  # noqa: E402
import run_blocks  # noqa: E402
import ste_lint  # noqa: E402


# --------------------------------------------------------------------------
# STE linter: one bad and one good text per rule
# --------------------------------------------------------------------------
def _problems(md: str) -> list[dict]:
    return ste_lint.lint_units(ste_lint.md_units(md), "x.md", markdown=True)


def _rules(md: str) -> set[str]:
    return {p["rule"] for p in _problems(md)}


@pytest.mark.parametrize("rule,bad,good", [
    ("STE-01", " ".join(["word"] * 26) + ".", " ".join(["word"] * 25) + "."),
    ("STE-02", "Load the data — then run it.", "Load the data. Then run it."),
    ("STE-02", "Load the data -- then run it.", "Run `--full` to check."),
    ("STE-03", "Load the data; run it.", "Load the data. Run it."),
    ("STE-04", "A a. B b. C c. D d. E e. F f. G g.", "A a. B b. C c. D d. E e. F f."),
    ("STE-06", "We utilize the harmonizer.", "We use the harmonizer."),
    ("STE-10", "The normalisation step.", "The normalization step."),
    ("STE-11", "The class will load the data.", "The class loads the data."),
])
def test_ste_rule(rule, bad, good):
    assert rule in _rules(bad)
    assert rule not in _rules(good)


def test_ste_skips_code_and_comments():
    md = "Text.\n\n```python\nx = 1; y = 2  # will\n```\n\n<!-- a; b — c -->\nUse `a; b`.\n"
    assert _rules(md) == set()


def test_ste_python_units():
    src = ('"""Load the data — fast."""\n'
           "# This will run.\n"
           "x = 1  # x = compute(y)\n")
    rules = {p["rule"] for p in ste_lint.lint_units(ste_lint.py_units(src), "x.py", markdown=False)}
    assert rules == {"STE-02", "STE-11"}  # the code-like comment is skipped


def test_ste_docstring_sections_are_not_prose():
    doc = "Load data.\n\nArgs:\n    path (str): The file.\n\nExample::\n\n    a = 1; b = 2\n"
    texts = [t for _, t, _ in ste_lint.docstring_units(doc, 1)]
    assert texts == ["Load data.", "path (str): The file.", "Example::"]


# --------------------------------------------------------------------------
# Known-problems list
# --------------------------------------------------------------------------
@pytest.fixture
def known_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(problems, "KNOWN_DIR", tmp_path)
    return tmp_path


def test_key_ignores_line_moves_and_addresses():
    a = problems.problem("ruff", "F401", "a.py", 3, "`os` imported but unused (line 3)")
    b = problems.problem("ruff", "F401", "a.py", 9, "`os` imported but unused (line 9)")
    assert problems.key(a) == problems.key(b)
    a = problems.problem("blocks_run", "BLOCK-RUN", "p.md", 1, "TypeError: <obj at 0x7f49a6778690>")
    b = problems.problem("blocks_run", "BLOCK-RUN", "p.md", 1, "TypeError: <obj at 0x7e9f2eb78690>")
    assert problems.key(a) == problems.key(b)


def test_ste_problem_keeps_only_its_sentence():
    before = _problems("Load it \u2014 fast. Then run it.")
    after = _problems("Load it \u2014 fast. Then run it. Text.")
    assert [problems.key(p) for p in before] == [problems.key(p) for p in after]


def test_compare_counts_copies(known_dir):
    p = problems.problem("ste", "STE-02", "a.md", 1, "em dash")
    problems.save_known("ste", [p, p])
    assert problems.compare("ste", [p, p]) == ([], 0)
    new, fixed = problems.compare("ste", [p, p, p])
    assert len(new) == 1 and fixed == 0
    assert problems.compare("ste", [p]) == ([], 1)


def test_shrink_errors(known_dir, monkeypatch):
    old = problems.problem("ste", "STE-02", "a.md", 1, "old")
    new = problems.problem("ste", "STE-02", "a.md", 2, "new")
    problems.save_known("ste", [old])
    old_list = problems.load_known("ste")
    problems.save_known("ste", [old, new])
    now = problems.load_known("ste")
    # A list that grew gives one error per new key.
    monkeypatch.setattr(problems, "load_known", lambda check, ref=None: old_list if ref else now)
    assert len(problems.shrink_errors("HEAD")) == 1
    monkeypatch.setattr(problems, "load_known", lambda check, ref=None: None if ref else now)
    assert problems.shrink_errors("HEAD") == []  # the list is new in this branch


def test_shrink_errors_follow_a_moved_file(known_dir, monkeypatch):
    """A moved page keeps its old problems: they are not new. A new problem in it still counts."""
    load = problems.load_known
    old = problems.problem("ste", "STE-02", "docs/wiki/a.md", 3, "em dash at line 3")
    problems.save_known("ste", [old])
    old_list = load("ste")
    moved = problems.problem("ste", "STE-02", "docs/wiki/how-to/a.md", 9, "em dash at line 9")
    problems.save_known("ste", [moved])
    now = load("ste")
    monkeypatch.setattr(problems, "load_known", lambda check, ref=None: old_list if ref else now)
    monkeypatch.setattr(problems, "_moved_files", lambda ref: {"docs/wiki/how-to/a.md": "docs/wiki/a.md"})
    assert problems.shrink_errors("HEAD") == []
    extra = problems.problem("ste", "STE-03", "docs/wiki/how-to/a.md", 5, "semicolon")
    problems.save_known("ste", [moved, extra])
    now = load("ste")
    assert len(problems.shrink_errors("HEAD")) == 1


def test_shrink_errors_moved_file_counts_are_summed(known_dir, monkeypatch):
    """A moved file fails when a problem gets more copies (C), also when 2 old entries share their first 100 chars (D)."""
    load = problems.load_known
    head = "x" * 100
    a = problems.problem("ste", "STE-01", "a.md", 1, head + " first")
    b = problems.problem("ste", "STE-01", "a.md", 2, head + " second")
    problems.save_known("ste", [a, b])
    old_list = load("ste")
    moved = [problems.problem("ste", "STE-01", "b.md", n, head + tail) for n, tail in
             [(1, " first"), (2, " second")]]
    monkeypatch.setattr(problems, "_moved_files", lambda ref: {"b.md": "a.md"})
    problems.save_known("ste", moved)
    now = load("ste")
    monkeypatch.setattr(problems, "load_known", lambda check, ref=None: old_list if ref else now)
    assert problems.shrink_errors("HEAD") == []
    problems.save_known("ste", moved * 2)  # each problem 2 times: 4 against 2 old ones
    now = load("ste")
    assert len(problems.shrink_errors("HEAD")) == 1


def test_moved_files_reads_git(tmp_path, monkeypatch):
    """_moved_files reads real git output, also a path with a non-ASCII letter."""
    import subprocess

    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=tmp_path,
                       check=True, capture_output=True)
    git("init", "-q")
    (tmp_path / "a.md").write_text("one line of text\n" * 5)
    (tmp_path / "c.md").write_text("another page\n" * 5)
    git("add", ".")
    git("commit", "-qm", "x")
    git("mv", "a.md", "how-to.md")
    git("mv", "c.md", "café.md")
    monkeypatch.setattr(problems, "REPO", tmp_path)
    assert problems._moved_files("HEAD") == {"how-to.md": "a.md", "café.md": "c.md"}


def test_check_shrink_says_when_the_ref_has_no_lists(monkeypatch, capsys):
    """A ref with no known-problems list compares nothing. The output must say so, not only 'did not grow'.

    Before 2026-11-03, `--check-shrink origin/main` passed in silence: main had no lists.
    """
    import run_all
    monkeypatch.setattr(problems, "load_known", lambda check, ref=None: None)
    assert run_all.main(["--check-shrink", "HEAD"]) == 0
    assert "nothing was compared" in capsys.readouterr().out


def test_shrink_errors_bad_ref():
    assert problems.shrink_errors("no_such_branch_xyz") == ["no_such_branch_xyz is not a commit"]


def test_shrink_errors_for_an_old_rule_with_no_entry(known_dir, monkeypatch):
    """V-A-38 (D-45): a rule with no entry in the old list joins it only when it is named as new."""
    old = problems.problem("lessons", "L-006", "a.md", 1, "old")
    nas = problems.problem("lessons", "L-002", "b.md", 1, "NAS path: /mnt/x")
    load = problems.load_known
    problems.save_known("lessons", [old])
    old_list = load("lessons")
    problems.save_known("lessons", [old, nas])
    now = load("lessons")
    monkeypatch.setattr(problems, "load_known", lambda check, ref=None: old_list if ref else now)
    assert len(problems.shrink_errors("HEAD")) == 1
    assert problems.shrink_errors("HEAD", new_rules=["L-002"]) == []
    # L-006 has an entry in the old list, so it is not new, even when the PR names it.
    more = problems.problem("lessons", "L-006", "c.md", 1, "a new L-006 problem")
    problems.save_known("lessons", [old, more])
    now = load("lessons")
    assert len(problems.shrink_errors("HEAD", new_rules=["L-006"])) == 1


def test_new_rules_come_from_the_pr_description():
    """D-45: each 'New rule: <ID>' line of the PR description names a new rule for CI."""
    body = "## Summary\nAdd L-013.\n\n- New rule: L-013\n**New rule:** L-014 (box labels)\nno New rule: here\n"
    assert problems.new_rules_in(body) == ["L-013", "L-014"]
    assert problems.new_rules_in("## Summary\nNo rule.\n") == []
    # A line in a code block is an example. An ID in backticks counts.
    assert problems.new_rules_in("```\nNew rule: L-002\n```\n- New rule: `L-013`\n") == ["L-013"]


def test_run_all_with_an_unknown_group_fails():
    """V-A-40: a wrong group name is an error, not a PASS with 0 checks."""
    import subprocess

    proc = subprocess.run([sys.executable, "scripts/evalkit/run_all.py", "--group", "lintt"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "lintt" in proc.stderr


# --------------------------------------------------------------------------
# Code blocks
# --------------------------------------------------------------------------
def test_block_notes_and_indent():
    md = ("<!-- blocks: run=local data=chexpert_train -->\n\n"
          "<!-- block: run=skip reason=\"needs a GPU\" -->\n```python\nx = 1\n```\n\n"
          "- item\n\n  ```python\n  y = 2\n  ```\n")
    b1, b2 = blocks.extract_blocks(md)
    assert b1.opts == {"run": "skip", "data": "chexpert_train", "reason": "needs a GPU"}
    assert b2.opts == {"run": "local", "data": "chexpert_train"}
    assert b2.code == "y = 2"


def test_static_block_problems(tmp_path, monkeypatch):
    monkeypatch.setattr(run_blocks, "REPO", tmp_path)
    monkeypatch.setattr(blocks, "REPO", tmp_path)
    page = tmp_path / "p.md"
    page.write_text("```python\nx = (\n```\n\n<!-- block: run=skip -->\n```python\nok = 1\n```\n")
    rules = sorted(p["rule"] for p in run_blocks.static_problems(page))
    assert rules == ["BLOCK-NOTE", "BLOCK-SYNTAX"]


def test_static_block_imports_and_keywords(tmp_path, monkeypatch):
    """L-005: a radharmony import that does not exist, and a keyword that the class does not take, are problems.

    The import in the first block counts for the calls in the other blocks. In a pycon block, the
    line is the Markdown line of the call, not of the output above it.
    """
    monkeypatch.setattr(run_blocks, "REPO", tmp_path)
    monkeypatch.setattr(blocks, "REPO", tmp_path)
    page = tmp_path / "p.md"
    page.write_text("```python\nfrom radharmony.harmonizer import NoSuchHarmonizer\n"
                    "from radharmony.harmonizer import ReXGradientHarmonizer\n```\n\n"
                    "```python\nh = ReXGradientHarmonizer(csv_path='a.json', output_mask=True)\n"
                    "h2 = ReXGradientHarmonizer(csv_path='a.json', base_image_dir='x')\n```\n\n"
                    "```pycon\n>>> 1 + 1\n2\n>>> ReXGradientHarmonizer(csv_path='a.json', mask=1)\n```\n")
    got = sorted((p["rule"], p["line"], p["text"]) for p in run_blocks.static_problems(page))
    assert got == [
        ("BLOCK-IMPORT", 2, "`from radharmony.harmonizer import NoSuchHarmonizer` does not resolve"),
        ("BLOCK-KWARG", 7, "`ReXGradientHarmonizer(...)` passes `output_mask=`, which is not a parameter"),
        ("BLOCK-KWARG", 14, "`ReXGradientHarmonizer(...)` passes `mask=`, which is not a parameter"),
    ]


def test_run_page_marks_later_blocks_not_run(tmp_path, monkeypatch):
    monkeypatch.setattr(run_blocks, "REPO", tmp_path)
    monkeypatch.setattr(blocks, "REPO", tmp_path)
    page = tmp_path / "p.md"
    page.write_text("<!-- blocks: run=ci -->\n"
                    "```python\nx = 1\n```\n\n```python\nraise TypeError('bad kwarg')\n```\n\n"
                    "```python\nprint(x)\n```\n")
    got = [(r["result"], r["detail"][:9]) for r in run_blocks.run_page(page, "ci")]
    assert got == [("ok", ""), ("failed", "TypeError"), ("not_run", "depends o")]


def test_classify_missing_data():
    rec = {"status": "error", "exc": "FileNotFoundError", "msg": "/data/vindr/train.csv", "module": ""}
    assert run_blocks._classify(rec, "full")[0] == "blocked"
    assert run_blocks._classify(rec, "ci")[0] == "failed"  # a CI block must not need data
    rec = {"status": "error", "exc": "ModuleNotFoundError", "msg": "no kagglehub", "module": "kagglehub"}
    assert run_blocks._classify(rec, "full") == ("blocked", "needs package kagglehub")


def test_block_output_does_not_hide_records(tmp_path, monkeypatch):
    monkeypatch.setattr(run_blocks, "REPO", tmp_path)
    monkeypatch.setattr(blocks, "REPO", tmp_path)
    page = tmp_path / "p.md"
    page.write_text("<!-- blocks: run=ci -->\n```python\nprint('loading...', end='')\n```\n")
    assert [r["result"] for r in run_blocks.run_page(page, "ci")] == ["ok"]


def test_stuck_block_times_out_once(tmp_path, monkeypatch):
    monkeypatch.setattr(run_blocks, "REPO", tmp_path)
    monkeypatch.setattr(blocks, "REPO", tmp_path)
    page = tmp_path / "p.md"
    page.write_text("<!-- blocks: run=ci timeout=1 -->\n"
                    "```python\nimport time\ntry:\n    time.sleep(30)\nexcept Exception:\n    pass\n```\n\n"
                    "```python\nx = 1\n```\n")
    assert [r["result"] for r in run_blocks.run_page(page, "ci")] == ["timeout", "not_run"]


def test_bad_timeout_note(tmp_path, monkeypatch):
    monkeypatch.setattr(run_blocks, "REPO", tmp_path)
    monkeypatch.setattr(blocks, "REPO", tmp_path)
    page = tmp_path / "p.md"
    page.write_text("<!-- block: timeout=2m -->\n```python\nx = 1\n```\n")
    assert [p["rule"] for p in run_blocks.static_problems(page)] == ["BLOCK-NOTE"]
    assert run_blocks.run_page(page, "full") == []


# --------------------------------------------------------------------------
# Claims
# --------------------------------------------------------------------------
def test_claim_pass_and_fail():
    md = ('<!-- claim: id=n cmd="python -c \'print(\\"4,555\\")\'" expect="4,555" -->\n'
          "The set has 4,555 images.\n")
    (line, note, para), = claims.find_claims(md)
    assert claims.check_claim("p.md", line, note, para) == []
    note["expect"] = "4,556"
    rules = sorted(p["rule"] for p in claims.check_claim("p.md", line, note, para))
    assert rules == ["CLAIM-TEXT", "CLAIM-WRONG"]


# --------------------------------------------------------------------------
# Base-code check
# --------------------------------------------------------------------------
def test_base_files_match_claude_md():
    text = (ROOT / "CLAUDE.md").read_text()
    section = text.split("## Base code needs explicit approval", 1)[1].split("First look", 1)[0]
    listed = set(re.findall(r"`(radharmony/[\w/]+\.py)`", section))
    assert listed == set(base_guard.BASE_FILES)


def test_base_guard_texts():
    body = "## ⚠️ Base code change\n- Files: x\n- Approved by: Frank on 2026-10-27\n"
    assert base_guard.pr_body_ok(body)
    assert not base_guard.pr_body_ok("## Summary\n" + body)  # the section must come first
    # V-A-39: an empty approver line or the template text of CLAUDE.md is no approval.
    assert not base_guard.pr_body_ok("## ⚠️ Base code change\n- Approved by:\n- Can affect: all\n")
    assert not base_guard.pr_body_ok(
        "## ⚠️ Base code change\n- Approved by: <name of the user who approved> on <date>\n")
    for line in ("Approved by: on 2026-11-01", "Approved by: TBD", "Approved by: pending"):
        assert not base_guard.pr_body_ok(f"## ⚠️ Base code change\n- {line}\n"), line
    for line in ("**Approved by**: Frank Li on 2026-11-01", "**Approved by:** Frank Li on 2026-11-01"):
        assert base_guard.pr_body_ok(f"## ⚠️ Base code change\n- {line}\n"), line
    base = ["radharmony/dataset/base.py"]
    assert base_guard.commit_msg_ok("Fix x\n\nBASE CODE CHANGE: radharmony/dataset/base.py", base)
    assert not base_guard.commit_msg_ok("Fix x\n\nBASE CODE CHANGE: transforms", base)
    assert not base_guard.commit_msg_ok("Fix x\n# BASE CODE CHANGE: radharmony/dataset/base.py", base)
    assert base_guard.base_files(["radharmony/dataset/base.py", "README.md"]) == ["radharmony/dataset/base.py"]


# --------------------------------------------------------------------------
# /evaluate merge: a finding ID means one finding in all runs
# --------------------------------------------------------------------------
def _run(old=(), correct=(), rules=(), status="still_open"):
    return {"old": [{"old_id": i, "status": status, "title": "old"} for i in old],
            "correct": [{"id": i, "title": "c"} for i in correct],
            "rules": [{"id": i, "why": "r"} for i in rules], "asked": []}


def test_merge_ids_stay_unique_across_runs(tmp_path):
    index = tmp_path / "index.json"
    merge.update_index(_run(old=["V-D-1"], correct=["V-D-2"], rules=["R-code-1"]), "run1", index, "audit")
    second = _run(old=["V-D-1", "V-D-2"], correct=["V-D-2", "V-D-3"], rules=["R-code-1"])
    merge.update_index(second, "run2", index, "run1")
    # V-D-2 and R-code-1 are taken by run1, so they get the next free numbers.
    assert [f["id"] for f in second["correct"]] == ["V-D-4", "V-D-3"]
    assert second["correct"][0]["agent_id"] == "V-D-2"
    assert second["rules"][0]["id"] == "R-code-2"
    ids = json.loads(index.read_text())["ids"]
    assert {k: v["run"] for k, v in ids.items()} == {
        "R-code-1": "run1", "R-code-2": "run2", "V-D-1": "audit", "V-D-2": "run1", "V-D-3": "run2", "V-D-4": "run2"}
    assert ids["V-D-2"]["status"] == "still_open" and ids["V-D-3"]["status"] == "open"
    # A second merge of the same run gives the same IDs.
    again = _run(old=["V-D-1", "V-D-2"], correct=["V-D-2", "V-D-3"], rules=["R-code-1"])
    merge.update_index(again, "run2", index, "run1")
    assert again == second and json.loads(index.read_text())["ids"] == ids


def test_merge_again_keeps_ids_after_a_later_run(tmp_path):
    index = tmp_path / "index.json"
    first = _run(old=["V-D-1"], correct=["V-D-1", "V-D-2"])
    merge.update_index(first, "run1", index, "audit")
    assert [f["id"] for f in first["correct"]] == ["V-D-3", "V-D-2"]
    merge.update_index(_run(old=["V-D-3"], status="fixed"), "run2", index, "run1")
    # Merge run1 again, with V-D-2 dropped (step 5 of /evaluate can drop a finding).
    again = _run(old=["V-D-1"], correct=["V-D-1"])
    merge.update_index(again, "run1", index, "audit")
    assert again["correct"][0]["id"] == "V-D-3"
    ids = json.loads(index.read_text())["ids"]
    assert ids["V-D-3"]["status"] == "fixed" and ids["V-D-3"]["checked_in"] == "run2"
    assert "V-D-2" not in ids


def test_merge_renames_lesson_evidence_and_rejects_a_bad_index(tmp_path):
    index = tmp_path / "index.json"
    merge.update_index(_run(correct=["V-D-1"]), "run1", index, "audit")
    data = _run(correct=["V-D-1"])
    data["lessons"] = [{"what": "x", "evidence": ["V-D-1"], "source": "correct/D"},
                       {"what": "y", "evidence": ["V-D-1"], "source": "rules/code"}]
    merge.update_index(data, "run2", index, "run1")
    assert [x["evidence"] for x in data["lessons"]] == [["V-D-2"], ["V-D-1"]]
    assert merge._label(data["correct"][0]) == "**V-D-2** (agent ID V-D-1)"
    index.write_text("")
    with pytest.raises(SystemExit, match="not valid JSON"):
        merge.update_index(_run(correct=["V-D-1"]), "run3", index, "run2")



def test_merge_keeps_bugs_outside_the_diff(tmp_path):
    out = tmp_path / "2026-10-28-branch"
    (out / "correct" / "branch_check").mkdir(parents=True)
    finding = {"title": "t", "severity": "low", "files": ["a.md:9"], "command": "c"}
    (out / "correct" / "branch_check" / "checked.json").write_text(json.dumps(
        {"part": "branch", "kept": [dict(finding, id="V-branch-1")],
         "outside_diff": [dict(finding, id="V-branch-2", title="old bug")]}))
    data = merge.collect(out)
    merge.update_index(data, out.name, tmp_path / "index.json", "earlier run")
    assert [f["id"] for f in data["outside"]] == ["V-branch-2"]
    assert json.loads((tmp_path / "index.json").read_text())["ids"]["V-branch-2"]["review"] == "outside"
    assert "### Outside the diff" in merge.draft(data, "x", None)


def test_l002_checks_the_skills():
    """D-21: L-002 (no NAS path) also reads .claude/skills/, because agents read and copy the skills."""
    import lessons

    files = lessons.l002_files()
    assert ".claude/skills/fix/SKILL.md" in files
    assert "docs/wiki/design.md" in files


def test_dataset_stat_reads_expected_stats(tmp_path, monkeypatch):
    """A claim prints a value of expected_stats as the docs write it."""
    cfg = {"expected_stats": {"a_train": {"images": 138358, "age_years": [0, 90]},
                              "a_valid": {"images": 234}}}
    (tmp_path / "a.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(dataset_stat, "CONFIGS", tmp_path)
    assert dataset_stat.fmt(dataset_stat.stat("a", "a_train.images")) == "138,358"
    assert dataset_stat.fmt(dataset_stat.stat("a", "a_train.images+a_valid.images")) == "138,592"
    assert dataset_stat.fmt(dataset_stat.stat("a", "a_train.age_years")) == "[0, 90]"
    with pytest.raises(KeyError):
        dataset_stat.stat("a", "a_test.images")


def test_dataset_stat_counts_a_source_file(tmp_path, monkeypatch):
    """--source counts the rows or the distinct values of a file named in data_paths.json."""
    (tmp_path / "labels.csv").write_text("Path,Sex\na,Male\nb,Female\nc,Male\n")
    (tmp_path / "boxes.json").write_text('[{"id": 1}, {"id": 2}]')
    sys.path.insert(0, str(ROOT / "scripts"))
    import dataset_paths
    monkeypatch.setattr(dataset_paths, "paths", lambda key, variant=None: {"csv_path": str(tmp_path / "labels.csv"),
                                                                          "base_image_dir": str(tmp_path)})
    assert dataset_stat.source_count("k", "csv_path", "rows") == 3
    assert dataset_stat.source_count("k", "csv_path", "unique:Sex") == 2
    assert dataset_stat.source_count("k", "base_image_dir/boxes.json", "rows") == 2
    with pytest.raises(ValueError):
        dataset_stat.source_count("k", "base_image_dir/boxes.json", "unique:id")
    assert dataset_stat.main(["--source", "k"]) == 2


def test_dataset_stat_reads_a_variant_and_filters_rows(tmp_path, monkeypatch):
    """``<key>:<variant>`` picks a variant of the entry. ``@<column>=<value>`` keeps the matching rows."""
    (tmp_path / "t.csv").write_text("ImageID,split\na,train\na,train\nb,train\nc,test\n")
    sys.path.insert(0, str(ROOT / "scripts"))
    import dataset_paths
    seen = []
    monkeypatch.setattr(dataset_paths, "paths",
                        lambda key, variant=None: seen.append((key, variant)) or {"csv_path": str(tmp_path / "t.csv")})
    assert dataset_stat.source_count("k:tables", "csv_path", "unique:ImageID@split=train") == 2
    assert dataset_stat.source_count("k", "csv_path", "rows@split=test") == 1
    assert seen == [("k", "tables"), ("k", None)]
    with pytest.raises(ValueError, match="no row has split"):
        dataset_stat.source_count("k", "csv_path", "rows@split=val")


def test_claim_with_a_paper_source_checks_only_the_text():
    """A paper number has source= in place of cmd=. Only its place in the text is checked."""
    note = {"id": "p", "source": "Reis et al. 2022", "expect": "19,351"}
    assert claims.check_claim("a.md", 1, note, "The paper gives 19,351 patients.") == []
    assert [p["rule"] for p in claims.check_claim("a.md", 1, note, "The paper gives 19,352.")] == ["CLAIM-TEXT"]
    both = dict(note, cmd="echo 19,351")
    assert [p["rule"] for p in claims.check_claim("a.md", 1, both, "19,351")] == ["CLAIM-NOTE"]
    neither = {"id": "p", "expect": "19,351"}
    assert [p["rule"] for p in claims.check_claim("a.md", 1, neither, "19,351")] == ["CLAIM-NOTE"]


def test_claim_number_is_a_whole_number():
    """V-A-47: 14,555 in the text does not match the expected 4,555."""
    note = {"id": "c", "cmd": "echo 4,555", "expect": "4,555"}
    ok = claims.check_claim("a.md", 1, note, "The dataset has 4,555 images.")
    assert not [p for p in ok if p["rule"] == "CLAIM-TEXT"]
    for para in ("The dataset has 14,555 images.", "The dataset has 64,555 images."):
        got = claims.check_claim("a.md", 1, note, para)
        assert any(p["rule"] == "CLAIM-TEXT" for p in got), para


def test_classify_missing_repo_file_fails():
    """V-A-41: a missing file in a repo folder is a typo in the doc, not missing data."""
    msg = "[Errno 2] No such file or directory: 'configs/this_file_is_a_typo_in_the_doc.csv'"
    rec = {"status": "error", "exc": "FileNotFoundError", "msg": msg, "module": ""}
    assert run_blocks._classify(rec, "full")[0] == "failed"


def test_golden_compare_finds_a_changed_column(tmp_path):
    """Plan 6b: the golden tool names a column whose values change, and passes when nothing changes."""
    import json

    import pandas as pd

    sys.path.insert(0, str(ROOT / "scripts" / "evalkit"))
    import golden

    df = pd.DataFrame({"image_path": ["a.png", "b.png"], "edema": [1.0, 0.0], "bbox": [[[0, 1, 0, 1]], []]})
    rec = {"table": golden.table_record(df), "samples": {}}
    changed = {"table": golden.table_record(df.assign(edema=[1.0, float("nan")])), "samples": {}}
    files = []
    for name, r in (("a", rec), ("b", rec), ("c", changed)):
        files.append(tmp_path / f"{name}.json")
        files[-1].write_text(json.dumps({"evaluators": {}, "datasets": {"x": r}}))
    assert golden.compare(files[0], files[1]) == 0
    assert golden.compare(files[0], files[2]) == 1


def test_golden_skips_the_monai_trace_keys():
    """Plan 6b: a sample key such as ``img_transforms`` is MONAI's record of the steps, so the golden tool skips it."""
    import torch

    sys.path.insert(0, str(ROOT / "scripts" / "evalkit"))
    import golden

    rec = golden.sample_record({"img": torch.zeros(1, 2, 2), "img_transforms": [{"id": 140234}]})
    assert list(rec) == ["img"]


def test_a_package_error_raised_again_by_repo_code_is_blocked(tmp_path, monkeypatch):
    """compute_report_metrics raises its own ImportError from the ModuleNotFoundError of radeval.
    The block needs a package, so it is blocked (and runs again in the model venv), not failed."""
    monkeypatch.setattr(run_blocks, "REPO", tmp_path)
    monkeypatch.setattr(blocks, "REPO", tmp_path)
    page = tmp_path / "p.md"
    page.write_text("```python\ntry:\n    import no_such_package_xyz\n"
                    "except ImportError as e:\n    raise ImportError('install the extra') from e\n```\n")
    got = [(r["result"], r["detail"]) for r in run_blocks.run_page(page, "full")]
    assert got == [("blocked", "needs package no_such_package_xyz")]


# --------------------------------------------------------------------------
# wiki_audit: the 6c page templates (outline step 4)
# --------------------------------------------------------------------------
_GLANCE = ("| | |\n|---|---|\n| Registry keys | x |\n| Task and outputs | x |\n| Images in the table | x |\n"
           "| Modality | x |\n| License | x |\n| Download | x |\n")
_DATASET_H2 = ["Quick use", "How it is harmonized", "Labels", "Boxes", "Splits", "Arguments", "Harmonizer",
               "Known data problems"]


def _page(h2s, glance=_GLANCE):
    return "# Name\n\n" + glance + "\n" + "".join(f"## {h}\n\ntext\n\n" for h in h2s)


@pytest.mark.parametrize("rel, text, kinds", [
    ("wiki/reference/datasets/cxr-labels/a.md", _page(_DATASET_H2), []),
    ("wiki/reference/datasets/cxr-labels/a.md", _page([h for h in _DATASET_H2 if h != "Splits"]),
     ["template-sections"]),
    ("wiki/reference/datasets/cxr-labels/a.md", _page(["How it is harmonized", "Quick use"] + _DATASET_H2[2:]),
     ["template-sections"]),
    ("wiki/reference/datasets/cxr-labels/a.md", _page(_DATASET_H2, _GLANCE.replace("License", "Licence")),
     ["template-glance"]),
    ("wiki/how-to/x.md", _page(["1. A", "2. B", "Check", "See also"], ""), []),
    ("wiki/how-to/x.md", _page(["1. A", "3. B", "Check", "See also"], ""), ["template-sections"]),
    ("wiki/how-to/x.md", _page(["1. A", "See also"], ""), ["template-sections"]),
    ("wiki/reference/datasets/index.md", _page(["Anything"], ""), []),
])
def test_wiki_audit_checks_the_page_template(rel, text, kinds):
    import wiki_audit

    assert [f.type for f in wiki_audit.check_template(rel, text)] == kinds


def test_every_wiki_page_follows_its_template():
    import wiki_audit

    wiki = ROOT / "docs" / "wiki"
    bad = [f"{f.page}:{f.line} {f.detail}" for p in wiki.rglob("*.md") if ".ipynb_checkpoints" not in p.parts
           for f in wiki_audit.check_template(str(p.relative_to(wiki.parent)), p.read_text())]
    assert bad == []


# --------------------------------------------------------------------------
# Round 3 of plan 6d: IDs for check problems, progress lines, the base ref, the repo root on sys.path
# --------------------------------------------------------------------------
def _results(new_by_check):
    return {name: {"group": group, "problems": list(new), "new": list(new), "fixed": 0, "info": ""}
            for name, (group, new) in new_by_check.items()}


def test_merge_gives_each_new_check_problem_an_id(tmp_path):
    """/fix names a check problem by ID, so merge.py gives each NEW problem one (C-<check>-<n>)."""
    ste = [problems.problem("ste", "STE-02", "docs/wiki/how-to/x.md", 14, "em dash")]
    blk = [problems.problem("blocks", "BLOCK-KWARG", "docs/wiki/y.md", 40, "bad kwarg")]
    out = tmp_path / "2026-11-04-branch"
    (out / "checks").mkdir(parents=True)
    (out / "checks" / "results.json").write_text(json.dumps(
        _results({"ste": ("lint", ste), "blocks": ("docs", blk), "ruff": ("lint", [])})))
    data = merge.collect(out)
    found = {f["id"]: f for f in data["check_findings"]}
    assert set(found) == {"C-ste-1", "C-blocks-1"}
    assert found["C-ste-1"]["command"] == ".venv/bin/python scripts/evalkit/run_all.py --group lint"
    assert found["C-ste-1"]["narrow_command"] == ".venv/bin/python scripts/evalkit/ste_lint.py docs/wiki/how-to/x.md"
    assert found["C-blocks-1"]["narrow_command"].endswith("run_blocks.py --level static --page docs/wiki/y.md")
    index = tmp_path / "index.json"
    merge.update_index(data, out.name, index, "earlier run")
    ids = json.loads(index.read_text())["ids"]
    assert ids["C-ste-1"]["run"] == out.name and ids["C-ste-1"]["status"] == "open"
    assert "**C-ste-1**" in merge.draft(data, "t", None)
    # A later run with the same problem gets a new ID. The old ID keeps its run.
    later = merge.collect(out)
    merge.update_index(later, "2026-11-05-branch", index, "earlier run")
    assert {f["id"] for f in later["check_findings"]} == {"C-ste-2", "C-blocks-2"}


def test_run_all_prints_each_check_when_it_ends(monkeypatch, capsys):
    """A long run shows one line when each check ends, before the table at the end."""
    import run_all

    def ok(files):
        return [], ""

    def bad(files):
        return [problems.problem("x", "R", "a.md", 1, "t")], ""
    monkeypatch.setattr(run_all, "build_checks", lambda args: [("lint", "one", ok, False), ("lint", "two", bad, False)])
    assert run_all.main(["--group", "lint"]) == 1
    out = capsys.readouterr().out
    assert out.index("lint/one: pass (0 new") < out.index("lint/two: FAIL (1 new") < out.index("group  check")


def test_base_check_starts_at_the_merge_base(tmp_path, monkeypatch):
    """The base check lists the base files of this branch only, not the later commits of the base ref."""
    import subprocess

    import run_all

    def git(*args):
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=tmp_path,
                              check=True, capture_output=True, text=True).stdout.strip()

    def write(rel, text):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    git("init", "-q", "-b", "main")
    write("radharmony/dataset/base.py", "x = 1\n")
    write("radharmony/harmonizer/base.py", "y = 1\n")
    git("add", ".")
    git("commit", "-qm", "a")
    git("checkout", "-qb", "dev")
    write("radharmony/dataset/base.py", "x = 2\n")
    git("commit", "-qam", "dev changes base code")
    git("checkout", "-qb", "feat")
    write("notes.md", "a doc change\n")
    git("add", ".")
    git("commit", "-qm", "feat changes a doc only")
    git("checkout", "-q", "main")
    write("radharmony/harmonizer/base.py", "y = 2\n")
    git("commit", "-qam", "main moves on")
    git("checkout", "-q", "feat")
    monkeypatch.setattr(base_guard, "REPO", tmp_path)
    monkeypatch.setattr(run_all, "REPO", tmp_path)
    _, info = run_all.make_check_base(False, "dev")(None)
    assert info.startswith("no base-code change since the merge-base of HEAD and dev")
    assert git("rev-parse", "dev")[:8] in info
    # Against main: the change of dev counts, the later change on the tip of main does not.
    _, info = run_all.make_check_base(False, "main")(None)
    assert info.endswith(": radharmony/dataset/base.py")


def test_evalkit_scripts_put_the_repo_root_first(tmp_path):
    """Each evalkit script imports radharmony from its own clone or worktree, not from an installed copy.

    The dev venv has an editable install of the main clone. A script run in a worktree read the main
    clone's code before (skill tests of plan 6d, round 3).
    """
    import os
    import shutil
    import subprocess

    clone = tmp_path / "clone"
    shutil.copytree(ROOT / "scripts" / "evalkit", clone / "scripts" / "evalkit",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "scripts" / "dataset_paths.py", clone / "scripts" / "dataset_paths.py")
    (clone / "radharmony").mkdir()
    (clone / "radharmony" / "__init__.py").write_text("")
    (clone / "pyproject.toml").write_text("")
    mains = sorted(p.name for p in (clone / "scripts" / "evalkit").glob("*.py") if "__main__" in p.read_text())
    assert len(mains) >= 10
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for name in mains:
        probe = ("import runpy, sys; runpy.run_path(sys.argv[1], run_name='probe'); "
                 "import radharmony; print(radharmony.__file__)")
        proc = subprocess.run([sys.executable, "-c", probe, str(clone / "scripts" / "evalkit" / name)],
                              cwd=tmp_path, env=env, capture_output=True, text=True)
        assert proc.returncode == 0, (name, proc.stderr[-500:])
        assert proc.stdout.strip().startswith(str(clone)), (name, proc.stdout)


def test_run_text_runs_the_blocks_of_a_text(tmp_path):
    """run_text runs every Python block of a string in one process, and keeps no file in the repo."""
    text = ("```python\nx = 2\nprint('hello', x)\n```\n\n"
            "<!-- block: run=skip -->\n```python\nraise SystemExit('skipped')\n```\n\n"
            "```python\nfrom radharmony.harmonizer import NoSuchHarmonizer\n```\n\n"
            "```python\nprint(x)\n```\n")
    before = set(ROOT.iterdir())
    records, output = run_blocks.run_text(text, level="ci", has_gpu=False)
    assert [(r["line"], r["result"]) for r in records] == [(2, "ok"), (12, "failed"), (16, "not_run")]
    assert "ImportError" in records[1]["detail"] and records[0]["page"] == "snippet"
    assert "hello 2" in output
    assert set(ROOT.iterdir()) == before
    assert run_blocks.run_text("no code here\n") == ([], "")
