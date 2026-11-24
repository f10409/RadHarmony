"""Problem records and the known-problems list.

A problem is a dict with the keys check, rule, file, line and text.
The known-problems list (``eval/known_problems/<check>.json``) holds the old
problems that we accept for now. A run fails only on a problem that is not on
the list. The list can only get shorter (see ``shrink_errors``).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from collections.abc import Collection
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KNOWN_DIR = REPO / "eval" / "known_problems"

# Parts of a message that change from run to run: line references, memory addresses
# and temporary folders. The key leaves them out.
_LINE_REF = re.compile(r"(\bline\s+\d+|\bL\d+\b|:\d+(:\d+)?\b)", re.IGNORECASE)
_VOLATILE = re.compile(r"0x[0-9a-fA-F]+|/tmp/\S+")


def problem(check: str, rule: str, file, line: int, text: str) -> dict:
    """Make one problem record. The file path is relative to the repo root."""
    path = Path(file)
    if path.is_absolute():
        try:
            path = path.relative_to(REPO)
        except ValueError:
            pass
    return {"check": check, "rule": rule, "file": path.as_posix(), "line": int(line or 0),
            "text": " ".join(str(text).split())}


def key(p: dict) -> str:
    """Give a key that stays the same when the problem moves to a different line."""
    text = _VOLATILE.sub("#", _LINE_REF.sub("#", p["text"]))
    raw = f"{p['check']}|{p['rule']}|{p['file']}|{text}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _known_path(check: str) -> Path:
    return KNOWN_DIR / f"{check}.json"


def load_known(check: str, ref: str | None = None) -> dict | None:
    """Read the known-problems list of a check: {key: entry}.

    With ``ref``, read the list as it is in that git commit.
    Returns None when that commit has no list for the check.
    """
    path = _known_path(check)
    if ref is None:
        if not path.is_file():
            return {}
        return json.loads(path.read_text())["problems"]
    rel = path.relative_to(REPO).as_posix()
    out = subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=REPO,
                         capture_output=True, text=True)
    return json.loads(out.stdout)["problems"] if out.returncode == 0 else None


def save_known(check: str, problems: list[dict]) -> None:
    """Write the current problems of a check as its known-problems list."""
    entries: dict[str, dict] = {}
    for p in problems:
        k = key(p)
        if k in entries:
            entries[k]["count"] += 1
        else:
            entries[k] = {"rule": p["rule"], "file": p["file"], "text": p["text"][:100], "count": 1}
    KNOWN_DIR.mkdir(parents=True, exist_ok=True)
    body = {"check": check, "problems": dict(sorted(entries.items(),
                                                    key=lambda kv: (kv[1]["file"], kv[1]["rule"], kv[0])))}
    _known_path(check).write_text(json.dumps(body, indent=1, sort_keys=False) + "\n")


def compare(check: str, problems: list[dict], files: set[str] | None = None) -> tuple[list[dict], int]:
    """Split problems into new ones and count the fixed ones.

    A key that occurs more often than the list allows is new (the extra copies).
    ``files`` limits the fixed count to those files (for a run on staged files).
    Returns (new problems, number of fixed problems).
    """
    known = load_known(check)
    allowed = {k: v["count"] for k, v in known.items()}
    seen: Counter = Counter()
    new = []
    for p in problems:
        k = key(p)
        seen[k] += 1
        if seen[k] > allowed.get(k, 0):
            new.append(p)
    fixed = sum(max(0, n - seen[k]) for k, n in allowed.items()
                if files is None or known[k]["file"] in files)
    return new, fixed


def new_rules_in(body: str) -> list[str]:
    """Return the rule IDs of the ``New rule: <ID>`` lines of a PR description (D-45).

    A line in a fenced code block is an example, so it does not count. The ID may be in backticks.
    """
    text = re.sub(r"^[ \t]*```.*?^[ \t]*```", "", body, flags=re.MULTILINE | re.DOTALL)
    return re.findall(r"^[-* \t]*New rule:[* \t]*`?([A-Za-z][\w-]*)", text, re.MULTILINE)


def _moved_files(ref: str) -> dict[str, str]:
    """Return {new path: old path} for each file that git sees as moved since the commit ``ref``.

    git sees a move only when it is staged (``git mv`` or ``git add``). A file that changed by
    more than half is a delete and an add, not a move.
    """
    out = subprocess.run(["git", "diff", "-M", "-z", "--name-status", ref, "--"], cwd=REPO,
                         capture_output=True, text=True)
    fields = out.stdout.split("\0")  # -z: no quotes around a path with a non-ASCII letter
    moved, i = {}, 0
    while i < len(fields) and fields[i]:
        if fields[i][0] in "RC":  # status, old path, new path
            if fields[i][0] == "R":
                moved[fields[i + 2]] = fields[i + 1]
            i += 3
        else:  # status, path
            i += 2
    return moved


def _loose_key(entry: dict, file: str) -> tuple[str, str, str]:
    """Return (rule, file, text) of a list entry for a moved file.

    The text has no line numbers, memory addresses or /tmp paths, as in ``key``.
    """
    return entry["rule"], file, _VOLATILE.sub("#", _LINE_REF.sub("#", entry["text"]))


def lists_at(ref: str) -> int:
    """Return how many of the current known-problems lists also exist in the commit ``ref``."""
    return sum(load_known(p.stem, ref) is not None for p in KNOWN_DIR.glob("*.json"))


def shrink_errors(ref: str, new_rules: Collection[str] = ()) -> list[str]:
    """List each way the known-problems lists grew since the git commit ``ref`` (D-45).

    A list that is not in ``ref`` belongs to a new check, so it cannot grow. In
    another list, each new entry or higher count is an error, also for a rule with
    no entry in the old list. Only a rule in ``new_rules`` may add old problems, and
    only when the old list has no entry for it. A problem of a moved file is compared
    with the problems of the file at its old path, so a move adds nothing.
    """
    ok = subprocess.run(["git", "rev-parse", "--verify", "-q", f"{ref}^{{commit}}"], cwd=REPO,
                        capture_output=True)
    if ok.returncode != 0:
        return [f"{ref} is not a commit"]
    errors = []
    moved = _moved_files(ref)
    for check in sorted(p.stem for p in KNOWN_DIR.glob("*.json")):
        old, now = load_known(check, ref), load_known(check)
        if old is None:
            continue
        old_rules = {v["rule"] for v in old.values()}
        old_loose = Counter()
        for v in old.values():
            old_loose[_loose_key(v, v["file"])] += v["count"]
        moved_now: Counter = Counter()
        first: dict = {}
        for k, v in now.items():
            if v["rule"] in new_rules and v["rule"] not in old_rules:
                continue
            if k not in old and v["file"] in moved:
                # Sum the copies first: 2 old entries can share the first 100 characters.
                lk = _loose_key(v, moved[v["file"]])
                moved_now[lk] += v["count"]
                first.setdefault(lk, v)
            elif v["count"] > old.get(k, {}).get("count", 0):
                errors.append(f"{check}: {v['file']}: [{v['rule']}] {v['text'][:100]}")
        for lk, n in moved_now.items():
            if n > old_loose[lk]:
                v = first[lk]
                errors.append(f"{check}: {v['file']}: [{v['rule']}] {v['text'][:100]}")
    return errors
