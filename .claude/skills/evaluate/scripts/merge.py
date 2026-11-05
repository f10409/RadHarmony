"""Merge the results of one /evaluate run into findings.json and report_draft.md.

Folder layout (``OUT`` is the run folder)::

    OUT/checks/results.json, summary.md    automatic checks (run_all.py --out OUT/checks)
    OUT/correct/<part>_check/checked.json   "Correct?" (checker output, one per repo part)
    OUT/rules/<scope>/findings.json        "Follows our rules?"
    OUT/asked/findings.json                "Does what was asked?"

Each NEW problem of the automatic checks gets an ID too (``C-<check>-<n>``, for example ``C-blocks-1``),
so that /fix can name it. Its command reruns the check group.

It also adds the finding IDs of the run to ``reports/eval/index.json`` (the folder above OUT).

Usage::

    python merge.py OUT [--previous OLD/findings.json] [--title "..."]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

RANK = {"high": 0, "medium": 1, "low": 2}
PY = ".venv/bin/python"


def _narrow_command(check: str, file: str) -> str:
    """Return a command that checks only the file of a check problem, or "" if no such command exists."""
    if check == "ste":
        return f"{PY} scripts/evalkit/ste_lint.py {file}"
    if check == "ruff":
        return f".venv/bin/ruff check {file}"
    if check == "wiki" and file.startswith("docs/wiki/"):
        return f"{PY} scripts/evalkit/wiki_audit.py --page {file.removeprefix('docs/wiki/')}"
    if check == "blocks":
        return f"{PY} scripts/evalkit/run_blocks.py --level static --page {file}"
    return ""


def check_findings(checks: dict) -> list[dict]:
    """Give each NEW problem of the automatic checks an ID (``C-<check>-<n>``) and a command.

    Args:
        checks: the ``results.json`` of ``run_all.py --out``.

    Returns:
        One finding for each new problem, in the order of the checks.
    """
    out = []
    for name, res in checks.items():
        group = res.get("group", "")
        for i, prob in enumerate(res.get("new") or [], 1):
            file, line, rule = prob.get("file", ""), prob.get("line", 0), prob.get("rule", "")
            out.append({"id": f"C-{name}-{i}", "check": name, "group": group, "rule": rule,
                        "file": file, "line": line, "text": prob.get("text", ""),
                        "title": f"[{rule}] {file}:{line}",
                        "command": f"{PY} scripts/evalkit/run_all.py --group {group}",
                        "narrow_command": _narrow_command(name, file)})
    return out


def _load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def _questions(v: dict | None, source: str) -> list[dict]:
    """Return the valid open questions of one agent file. Bad items are skipped."""
    out = []
    for q in (v or {}).get("open_questions") or []:
        if not isinstance(q, dict) or not q.get("question"):
            continue
        opts = q.get("options") or []
        out.append(dict(q, options=[opts] if isinstance(opts, str) else list(opts), source=source))
    return out


def _merge_questions(questions: list[dict]) -> list[dict]:
    """Merge questions with the same text, and give each one an ID (Q-1, Q-2, ...)."""
    merged: dict[str, dict] = {}
    for q in questions:
        q = dict(q)
        src = q.pop("source")
        k = " ".join(str(q["question"]).lower().split())
        if k in merged:
            merged[k]["sources"].append(src)
        else:
            merged[k] = dict(q, sources=[src])
    return [dict(q, id=f"Q-{i}") for i, q in enumerate(merged.values(), 1)]


def _split_id(fid: str) -> tuple[str, int] | None:
    """Split "V-D-19" into ("V-D", 19). Return None if the ID does not end in a number."""
    head, _, tail = fid.rpartition("-")
    return (head, int(tail)) if head and tail.isdigit() else None


def update_index(data: dict, run: str, index_path: Path, old_run: str) -> None:
    """Give each new finding an ID that no other run used, and write the ID index.

    The index maps each finding ID of each run to its run folder, review part, title and status.
    An old finding also gets ``checked_in``, the last run that ran it again. A new finding keeps its
    agent's ID if no other run used it. Else it gets the next free number, and ``agent_id`` keeps the
    agent's ID. A new merge of a run keeps the IDs of its last merge. The lessons get the new IDs too.
    """
    try:
        ids = (_load(index_path) or {}).get("ids") or {}
    except json.JSONDecodeError as e:
        raise SystemExit(f"{index_path} is not valid JSON ({e}). Restore it, then merge again.") from e
    mine = {k: v for k, v in ids.items() if v.get("run") == run}  # this run, from its last merge
    last = {(v.get("review"), v.get("agent_id", k)): k for k, v in mine.items()}
    index = {k: v for k, v in ids.items() if v.get("run") != run}
    for o in data["old"]:  # old findings keep their IDs
        entry = index.setdefault(o["old_id"], {"run": old_run, "review": "correct",
                                               "title": str(o.get("title", ""))[:120]})
        entry.update(status=o["status"], checked_in=run)
    top: dict[str, int] = {}
    for fid in index:
        if parts := _split_id(fid):
            top[parts[0]] = max(top.get(parts[0], 0), parts[1])
    new = [(review, f) for review in ("correct", "outside", "rules", "asked", "check_findings")
           for f in data.get(review, [])]
    for review, f in new:  # keep the ID of the last merge of this run
        k = last.get((review, f["id"]))
        if k and k != f["id"]:
            f["agent_id"], f["id"] = f["id"], k
    used, clash = set(index), []
    for _, f in new:
        if f["id"] in used:
            clash.append(f)
            continue
        used.add(f["id"])
        if parts := _split_id(f["id"]):
            top[parts[0]] = max(top.get(parts[0], 0), parts[1])
    for f in clash:
        head = (_split_id(f["id"]) or (f["id"], 0))[0]
        top[head] = top.get(head, 0) + 1
        f["agent_id"], f["id"] = f.get("agent_id", f["id"]), f"{head}-{top[head]}"
    renames: dict[str, dict[str, str]] = {}
    for review, f in new:
        title = f.get("title") or f.get("why") or f.get("request") or ""
        entry = {"run": run, "review": review, "title": str(title)[:120], "status": "open"}
        if "agent_id" in f:
            entry["agent_id"] = f["agent_id"]
            # Lessons of the checker cite the IDs of both its lists.
            renames.setdefault("correct" if review == "outside" else review, {})[f["agent_id"]] = f["id"]
        prev = mine.get(f["id"], {})
        if prev.get("checked_in", run) != run:  # a later run ran it again
            entry.update(status=prev.get("status", "open"), checked_in=prev["checked_in"])
        index[f["id"]] = entry
    for x in data.get("lessons", []):  # a lesson cites the IDs of its own review part
        m = renames.get(str(x.get("source", "")).split("/")[0], {})
        if isinstance(x.get("evidence"), list):
            x["evidence"] = [m.get(e, e) if isinstance(e, str) else e for e in x["evidence"]]
    order = sorted(index, key=lambda k: (_split_id(k) or (k, 0)))
    index_path.write_text(json.dumps({"ids": {k: index[k] for k in order}}, indent=1) + "\n")


def collect(out: Path) -> dict:
    """Read every part of the run into one dict."""
    correct, outside, old, coverage, dropped, lessons, questions = [], [], [], [], {}, [], []
    for f in sorted((out / "correct").glob("*_check/checked.json")):
        v = _load(f)
        part = v.get("part") or f.parent.name.removesuffix("_check")
        for k in v.get("kept", []):
            correct.append(dict(k, part=part))
        outside += [dict(k, part=part) for k in v.get("outside_diff", [])]  # branch review only
        old += [dict(o, part=part) for o in v.get("old", [])]
        coverage += [dict(c, part=part) for c in v.get("coverage", [])]
        dropped[part] = dict(Counter(d.get("hunter", "?") for d in v.get("dropped", [])))
        lessons += [dict(x, source=f"correct/{part}") for x in v.get("lessons", [])]
        questions += _questions(v, f"correct/{part}")
    rules = []
    for f in sorted((out / "rules").glob("*/findings.json")):
        v = _load(f)
        rules += [dict(x, scope=v.get("scope", f.parent.name)) for x in v.get("findings", [])]
        lessons += [dict(x, source=f"rules/{f.parent.name}") for x in v.get("lessons", [])]
        questions += _questions(v, f"rules/{f.parent.name}")
    asked = _load(out / "asked" / "findings.json") or {}
    lessons += [dict(x, source="asked") for x in asked.get("lessons", [])]
    checks = _load(out / "checks" / "results.json") or {}
    # Open questions: behavior that no decision in docs/wiki/design.md covers.
    questions = _merge_questions(questions + _questions(asked, "asked"))
    correct.sort(key=lambda f: (RANK.get(f.get("severity"), 3), f["part"]))
    rules.sort(key=lambda f: (RANK.get(f.get("severity"), 3), f.get("rule", "")))
    return {"checks": {k: {kk: vv for kk, vv in v.items() if kk not in ("problems", "output")}
                       for k, v in checks.items()},
            "check_findings": check_findings(checks),
            "correct": correct, "outside": outside, "rules": rules, "asked": asked.get("findings", []),
            "asked_source": asked.get("source", ""), "old": old, "coverage": coverage,
            "dropped": dropped, "open_questions": questions, "lessons": lessons}


def _label(f: dict) -> str:
    """Return the bold ID of a finding, with the agent's ID if merge gave it a new one."""
    return f"**{f['id']}**" + (f" (agent ID {f['agent_id']})" if f.get("agent_id") else "")


def draft(data: dict, title: str, previous: dict | None) -> str:
    """Write the report draft. The lessons section is left for the main agent."""
    md = [f"# {title}", ""]
    md += ["## 1. Automatic checks", "", "See `checks/summary.md` for the table.", ""]
    found = data.get("check_findings", [])
    for name, res in data["checks"].items():
        if "crash" in res:
            md.append(f"- **{name}: CRASH** {res['crash']}")
        elif res.get("new"):
            mine = [f for f in found if f["check"] == name]
            md.append(f"- **{name}: {len(res['new'])} new problems** (command: `{mine[0]['command'] if mine else ''}`)")
            md += [f"  - {_label(f)} `{f['file']}:{f['line']}` [{f['rule']}] {f['text'][:140]}" for f in mine[:30]]
            if len(mine) > 30:
                md.append(f"  - and {len(mine) - 30} more (see `findings.json`, `check_findings`)")
    recs = data["checks"].get("blocks_run", {}).get("records", [])
    if recs:
        c = Counter(r["result"] for r in recs)
        md.append("- doc code blocks run: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    md.append("")

    md += ["## 2. Correct?", "", "| Repo part | High | Medium | Low | Base code | Dropped (opus / sonnet) |",
           "|---|---|---|---|---|---|"]
    parts = sorted({f["part"] for f in data["correct"]} | set(data["dropped"]))
    for p in parts:
        sev = Counter(f["severity"] for f in data["correct"] if f["part"] == p)
        nb = sum(1 for f in data["correct"] if f["part"] == p and f.get("touches_base_code"))
        d = data["dropped"].get(p, {})
        md.append(f"| {p} | {sev['high']} | {sev['medium']} | {sev['low']} | {nb} | "
                  f"{d.get('opus', 0)} / {d.get('sonnet', 0)} |")
    md.append("")
    for f in data["correct"]:
        base = " (base code)" if f.get("touches_base_code") else ""
        md.append(f"- {_label(f)} ({f['severity']}{base}) {f['title']}. "
                  f"Files: {', '.join(f.get('files', [])[:3])}. Command: `{f.get('command', '')[:160]}`")
    md.append("")

    if data.get("outside"):
        md += ["### Outside the diff", "", "Real failures in lines that the branch did not change.", ""]
        md += [f"- {_label(f)} ({f['severity']}) {f['title']}. Files: {', '.join(f.get('files', [])[:3])}. "
               f"Command: `{f.get('command', '')[:160]}`" for f in data["outside"]]
        md.append("")

    md += ["## 3. Follows our rules?", ""]
    md.append(", ".join(f"{r}: {n}" for r, n in sorted(Counter(f.get('rule') for f in data['rules']).items()))
              or "No findings.")
    md.append("")
    md += [f"- {_label(f)} [{f.get('rule')}] ({f.get('severity')}) `{f.get('file')}:{f.get('line')}`: {f.get('why')}"
           for f in data["rules"]]
    md.append("")

    md += ["## 4. Does what was asked?", "", f"Source: {data['asked_source'] or 'none'}", ""]
    md += [f"- {_label(f)} ({f.get('type')}, {f.get('severity')}) {f.get('request', '')[:160]}: {f.get('why')}"
           for f in data["asked"]] or ["No findings."]
    md.append("")

    if previous is not None or data["old"]:
        st = Counter(o["status"] for o in data["old"])
        # The old audit format has "findings". The format of this script has "correct".
        n_prev = len(previous.get("correct", previous.get("findings", []))) if previous else len(data["old"])
        md += ["## 5. Compared with the last whole-repo review", "",
               f"Last run: {n_prev} findings. Still open: {st['still_open']}. Fixed: {st['fixed']}. "
               f"Could not run: {st['cannot_run']}. New this run: {len(data['correct'])} (Correct?).", ""]
        md += [f"- fixed: {o['old_id']}: {o.get('note', '')}" for o in data["old"] if o["status"] == "fixed"]
        md.append("")

    blocked = [c for c in data["coverage"] if "BLOCKED" in (c.get("opus"), c.get("sonnet"))]
    md += ["## 6. Coverage and BLOCKED", "",
           f"Items checked: {len(data['coverage'])}. BLOCKED: "
           + (", ".join(sorted({c['item'] for c in blocked})) or "none") + ".", ""]
    md += ["## 7. Open questions", "", "(Merge copies with other words. Ask the user with `references/decisions.md`.)", ""]
    md += [f"- **{q['id']}** [{', '.join(q['sources'])}] {q['question']} Today: {q.get('today') or '?'} "
           f"Options: {'; '.join(map(str, q['options'])) or '?'}"
           for q in data["open_questions"]] or ["None."]
    md.append("")
    md += ["## 8. Lessons", "", "(Write with `references/lessons.md`. Agent input below.)", ""]
    md += [f"- [{x['source']}] {x['what']} (evidence: {', '.join(map(str, x.get('evidence', [])))})"
           for x in data["lessons"]]
    return "\n".join(md) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", help="run folder")
    ap.add_argument("--previous", help="findings.json of the last whole-repo review")
    ap.add_argument("--title", default="RadHarmony evaluation")
    args = ap.parse_args()
    out = Path(args.out).resolve()  # "." has no folder name
    data = collect(out)
    previous = _load(Path(args.previous)) if args.previous else None
    old_run = Path(args.previous).parent.name if args.previous else "earlier run"
    update_index(data, out.name, out.parent / "index.json", old_run)
    (out / "findings.json").write_text(json.dumps(data, indent=1))
    (out / "report_draft.md").write_text(draft(data, args.title, previous))
    sev = Counter(f["severity"] for f in data["correct"])
    print(f"checks: {len(data['check_findings'])} new problems  Correct?: {dict(sev)}  "
          f"outside the diff: {len(data['outside'])}  "
          f"Follows our rules?: {len(data['rules'])}  Does what was asked?: {len(data['asked'])}  "
          f"open questions: {len(data['open_questions'])}  "
          f"old: {dict(Counter(o['status'] for o in data['old']))}")
    print(f"wrote {out / 'findings.json'} and {out / 'report_draft.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
