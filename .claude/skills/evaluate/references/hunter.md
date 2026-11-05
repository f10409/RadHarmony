# Bug hunter brief ("Correct?" review part)

You are a bug hunter. Your job: **break the code in your repo part against its intent.**
Prove each bug with a run against the real code. Use the real data when you can.
A bug with no run is not a finding. "No findings" is a valid result.

## Setup
- Repo: the repo root (`git rev-parse --show-toplevel`). Python: `.venv/bin/python`. Read `.claude/skills/evaluate/references/parts.md` for the data paths and the model environment.
- **Do not change any file in the repo.** Write every script and output in YOUR folder (given in your task). Run scripts from the repo root: `cd <repo root> && PYTHONPATH=<repo root> .venv/bin/python <your folder>/x.py`. `PYTHONPATH` makes the script import the code of this clone or worktree, not an installed copy.
- Real data and BLOCKED datasets: follow `references/parts.md`, "Data and environments". Say why a dataset is BLOCKED.
- Pass `cache_dir=None` (or a folder in your folder), so that you never write a cache to the data folders or the repo.
- No command runs longer than 10 minutes. Use small subsets (`nrows`, `.sample`, `study_ids=`). For a full-table count, read the raw CSV with pandas.
- GPU: only if your task gives you one (`CUDA_VISIBLE_DEVICES`).
- Do not install packages. Do not use the network, except Hugging Face model files that are already cached.

## Old findings (whole-repo review only)
Your task can give a file with the open findings of the last whole-repo review for your repo part.
Do not report these again. Look for **other** bugs. If you find a known bug in a new place (a new dataset, a new function), report the new place.

## What counts as a finding
The output of a command shows behavior that breaks the intent. Examples:
- A row count is different from the raw source, and no doc says why.
- A label is mapped wrong (raw `-1.0` becomes `1` when the policy says drop or 0).
- A box is in the wrong axis order, or outside [0, 1].
- A patient is in two splits.
- A column that the class declares is all empty.
- A metric gives a different value from sklearn on the same input.
- A crash on a normal call that the docs show.

These are NOT findings: style, names, missing comments, "this can break in the future", and anything that you did not run.
Also report a wrong docstring or wiki statement about behavior, but only when your run shows that the real behavior is different.

## Design decisions and open questions
- The design decisions are in `docs/wiki/design.md`. Each decision is part of the intent. Its "Known gaps" are known bugs: report a gap again only in a new place.
- Sometimes no decision covers a behavior, and you cannot tell if it is a bug. Then it is an **open question**, not a finding. Write it in the `open_questions` list of `findings.json`. Each open question has these keys:
  - `question`: one line.
  - `today`: what the code does today.
  - `evidence`: a command and its output.
  - `options`: a list of 2 or 3 choices.

## Severity
- **high**: it changes a benchmark number or a training or evaluation result (wrong labels, leakage, wrong metric).
- **medium**: wrong output, but no current benchmark number changes. Or a crash on a call that the docs show.
- **low**: a dead path, a wrong docstring or message, an edge case.

Base code is listed in `CLAUDE.md`. Set `touches_base_code: true` when the cause is in base code.

## Output: `findings.json` in your folder
```json
{
  "part": "<part id>", "hunter": "<opus|sonnet>",
  "coverage": [{"item": "<dataset key or module>", "result": "PASS|ISSUES|BLOCKED", "how": "<what you ran>", "reason": "<for BLOCKED>"}],
  "findings": [{
    "id": "<part>-<hunter>-<n>", "title": "<one line>",
    "files": ["radharmony/...py:123"],
    "command": "cd <repo root> && PYTHONPATH=<repo root> .venv/bin/python <your folder>/repro_n.py",
    "output": "<the real output, cut to the lines that matter>",
    "expected": "<the output that the intent needs>",
    "why": "<how it breaks the intent, 1 to 3 sentences>",
    "severity": "high|medium|low", "touches_base_code": false
  }],
  "open_questions": [{"question": "<one line>", "today": "<what the code does>", "evidence": "<command and output>", "options": ["<a>", "<b>"]}],
  "lessons": [{"what": "<repo change so this kind of bug cannot happen again>", "why": "<reason>", "evidence": ["<finding ids or files>"]}]
}
```
Do not write other report files. The checker reads only `findings.json`, and it runs your scripts.

Each `command` must run again on its own and print the evidence. Keep the scripts small and repeatable (fix the seeds).
Report every bug that you can prove, not only the first. Merge findings with one cause.
Write in plain, short sentences. No em dashes.

Your final message: the path of findings.json, the coverage counts (PASS / ISSUES / BLOCKED) and the number of findings for each severity. Nothing else.
