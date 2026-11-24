---
name: evaluate
description: >
  Evaluate RadHarmony code and docs, and write a report with proof for each finding.
  It runs the automatic checks (lint, docs, test), then a 3-part review that runs in parallel
  (Correct?, Follows our rules?, Does what was asked?), then writes lessons. It reports only.
  It does not fix. Use when the user says "evaluate", "/evaluate", "evaluate full", "audit the repo",
  "review this branch", "review my changes" of the RadHarmony repo, or "check the repo before a release".
  To fix findings after a report, use the fix skill. For a quick diff review with no report, use code-review.
---

# /evaluate

Arguments:
- none: **branch review**. Evaluate the changes on this branch (the diff against `main`, plus uncommitted changes).
- `--base <branch or commit>`: in a branch review, compare with this branch or commit instead of `main`. Example: `--base v0.2-dev` for a branch made from `v0.2-dev`. The fix skill gives a commit.
- `--full`: **whole-repo review**. Evaluate the whole repo, one repo part at a time. Use it before a release. `/evaluate --full` is not `run_all.py --full` (step 3).
- `--lessons-only`: give only the lessons for the work in this session.

The chart and the word list are in `eval/PIPELINE.md`. The rules are in `eval/RULES.md`.
All `references/` and `scripts/` paths below are in `.claude/skills/evaluate/`.

## Rules for every run
- **Report only.** Do not change code or docs. Review agents write only in their own folders. You write only in `OUT`, with 2 exceptions. `merge.py` updates `reports/eval/index.json`. In step 6, you record the user's answers in `docs/wiki/design.md` (a D- entry, with the status "Decided" or "Open"), as `references/decisions.md` says.
- Every finding needs proof: a command and its output (Correct?), a quote (Follows our rules?), or a quote of the request (Does what was asked?).
- Mark findings whose cause is in base code (`CLAUDE.md`). Never fix them here.
- Do not install packages.
- **Real data.** The data folders come from the data paths file: `configs/data_paths.json`, or the file that `RADHARMONY_DATA_PATHS` names (`docs/wiki/get-started/install.md`, "Data paths for the checks"). Read them with `paths(key)` of `scripts/dataset_paths.py`.
  If the file or the key is missing, `paths()` raises `MissingPathsError` with a hint. Then ask the user for the folder, with 3 choices:
  1. Add the entry to their data paths file. If the file is missing, copy `configs/data_paths.example.json` first.
  2. Give the folder for this run only.
  3. Skip the real-data check, and say so in the report.
- Use plain words in the report. No em dashes. Use the names in the `eval/PIPELINE.md` word list.
- **No user to ask.** A sub-agent or a headless run has no user. Then write the question and the chosen answer in the report, and go on.

## Step 1. Scope
1. Make the run folder `reports/eval/<YYYY-MM-DD>-<branch|full>/`. Replace each `/` in the branch name with `-`. If the folder exists, add `-2` (or the next free number). `reports/` is in `.gitignore`. Below, this folder is `OUT`.
2. Branch review: list the changed files with `git diff --name-only <base>...HEAD` and `git status --short --untracked-files=all`. `<base>` is `main`, or the `--base` branch. If the list is empty, tell the user "nothing to evaluate" and stop. The repo part name is `branch`.
3. Whole-repo review: the repo parts are in `references/parts.md`. Find the last whole-repo review: the newest `reports/eval/*-full/findings.json`, or else `reports/eval/audit-*/findings.json`. Its open findings are its `correct` list (or `findings` list in the old audit format) plus its `old` entries with the status `still_open` or `cannot_run`. Write the open findings of each repo part to `OUT/correct/<part>_old.json`. Write its `rules` findings to `OUT/rules/old.json`.

## Step 2. Find what was asked
- Branch review: look in these places. Issue numbers in the commit messages (`gh issue view`). The PR text (`gh pr view`). A plan file that the user names. The user's request in this session.
- Whole-repo review: the promises in the README and the main wiki pages (see `references/asked_reviewer.md`).
- If you find nothing, the Does what was asked? part says so. Do not guess a request.

## Step 3. Automatic checks
Run `.venv/bin/python scripts/evalkit/run_all.py --full --out OUT/checks --base <base>`. `<base>` is the base of step 1 (`origin/main` for a whole-repo review). The base check then lists only the base files that changed since the merge-base of HEAD and `<base>`.

Use `--full` only in the dev repo (`git ls-files configs/data_paths.json` prints the file) with the NAS mounted: `.venv/bin/python -m pytest tests/test_data_paths.py -k paths_exist_on_nas -q` reports passed tests, not skipped ones. Else leave out `--full`.

Without `--full`, the run still does ruff, the STE linter, the wiki audit and the static doc block checks. It also does the lesson checks, the base-code check and pytest. Tests that need data skip. It skips the claims and the doc blocks that need data. Write this in the run facts of `report.md`.

The run takes more than 1 hour on the lab server, because it runs all of pytest. Run it in the background (`run_in_background`), and wait for the notice. It prints one line when each check ends. While it runs, you can start the Correct? and Does what was asked? agents of step 4. Start the Follows our rules? agents after it ends, because they need `OUT/checks/results.json`.

Read `OUT/checks/summary.md`. If `run_all.py` crashes, write the command and the error in the run facts of `report.md`. If `OUT/checks/results.json` is missing, tell the rules reviewers that there are no check results. Do step 4 also after a crash or a failed check.

## Step 4. The 3-part review, in parallel
Start the agents with the Agent tool. Each review part is its own agent, so the parts do not mix. Start at most 10 agents at a time.
Give each agent its brief path, its folder, its scope and the paths it needs. Do not give an agent your own guess about the bugs.

A sub-agent cannot ask the user. So before you start the agents, get the data folders of the datasets in scope (see "Real data" above). Pass them in each task.

Do not send the decisions of `docs/wiki/design.md` to the agents. Each brief already sends its agent there.

**Correct?** For each repo part (whole-repo review), or for the changed files (branch review):
1. 2 bug hunters, in parallel: one on opus and one on sonnet. Brief: `references/hunter.md`, plus the intent file from `references/parts.md`. Folders: `OUT/correct/<part>_opus/` and `OUT/correct/<part>_sonnet/`. Whole-repo review: also give `OUT/correct/<part>_old.json`. Give the backbone part a free GPU.
2. When both bug hunters are done: 1 checker on haiku. Brief: `references/checker.md`. Folder: `OUT/correct/<part>_check/`. Whole-repo review: also give the old findings file and the old run folder.

**Follows our rules?** Brief: `references/rules_reviewer.md`. Give each agent `OUT/checks/results.json`. Whole-repo review: also give the old findings files `OUT/correct/*_old.json` and `OUT/rules/old.json`.
- Whole-repo review: 2 agents on sonnet. Folder `OUT/rules/code/` for `radharmony/`, `radharmony_internal/` (if the repo has it), `app.py` and `scripts/`. Folder `OUT/rules/docs/` for `docs/wiki/`, `README.md`, `CONTRIBUTING.md` and `.claude/skills/`.
- Branch review: 1 agent for the changed files. Folder `OUT/rules/branch/`.

**Does what was asked?** 1 agent on opus. Brief: `references/asked_reviewer.md`. Folder: `OUT/asked/`. Give it the request sources from step 2.

If an agent stops before it writes its file, send it a message with SendMessage. Ask it to finish.
An agent can ask for an action that its permissions do not allow, for example to kill a process. Do not do the action for the agent. Tell the user what the agent asked for.

## Step 5. Merge
Run `.venv/bin/python .claude/skills/evaluate/scripts/merge.py OUT --title "<title>"`.
Whole-repo review: add `--previous <findings.json of the last whole-repo review>`.
It writes `OUT/findings.json` and `OUT/report_draft.md`. It also adds the finding IDs to `reports/eval/index.json`. Each new problem of the automatic checks gets an ID too (`C-<check>-<n>`, in the `check_findings` list), with its command.

If an earlier run already used the ID of a new finding, the new finding gets the next free number. The agent's own ID goes to `agent_id`.

Run the command of each **high** finding yourself. If the output does not show the failure, remove the finding from `OUT/findings.json` after the last `merge.py` run. Set its status to `dropped` in `reports/eval/index.json`. Say so in `report.md`.

## Step 6. Lessons and report
1. Write the lessons with `references/lessons.md`. Use the agents' lesson input at the end of the draft as raw material.
2. Write `OUT/report.md` from the draft:
   - The run facts at the top: date, branch, commit, scope and agents used.
   - The 3 review parts side by side, each with its own findings. Do not merge or rank them across parts.
   - Branch review: the bugs outside the diff, from the checker. The branch did not cause them, but they are real.
   - "Compared with the last whole-repo review" (whole-repo review only).
   - "Open questions": the `open_questions` of all agents from the draft, with copies merged.
   - The lessons last.
3. Tell the user in a few lines. Give the result of the automatic checks and the counts for each review part. Give each high finding in one line. Give the report path and the top 3 lessons.
4. Go through the items that need a choice from the user, one at a time, with `references/decisions.md`. The order is: open questions, then severity changes, then draft rules from the lessons. If there is no such item, go to item 5.
5. Then ask which findings to fix, as a short list grouped by repo part. Add the C- IDs of the automatic checks. The fix skill fixes them (`/fix <finding ids>`).

## Done when
- `OUT/checks/summary.md`, `OUT/findings.json` and `OUT/report.md` exist.
- Each repo part in scope has a checker file, or the report says why not.
- You ran the command of each high finding again.
- Each open question is a D- entry in `docs/wiki/design.md`, with the status "Decided" or "Open".
- Each draft rule has the user's answer in the lessons of `report.md`.
- The user knows the list of findings to fix (item 5 of step 6).
- The report ends with lessons, and each lesson points to findings.

## `--lessons-only`
Skip steps 3 to 5. Read the work of this session (the diff, the errors you hit, the user's corrections). Write lessons with `references/lessons.md` and show them. Ask about each draft rule with `references/decisions.md`. Do not write a report file.
