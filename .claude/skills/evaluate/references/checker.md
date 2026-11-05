# Checker brief ("Correct?" review part)

You check the findings of 2 bug hunters for one repo part. You did not write them. Be strict.
**Do not change any file in the repo.** Write only in your checker folder.

Run each command in the foreground. Do not start background jobs (`nohup`, `&`). Your run can end before the job ends, and then you lose the result.
Write `checked.json` before your final message.

## New findings
For each finding in `<part>_opus/findings.json` and `<part>_sonnet/findings.json`:
1. Run its `command` again, exactly (from the repo root, 10 minute limit).
2. **Keep** the finding only if the new output shows the same failure as the claimed `output`. The failure must also break the intent (the intent file of the part). Small changes in time or memory addresses are fine.
3. **Drop** it if the command does not run or the output does not show the failure. Also drop it if the "failure" is style, a name or a future risk. Write the reason.
4. **Merge** findings with one cause (the same file and the same wrong logic), across both hunters. Keep the clearest command. List every hunter that found it.
5. Check the severity against the hunter brief. Change it if it is wrong, and say why.
6. Branch review: a finding can show a real failure in a line that the branch did not change. Do not drop it. Put it in `outside_diff`, with the keys of a kept finding.

## Design decisions and open questions
- The intent is the intent file of the part plus the decisions in `docs/wiki/design.md`.
- A finding can show a behavior that no decision covers, and you cannot tell if it is a bug. Then do not keep it and do not drop it. Move it to `open_questions`, with the keys of the hunter brief.
- Copy the `open_questions` of both hunters into `checked.json`. Merge copies: 2 questions about the same behavior in the same file are one question. List every hunter that asked it in `from`.

## Old findings (whole-repo review only)
Your task can give a file with the open findings of the last whole-repo review for this repo part.
For each old finding, run its `command` again. If the script path does not exist, look for the same file name in the old report folder (given in your task).
- `still_open`: the output shows the same failure.
- `fixed`: the command runs and the failure is gone.
- `cannot_run`: the command does not run (say why).

## Output: `checked.json` in your folder
```json
{
  "part": "<part>",
  "kept": [{"id": "V-<part>-<n>", "source_ids": ["..."], "found_by": ["opus","sonnet"],
            "title": "...", "files": ["..."], "command": "...", "output": "<YOUR output, cut>",
            "expected": "...", "why": "...", "severity": "high|medium|low", "severity_changed": "<reason or empty>",
            "touches_base_code": false}],
  "outside_diff": [{<the keys of a kept finding>}],
  "dropped": [{"source_id": "...", "hunter": "opus|sonnet", "reason": "..."}],
  "old": [{<all fields of the old finding>, "old_id": "<its id>", "status": "still_open|fixed|cannot_run", "note": "<one line>"}],
  "coverage": [{"item": "...", "opus": "PASS|ISSUES|BLOCKED", "sonnet": "PASS|ISSUES|BLOCKED"}],
  "open_questions": [{"question": "...", "today": "...", "evidence": "...", "options": ["...", "..."], "from": ["opus","sonnet"]}],
  "lessons": [{"what": "...", "why": "...", "evidence": ["..."], "from": ["opus","sonnet"]}]
}
```
Merge the two coverage lists and the two lesson lists (remove copies).

Your final message: the path of checked.json and the counts (kept for each severity, dropped for each hunter, open questions, old findings for each status). Nothing else. No em dashes.
