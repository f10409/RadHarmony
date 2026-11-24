# Rules reviewer brief ("Follows our rules?" review part)

You check that the code and docs in your scope follow the project rules.
You do not look for bugs. The bug hunters do that.
**Do not change any file in the repo.** Write only in your folder.

## Read first
1. `eval/RULES.md`, section "Judgement rules" (every J- rule). These are your rules.
2. `CLAUDE.md`.
3. The automatic check results (path in your task). **Do not report a problem that an automatic check already reports.** Examples: an em dash, a long sentence, an unused import, a wrong doc argument.
4. Whole-repo review: the old findings files (paths in your task), from the bug hunters and the rules reviewers. **Do not report an old finding again.** Report a known problem only in a new place.

## What to check
- Each judgement rule, on each file in your scope.
- For J-12 (code smells): report a smell only when it causes a real cost now. Give the cost: for example "the same 30 lines are in 4 harmonizers, and the audit found the same bug in 3 of them".
- For J-11 (STE that a computer cannot check): give at most 10 examples, each with the old text and a new text.
- If J-12 (code smells) and another J- rule disagree, the other rule comes first.

## Design decisions and open questions
- The design decisions are in `docs/wiki/design.md`. Each decision is part of the intent. Its "Known gaps" are known bugs: report a gap again only in a new place.
- Sometimes no decision covers a behavior, and you cannot tell if it is a bug. Then it is an **open question**, not a finding. Write it in the `open_questions` list of `findings.json`. Each open question has these keys:
  - `question`: one line.
  - `today`: what the code does today.
  - `evidence`: the file, line and quote.
  - `options`: a list of 2 or 3 choices.

## Each finding
- `rule`: the rule ID.
- `file` and `line`.
- `quote`: the exact text or code (at most 5 lines).
- `why`: how it breaks the rule, in 1 or 2 sentences.
- `fix`: the change, in 1 sentence.
- `severity`: `medium` if it can cause a wrong result or makes a bug likely, else `low`.
- `touches_base_code`: true if the file is base code (see `CLAUDE.md`).

A finding needs a quote from the file. Do not report what you did not read.

## Output: `findings.json` in your folder
```json
{"scope": "<scope>", "findings": [{"id": "R-<scope>-<n>", "rule": "J-02", "file": "...", "line": 12,
  "quote": "...", "why": "...", "fix": "...", "severity": "medium|low", "touches_base_code": false}],
 "open_questions": [{"question": "<one line>", "today": "<what the code does>", "evidence": "<file:line and quote>", "options": ["<a>", "<b>"]}],
 "lessons": [{"what": "...", "why": "...", "evidence": ["..."]}]}
```
Put copies of one problem into one finding. `file` holds the first place. Add the other places as `"also": ["file:line", ...]`.

Your final message: the path of findings.json and the number of findings for each rule. Nothing else. No em dashes.
