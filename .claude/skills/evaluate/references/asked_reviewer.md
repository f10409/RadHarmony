# "Does what was asked?" reviewer brief

You compare what was asked with what the code does.
**Do not change any file in the repo.** Write only in your folder.

## What was asked
Your task gives the source of the request:
- **Branch review:** the issue, the PR text, the plan file or the user request.
- **Whole-repo review:** the promises that the project makes to its users: `README.md`, `docs/wiki/index.md`, `docs/wiki/get-started/install.md`, `docs/wiki/reference/datasets/index.md`, `docs/wiki/reference/evaluators/index.md` and `docs/wiki/reference/backbones/index.md`. A promise is a sentence that says what the user can do, for example "every dataset supports `output_bbox`" or "`list_datasets()` lists all datasets".

If there is no request, write one finding: "no request found", and stop.

## What to check
For each request or promise, find the code that does it. Run a small command when you can.
Give each finding one of these types:
- `missing`: it was asked or promised, and the code does not do it.
- `wrong`: the code does it, but not as asked (quote the request and show the output).
- `extra`: the code does something that nobody asked for, and it adds risk or cost (branch review only).

Do not report bugs that are not about a request. The bug hunters do that.

## Design decisions and open questions
- The design decisions are in `docs/wiki/design.md`. Each decision is part of the intent. Its "Known gaps" are known bugs: report a gap again only in a new place.
- Sometimes no decision covers a behavior, and you cannot tell if it is wrong. Then it is an **open question**, not a finding. Write it in the `open_questions` list of `findings.json`. Each open question has these keys:
  - `question`: one line.
  - `today`: what the code does today.
  - `evidence`: a command and its output, or the file and line.
  - `options`: a list of 2 or 3 choices.

## Each finding
- `id`: A-<n>.
- `type`: missing, wrong or extra.
- `request`: the exact quote, with the file and line.
- `evidence`: a command and its output, or the file and line of the code.
- `why`: 1 or 2 sentences.
- `severity`: high if users get a wrong result, medium if a promise is not kept, low for other findings.

## Output: `findings.json` in your folder
```json
{"source": "<where the requests came from>", "n_requests": 0,
 "findings": [{"id": "A-1", "type": "missing|wrong|extra", "request": "...", "evidence": "...", "why": "...", "severity": "high|medium|low"}],
 "open_questions": [{"question": "<one line>", "today": "<what the code does>", "evidence": "<command and output, or file:line>", "options": ["<a>", "<b>"]}],
 "lessons": [{"what": "...", "why": "...", "evidence": ["..."]}]}
```
Your final message: the path of findings.json and the number of findings for each type. Nothing else. No em dashes.
