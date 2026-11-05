# Decisions checklist (one item at a time)

Use this checklist when the user must make a choice for each item. Both `/evaluate` and `/fix` use it.

## When to use it
- An **open question**: a review agent found a behavior that no decision in `docs/wiki/design.md` covers.
- A **base-code change**, before the edit (see `CLAUDE.md`).
- A **draft rule** for `eval/RULES.md` (L- or J-), from the lessons.
- A **fix with more than one correct behavior**, when `docs/wiki/design.md` does not choose one.
- A **severity change**: a severity that does not match the definitions in `references/hunter.md`
  ("Severity"). The checker's own changes (`severity_changed`) are already done. Do not ask about them.

## Order
Keep the order that the skill gives. Inside one group, start with the item with the highest severity
(high, then medium, then low).

## For each item
1. **Check the facts first.** Read the code or run a short command. Do not ask about a claim that you did
   not check.
2. **Say what the code does today**, in plain words, with a concrete example: numbers, a short code
   snippet or a `file:line`.
3. **Give your recommendation**, the reason and the cost. For example: "This needs a base-code change.
   It needs your approval at the time of the edit."
4. **Ask one question** with the AskUserQuestion tool. Give 2 or 3 options. Put your recommendation first
   and add "(Recommended)" to its label. For an open question, one option is "Leave open". If the tool is
   not available, ask in the chat and wait.
5. **Read the whole answer**, also any note that the user adds. A note can add a rule or a time, for
   example "this applies to all datasets" or "do it after the pipeline is done". Write that note in the
   record, as D-31 in `docs/wiki/design.md` does.
6. **Record the answer before the next item:**
   - Design decision: add a D- entry to `docs/wiki/design.md`. Use the lines in its section "How to read an
     entry", with "Status: Decided (<name>, <date>)". Add the entry to its index.
   - Left open: add a D- entry with the status "Open" to `docs/wiki/design.md`. Put it in its topic
     section, not under "Open questions", and add it to the index. If the user says "skip the questions", every open question
     gets such an entry. When the user decides it later, change its status. The ID stays the same.
   - Draft rule in `/evaluate`: write "approved, to build with `/fix`" next to the rule in the lessons of
     `report.md`. Do not add the row to `eval/RULES.md` yet. `/fix` adds it after the check is built and
     proved (`eval/RULES.md`, "How to add a rule").
   - Base-code approval: follow the steps in `CLAUDE.md`.
   - Severity change: update `findings.json` and `report.md` of the run. Do this after the last run of
     `merge.py`, because `merge.py` writes `findings.json` again.
7. Go to the next item.

At the end, list the answers in one table: item, answer, and what still needs work.
