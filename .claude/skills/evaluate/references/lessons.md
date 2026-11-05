# Lessons checklist

Lessons are changes to the repo, so that the same mistake cannot happen again.
Write them after the 3 review parts. Each lesson says what to change, why, and which findings show it.
Use these 7 groups. Leave out a group with no lesson.

1. **Where the findings were.** Which repo parts or files had the most findings? Is one design the cause of many findings (for example one base method)?
2. **Automatic checks.** Which findings can a computer check? Write a draft row for `eval/RULES.md` ("Checked by the computer"). Say which check to add: a test in `tests/test_lessons.py`, a check in `scripts/evalkit/lessons.py`, a ruff rule or an STE rule.
3. **Judgement rules.** Which findings need a new judgement rule, or a clearer one? Write the draft row for `eval/RULES.md`.
4. **CLAUDE.md size.** Did a rule get missed because it is hidden in a long file? Move it to the right place (`CLAUDE.md` only for rules for every task).
5. **Tool cost.** Which steps were slow or failed (an agent stopped, a command timed out, a missing package)? What change makes the next run cheaper?
6. **Rules that change nothing.** Which rule in `eval/RULES.md` found nothing for a long time, or is always ignored? Suggest to remove or change it.
7. **Missing information.** What did the agents need and not find (a data path, a model environment, a spec)? Where must it go?

Do not write a lesson with no evidence. Do not commit a draft rule. The user approves each one.
