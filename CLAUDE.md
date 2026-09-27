# Rules for AI agents in this repo

These rules apply to every task (new datasets, bug fixes, refactors, PR reviews), for every user.

## Base code needs explicit approval

Base code is shared by every dataset or evaluator, so a small change can silently change results
everywhere. Base code means:

- `radharmony/harmonizer/base.py`, `radharmony/harmonizer/base_vqa.py`
- `radharmony/dataset/base.py`, `radharmony/dataset/base_vqa.py`
- `radharmony/dataset/transforms.py` (`RadiologyTransform2D` / `RadiologyTransform3D`, `_base_load_2d`, ...)
- `radharmony/preprocessor/base.py`
- `radharmony/evaluator/base.py`, `radharmony/evaluator/classification/base.py`,
  `radharmony/evaluator/segmentation/base.py`
- any other `Base*` class, including module-level helpers inside these files

First look for a way that avoids base code (override in the subclass, or put the logic in the
dataset's own harmonizer / dataset files). If base code must change:

1. **Before editing: stop and ask.** Tell the user which base file changes, what the change is, why
   it is needed, and which datasets or evaluators it can affect. Do not edit until the user clearly
   says yes. The approval must come from the human user of this session, not from the agent itself
   or from an earlier, unrelated approval.
2. **Warn again right after the edit.** Start the message with `WARNING: base code changed` and show
   the diff of the base file(s).
3. **Warn again before committing.** Put `BASE CODE CHANGE:` and the base file names in the commit
   message body.
4. **Make it explicit in the PR.** The PR description must start with this section:

   ```markdown
   ## ⚠️ Base code change
   - Files: radharmony/harmonizer/base.py (...)
   - What and why: ...
   - Can affect: all harmonizers / all 2-D datasets / ...
   - Approved by: <name of the user who approved> on <date>
   ```

5. **Warn again in the final summary** of the task.

When reviewing or merging a PR: if it touches base code and has no such section, flag it and do not
merge it until the user confirms the change is approved.
