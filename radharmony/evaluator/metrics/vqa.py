"""Metrics for visual question-answering (VQA) evaluation.

Follows the MedGemma technical report (§3.4, Table 9): the headline numbers
are the average **tokenized F1** across *all* QAs and **accuracy on the
yes/no subset**; SLAKE additionally reports open-ended **token recall**.

"Tokenized F1" is the standard VQA-RAD / SLAKE token-overlap F1: normalize
each string (lowercase, strip punctuation and articles, collapse
whitespace), then compute the multiset token-overlap F1 between the
prediction and the reference. These are the same normalization conventions
used for SQuAD-style token F1.
"""

from __future__ import annotations

import re

_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[^a-z0-9 ]")
_YES_NO = ("yes", "no")


def normalize_answer(s) -> list[str]:
    """Lowercase, strip punctuation + articles, split into tokens."""
    return [t for t in _PUNCT.sub(" ", str(s).lower()).split() if t not in _ARTICLES]


def _overlap(pred_tokens: list[str], ref_tokens: list[str]) -> int:
    left = list(ref_tokens)
    common = 0
    for t in pred_tokens:
        if t in left:
            common += 1
            left.remove(t)
    return common


def tokenized_f1(pred, ref) -> float:
    """Token-overlap F1 between prediction and reference (SQuAD-style)."""
    p, r = normalize_answer(pred), normalize_answer(ref)
    if not p or not r:
        # Both empty → perfect; one empty → zero.
        return float(p == r)
    common = _overlap(p, r)
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def token_recall(pred, ref) -> float:
    """Fraction of reference tokens recovered by the prediction."""
    p, r = normalize_answer(pred), normalize_answer(ref)
    if not r:
        return 0.0
    return _overlap(p, r) / len(r)


def extract_yes_no(text):
    """Return the first ``"yes"``/``"no"`` token in *text*, else ``None``.

    Used to grade closed (yes/no) answers even when the model replies with a
    fuller sentence (the concise-answer prompt keeps this rare).
    """
    for t in normalize_answer(text):
        if t in _YES_NO:
            return t
    return None


def is_yes_no_answer(ref) -> bool:
    """True when the *reference* answer is exactly ``"yes"`` or ``"no"``.

    This is MedGemma's definition of the closed-ended subset for accuracy.
    """
    return str(ref).strip().lower() in _YES_NO
