"""Clean a generated report before scoring (D-50).

This module imports only ``re``. The Stage B script
(``notebooks/evaluator/score_reports_radeval.py``) loads this file by its path,
so it runs in the RadEval env with no ``radharmony`` install.
"""

from __future__ import annotations

import re

# CheXagent-2 writes 2 grounded formats.
# MIMIC DICOM input gives "[Region: Subregion] **finding**".
# Emory PNG input gives "<|ref|> finding <|/ref|> <|box|> (x,y),(x,y) <|/box|>".
# Both lower BLEU, ROUGE and F1-CheXbert against free-text references.
_GROUNDED_TAG = re.compile(r"\[[A-Za-z][A-Za-z _/&\-]+:\s*[A-Za-z][A-Za-z _/&\-]+\]\s*")
_REF_OPEN = re.compile(r"<\|ref\|>\s*")
_REF_CLOSE = re.compile(r"\s*<\|/ref\|>")
_BOX = re.compile(r"<\|box\|>.*?<\|/box\|>", re.DOTALL)
_BOLD_MARK = re.compile(r"\*+")
_WS = re.compile(r"\s+")


def normalize_hypothesis(s: str) -> str:
    """Remove the grounding tags, boxes, ref marks and bold marks from a generated report.

    Parameters
    ----------
    s :
        The generated report.

    Returns
    -------
    str
        The text with single spaces. Example:
        ``"[Lungs: Pleura] **small effusion**."`` gives ``"small effusion."``.
    """
    s = _GROUNDED_TAG.sub("", s)
    s = _BOX.sub("", s)
    s = _REF_OPEN.sub("", s)
    s = _REF_CLOSE.sub("", s)
    s = _BOLD_MARK.sub("", s)
    return _WS.sub(" ", s).strip()
