"""Free-text radiology-report section parser.

RadHarmony datasets expose ``sample["report"]`` as a **file path** (see
``radharmony/dataset/base.py`` — the report path is carried verbatim, e.g.
MIMIC-CXR ``files/p10/.../s5XXXXXXX.txt``). Report-generation metrics are
conventionally computed on the FINDINGS and/or IMPRESSION section, not the
raw note (which also carries indication, technique, comparison, history).

:func:`parse_report_section` reads the file (or treats the input as the
report text itself if it is not an existing path — some datasets may carry
inline text) and extracts the requested section. It is intentionally
permissive: section headers vary across institutions, so unmatched input
falls back to the whole de-headered body rather than raising.
"""

from __future__ import annotations

import os
import re

# Headers that introduce the two sections we care about, plus the headers
# that *terminate* them. Matched case-insensitively at line start.
_FINDINGS_RE = re.compile(
    r"(?is)\bFINDINGS?\s*:(.*?)(?=\n\s*(?:IMPRESSION|CONCLUSION|RECOMMENDATION|"
    r"SUMMARY|NOTIFICATION)S?\s*:|\Z)"
)
_IMPRESSION_RE = re.compile(
    r"(?is)\b(?:IMPRESSION|CONCLUSION)S?\s*:(.*?)(?=\n\s*(?:RECOMMENDATION|"
    r"NOTIFICATION|END OF IMPRESSION)S?\s*:|\Z)"
)
# Leading administrative sections to strip when nothing else matches.
_PREAMBLE_RE = re.compile(
    r"(?is)^.*?\b(?:FINDINGS?|IMPRESSION|CONCLUSION)S?\s*:"
)
# Clinical-context header (the reason for exam), used as generation *input*,
# never as a scoring target. Terminates at the next section header.
_INDICATION_RE = re.compile(
    r"(?is)\b(?:INDICATION|CLINICAL INDICATION|CLINICAL HISTORY|HISTORY|"
    r"REASON FOR (?:THE )?EXAM(?:INATION)?)S?\s*:(.*?)"
    r"(?=\n\s*(?:TECHNIQUE|COMPARISON|FINDINGS?|IMPRESSION|CONCLUSION|"
    r"HISTORY|INDICATION)S?\s*:|\Z)"
)

_VALID_SECTIONS = {"findings", "impression", "both", "full", "indication"}


def _clean(text: str) -> str:
    """Collapse whitespace/newlines to single-spaced text."""
    return re.sub(r"\s+", " ", text).strip()


def _read(source: str) -> str:
    """Return report text from a path, or the string itself if not a path."""
    try:
        if os.path.isfile(source):
            with open(source, "r", errors="ignore") as f:
                return f.read()
    except (OSError, ValueError):
        pass
    return source


def parse_report_section(source: str, section: str = "findings") -> str:
    """Extract a section from a report file path (or inline report text).

    Parameters
    ----------
    source :
        Path to a ``.txt`` report (the value RadHarmony puts at
        ``sample["report"]``), or the raw report text.
    section :
        One of ``"findings"``, ``"impression"``, ``"both"`` (findings then
        impression, space-joined), or ``"full"`` (whole note, whitespace
        normalized).

    Returns
    -------
    str
        The cleaned section text, or ``""`` if nothing usable is found
        (the caller drops empty-reference pairs before scoring).
    """
    section = section.lower()
    if section not in _VALID_SECTIONS:
        raise ValueError(
            f"section must be one of {sorted(_VALID_SECTIONS)}, got {section!r}"
        )

    raw = _read(source)
    if not raw or not raw.strip():
        return ""

    if section == "full":
        return _clean(raw)

    if section == "indication":
        # No body fallback: absence must read as "no indication" (empty),
        # never the whole note — otherwise findings would leak into the input.
        m = _INDICATION_RE.search(raw)
        return _clean(m.group(1)) if m else ""

    f_match = _FINDINGS_RE.search(raw)
    i_match = _IMPRESSION_RE.search(raw)
    findings = _clean(f_match.group(1)) if f_match else ""
    impression = _clean(i_match.group(1)) if i_match else ""

    if section == "findings":
        chosen = findings or impression
    elif section == "impression":
        chosen = impression or findings
    else:  # both
        chosen = " ".join(p for p in (findings, impression) if p)

    if chosen:
        return chosen
    # No recognizable headers — fall back to the body minus any preamble.
    return _clean(_PREAMBLE_RE.sub("", raw))
