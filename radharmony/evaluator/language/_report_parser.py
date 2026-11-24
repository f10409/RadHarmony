"""Free-text radiology-report section parser.

RadHarmony datasets give ``sample["report"]`` as the report text. The
harmonizer reads the report files into the ``report`` column. Report-generation
metrics are conventionally computed on the FINDINGS and/or IMPRESSION section,
not the raw note (which also carries indication, technique, comparison, history).

:func:`parse_report_section` takes the report text, or reads the file when
the input is a path to an existing file, and extracts the requested section. It is intentionally
permissive: section headers vary across institutions, so unmatched input
falls back to the whole de-headered body rather than raising.
"""

from __future__ import annotations

import os
import re


def _header(names: str) -> str:
    """Return a regex for a section header: a name, an optional "S", then ":".

    A header is at the start of a line (any case), or in capitals after other
    text on the line (one-line reports). A lowercase word in a sentence, for
    example "new findings: none", is not a header (V-E-23).
    """
    return rf"(?:^[ \t]*(?i:(?:{names})S?)|\b(?:{names})S?)[ \t]*:"


# Headers of other sections. A FINDINGS or IMPRESSION section stops at them (V-E-8).
_OTHER = "COMPARISON|TECHNIQUE|INDICATION|HISTORY|EXAMINATION"
# Headers that start the two sections we need, plus the headers
# that *end* them.
_FINDINGS_RE = re.compile(
    rf"(?ms){_header('FINDING')}(.*?)"
    rf"(?={_header('IMPRESSION|CONCLUSION|RECOMMENDATION|SUMMARY|NOTIFICATION|' + _OTHER)}|\Z)"
)
# The impression also stops at FINDINGS, so a FINDINGS section after it is not in it (D-38, Q-9).
_IMPRESSION_RE = re.compile(
    rf"(?ms){_header('IMPRESSION|CONCLUSION')}(.*?)"
    rf"(?={_header('FINDING|RECOMMENDATION|NOTIFICATION|END OF IMPRESSION|' + _OTHER)}|\Z)"
)
# Leading administrative sections to strip when nothing else matches. It uses the
# same header rule, so a lowercase "findings:" in a sentence does not cut the report (V-E-28).
_PREAMBLE_RE = re.compile(
    rf"(?ms)\A.*?{_header('FINDING|IMPRESSION|CONCLUSION')}"
)
# The indication header (the reason for the exam). It is generation input only,
# never a scoring target. It uses the same header rule as above, so a lowercase
# "history:" in a sentence is not a header (V-E-23). It stops at the next header,
# or at a blank line (a MIMIC report can start its findings with no header).
_INDICATION_RE = re.compile(
    rf"(?ms){_header('INDICATION|CLINICAL INDICATION|CLINICAL HISTORY|HISTORY|REASON FOR (?:THE )?EXAM(?:INATION)?')}"
    rf"(.*?)(?={_header('FINDING|IMPRESSION|CONCLUSION|RECOMMENDATION|NOTIFICATION|' + _OTHER)}|\n[ \t]*\n|\Z)"
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
    """Extract a section from report text (or from a report file path).

    Parameters
    ----------
    source :
        The report text (the value RadHarmony puts at ``sample["report"]``),
        or a path to a ``.txt`` report.
    section :
        One of ``"findings"``, ``"impression"``, ``"both"`` (findings then
        impression, space-joined), or ``"full"`` (whole note, whitespace
        normalized). ``"indication"`` gives the reason for the exam. It is a
        generation input only, and ``""`` when the report has none.

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
        # No fallback to the body: a missing header gives "" (no indication).
        # The whole note would put the findings into the generation input.
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
    # No recognizable headers: use the body minus any preamble.
    return _clean(_PREAMBLE_RE.sub("", raw))
