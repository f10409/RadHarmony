"""Controlled vocabulary for imaging modality, and the precedence rule for labels.

Two spellings, one concept. RadHarmony stores modality in plain lowercase
(``mri``, ``ct``, ``xray``). These strings are in the tag tables of 11M rows,
in the clinician labels of 2026-08-22, and in every call such as
``modalities=["ct"]``. Clinical systems use the DICOM codes (``MR``, ``CT``,
``CR``/``DX``). A rename would break all of these without an error. A missed
call returns zero rows.

So this table is the one source for both spellings. Storage keeps the old
names, display uses the clinical names, and ``normalize()`` accepts both.

Label precedence (set by Judy, 2026-08-22): ``human > clip > other``. On the
same day, a clinician checked 151 tagger conflicts. CLIP named the right
modality in 72.8% and the caption lexicon in 10.6%. See
:data:`SOURCE_PRECEDENCE`.
"""
from __future__ import annotations

#: canonical key -> (DICOM code, display label, full name, aliases accepted on input)
MODALITY_VOCAB: dict[str, dict] = {
    "ct":          {"dicom": "CT",      "display": "CT",
                    "name": "Computed Tomography",
                    "aliases": ("ct", "CT", "computed tomography", "cat scan")},
    "mri":         {"dicom": "MR",      "display": "MR",
                    "name": "Magnetic Resonance",
                    "aliases": ("mri", "MRI", "MR", "mr", "magnetic resonance")},
    "xray":        {"dicom": "CR/DX",   "display": "XR",
                    "name": "Radiography",
                    # DICOM separates computed (CR) from digital (DX) radiography.
                    # Nothing in the tag data distinguishes them, so "XR" is honest
                    # and picking CR or DX would be a guess.
                    "aliases": ("xray", "x-ray", "XR", "xr", "CR", "DX",
                                "radiograph", "radiography")},
    "ultrasound":  {"dicom": "US",      "display": "US",
                    "name": "Ultrasound",
                    "aliases": ("ultrasound", "US", "us", "sonography", "echo")},
    "angiography": {"dicom": "XA",      "display": "XA",
                    "name": "X-Ray Angiography",
                    "aliases": ("angiography", "XA", "xa", "angiogram", "angio")},
    "pet_spect":   {"dicom": "PT/NM",   "display": "PT/NM",
                    "name": "PET / Nuclear Medicine",
                    # This merge was already there. PET (PT) and SPECT (NM) are
                    # different modalities, but the tag data cannot separate them.
                    # The new name only makes the merge easier to see.
                    "aliases": ("pet_spect", "pet", "PET", "spect", "SPECT",
                                "PT", "NM", "PT/NM", "nuclear medicine")},
    "mammography": {"dicom": "MG",      "display": "MG",
                    "name": "Mammography",
                    "aliases": ("mammography", "MG", "mg", "mammogram")},
}

#: All values that the harmonizer accepts as a modality filter.
RADIOLOGY_MODALITIES = tuple(MODALITY_VOCAB)

#: Lowercased alias -> canonical key.
_ALIAS = {al.lower(): k for k, v in MODALITY_VOCAB.items() for al in v["aliases"]}

#: Which label wins when sources disagree. Lower index wins.
#: Set by Judy 2026-08-22: human > clip > other.
SOURCE_PRECEDENCE = ("human", "clip", "figure_rollup", "caption")


def normalize(value: str | None) -> str | None:
    """Canonical key for any accepted spelling; None for unknown or empty.

    Returns None rather than raising so a corpus scan is not stopped by one odd
    string -- callers that need strictness should compare against the result.
    """
    if value is None:
        return None
    return _ALIAS.get(str(value).strip().lower())


def display(value: str | None) -> str | None:
    """Clinical display label (``MR``, ``CT``, ``XR``) for any accepted spelling."""
    k = normalize(value)
    return MODALITY_VOCAB[k]["display"] if k else None


def dicom(value: str | None) -> str | None:
    """DICOM modality code. ``xray`` and ``pet_spect`` return merged codes."""
    k = normalize(value)
    return MODALITY_VOCAB[k]["dicom"] if k else None


def resolve(labels: dict[str, str | None]) -> tuple[str | None, str | None]:
    """Pick the winning modality from ``{source: value}`` by SOURCE_PRECEDENCE.

    Returns ``(canonical_modality, winning_source)``; ``(None, None)`` when no
    source carries a recognisable value.

    >>> resolve({"caption": "ct", "clip": "mri"})
    ('mri', 'clip')
    >>> resolve({"human": "xray", "clip": "mri"})
    ('xray', 'human')
    """
    for src in SOURCE_PRECEDENCE:
        v = normalize(labels.get(src))
        if v is not None:
            return v, src
    return None, None
