"""Specialty and modality tags for MedPMC.

The MedPMC release has no specialty or modality field. Its metadata index has
only source_type, pmcid, image_id, parent_image_id, subfigure_index,
figure_label, shard, key and image_ext. The per-sample JSON adds caption and
parent_caption. The paper's specialty shares (radiology 25.6%, pathology
26.6%, ...) come from manual labels of 2,906 sampled images. They are not
a per-image label in the data.

So these tags are ours, not the authors'. Never call them MedPMC ground truth.
``tag_method`` on every row of the tag table records how a tag was made.

Method: a lexicon over the caption, applied in priority order. It is simple on
purpose. Each tag traces to a matched term, so a wrong tag can be found and
the lexicon fixed. Captions are a good start, because the authors' own
screening used captions and inline text (F1 93.2).

The taxonomy follows Figure 2 of the paper, so the shares can be compared with
their manual audit. If our radiology share is far from 25.6%, the tagger is wrong.

``tag_caption`` is the only function that the harmonizer uses. A model-based
tagger can replace it without an API change (for example zero-shot
MedPMC-CLIP on the images, as the authors did for skin photos). It must write a
different ``tag_method``.
"""
from __future__ import annotations

import re

TAG_METHOD = "caption-lexicon-v2"

SPECIALTIES = ("radiology", "pathology", "microscopy_nonpathology", "ophthalmology",
               "endoscopy", "dermatology", "clinical_photography",
               "physiological_signals", "non_medical", "unknown")

RADIOLOGY_MODALITIES = ("ct", "mri", "fmri", "xray", "ultrasound", "pet_spect",
                        "angiography", "mammography", "radiology_other")

# ---- radiology sub-modality lexicons (checked before the generic radiology terms)
_MOD = [
    ("fmri",        r"\bfunctional MRI\b|\bfMRI\b|\bBOLD\b|resting[- ]state"),
    ("mri",         r"\bMRI\b|\bMR imaging\b|magnetic resonance|\bT1[- ]weighted|\bT2[- ]weighted|\bFLAIR\b|\bDWI\b|\bADC map"),
    ("ct",          r"\bCT\b|computed tomograph|\bCTA\b|\bHRCT\b|\bCBCT\b|cone[- ]beam"),
    ("pet_spect",   r"\bPET\b|\bSPECT\b|\bFDG\b|scintigraph|radionuclide|\bPET/CT\b|\bPET-CT\b"),
    ("mammography", r"mammogra|\btomosynthes"),
    ("angiography", r"angiogra|\bDSA\b|arteriogra|venogra|coronary angio"),
    ("ultrasound",  r"ultrasound|ultrasonograph|\bUS image\b|sonogra|echocardiogra|\bdoppler\b"),
    ("xray",        r"\bx-?ray\b|radiograph|\bCXR\b|\bKUB\b|fluoroscop|orthopantomogram|\bOPG\b"),
]
_MOD = [(m, re.compile(p, re.I)) for m, p in _MOD]

_RADIOLOGY_GENERIC = re.compile(
    r"\bradiolog|\bimaging (?:study|studies|finding)|axial|coronal|sagittal|"
    r"contrast[- ]enhanc|\bmyelogra|\burogra|\bcholangiogra", re.I)

# ---- other specialties. Order matters: earlier entries win.
_SPEC = [
    ("pathology", r"histopatholog|histolog|\bH&E\b|haematoxylin|hematoxylin|immunohistochem|"
                  r"\bIHC\b|cytolog|biopsy specimen|\bgross (?:specimen|pathology)|"
                  r"microscopic examination|magnification\s*[x×]?\s*\d+|\b\d+\s*[x×]\s*magnif"),
    ("ophthalmology", r"fundus|retinal? (?:image|photograph|scan)|\bOCT\b|slit[- ]lamp|"
                      r"optic (?:disc|nerve head)|fluorescein angiogra|cornea|anterior segment"),
    ("endoscopy", r"endoscop|colonoscop|gastroscop|bronchoscop|laparoscop|arthroscop|"
                  r"cystoscop|hysteroscop|\bEGD\b|capsule endoscop"),
    ("dermatology", r"dermoscop|dermatoscop|skin lesion|cutaneous|\brash\b|erythema|"
                    r"papule|plaque|pustule|clinical photograph of the skin"),
    ("physiological_signals", r"\bECG\b|\bEKG\b|electrocardiogra|\bEEG\b|electroencephalogra|"
                              r"\bEMG\b|spirometr|waveform|polysomnogra"),
    # v1 caught 21.4% against 31.9%. Most non-pathology microscopy never says
    # "microscopy". It names the stain, the marker or the assay.
    ("microscopy_nonpathology", r"electron microscop|\bSEM\b|\bTEM\b|confocal|fluorescence microscop|"
                                r"immunofluorescen|scanning electron|transmission electron|"
                                r"\bDAPI\b|\bGFP\b|\bmCherry\b|phalloidin|\bDIC\b|bright[- ]?field|"
                                r"phase[- ]contrast|scale bar|\bum\b scale|stained with|counterstain|"
                                r"cell culture|cultured cells|\bin vitro\b|transfected|knockdown|"
                                r"immunostain|\bWB\b|colony formation|flow cytometr"),
    # v1 caught 0.4% against 6.2% in the paper, the worst miss. Captions rarely
    # say "clinical photography". They describe what the photo shows.
    ("clinical_photography", r"intraoperative photograph|clinical photograph|gross appearance|"
                             r"surgical (?:field|specimen|photograph|view)|pre-?operative photograph|"
                             r"post-?operative photograph|intra-?operative (?:view|image|finding)|"
                             r"occlusal view|intra-?oral|extra-?oral|clinical (?:image|view|appearance|photo)|"
                             r"macroscopic (?:view|appearance)|resected specimen|excised specimen|"
                             r"operative field|photograph of the (?:patient|lesion|wound|limb|face)|"
                             r"wound|prosthes[ie]s in place|cast|physical examination"),
]
_SPEC = [(s, re.compile(p, re.I)) for s, p in _SPEC]

# Explicit non-medical cues. The corpus is already medically filtered, so this is a
# backstop for panels that survived that filter, not a primary classifier.
_NONMED = re.compile(
    r"\b(?:flow ?chart|flowchart|schematic|study design|consort diagram|"
    r"bar (?:chart|graph)|scatter ?plot|kaplan[- ]meier|forest plot|"
    r"box ?plot|histogram|western blot|gel electrophoresis)\b", re.I)


def _match(text: str) -> dict | None:
    """Run the lexicons over one text. Returns None when nothing matches."""
    if not text or not text.strip():
        return None
    for mod, pat in _MOD:
        m = pat.search(text)
        if m:
            return {"specialty": "radiology", "modality": mod,
                    "tag_confidence": "high", "tag_evidence": m.group(0)[:40]}
    for spec, pat in _SPEC:
        m = pat.search(text)
        if m:
            return {"specialty": spec, "modality": None,
                    "tag_confidence": "high", "tag_evidence": m.group(0)[:40]}
    m = _RADIOLOGY_GENERIC.search(text)
    if m:
        return {"specialty": "radiology", "modality": "radiology_other",
                "tag_confidence": "low", "tag_evidence": m.group(0)[:40]}
    m = _NONMED.search(text)
    if m:
        return {"specialty": "non_medical", "modality": None,
                "tag_confidence": "low", "tag_evidence": m.group(0)[:40]}
    return None


def tag_caption(caption: str | None, parent_caption: str | None = None) -> dict:
    """Return {specialty, modality, tag_confidence, tag_evidence, tag_source}.

    The own caption comes first. v1 joined the panel caption with the parent
    caption, and the modality leaked between panels. Example: a panel with
    "(a) Occlusal view showing supernumerary teeth" (an intraoral photo) got
    the tag X-ray with high confidence. Another panel of the figure said
    "(b) Occlusal radiograph of the maxilla". This raised radiology and lowered
    clinical photography, the same error as in the paper's audit.

    The parent caption helps when the own caption is a short fragment such as
    "(e) Axial section". So it is a fallback only. Its tag gets low confidence
    and ``tag_source="parent"``. ``min_tag_confidence="high"`` removes these tags.
    """
    own = _match(caption or "")
    if own is not None:
        own["tag_source"] = "own"
        return own

    inherited = _match(parent_caption or "")
    if inherited is not None:
        inherited["tag_confidence"] = "low"      # never trust an inherited tag
        inherited["tag_source"] = "parent"
        return inherited

    return {"specialty": "unknown", "modality": None, "tag_confidence": None,
            "tag_evidence": None, "tag_source": None}
