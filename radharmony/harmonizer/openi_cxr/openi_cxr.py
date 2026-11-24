"""Harmonizer for the OpenI Indiana University Chest X-Ray dataset (IU X-Ray).

3,955 reports with 7,470 chest X-ray images from the Indiana University hospital network. The
NLM/LHNCBC released them under an open-access license. Source:
https://openi.nlm.nih.gov/faq?download=true (PNG: ``NLMCXR_png.tgz``, DICOM: ``NLMCXR_dcm.tgz``,
reports: ``NLMCXR_reports.tgz``). Cite: Demner-Fushman D, et al. *JAMIA* 2016;23(2):304-310.

``base_dir`` has ``ecgen-radiology/<n>.xml`` (reports), ``images/<id>.png`` and one DICOM
folder for each patient (``<n>/<n>_IM-XXXX-XXXX.dcm``). The ``parentImage id`` of the XML (for
example ``CXR1_1_IM-0001-3001``) is the file name of the PNG.

The table has one row for each image (7,470 rows). A report has about 2 images (frontal and
lateral), and both rows get the same report text.

Labels come from the ``<MeSH><major>`` tags. A term matches whole words only. The terms give
CheXpert-style binary columns. The raw terms stay in ``mesh_major`` (JSON list). A report with
only the term "No Indexing" has no label, so all its labels are NaN (D-05).

The figure number gives a first view (F1 frontal, F2 lateral). The DICOM ``ViewPosition`` tag
corrects it if the tag says AP or a lateral view. A PA tag counts only on F1, because many
lateral images also have PA (V-H2-40).

Text columns: ``findings``, ``impression``, ``indication`` (``AbstractText`` labels) and
``report`` (findings and impression joined).

Details: ``docs/wiki/reference/datasets/vqa-text/openi_cxr.md``.
"""

import json
import os
import re
import xml.etree.ElementTree as ET

import pandas as pd

from .._dicom_tags import read_dicom_tags, tag_values
from ..base import BaseHarmonizer


# MeSH terms and the label that each one sets: (label column, phrases).
# A phrase matches whole words of the MeSH term, in any case. Part of a word does not match:
# "tube" is not in "Tuberculosis" and "normal" is not in "Abnormal" (V-H2-7). A final "s"
# can follow a phrase ("Tubes").
_MESH_LABEL_MAP = [
    ("no_finding",       ["normal"]),
    ("cardiomegaly",     ["cardiomegaly"]),
    ("edema",            ["pulmonary edema", "pulmonary congestion", "edema"]),
    ("atelectasis",      ["atelectasis", "pulmonary atelectasis"]),
    ("consolidation",    ["consolidation", "pneumonia", "airspace disease"]),
    ("pleural_effusion", ["pleural effusion", "effusion/pleural"]),
    ("pneumothorax",     ["pneumothorax"]),
    # The MeSH terms of a pacemaker study are "Implanted Medical Device" or "Medical Device" (V-H2-42).
    ("support_devices",  ["catheter", "tube", "medical device", "stent"]),
]

# A report with only these terms has no label at all (D-05, V-H2-41).
_NO_LABEL_TERMS = {"no indexing"}

# A device term on the breast ("Medical Device/breast/bilateral") is a breast implant.
_BREAST_RE = re.compile(r"\bbreast\b")

# DICOM ViewPosition values of a lateral image.
_LATERAL_TAGS = {"LL", "RL", "LAT", "LATERAL", "LARGE LATERAL"}

_LABEL_COLS = [col for col, _ in _MESH_LABEL_MAP]

_MESH_LABEL_RE = [
    (col, re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")s?\b"))
    for col, phrases in _MESH_LABEL_MAP
]


def _mesh_to_labels(terms: list[str]) -> dict:
    """Map the MeSH major terms of a report to a dict of binary labels.

    A list with only "No Indexing" gives NaN in every label (D-05).
    """
    lower_terms = [t.lower() for t in terms]
    if all(t in _NO_LABEL_TERMS for t in lower_terms):
        return {col: float("nan") for col, _ in _MESH_LABEL_RE}
    return {col: int(any(rx.search(lt) and not (col == "support_devices" and _BREAST_RE.search(lt))
                         for lt in lower_terms))
            for col, rx in _MESH_LABEL_RE}


def _view_from_tag(figure_view: str, tag: str) -> str:
    """Return the view of one image from its figure view and its DICOM ``ViewPosition`` tag.

    The tag wins when it says AP or a lateral view. A PA tag counts only on a frontal
    figure, because many lateral images also have PA. With no tag, the figure view stays.
    """
    tag = (tag or "").strip().upper()
    if tag == "AP":
        return "AP"
    if tag in _LATERAL_TAGS:
        return "lateral"
    if tag == "PA" and figure_view == "frontal":
        return "PA"
    return figure_view


def _parse_xml(path: str) -> list[dict]:
    """Parse one XML report into a list of rows, one for each ``parentImage``."""
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return []

    root = tree.getroot()

    def _text(label: str) -> str | None:
        el = root.find(f'.//AbstractText[@Label="{label}"]')
        t = el.text.strip() if el is not None and el.text else None
        return t if t and t.lower() not in ("none.", "none", "") else None

    findings   = _text("FINDINGS")
    impression = _text("IMPRESSION")
    indication = _text("INDICATION")
    report     = " ".join(filter(None, [findings, impression])) or None

    mesh_terms = [
        m.text.strip()
        for m in root.findall(".//MeSH/major")
        if m.text and m.text.strip()
    ]
    labels = _mesh_to_labels(mesh_terms)

    uid = root.findtext("uId[@id]") or root.find("uId").get("id") if root.find("uId") is not None else None
    study_id = root.findtext("pmcId") or (root.find("pmcId").get("id") if root.find("pmcId") is not None else None)
    patient_id = uid  # one patient for each report

    images = root.findall(".//parentImage")
    rows = []
    for img in images:
        img_id = img.get("id")
        if not img_id:
            continue
        # First view from the figure number. _build_view_position() corrects it with the DICOM tag.
        fig_id = img.findtext("figureId", "").strip()
        view = "frontal" if fig_id == "F1" else ("lateral" if fig_id == "F2" else "other")

        # Two XML id patterns give {n}/{n}_IM-XXXX-XXXX.dcm:
        #   CXR1_1_IM-0001-3001 gives 1/1_IM-0001-3001.dcm
        #   CXR2_IM-0652-1001 gives 2/2_IM-0652-1001.dcm
        _dcm = re.match(r'^CXR(\d+)_(?:\1_)?(.*)', img_id)
        if _dcm:
            _n, _rest = _dcm.group(1), _dcm.group(2)
            dicom_path = f"{_n}/{_n}_{_rest}.dcm"
        else:
            dicom_path = None

        row = {
            "patient_id":    patient_id,
            "study_id":      study_id,
            "image_path":    f"images/{img_id}.png",
            "dicom_path":    dicom_path,
            "view_position": view,
            "report":        report,
            "findings":      findings,
            "impression":    impression,
            "indication":    indication,
            "mesh_major":    json.dumps(mesh_terms),
        }
        row.update(labels)
        rows.append(row)

    return rows


class OpenICXRHarmonizer(BaseHarmonizer):
    """Harmonize the OpenI Indiana University CXR dataset.

    Args:
        base_dir: Dataset root, with ``ecgen-radiology/`` and ``images/``, for example
            ``.../OpenI-IU-CXR/``.
    """

    LABEL_COLS = _LABEL_COLS

    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    EXTRA_OUTPUT_COLS = ["dicom_path", "findings", "impression", "indication", "mesh_major"]

    def __init__(self, base_dir: str):
        self.base_dir = os.path.expanduser(base_dir)
        super().__init__(csv_path=None)

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["base_dir"] = self.base_dir
        snap.pop("csv_path", None)
        return snap

    def _read_source(self) -> pd.DataFrame:
        """Parse the XML reports into one row for each image, in the order of the report number."""
        report_dir = os.path.join(self.base_dir, "ecgen-radiology")
        if not os.path.isdir(report_dir):
            raise FileNotFoundError(
                f"ecgen-radiology/ not found under {self.base_dir!r}. "
                "Extract NLMCXR_reports.tgz first."
            )

        rows = []
        for fname in sorted(os.listdir(report_dir), key=lambda f: int(f.split(".")[0]) if f.split(".")[0].isdigit() else 0):
            if not fname.endswith(".xml"):
                continue
            rows.extend(_parse_xml(os.path.join(report_dir, fname)))
        return pd.DataFrame(rows)

    def _build_patient_id(self) -> None:
        """``_parse_xml`` sets ``patient_id``."""

    def _build_study_id(self) -> None:
        """``_parse_xml`` sets ``study_id``."""

    def _build_image_path(self) -> None:
        """``_parse_xml`` sets ``image_path``."""

    def _build_view_position(self) -> None:
        """Correct the figure view of each row with the DICOM ``ViewPosition`` tag (V-H2-40).

        A row whose DICOM file is missing or cannot be read keeps its figure view.
        ``read_dicom_tags`` warns with the number of these files.
        """
        if not len(self.df):
            return
        headers = read_dicom_tags(self.base_dir, self.df["dicom_path"].dropna().unique().tolist(), ["ViewPosition"])
        tags = tag_values(headers, self.df["dicom_path"], "ViewPosition")
        self.df["view_position"] = [_view_from_tag(v, t if isinstance(t, str) else "")
                                    for v, t in zip(self.df["view_position"], tags, strict=True)]
