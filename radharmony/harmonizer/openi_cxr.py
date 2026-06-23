"""Harmonizer for the OpenI Indiana University Chest X-Ray dataset (IU X-Ray).

3,955 radiology reports with 7,470 paired chest X-ray images from the Indiana
University hospital network, released by the NLM/LHNCBC under open-access license.

**Source:**
  https://openi.nlm.nih.gov/faq?download=true
  - PNG images: NLMCXR_png.tgz  (~1.36 GB)
  - DICOM:      NLMCXR_dcm.tgz  (~75 GB)
  - Reports:    NLMCXR_reports.tgz (~1.1 MB)

**Cite:**
  Demner-Fushman D, et al. Preparing a collection of radiology examinations for
  distribution and retrieval. *JAMIA*. 2016;23(2):304–310.
  https://pubmed.ncbi.nlm.nih.gov/26133894/

**Layout after extraction:**

    <base_dir>/
      ecgen-radiology/
        1.xml  ...  3955.xml
      images/
        CXR1_1_IM-0001-3001.png
        ...
      1/                        ← DICOM folders (one per patient)
        1_IM-0001-3001.dcm
        1_IM-0001-4001.dcm
      2/
        ...

**One row per image** (7,470 rows). Each report contributes one row per associated
image (usually 2: frontal PA + lateral). The report text (FINDINGS + IMPRESSION)
is duplicated across both rows.

**Image IDs and filenames:** the ``parentImage id`` attribute in the XML (e.g.
``CXR1_1_IM-0001-3001``) corresponds to ``images/<id>.png``.

**Labels:** extracted from ``<MeSH><major>`` tags using prefix matching to
produce CheXpert-compatible binary columns. Raw terms are preserved as a
JSON list in ``mesh_major``.

**Report text fields:**
  ``findings``   — ``<AbstractText Label="FINDINGS">``
  ``impression`` — ``<AbstractText Label="IMPRESSION">``
  ``report``     — findings + " " + impression (concatenated)
  ``indication`` — ``<AbstractText Label="INDICATION">``

**Harmonized columns:**
  ``patient_id``, ``study_id``, ``image_path``, ``dicom_path``,
  ``view_position``, ``report``, ``findings``, ``impression``, ``indication``,
  ``mesh_major`` (JSON list),
  ``no_finding``, ``cardiomegaly``, ``edema``, ``atelectasis``,
  ``consolidation``, ``pleural_effusion``, ``pneumothorax``,
  ``support_devices``
"""

import json
import os
import re
import xml.etree.ElementTree as ET

import pandas as pd

from .base import BaseHarmonizer


# ------------------------------------------------------------------
# MeSH → CheXpert-compatible binary label mapping
# Each entry: (binary_col, list_of_mesh_prefixes_that_trigger_it)
# Matching is case-insensitive prefix/substring on the raw MeSH term.
# ------------------------------------------------------------------
_MESH_LABEL_MAP = [
    ("no_finding",       ["normal"]),
    ("cardiomegaly",     ["cardiomegaly"]),
    ("edema",            ["pulmonary edema", "pulmonary congestion", "edema"]),
    ("atelectasis",      ["atelectasis", "pulmonary atelectasis", "lung/hypoinflation"]),
    ("consolidation",    ["consolidation", "pneumonia", "airspace disease"]),
    ("pleural_effusion", ["pleural effusion", "effusion/pleural"]),
    ("pneumothorax",     ["pneumothorax"]),
    ("support_devices",  ["catheters", "pacemaker", "leads", "tube", "defibrillator"]),
]

_LABEL_COLS = [col for col, _ in _MESH_LABEL_MAP]


def _mesh_to_labels(terms: list[str]) -> dict:
    """Map a list of raw MeSH major terms to binary label dict."""
    lower_terms = [t.lower() for t in terms]
    result = {col: 0 for col, _ in _MESH_LABEL_MAP}
    for col, prefixes in _MESH_LABEL_MAP:
        for lt in lower_terms:
            if any(lt.startswith(p) or p in lt for p in prefixes):
                result[col] = 1
                break
    return result


def _parse_xml(path: str) -> list[dict]:
    """Parse one XML report file into a list of rows (one per parentImage)."""
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return []

    root = tree.getroot()

    # --- report text fields ---
    def _text(label: str) -> str | None:
        el = root.find(f'.//AbstractText[@Label="{label}"]')
        t = el.text.strip() if el is not None and el.text else None
        return t if t and t.lower() not in ("none.", "none", "") else None

    findings   = _text("FINDINGS")
    impression = _text("IMPRESSION")
    indication = _text("INDICATION")
    report     = " ".join(filter(None, [findings, impression])) or None

    # --- MeSH labels ---
    mesh_terms = [
        m.text.strip()
        for m in root.findall(".//MeSH/major")
        if m.text and m.text.strip()
    ]
    labels = _mesh_to_labels(mesh_terms)

    # --- IDs ---
    uid = root.findtext("uId[@id]") or root.find("uId").get("id") if root.find("uId") is not None else None
    study_id = root.findtext("pmcId") or (root.find("pmcId").get("id") if root.find("pmcId") is not None else None)
    patient_id = uid  # one patient per report in this dataset

    # --- images ---
    images = root.findall(".//parentImage")
    rows = []
    for idx, img in enumerate(images):
        img_id = img.get("id")
        if not img_id:
            continue
        # view position: F1 = frontal, F2+ = lateral (heuristic)
        fig_id = img.findtext("figureId", "").strip()
        view = "frontal" if fig_id == "F1" else ("lateral" if fig_id == "F2" else "other")

        # DICOM path: two XML patterns both map to {n}/{n}_IM-XXXX-XXXX.dcm
        #   CXR1_1_IM-0001-3001  →  1/1_IM-0001-3001.dcm
        #   CXR2_IM-0652-1001    →  2/2_IM-0652-1001.dcm
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
        base_dir: Dataset root — the directory containing ``ecgen-radiology/``
            and ``images/``, e.g. ``.../OpenI-IU-CXR/``.
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

    def harmonize(self, *args, **kwargs) -> pd.DataFrame:
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

        self.df = pd.DataFrame(rows)
        self.df.dropna(subset=["image_path"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self._select_harmonized_columns(self.df)
