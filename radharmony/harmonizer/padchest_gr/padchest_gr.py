"""Harmonizer for PadChest-GR (Castro et al., 2024; BIMCV).

The grounded-report release of PadChest: 4,555 chest radiographs of 4,459 patients. Each has a
radiologist report split into sentences. A sentence has a type (abnormal or not), labels from
the PadChest vocabulary and, if the finding has a place, boxes on the image. The images are
PadChest images (same ``ImageID`` and pixels), so this set overlaps ``padchest`` completely.
Do not train on one and test on the other without a check (see "Overlap").

Counts (release 2024-08-19, checked on 2026-09-28): 4,555 images. They are the same in the
JSON and in ``master_table.csv`` (one row for each finding, 8,787 rows). 10,479 finding
sentences, 7,037 abnormal. 7,733 boxes on 6,217 abnormal findings, on 3,008 images. The split
has no patient in two parts: train 3,185, validation 455, test 915. 1,456 images have no
abnormal finding. They are normal rows, with all labels 0 and ``bbox == "[]"``.

Data facts (details and checks: ``docs/wiki/reference/datasets/cxr-boxes-masks/padchest_gr.md``):

1. The zip extracts to ``BIMCV-Padchest-GR `` with a trailing space. ``base_image_dir`` is
   this folder or a link to it.
2. The images are in a 37-volume split zip. ``unzip`` fails, ``7z x PadChest_GR.zip.001``
   works. The result must be ``<base_image_dir>/PadChest_GR/``.
3. A release box is ``[x0, y0, x1, y1]`` (fractions of the original image). RadHarmony uses
   ``[y0, y1, x0, x1]``. Without the swap, every box is transposed. Laterality confirms it: of
   2,169 boxes whose sentence names one side, 93.0% have their center on that side (patient
   left is image right).
4. No box does not mean no finding. 820 abnormal findings have no box (``pectum carinatum``,
   ``pectum excavatum`` and ``vertebral fracture`` never have one).
5. ``master_table.csv`` has one row for each finding. Patient, split, sex and age must agree
   on all rows of an image. The code checks this.
6. 288 findings have 2 or 3 labels. Their boxes belong to the sentence, so ``bbox_labels``
   joins the labels with `` + ``.
7. ``extra_boxes`` is a second, looser box set that the release does not describe (6,265
   boxes). It goes to ``extra_bbox`` and is not merged into ``bbox``.

Overlap: each ``ImageID`` is also in ``padchest``, with other splits. A GR test patient is
probably a ``padchest`` train patient. Drop them before you train on ``padchest`` and test on
GR test.

The prior studies (``PadChest_GR_progression_prior_studies/``, 1,446 images) are not read.
``prior_image_id`` names the prior of each image.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .._bbox import box_label, clip_box, warn_dropped
from ..base import BaseHarmonizer

JSON_NAME = "grounded_reports_20240819.json"
MASTER_NAME = "master_table.csv"
IMAGE_DIR = "PadChest_GR"
SPLITS = ("train", "validation", "test")
REPORT_LANGUAGES = ("en", "es")

# The 154 labels of the abnormal findings in the 2024-08-19 release, spelled as in PadChest.
# A label outside this list raises an error. A dropped label would give a smaller vocabulary.
LABELS = [
    "abnormal foreign body", "abscess", "adenopathy", "air bronchogram",
    "air fluid level", "air trapping", "alveolar pattern", "aortic aneurysm",
    "aortic atheromatosis", "aortic button enlargement", "aortic elongation",
    "aortic endoprosthesis", "apical pleural thickening",
    "artificial aortic heart valve", "artificial heart valve",
    "artificial mitral heart valve", "ascendent aortic elongation", "atelectasis",
    "atelectasis basal", "axial hyperostosis", "azygoesophageal recess shift",
    "azygos lobe", "blastic bone lesion", "bronchiectasis", "bronchovascular markings",
    "bullas", "calcified adenopathy", "calcified densities", "calcified fibroadenoma",
    "calcified granuloma", "calcified pleural plaques", "calcified pleural thickening",
    "callus rib fracture", "cardiomegaly", "catheter", "cavitation",
    "central vascular redistribution", "central venous catheter",
    "central venous catheter via jugular vein",
    "central venous catheter via subclavian vein", "cervical rib", "chest drain tube",
    "Chilaiditi sign", "chronic changes", "clavicle fracture", "consolidation",
    "costochondral junction hypertrophy", "costophrenic angle blunting", "cyst", "dai",
    "descendent aortic elongation", "diaphragmatic eventration", "dual chamber device",
    "electrical device", "end on vessel", "endoprosthesis", "endotracheal tube",
    "fibrotic band", "fissure thickening", "flattened diaphragm", "fracture",
    "gastrostomy tube", "goiter", "granuloma", "ground glass pattern", "gynecomastia",
    "heart valve calcified", "hemidiaphragm elevation", "hiatal hernia",
    "hilar congestion", "hilar enlargement", "humeral fracture", "humeral prosthesis",
    "hydropneumothorax", "hyperinflated lung", "hypoexpansion", "increased density",
    "infiltrates", "interstitial pattern", "kerley lines", "kyphosis",
    "laminar atelectasis", "lobar atelectasis", "loculated fissural effusion",
    "loculated pleural effusion", "lung vascular paucity", "lytic bone lesion",
    "major fissure thickening", "mammary prosthesis", "mass", "mastectomy",
    "mediastinal enlargement", "mediastinal mass", "mediastinal shift",
    "mediastinic lipomatosis", "metal", "miliary opacities", "minor fissure thickening",
    "multiple nodules", "nipple shadow", "nodule",
    "non axial articular degenerative changes", "NSG tube", "obesity", "osteopenia",
    "osteoporosis", "osteosynthesis material", "pacemaker", "pectum carinatum",
    "pectum excavatum", "pericardial effusion", "pleural effusion", "pleural mass",
    "pleural thickening", "pneumoperitoneo", "pneumothorax", "prosthesis",
    "pseudonodule", "pulmonary artery enlargement", "pulmonary mass",
    "reservoir central venous catheter", "reticular interstitial pattern",
    "reticulonodular interstitial pattern", "rib fracture", "sclerotic bone lesion",
    "scoliosis", "segmental atelectasis", "single chamber device", "soft tissue mass",
    "sternoclavicular junction hypertrophy", "sternotomy",
    "subacromial space narrowing", "subcutaneous emphysema",
    "superior mediastinal enlargement", "supra aortic elongation", "surgery",
    "surgery breast", "surgery heart", "surgery humeral", "surgery lung",
    "surgery neck", "suture material", "thoracic cage deformation", "total atelectasis",
    "tracheal shift", "tracheostomy tube", "vascular hilar enlargement",
    "vascular redistribution", "ventriculoperitoneal drain tube",
    "vertebral anterior compression", "vertebral compression",
    "vertebral degenerative changes", "vertebral fracture", "volume loss",
]

LABEL_COLUMNS = [c.lower().replace(" ", "_") for c in LABELS]


def _to_dim0_first(box) -> list:
    """``[x0, y0, x1, y1]`` -> ``[y0, y1, x0, x1]``. ``clip_box`` clips it after (D-07)."""
    x0, y0, x1, y1 = (float(v) for v in box)
    return [y0, y1, x0, x1]


class PadChestGRHarmonizer(BaseHarmonizer):
    """Harmonize PadChest-GR grounded reports.

    Args:
        csv_path: ``master_table.csv``. Defaults to the one in ``base_image_dir``.
        base_image_dir: The release folder, with ``grounded_reports_*.json``,
            ``master_table.csv`` and ``PadChest_GR/`` (the PNG files).
        json_path: ``grounded_reports_20240819.json``. The default is the one in
            ``base_image_dir``.
        padchest_csv_path: The PadChest file
            ``PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz``. Optional. With it,
            ``view_position`` comes from its ``Projection`` column, because the GR release has
            no view (4,257 PA, 242 AP_horizontal, 56 AP). The table writes ``AP_HORIZONTAL``
            (Q-13). An image that is not in the file raises an error, not NaN.
        split: ``train``, ``validation`` or ``test`` keeps one official split. ``None`` keeps
            all.
        report_language: ``en`` (default) or ``es``, the language of ``report``. Both languages
            are in the ``findings`` column.
    """

    LABEL_COLS = LABELS
    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None
    EXTRA_OUTPUT_COLS = [
        "split", "n_boxes", "n_extra_boxes", "bbox_findings",
        "extra_bbox", "findings", "patient_age", "study_year",
        "prior_image_id",
    ]

    #: ``PatientAge`` of the master table is the age in years at the study (median 64, range
    #: 19 to 100). Birth year and study year give the same value.
    AGE_SOURCE_COL = "patient_age"
    AGE_SOURCE_UNIT = "years"
    #: The master table's ``PatientSex_DICOM``. The base step makes ``sex`` (D-26).
    SEX_SOURCE_COL = "PatientSex_DICOM"

    _META_COLS = ["PatientID", "split", "PatientSex_DICOM", "PatientAge", "StudyDate"]

    def __init__(self, csv_path: str = None, base_image_dir: str = None,
                 json_path: str = None, padchest_csv_path: str = None,
                 split: str = None, report_language: str = "en"):
        if split is not None and split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS} or None, got {split!r}")
        if report_language not in REPORT_LANGUAGES:
            raise ValueError(
                f"report_language must be one of {REPORT_LANGUAGES}, got {report_language!r}")
        if base_image_dir:
            base_image_dir = os.path.expanduser(base_image_dir)
            csv_path = csv_path or os.path.join(base_image_dir, MASTER_NAME)
            json_path = json_path or os.path.join(base_image_dir, JSON_NAME)
        if not csv_path or not json_path:
            raise ValueError(
                "PadChestGRHarmonizer needs base_image_dir, or both csv_path "
                "(master_table.csv) and json_path (grounded_reports_*.json)")
        super().__init__(csv_path=csv_path)
        self.base_image_dir = base_image_dir
        self.json_path = json_path
        self.padchest_csv_path = padchest_csv_path
        self.split = split
        self.report_language = report_language

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["json_path"] = self.json_path
        snap["split"] = self.split
        snap["report_language"] = self.report_language
        if self.padchest_csv_path is not None:
            snap["padchest_csv_path"] = self.padchest_csv_path
        return snap

    def _read_source(self) -> pd.DataFrame:
        """Return one row for each image: the JSON findings and the master table metadata."""
        with open(self.json_path, encoding="utf-8") as fh:
            reports = json.load(fh)

        recs, seen = [], set()
        for rep in reports:
            image_id = rep["ImageID"]
            if image_id in seen:
                raise ValueError(f"duplicate ImageID in {self.json_path}: {image_id}")
            seen.add(image_id)
            recs.append(self._image_record(rep))
        df = pd.DataFrame(recs)
        if len(df):
            warn_dropped("PadChest-GR", int(df["_dropped"].sum()))
            warn_dropped("PadChest-GR extra boxes", int(df["_dropped_extra"].sum()), labels=False)
            df = df.drop(columns=["_dropped", "_dropped_extra"])

        master = pd.read_csv(self.csv_path, usecols=["ImageID"] + self._META_COLS)
        # The table has one row for each finding. The fields here belong to the image, so
        # they must agree on all its rows.
        spread = master.groupby("ImageID")[self._META_COLS].nunique()
        if (spread > 1).any().any():
            bad = spread[(spread > 1).any(axis=1)].index[:3].tolist()
            raise ValueError(
                f"{self.csv_path}: per-image metadata differs across an image's rows, "
                f"e.g. {bad}")
        meta = master.drop_duplicates("ImageID").set_index("ImageID")[self._META_COLS]
        missing = set(df["ImageID"]) - set(meta.index)
        if missing:
            raise ValueError(
                f"{len(missing)} images in the JSON are absent from the master table, "
                f"e.g. {sorted(missing)[:3]}")
        return df.join(meta, on="ImageID")

    def _image_record(self, rep: dict) -> dict:
        findings = rep["findings"]
        labels, boxes, box_labels, box_findings, extra, out = set(), [], [], [], [], []
        dropped = dropped_extra = 0
        for f in findings:
            abnormal = bool(f["abnormal"])
            flabels = list(f.get("labels") or [])
            if abnormal:
                if not flabels:
                    raise ValueError(
                        f"abnormal finding without a label in {rep['ImageID']}: {f['sentence_en']!r}")
                unknown = [l for l in flabels if l not in _LABEL_SET]
                if unknown:
                    raise ValueError(
                        f"label(s) {unknown} in {rep['ImageID']} are not in PadChestGR LABELS; "
                        "the release has changed -- extend LABELS deliberately")
                labels.update(flabels)
            # Clip here, so that the box count of each sentence is right (D-07).
            raw = [_to_dim0_first(b) for b in f.get("boxes") or []]
            fboxes = [b for b in map(clip_box, raw) if b is not None]
            dropped += len(raw) - len(fboxes)
            boxes.extend(fboxes)
            box_labels.extend([" + ".join(map(box_label, flabels))] * len(fboxes))
            box_findings.extend([f["sentence_en"]] * len(fboxes))
            raw_extra = [_to_dim0_first(b) for b in f.get("extra_boxes") or []]
            kept_extra = [b for b in map(clip_box, raw_extra) if b is not None]
            dropped_extra += len(raw_extra) - len(kept_extra)
            extra.extend(kept_extra)
            out.append({
                "sentence_en": f["sentence_en"], "sentence_es": f["sentence_es"],
                "abnormal": abnormal, "labels": flabels,
                "locations": list(f.get("locations") or []),
                "progression": f.get("progression"), "n_boxes": len(fboxes),
            })
        return {
            "ImageID": rep["ImageID"], "StudyID": str(rep["StudyID"]),
            "PreviousImageID": rep.get("PreviousImageID"),
            "_labels": labels, "_boxes": boxes, "_box_labels": box_labels,
            "_box_findings": box_findings, "_extra": extra, "_findings": out,
            "_dropped": dropped, "_dropped_extra": dropped_extra,
        }

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["PatientID"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["StudyID"]

    def _build_image_path(self) -> None:
        self.df["image_path"] = IMAGE_DIR + "/" + self.df["ImageID"]

    def _build_labels(self) -> None:
        idx = {l: i for i, l in enumerate(LABELS)}
        mat = np.zeros((len(self.df), len(LABELS)), dtype=np.int8)
        for r, labs in enumerate(self.df["_labels"]):
            for l in labs:
                mat[r, idx[l]] = 1
        self.df = pd.concat(
            [self.df, pd.DataFrame(mat, columns=LABEL_COLUMNS, index=self.df.index)], axis=1)

    def _build_bbox(self) -> None:
        # JSON text, as in NODE21, MS-CXR and ChestX-Det10. The lists have different lengths,
        # and MONAI collation fails on them.
        self.df["bbox"] = [json.dumps(b) for b in self.df["_boxes"]]
        self.df["bbox_labels"] = [json.dumps(b) for b in self.df["_box_labels"]]
        self.df["bbox_findings"] = [json.dumps(b) for b in self.df["_box_findings"]]
        self.df["extra_bbox"] = [json.dumps(b) for b in self.df["_extra"]]
        self.df["n_boxes"] = self.df["_boxes"].apply(len)
        self.df["n_extra_boxes"] = self.df["_extra"].apply(len)

    def _build_report(self) -> None:
        """Make ``report`` and ``findings``. Then make the metadata and keep one split.

        This is the last build step. So the check in :meth:`_build_view_position` sees the rows of
        all splits. It raises for an image that is not in ``padchest_csv_path``.
        """
        key = "sentence_" + self.report_language
        self.df["report"] = [" ".join(f[key] for f in fs) for fs in self.df["_findings"]]
        self.df["findings"] = [json.dumps(fs, ensure_ascii=False) for fs in self.df["_findings"]]
        self._build_metadata()
        if self.split is not None:
            self.df = self.df[self.df["split"] == self.split]

    def _build_view_position(self) -> None:
        if not self.padchest_csv_path:
            return
        pc = pd.read_csv(self.padchest_csv_path, usecols=["ImageID", "Projection"])
        proj = pc.drop_duplicates("ImageID").set_index("ImageID")["Projection"]
        missing = self.df.loc[~self.df["ImageID"].isin(proj.index), "ImageID"]
        if len(missing):
            raise ValueError(
                f"{len(missing)} PadChest-GR images are not in {self.padchest_csv_path} "
                f"(e.g. {missing.iloc[0]}); it is not the matching PadChest label file")
        self.df["view_position"] = self.df["ImageID"].map(proj)

    def _build_metadata(self) -> None:
        self.df["patient_age"] = self.df["PatientAge"].astype(float)
        self.df["study_year"] = self.df["StudyDate"].astype(int)
        self.df["prior_image_id"] = self.df["PreviousImageID"]


_LABEL_SET = frozenset(LABELS)
