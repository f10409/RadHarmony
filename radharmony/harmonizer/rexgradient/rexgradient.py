"""Harmonizer for the ReXGradient-160K dataset."""

import json
import os

import pandas as pd

from ..base import BaseHarmonizer


_IMAGE_PATH_PREFIX = "../deid_png/"


def _format_report(indication, comparison, findings, impression) -> str | None:
    """Join the 4 report sections into one text.

    Returns ``None`` if every section is empty or NaN.
    """
    parts = []
    for header, body in (
        ("INDICATION", indication),
        ("COMPARISON", comparison),
        ("FINDINGS", findings),
        ("IMPRESSION", impression),
    ):
        if body is None:
            continue
        if isinstance(body, float) and pd.isna(body):
            continue
        text = str(body).strip()
        if not text or text.lower() == "none":
            continue
        parts.append(f"{header}:\n{text}")
    if not parts:
        return None
    return "\n\n".join(parts)


class ReXGradientHarmonizer(BaseHarmonizer):
    """Harmonize ReXGradient-160K into the standard RadHarmony format.

    The release has metadata in two forms for each split:

    - ``<split>_metadata.csv``: one row for each study (140K / 10K / 10K).
    - ``<split>_metadata_view_position.json``: one entry for each study, with an ``ImagePath``
      list of all its images (238,968 / 17,007 / 17,029).

    The harmonizer reads the JSON, because the CSV cannot give one row for each image.
    ``csv_path`` is the JSON file. The name stays because :class:`BaseHarmonizer` uses it. A
    ``.csv`` path raises ``ValueError``.

    Example::

        csv_path = ".../ReXGradient-160K/download/metadata/train_metadata_view_position.json"
        base_image_dir = ".../ReXGradient-160K/deid_png/"
        # image_path: "GRDNLZHK1CJMB9DS/GRDNLD4ATLU63FN8/studies/<study>/series/<series>/instances/<sop>.png"

    The report joins the fields ``Indication``, ``Comparison``, ``Findings`` and ``Impression``
    with section headers (:func:`_format_report`). The release has no labels. Optional
    GPT-OSS labels come from ``labels_csv_path``. Without it, ``LABEL_COLS`` is empty.

    Args:
        csv_path: Path to the ``*_metadata_view_position.json`` file of the split.
        base_image_dir: Root of the extracted PNG tree, for example
            ``.../ReXGradient-160K/deid_png/``.
        labels_csv_path: Path to the GPT-OSS label table. Optional.
    """

    #: The 14 CheXpert labels. The GPT-OSS table has ``Abnormal_response``, the opposite of
    #: ``no_finding``. The harmonizer inverts it (Abnormal Present 61.1% gives no_finding 38.9%
    #: on the 160K reports). All other CheXpert-14 datasets here use ``No Finding``. A column
    #: with the opposite meaning would flip the sign of a metric over many datasets, and
    #: nothing would raise.
    LABEL_COLS = [
        "atelectasis", "cardiomegaly", "consolidation", "edema",
        "enlarged_cardiomediastinum", "fracture", "lung_lesion", "lung_opacity",
        "no_finding", "pleural_effusion", "pleural_other", "pneumonia",
        "pneumothorax", "support_devices",
    ]
    LABEL_JOIN_COLS = None

    #: GPT-OSS-120B labels for all 160,000 reports (run 2026-09-13 to 09-16). 99.987% of the
    #: rows give Present or Absent for all 14 columns. They are optional. Without
    #: ``labels_csv_path`` there are no label columns.
    _GPTOSS_TO_CANON = {
        "Atelectasis": "atelectasis", "Cardiomegaly": "cardiomegaly",
        "Consolidation": "consolidation", "Edema": "edema",
        "EnlargedCardiomediastinum": "enlarged_cardiomediastinum",
        "Fracture": "fracture", "LungLesion": "lung_lesion",
        "LungOpacity": "lung_opacity", "PleuralEffusion": "pleural_effusion",
        "PleuralOther": "pleural_other", "Pneumonia": "pneumonia",
        "Pneumothorax": "pneumothorax", "SupportDevices": "support_devices",
    }

    # _read_source() reads view_position from the JSON.
    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None

    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None

    # The reports are in the JSON.
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    #: The split of a split class. Its JSON must be ``<SPLIT>_metadata_view_position.json`` (D-09).
    SPLIT: str = None

    #: ``EthnicGroup`` and ``InstitutionName`` are only in the ``*_metadata_view_position`` JSON,
    #: not in the plain ``*_metadata`` file. The data has 79 U.S. sites (67 in the test split
    #: alone). The site is needed for tests across institutions.
    EXTRA_OUTPUT_COLS = ["accession_number", "patient_sex", "patient_age",
                         "patient_ethnic_group", "institution", "study_date"]

    #: ``PatientAge`` is a DICOM AS string, for example '002D' or '029Y' (see ``_age.py``).
    #: Without the unit, a 2-day-old child reads as 2 years.
    AGE_SOURCE_COL = "patient_age"
    AGE_SOURCE_UNIT = "dicom_as"
    #: metadata PatientSex: M, F or O. The base step makes ``sex``, and O gives NaN (D-26).
    SEX_SOURCE_COL = "patient_sex"

    def __init__(self, csv_path: str, base_image_dir: str = None,
                 labels_csv_path: str = None):
        if csv_path is not None and str(csv_path).lower().endswith(".csv"):
            raise ValueError(
                "ReXGradientHarmonizer expects the JSON metadata path "
                "(e.g. '.../metadata/train_metadata_view_position.json'), "
                f"got a CSV: {csv_path!r}. The per-study CSV does not "
                "contain image paths; pass the matching "
                "'*_metadata_view_position.json' instead."
            )
        super().__init__(csv_path=csv_path)
        self.base_image_dir = base_image_dir
        self.labels_csv_path = (
            os.path.expanduser(labels_csv_path) if labels_csv_path else None
        )

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        if self.base_image_dir is not None:
            snap["base_image_dir"] = self.base_image_dir
        if self.labels_csv_path is not None:
            snap["labels_csv_path"] = self.labels_csv_path
        return snap

    def _read_source(self) -> pd.DataFrame:
        """Read the JSON and return one row for each image.

        Each ``ImagePath`` list of a study becomes rows. The rows have the IDs, the image
        path, the view and the report already.
        """
        if self.csv_path is None or not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"ReXGradient JSON metadata not found: {self.csv_path!r}"
            )
        name = os.path.basename(self.csv_path)
        if self.SPLIT and not name.startswith(f"{self.SPLIT}_"):
            # A split class with the JSON of another split gave that split's rows (V-H2-46).
            raise ValueError(f"{type(self).__name__} needs {self.SPLIT}_metadata_view_position.json, got {name}.")

        with open(self.csv_path, "r") as f:
            entries = json.load(f)

        # One JSON entry gives one row for each image.
        rows = []
        for e in entries.values():
            image_paths = e.get("ImagePath") or []
            if not image_paths:
                continue
            view_positions = e.get("ImageViewPosition") or [None] * len(image_paths)
            # The view list can be shorter than the path list.
            if len(view_positions) < len(image_paths):
                view_positions = list(view_positions) + [None] * (
                    len(image_paths) - len(view_positions)
                )
            report = _format_report(
                e.get("Indication"),
                e.get("Comparison"),
                e.get("Findings"),
                e.get("Impression"),
            )
            patient_id = str(e.get("PatientID"))
            study_id = str(e.get("StudyInstanceUid"))
            accession = e.get("AccessionNumber")
            sex = e.get("PatientSex")
            age = e.get("PatientAge")
            ethnic_group = e.get("EthnicGroup")
            study_date = e.get("StudyDate")
            institution = e.get("InstitutionName")
            for img_path, vp in zip(image_paths, view_positions, strict=False):
                if pd.isna(img_path):
                    continue  # no image, as the base dropna step (the label message counts image rows)
                # The JSON paths start with '../deid_png/'. Cut it, so the path is relative
                # to base_image_dir.
                if isinstance(img_path, str) and img_path.startswith(_IMAGE_PATH_PREFIX):
                    rel = img_path[len(_IMAGE_PATH_PREFIX):]
                else:
                    rel = img_path
                rows.append(
                    {
                        "patient_id": patient_id,
                        "study_id": study_id,
                        "patient_ethnic_group": ethnic_group,
                        "image_path": rel,
                        "view_position": vp,
                        "report": report,
                        "accession_number": accession,
                        "patient_sex": sex,
                        "patient_age": age,
                        "institution": institution,
                        "study_date": study_date,
                    }
                )

        return pd.DataFrame(rows)

    # _read_source() builds patient_id, study_id and image_path. These hooks stay empty.
    def _build_patient_id(self) -> None:
        pass

    def _build_study_id(self) -> None:
        pass

    def _build_image_path(self) -> None:
        pass

    def _build_labels(self) -> None:
        """Join the GPT-OSS CheXpert-14 labels on ``accession_number``.

        The label table has ``AccessionNumber_anon`` and a ``study_id`` like
        ``p<patient>_a<accession>_s<StudyInstanceUID>``. The own ``study_id`` here is the bare
        ``StudyInstanceUid``, so the join uses the accession. It is unique in all 160,000 rows.

        ``no_finding`` is the inverse of ``Abnormal_response`` (see LABEL_COLS). A value that
        is not Present or Absent becomes NaN, not negative. A blank is not an absent finding.
        Making it negative halved a worst group of Support Devices once.
        """
        if self.labels_csv_path is None:
            self.LABEL_COLS = []
            return
        if not os.path.isfile(self.labels_csv_path):
            raise FileNotFoundError(f"ReXGradient labels not found: {self.labels_csv_path}")
        lab = pd.read_csv(self.labels_csv_path, low_memory=False)
        need = ["AccessionNumber_anon", "Abnormal_response"]
        missing = [c for c in need if c not in lab.columns]
        if missing:
            raise ValueError(f"label table missing {missing}; found {list(lab.columns)[:8]}")

        def _tri(series):
            v = series.astype(str).str.strip().str.lower()
            return pd.Series(
                pd.NA, index=v.index, dtype="Float64"
            ).mask(v == "present", 1.0).mask(v == "absent", 0.0)

        out = pd.DataFrame({"accession_number": lab["AccessionNumber_anon"].astype(str)})
        for src, dst in self._GPTOSS_TO_CANON.items():
            col = f"{src}_response"
            out[dst] = _tri(lab[col]) if col in lab.columns else pd.NA
        # The table stores the opposite of No Finding.
        out["no_finding"] = 1.0 - _tri(lab["Abnormal_response"])
        # Use float64 with NaN, because torch cannot convert the nullable Float64.
        label_cols = list(self._GPTOSS_TO_CANON.values()) + ["no_finding"]
        for c in label_cols:
            out[c] = out[c].astype("Float64").to_numpy(dtype="float64", na_value=float("nan"))
        out = out.drop_duplicates("accession_number")

        before = len(self.df)
        self.df["accession_number"] = self.df["accession_number"].astype(str)
        self.df = self.df.merge(out, on="accession_number", how="left")
        assert len(self.df) == before, (
            "label join changed the row count -- duplicate accession in the label table"
        )
        cov = float(self.df["no_finding"].notna().mean())
        print(f"[rexgradient] GPT-OSS labels joined on {cov:.1%} of {before:,} rows "
              f"({len(out):,} labelled studies); no_finding positive rate "
              f"{float(self.df['no_finding'].mean(skipna=True)):.3f}")
