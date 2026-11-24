"""Harmonizer for the Montgomery County Tuberculosis CXR dataset.

From the National Library of Medicine (LHNCBC), the Department of Health and
Human Services of Montgomery County, Maryland, and Shenzhen No.3 People's
Hospital, China. Source:
https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/

138 PA chest radiographs (80 normal, 58 TB-positive). All images are 4892 x
4020 PNG. There are left and right lung masks and a clinical reading text file
for each patient.

Layout after download::

    <dataset_root>/                 # MontgomerySet/
      CXR_png/                      # base_dir points here
        MCUCXR_0001_0.png           # normal  (_0 suffix)
        MCUCXR_0104_1.png           # TB      (_1 suffix)
        ...
      ManualMask/                   # sibling of CXR_png/
        leftMask/
          MCUCXR_0001_0.png
          ...
        rightMask/
          MCUCXR_0001_0.png
          ...
      ClinicalReadings/             # sibling of CXR_png/
        MCUCXR_0001_0.txt           # one file per image
        ...
      montgomery_consensus_roi.csv
      NLM-MontgomeryCXRSet-ReadMe.pdf

``base_dir`` is the ``CXR_png/`` folder itself, the last common folder of all
images. The code reads ``ClinicalReadings/`` and ``ManualMask/`` as sibling
folders of ``base_dir``.

Format of ``ClinicalReadings/<stem>.txt`` (one file for each image)::

    Patient's Sex: F
    Patient's Age: 027Y
    <free-text clinical report, one or more lines>

  The file name suffix ``_0`` is normal (TB-negative). ``_1`` is TB-positive.

Harmonized columns:
  ``patient_id``, ``study_id``, ``image_path``, ``tuberculosis`` (0/1 Int64),
  ``sex``, ``age`` (int years), ``report`` (str),
  ``mask_path_left``, ``mask_path_right``
"""

import os
import re

import pandas as pd

from ..base import BaseHarmonizer


# Any sex letter: one file (MCUCXR_0080) has "O". The base step maps it (D-26).
_SEX_RE  = re.compile(r"Patient['']s Sex:\s*([A-Z]+)", re.IGNORECASE)
_AGE_RE  = re.compile(r"Patient['']s Age:\s*(\d+)\s*Y", re.IGNORECASE)


def _parse_clinical_txt(path: str) -> dict:
    """Parse a single ClinicalReadings/<stem>.txt file."""
    result = {"sex": None, "age": None, "report": None}
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = [l.rstrip() for l in fh.readlines()]
    except OSError:
        return result

    report_lines = []
    for line in lines:
        m_sex = _SEX_RE.match(line)
        m_age = _AGE_RE.match(line)
        if m_sex:
            result["sex"] = m_sex.group(1).upper()
        elif m_age:
            result["age"] = int(m_age.group(1))
        else:
            if line.strip():
                report_lines.append(line.strip())

    result["report"] = " ".join(report_lines) if report_lines else None
    return result


class MontgomeryCXRHarmonizer(BaseHarmonizer):
    """Harmonize the Montgomery County TB CXR set into the RadHarmony format.

    Args:
        base_dir: The ``CXR_png/`` folder itself, the last common folder of
            all images. The code reads ``ClinicalReadings/`` and ``ManualMask/``
            as sibling folders. ``image_path`` values are bare file names
            (relative to ``base_dir``). Mask paths start with
            ``../ManualMask/`` to reach the sibling folder.
    """

    LABEL_COLS = ["tuberculosis"]

    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    EXTRA_OUTPUT_COLS = ["age", "mask_path_left", "mask_path_right"]

    #: Parsed from "Patient's Sex: F". The base step maps it in place (D-26).
    SEX_SOURCE_COL = "sex"


    #: Age unit declaration -- see radharmony/harmonizer/_age.py.

    #: Parsed from "Patient's Age: 027Y" by _AGE_RE, which requires the Y and
    #: captures only the number, so already years.

    AGE_SOURCE_COL = "age"

    AGE_SOURCE_UNIT = "years"
    def __init__(self, base_dir: str):
        self.base_dir = os.path.expanduser(base_dir)
        super().__init__(csv_path=None)

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["base_dir"] = self.base_dir
        snap.pop("csv_path", None)  # not an argument of this class (V-H2-1)
        return snap

    # ------------------------------------------------------------------
    # Build hooks
    # ------------------------------------------------------------------

    def _read_source(self) -> pd.DataFrame:
        """List the PNG files of ``base_dir`` and read their clinical readings. Return one row per image."""
        img_dir = self.base_dir
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        stems = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(img_dir)
            if f.endswith(".png")
        )
        if not stems:
            raise FileNotFoundError(f"No .png file in {img_dir}. Check the path of the image folder (D-35).")
        # The patient ID step reads the reports, so read them here.
        return self._read_clinical(pd.DataFrame({"_stem": stems}))

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["_stem"].str.replace(r"_[01]$", "", regex=True)
        self._link_same_patient()

    def _build_study_id(self) -> None:
        # One study per image. Not patient_id: _link_same_patient can change it.
        self.df["study_id"] = self.df["_stem"].str.replace(r"_[01]$", "", regex=True)

    def _build_image_path(self) -> None:
        self.df["image_path"] = self.df["_stem"] + ".png"

    def _build_labels(self) -> None:
        self.df["tuberculosis"] = (
            self.df["_stem"]
            .str.extract(r"_([01])$")[0]
            .astype("Int64")
        )

    def _read_clinical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sex, age and report from the ClinicalReadings/<stem>.txt files (sibling dir) to ``df``."""
        cr_dir = os.path.join(self.base_dir, "..", "ClinicalReadings")
        if not os.path.isdir(cr_dir):
            return df

        records = []
        for stem in df["_stem"]:
            path = os.path.join(cr_dir, stem + ".txt")
            records.append(_parse_clinical_txt(path))

        clinical_df = pd.DataFrame(records)
        for col in ("sex", "age", "report"):
            if col in clinical_df.columns:
                df[col] = clinical_df[col].values
        return df

    def _build_mask_path(self) -> None:
        """Populate mask_path_left / mask_path_right from sibling ManualMask/.

        Mask paths are stored as ``../ManualMask/<side>/<stem>.png`` so they
        resolve correctly when joined to ``base_image_dir`` (= ``CXR_png/``).
        """
        for side, col in (("leftMask", "mask_path_left"), ("rightMask", "mask_path_right")):
            mask_dir = os.path.join(self.base_dir, "..", "ManualMask", side)
            if not os.path.isdir(mask_dir):
                continue
            self.df[col] = self.df["_stem"].apply(
                lambda s, d=mask_dir, side=side: (
                    os.path.join("..", "ManualMask", side, s + ".png")
                    if os.path.exists(os.path.join(d, s + ".png"))
                    else None
                )
            )

    def _link_same_patient(self) -> None:
        """Give one ``patient_id`` to the images of one patient (V-H2-15).

        The file names have no patient ID. 3 clinical readings name the other
        images of the same patient, for example ``(same pt as MCUCXR_0113_1)``
        in ``MCUCXR_0117_1.txt``. The linked images get the smallest ID of
        their group, so 138 images give 135 patients.
        """
        if "report" not in self.df.columns:
            return
        group = {pid: pid for pid in self.df["patient_id"]}

        def _root(pid):
            while group[pid] != pid:
                pid = group[pid]
            return pid

        for pid, report in zip(self.df["patient_id"], self.df["report"], strict=True):
            if not isinstance(report, str) or "same pt as" not in report.lower():
                continue
            for other in re.findall(r"(MCUCXR_\d{4})_[01]", report):
                if other in group:
                    a, b = sorted((_root(pid), _root(other)))
                    group[b] = a
        self.df["patient_id"] = self.df["patient_id"].map(_root)

    def _build_report(self) -> None:
        # "NO REPORT (same pt as ...)" is no report (D-33, Q-16). The patient ID step reads it, so clear it here.
        if "report" in self.df.columns:
            self.df.loc[self.df["report"].str.match(r"(?i)\s*no report\b", na=False), "report"] = None

    # ------------------------------------------------------------------
    # Mask preprocessing
    # ------------------------------------------------------------------

    def preprocess_masks(
        self,
        output_dir: str,
        base_image_dir: str = None,
        num_cores: int = 1,  # noqa: ARG002 — accepted for signature parity; loop is small
    ) -> None:
        """Fuse left+right manual lung masks into a single binary PNG per image.

        Reads ``mask_path_left`` and ``mask_path_right`` (paths relative to
        ``self.base_dir``), ORs them as binary, and writes the result to
        ``<output_dir>/<stem>.png``. The ``mask_path`` column is then populated
        with the **absolute** path to the fused PNG (so it bypasses
        ``base_image_dir`` prepending in the data-dict builder).

        Idempotent: existing fused PNGs are not rewritten.
        """
        import numpy as np  # type: ignore[import-untyped]
        from PIL import Image
        from tqdm import tqdm

        if self.df is None:
            raise RuntimeError("Call harmonize() before preprocess_masks().")
        if "mask_path_left" not in self.df.columns or "mask_path_right" not in self.df.columns:
            # The masks need this folder, so it is a missing file (D-35, V-H2-36).
            folder = os.path.normpath(os.path.join(self.base_dir, "..", "ManualMask"))
            raise FileNotFoundError(
                f"No {folder}/leftMask/ or rightMask/ folder, so Montgomery has no lung masks. "
                "Download ManualMask/ next to CXR_png/, or leave mask_output_dir as None."
            )

        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(output_dir, exist_ok=True)
        base_dir = base_image_dir or self.base_dir

        fused_paths: list[str | None] = []
        for _, row in tqdm(
            self.df.iterrows(), total=len(self.df), desc="Fusing lung masks"
        ):
            left_rel = row.get("mask_path_left")
            right_rel = row.get("mask_path_right")
            if not (isinstance(left_rel, str) and isinstance(right_rel, str)):
                fused_paths.append(None)
                continue

            stem = os.path.splitext(os.path.basename(row["image_path"]))[0]
            out_path = os.path.join(output_dir, stem + ".png")
            if not os.path.isfile(out_path):
                left = np.array(
                    Image.open(os.path.normpath(os.path.join(base_dir, left_rel))).convert("L")
                )
                right = np.array(
                    Image.open(os.path.normpath(os.path.join(base_dir, right_rel))).convert("L")
                )
                fused = ((left > 0) | (right > 0)).astype(np.uint8) * 255
                Image.fromarray(fused).save(out_path)
            fused_paths.append(out_path)

        self.df["mask_path"] = fused_paths
