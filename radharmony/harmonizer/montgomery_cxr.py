"""Harmonizer for the Montgomery County Tuberculosis CXR dataset.

A collaboration between the National Library of Medicine (LHNCBC), the
Department of Health and Human Services of Montgomery County, Maryland,
and Shenzhen No.3 People's Hospital, Guangdong Medical College, Shenzhen,
China.

**Source:**
  https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/
  Montgomery-County-CXR-Set/

138 PA chest radiographs (80 normal, 58 TB-positive).  All images are
4892 × 4020 PNG.  Left- and right-lung binary masks are included, along
with per-patient clinical reading text files.

**Expected layout after download:**::

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
        MCUCXR_0001_0.txt           # one file per patient
        ...
      montgomery_consensus_roi.csv
      NLM-MontgomeryCXRSet-ReadMe.pdf

``base_dir`` points at the ``CXR_png/`` image directory itself — the last
common folder containing every image. ``ClinicalReadings/`` and
``ManualMask/`` are read as sibling directories of ``base_dir``.

**ClinicalReadings/<stem>.txt format (one file per image):**::

    Patient's Sex: F
    Patient's Age: 027Y
    <free-text clinical report — one or more lines>

  ``_0`` filename suffix = normal (TB-negative).
  ``_1`` filename suffix = TB-positive.

**Harmonized columns produced:**
  ``patient_id``, ``study_id``, ``image_path``, ``tb`` (0/1 Int64),
  ``sex``, ``age`` (int years), ``report`` (str),
  ``mask_path_left``, ``mask_path_right``
"""

import os
import re

import pandas as pd

from .base import BaseHarmonizer


_SEX_RE  = re.compile(r"Patient['']s Sex:\s*([MF])", re.IGNORECASE)
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
        base_dir: The ``CXR_png/`` images directory itself — the last common
            folder containing every image. ``ClinicalReadings/`` and
            ``ManualMask/`` are read as sibling directories. ``image_path``
            entries in the harmonized DataFrame are bare filenames (relative
            to ``base_dir``); mask paths use ``../ManualMask/...`` to reach
            the sibling tree.
    """

    LABEL_COLS = ["tb"]

    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    EXTRA_OUTPUT_COLS = ["sex", "age", "mask_path_left", "mask_path_right"]

    def __init__(self, base_dir: str):
        self.base_dir = os.path.expanduser(base_dir)
        super().__init__(csv_path=None)

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["base_dir"] = self.base_dir
        return snap

    # ------------------------------------------------------------------
    # Build hooks
    # ------------------------------------------------------------------

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["_stem"].str.replace(r"_[01]$", "", regex=True)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["patient_id"]

    def _build_image_path(self) -> None:
        self.df["image_path"] = self.df["_stem"] + ".png"

    def _build_labels(self) -> None:
        self.df["tb"] = (
            self.df["_stem"]
            .str.extract(r"_([01])$")[0]
            .astype("Int64")
        )

    def _build_clinical(self) -> None:
        """Parse per-patient ClinicalReadings/<stem>.txt files (sibling dir)."""
        cr_dir = os.path.join(self.base_dir, "..", "ClinicalReadings")
        if not os.path.isdir(cr_dir):
            return

        records = []
        for stem in self.df["_stem"]:
            path = os.path.join(cr_dir, stem + ".txt")
            records.append(_parse_clinical_txt(path))

        clinical_df = pd.DataFrame(records)
        for col in ("sex", "age", "report"):
            if col in clinical_df.columns:
                self.df[col] = clinical_df[col].values

    def _build_masks(self) -> None:
        """Populate mask_path_left / mask_path_right from sibling ManualMask/.

        Mask paths are stored as ``../ManualMask/<side>/<stem>.png`` so they
        resolve correctly when joined to ``base_image_dir`` (= ``CXR_png/``).
        """
        for side, col in (("leftMask", "mask_path_left"), ("rightMask", "mask_path_right")):
            mask_dir = os.path.join(self.base_dir, "..", "ManualMask", side)
            if not os.path.isdir(mask_dir):
                continue
            self.df[col] = self.df["_stem"].apply(
                lambda s, d=mask_dir: (
                    os.path.join("..", "ManualMask", side, s + ".png")
                    if os.path.exists(os.path.join(d, s + ".png"))
                    else None
                )
            )

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
            raise ValueError(
                "mask_path_left / mask_path_right not in harmonized DataFrame; "
                "ManualMask/ subdir was not found at harmonize() time."
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

    # ------------------------------------------------------------------
    # Custom harmonize
    # ------------------------------------------------------------------

    def harmonize(
        self,
        *args,
        mask_output_dir: str = None,
        mask_num_cores: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        img_dir = self.base_dir
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        stems = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(img_dir)
            if f.endswith(".png")
        )
        self.df = pd.DataFrame({"_stem": stems})

        self._build_patient_id()
        self._build_study_id()
        self._build_image_path()
        self._build_labels()
        self._build_clinical()
        self._build_masks()
        self._build_view_position()
        self._build_mask_path()
        self._build_bbox()
        self._build_report()

        self.df.drop(columns=["_stem"], inplace=True, errors="ignore")
        self.df.dropna(subset=["image_path"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        if mask_output_dir is not None:
            self.preprocess_masks(
                output_dir=mask_output_dir,
                num_cores=mask_num_cores,
            )

        return self._select_harmonized_columns(self.df)
