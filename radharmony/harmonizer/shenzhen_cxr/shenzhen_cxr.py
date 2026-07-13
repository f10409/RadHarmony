"""Harmonizer for the NLM Shenzhen Hospital Chest X-Ray Set.

A tuberculosis screening dataset created by the National Library of Medicine
in collaboration with Shenzhen No.3 People's Hospital, Guangdong Medical
College, Shenzhen, China.  Images were captured as part of daily out-patient
clinical routine using a Philips DR Digital Diagnose system.

**No primary CSV exists.**  The harmonizer scans ``base_dir`` (the
``CXR_png/`` image directory) for all ``.png`` files and builds the
DataFrame from filenames.

Layout::

    <dataset_root>/                       # Shenzhen-Hospital-CXR-Set/
      CXR_png/                            # base_dir points here (662 PNGs)
        CHNCXR_0001_0.png                 # Normal (_0 suffix)
        CHNCXR_0327_1.png                 # TB-positive (_1 suffix)
        ...
      ClinicalReadings/                   # sibling of CXR_png/
        CHNCXR_0001_0.txt                 # line 1: "<sex> <age>yrs"; rest: diagnosis
        CHNCXR_0327_1.txt
        ...
      Annotations-2/                      # sibling of CXR_png/
        Statistics_ShenzhenDataset_US.csv # 19 binary TB finding columns (TB patients only)
        masks/                            # per-finding region masks (TB patients only)
          CHNCXR_0327_1_Small_Infiltrate_(non-linear)_1.png
          CHNCXR_0330_1_Pleural_Effusion_2.png
          ...
      shenzhen_consensus_roi.csv          # consensus ROI bounding boxes (TB patients)
      NLM-ChinaCXRSet-ReadMe.docx

``base_dir`` points at the ``CXR_png/`` image directory itself — the last
common folder containing every image. ``ClinicalReadings/`` and
``Annotations-2/`` are read as sibling directories of ``base_dir``.
``image_path`` entries in the harmonized DataFrame are bare filenames
(e.g. ``CHNCXR_0327_1.png``); finding-mask paths use ``../Annotations-2/...``
to reach the sibling tree.

This mirrors the :class:`MontgomeryCXRHarmonizer` layout, which is the
companion dataset from the same NLM collaboration.

Label encoding (from filename suffix):
  * ``_0`` → ``tuberculosis = 0`` (normal)
  * ``_1`` → ``tuberculosis = 1`` (active TB)

When ``ClinicalReadings/`` is present, each ``.txt`` file is parsed for
``patient_age`` (int), ``patient_sex`` (str), and ``report`` (raw text).

When ``Annotations-2/Statistics_ShenzhenDataset_US.csv`` is present, 19
binary TB finding columns are joined in (NaN for normal patients).

When ``Annotations-2/masks/`` is present, a JSON-encoded list of relative
mask paths is stored in ``finding_masks`` for each TB-positive patient.

**Harmonized columns produced:**
  ``patient_id``, ``study_id``, ``image_path``, ``report``, ``tuberculosis``,
  ``patient_age``, ``patient_sex``,
  ``Pleural_Effusion``, ``Apical_Thickening``, ``Single_Nodule_(non-calcified)``,
  ``Pleural_Thickening_(non-apical)``, ``Calcified_Nodule``,
  ``Small_Infiltrate_(non-linear)``, ``Cavity``, ``Linear_Density``,
  ``Severe_Infiltrate_(Consolidation)``, ``Thickening_of_interlobar_fissure``,
  ``Clustered_Nodule_(2mm-5mm_apart)``, ``Moderate_Infiltrate_(non-linear)``,
  ``Adenopathy``, ``Calcification_(other_than_nodule&lymphnod)``,
  ``Calcified_lymph_node``, ``Miliary``, ``Retraction``, ``Other``, ``Unknown``,
  ``finding_masks``
"""

import json
import os
import re
from collections import defaultdict

import pandas as pd

from ..base import BaseHarmonizer


# Matches e.g. CHNCXR_0001_0.png  →  groups: ('0001', '0')
_FILENAME_RE = re.compile(r"^CHNCXR_(\d+)_([01])\.png$", re.IGNORECASE)

_AGE_RE = re.compile(r"(\d+)\s*yrs?", re.IGNORECASE)
_SEX_RE = re.compile(r"\b(male|female|M|F)\b", re.IGNORECASE)

# 19 TB finding types present in Statistics_ShenzhenDataset_US.csv
TB_FINDING_COLS = [
    "Pleural_Effusion",
    "Apical_Thickening",
    "Single_Nodule_(non-calcified)",
    "Pleural_Thickening_(non-apical)",
    "Calcified_Nodule",
    "Small_Infiltrate_(non-linear)",
    "Cavity",
    "Linear_Density",
    "Severe_Infiltrate_(Consolidation)",
    "Thickening_of_interlobar_fissure",
    "Clustered_Nodule_(2mm-5mm_apart)",
    "Moderate_Infiltrate_(non-linear)",
    "Adenopathy",
    "Calcification_(other_than_nodule&lymphnod)",
    "Calcified_lymph_node",
    "Miliary",
    "Retraction",
    "Other",
    "Unknown",
]


def _parse_clinical_file(path: str) -> tuple:
    """Return (age, sex, full_text) from a clinical reading text file."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
    except OSError:
        return None, None, None

    if not text:
        return None, None, None

    first_line = text.splitlines()[0]

    age = None
    m_age = _AGE_RE.search(first_line)
    if m_age:
        try:
            age = int(m_age.group(1))
        except ValueError:
            pass

    sex = None
    m_sex = _SEX_RE.search(first_line)
    if m_sex:
        raw = m_sex.group(1).lower()
        sex = "M" if raw.startswith("m") else "F"

    return age, sex, text


class ShenzhenCXRHarmonizer(BaseHarmonizer):
    """Harmonize the Shenzhen Hospital CXR Set into the standard RadHarmony format.

    Args:
        base_dir: The ``CXR_png/`` images directory itself — the last common
            folder containing every image. ``ClinicalReadings/`` and
            ``Annotations-2/`` are read as sibling directories.
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

    EXTRA_OUTPUT_COLS = ["patient_age", "patient_sex"] + TB_FINDING_COLS + ["finding_masks"]

    def __init__(self, base_dir: str):
        self.base_dir = os.path.expanduser(base_dir)
        super().__init__(csv_path=None)

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["base_dir"] = self.base_dir
        snap.pop("csv_path", None)
        return snap

    # ------------------------------------------------------------------
    # Directory scan
    # ------------------------------------------------------------------

    def _scan_image_dir(self) -> pd.DataFrame:
        img_dir = self.base_dir
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"Image directory not found: {img_dir!r}."
            )

        rows = []
        for fname in sorted(os.listdir(img_dir)):
            m = _FILENAME_RE.match(fname)
            if m is None:
                continue
            rows.append({
                "_stem": os.path.splitext(fname)[0],
                "_numeric_id": m.group(1),
                "_label_digit": int(m.group(2)),
            })

        if not rows:
            raise FileNotFoundError(
                f"No CHNCXR_*.png files found in {img_dir!r}."
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Build hooks
    # ------------------------------------------------------------------

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["_numeric_id"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["_numeric_id"].astype(str)

    def _build_image_path(self) -> None:
        self.df["image_path"] = self.df["_stem"] + ".png"

    def _build_labels(self) -> None:
        self.df["tuberculosis"] = self.df["_label_digit"]

    def _build_clinical_readings(self) -> None:
        readings_dir = os.path.join(self.base_dir, "..", "ClinicalReadings")
        if not os.path.isdir(readings_dir):
            return

        ages, sexes, reports = [], [], []
        for stem in self.df["_stem"]:
            txt_path = os.path.join(readings_dir, stem + ".txt")
            age, sex, text = _parse_clinical_file(txt_path)
            ages.append(age)
            sexes.append(sex)
            reports.append(text)

        self.df["patient_age"] = ages
        self.df["patient_sex"] = sexes
        self.df["report"] = reports

    def _build_tb_findings(self) -> None:
        """Join 19 binary TB finding columns from Statistics_ShenzhenDataset_US.csv (sibling dir)."""
        stats_path = os.path.join(
            self.base_dir, "..", "Annotations-2", "Statistics_ShenzhenDataset_US.csv"
        )
        if not os.path.isfile(stats_path):
            return

        stats = pd.read_csv(stats_path)
        # imagename column: "CHNCXR_0327_1.png" → stem "CHNCXR_0327_1"
        stats["_stem"] = stats["imagename"].str.replace(r"\.png$", "", regex=True)

        keep = [c for c in TB_FINDING_COLS if c in stats.columns]
        stats = stats[["_stem"] + keep]

        self.df = self.df.merge(stats, on="_stem", how="left")

    def _build_finding_masks(self) -> None:
        """Build JSON list of mask paths per patient from sibling Annotations-2/masks/.

        Paths are stored as ``../Annotations-2/masks/<fname>`` so they
        resolve correctly when joined to ``base_image_dir`` (= ``CXR_png/``).
        """
        masks_dir = os.path.join(self.base_dir, "..", "Annotations-2", "masks")
        if not os.path.isdir(masks_dir):
            self.df["finding_masks"] = None
            return

        # group mask filenames by patient stem
        stem_to_masks: dict = defaultdict(list)
        for fname in sorted(os.listdir(masks_dir)):
            if not fname.endswith(".png"):
                continue
            # filename: CHNCXR_0327_1_Small_Infiltrate_(non-linear)_1.png
            # patient stem = first three underscore-separated tokens
            parts = fname.split("_")
            if len(parts) < 3:
                continue
            stem = "_".join(parts[:3])  # CHNCXR_0327_1
            rel_path = os.path.join("..", "Annotations-2", "masks", fname)
            stem_to_masks[stem].append(rel_path)

        self.df["finding_masks"] = self.df["_stem"].map(
            lambda s: json.dumps(stem_to_masks[s]) if stem_to_masks[s] else None
        )

    # ------------------------------------------------------------------
    # Mask preprocessing
    # ------------------------------------------------------------------

    def preprocess_masks(
        self,
        output_dir: str,
        base_image_dir: str = None,
        num_cores: int = 1,  # noqa: ARG002 — accepted for signature parity
    ) -> None:
        """Union per-finding masks into a single binary "TB region" PNG per image.

        Matches the SIIM-ACR PTX convention: a PNG is written for **every**
        image, even when there are no finding masks. For TB-positive rows
        whose ``finding_masks`` JSON lists one or more mask paths, the union
        of those masks (relative to ``self.base_dir``) is saved. For
        TB-negative rows (or any row without finding_masks), a same-size
        all-zero mask is saved instead — these rows stay in the sample stream
        so segmentation eval can include them as true-negative supervision.

        ``mask_path`` is populated with the **absolute** path to the saved
        PNG for every row. Idempotent: existing PNGs are not rewritten.
        """
        import numpy as np  # type: ignore[import-untyped]
        from PIL import Image
        from tqdm import tqdm

        if self.df is None:
            raise RuntimeError("Call harmonize() before preprocess_masks().")
        if "finding_masks" not in self.df.columns:
            raise ValueError(
                "finding_masks not in harmonized DataFrame; "
                "Annotations-2/masks/ subdir was not found at harmonize() time."
            )

        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(output_dir, exist_ok=True)
        base_dir = base_image_dir or self.base_dir

        fused_paths: list[str | None] = []
        for _, row in tqdm(
            self.df.iterrows(), total=len(self.df), desc="Fusing TB-finding masks"
        ):
            stem = os.path.splitext(os.path.basename(row["image_path"]))[0]
            out_path = os.path.join(output_dir, stem + ".png")

            # Parse the finding_masks list (may be missing → TB-negative)
            raw = row.get("finding_masks")
            mask_rels: list = []
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        mask_rels = parsed
                except (ValueError, TypeError):
                    pass

            if not os.path.isfile(out_path):
                if mask_rels:
                    # TB-positive with finding masks: union them.
                    union = None
                    for rel in mask_rels:
                        arr = np.array(
                            Image.open(
                                os.path.normpath(os.path.join(base_dir, rel))
                            ).convert("L")
                        )
                        union = (arr > 0) if union is None else (union | (arr > 0))
                    fused = union.astype(np.uint8) * 255  # type: ignore[union-attr]
                else:
                    # TB-negative (or no finding masks): write a black PNG of
                    # the same size as the source image — match SIIM convention.
                    img_path = os.path.normpath(
                        os.path.join(base_dir, row["image_path"])
                    )
                    w, h = Image.open(img_path).size  # PIL returns (W, H)
                    fused = np.zeros((h, w), dtype=np.uint8)
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
        self._harmonized_df_override = None
        self.df = self._scan_image_dir()

        self._build_patient_id()
        self._build_study_id()
        self._build_image_path()
        self._build_labels()
        self._build_clinical_readings()
        self._build_tb_findings()
        self._build_finding_masks()

        self.df.drop(
            columns=["_stem", "_numeric_id", "_label_digit"],
            inplace=True,
            errors="ignore",
        )
        self.df.dropna(subset=["image_path"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        if mask_output_dir is not None:
            self.preprocess_masks(
                output_dir=mask_output_dir,
                num_cores=mask_num_cores,
            )

        return self._select_harmonized_columns(self.df)
