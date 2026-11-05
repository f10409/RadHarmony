"""Harmonizer for the NLM Shenzhen Hospital Chest X-Ray Set.

A tuberculosis screening set from the National Library of Medicine and Shenzhen No.3 People's
Hospital (Guangdong Medical College). The images are from routine out-patient care, taken with a
Philips DR system. The release has no main CSV. The harmonizer lists the ``.png`` files of
``base_dir`` (the ``CXR_png/`` folder, 662 images) and builds the table from the file names.

Layout::

    <dataset_root>/                       # Shenzhen-Hospital-CXR-Set/
      CXR_png/                            # base_dir
        CHNCXR_0001_0.png                 # _0: normal
        CHNCXR_0327_1.png                 # _1: TB
      ClinicalReadings/                   # next to CXR_png/
        CHNCXR_0001_0.txt                 # line 1: "<sex> <age>yrs", then the diagnosis
      Annotations-2/                      # next to CXR_png/
        Statistics_ShenzhenDataset_US.csv # 19 TB finding columns (TB patients only)
        masks/CHNCXR_0327_1_Small_Infiltrate_(non-linear)_1.png   # region masks (TB only)
      shenzhen_consensus_roi.csv          # consensus ROI boxes (TB patients)

``image_path`` is the bare file name, for example ``CHNCXR_0327_1.png``. The finding masks use
``../Annotations-2/...``. The layout is the same as in :class:`MontgomeryCXRHarmonizer`, the
companion set from the same NLM team.

``tuberculosis`` comes from the file name suffix (``_0`` is 0, ``_1`` is 1).

If ``ClinicalReadings/`` exists, each ``.txt`` file gives ``patient_age`` (float years),
``patient_sex`` ("M" or "F") and ``report`` (the raw text). The first line has the sex and the
age, for example ``male 35yrs``, ``femal 32yrs`` (a typo in 3 files), ``male35yrs``,
``male 16month`` or ``female 64days``. One file (``CHNCXR_0018_0``) says ``male 42`` with no
unit, so its age is NaN (D-06: the unit is never guessed). The base steps make ``sex`` (D-26)
and ``age_years`` (D-06).

If ``Annotations-2/Statistics_ShenzhenDataset_US.csv`` exists, its 19 TB finding columns join
the table (NaN for normal patients). They count the annotated regions (0 to 75). They are not
0/1 flags.

If ``Annotations-2/masks/`` exists, ``finding_masks`` has a JSON list of relative mask paths for
each TB patient.

Columns: ``patient_id``, ``study_id``, ``image_path``, ``report``, ``tuberculosis``,
``age_years``, ``sex``, ``patient_age``, ``patient_sex``, the 19 columns of ``TB_FINDING_COLS``
and ``finding_masks``.
"""

import json
import os
import re
from collections import defaultdict

import pandas as pd

from .._age import to_years
from ..base import BaseHarmonizer


# For example CHNCXR_0001_0.png gives the groups ('0001', '0').
_FILENAME_RE = re.compile(r"^CHNCXR_(\d+)_([01])\.png$", re.IGNORECASE)

# The age and its unit word. A number with no unit word gives no age (D-06).
_AGE_RE = re.compile(r"(\d+)\s*(yrs?|months?|days?)?", re.IGNORECASE)
_AGE_UNITS = {"yr": "years", "month": "months", "day": "days"}
# The first word. It can touch the age ("male35yrs"). "femal" is a typo in the source.
_SEX_RE = re.compile(r"^\s*(female|femal|male)", re.IGNORECASE)

# The 19 TB finding types of Statistics_ShenzhenDataset_US.csv.
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
    """Read a clinical reading file and return (age, sex, full_text)."""
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
    if m_age and m_age.group(2):
        unit = _AGE_UNITS[m_age.group(2).lower().rstrip("s")]
        age = float(to_years(pd.Series([int(m_age.group(1))]), unit).iloc[0])

    sex = None
    m_sex = _SEX_RE.match(first_line)
    if m_sex:
        sex = "M" if m_sex.group(1).lower() == "male" else "F"

    return age, sex, text


class ShenzhenCXRHarmonizer(BaseHarmonizer):
    """Harmonize the Shenzhen Hospital CXR Set into the standard RadHarmony format.

    Args:
        base_dir: The ``CXR_png/`` image folder itself. ``ClinicalReadings/`` and
            ``Annotations-2/`` are the folders next to it.
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

    #: Parsed from '<sex> <age>yrs' by _AGE_RE. _parse_clinical_file converts months and days
    #: to years.
    AGE_SOURCE_COL = "patient_age"
    AGE_SOURCE_UNIT = "years"
    #: "M" or "F", parsed from the first word. The base step makes ``sex`` (D-26).
    SEX_SOURCE_COL = "patient_sex"

    def __init__(self, base_dir: str):
        self.base_dir = os.path.expanduser(base_dir)
        super().__init__(csv_path=None)

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["base_dir"] = self.base_dir
        snap.pop("csv_path", None)
        return snap

    def verify_images(self, base_image_dir: str = None, drop_missing: bool = True) -> pd.DataFrame:
        """Check the image files. The folder defaults to ``base_dir`` (D-04, V-H3-36)."""
        return super().verify_images(base_image_dir or self.base_dir, drop_missing)

    def _read_source(self) -> pd.DataFrame:
        """List the ``CHNCXR_*.png`` files of ``base_dir`` (one row for each image).

        It also reads the other source files: the clinical readings, the TB finding columns and
        the list of finding masks. They are read for all rows before the id filter. Then a table
        of a few studies has the column types of the full table (D-02).
        """
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
        df = pd.DataFrame(rows)
        df = self._add_clinical_readings(df)
        df = self._add_tb_findings(df)
        return self._add_finding_masks(df)

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["_numeric_id"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["_numeric_id"].astype(str)

    def _build_image_path(self) -> None:
        self.df["image_path"] = self.df["_stem"] + ".png"

    def _build_labels(self) -> None:
        self.df["tuberculosis"] = self.df["_label_digit"]

    def _add_clinical_readings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``patient_age``, ``patient_sex`` and ``report`` from ``ClinicalReadings/``."""
        readings_dir = os.path.join(self.base_dir, "..", "ClinicalReadings")
        if not os.path.isdir(readings_dir):
            return df

        ages, sexes, reports = [], [], []
        for stem in df["_stem"]:
            txt_path = os.path.join(readings_dir, stem + ".txt")
            age, sex, text = _parse_clinical_file(txt_path)
            ages.append(age)
            sexes.append(sex)
            reports.append(text)

        df["patient_age"] = ages
        df["patient_sex"] = sexes
        df["report"] = reports
        return df

    def _add_tb_findings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Join the 19 TB finding columns of ``Annotations-2/Statistics_ShenzhenDataset_US.csv``."""
        stats_path = os.path.join(
            self.base_dir, "..", "Annotations-2", "Statistics_ShenzhenDataset_US.csv"
        )
        if not os.path.isfile(stats_path):
            return df

        stats = pd.read_csv(stats_path)
        # "CHNCXR_0327_1.png" gives the stem "CHNCXR_0327_1".
        stats["_stem"] = stats["imagename"].str.replace(r"\.png$", "", regex=True)

        keep = [c for c in TB_FINDING_COLS if c in stats.columns]
        stats = stats[["_stem"] + keep]

        return df.merge(stats, on="_stem", how="left")

    def _add_finding_masks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``finding_masks``: a JSON list of mask paths for each patient from ``Annotations-2/masks/``.

        A path is ``../Annotations-2/masks/<fname>``, so it works when joined to
        ``base_image_dir`` (``CXR_png/``). :meth:`preprocess_masks` joins the masks of
        each image into one PNG. It sets ``mask_path``.
        """
        masks_dir = os.path.join(self.base_dir, "..", "Annotations-2", "masks")
        if not os.path.isdir(masks_dir):
            df["finding_masks"] = None
            return df

        stem_to_masks: dict = defaultdict(list)
        for fname in sorted(os.listdir(masks_dir)):
            if not fname.endswith(".png"):
                continue
            # Example: CHNCXR_0327_1_Small_Infiltrate_(non-linear)_1.png. The patient stem is
            # the first 3 tokens.
            parts = fname.split("_")
            if len(parts) < 3:
                continue
            stem = "_".join(parts[:3])  # CHNCXR_0327_1
            rel_path = os.path.join("..", "Annotations-2", "masks", fname)
            stem_to_masks[stem].append(rel_path)

        df["finding_masks"] = df["_stem"].map(
            lambda s: json.dumps(stem_to_masks[s]) if stem_to_masks[s] else None
        )
        return df

    def preprocess_masks(
        self,
        output_dir: str,
        base_image_dir: str = None,
        num_cores: int = 1,  # noqa: ARG002 (same signature as the base method)
    ) -> None:
        """Join the finding masks of each image into one binary "TB region" PNG.

        As in SIIM-ACR PTX, a TB-negative image also gets a PNG. For a TB-positive row with
        mask paths in ``finding_masks`` (relative to ``self.base_dir``), the PNG is the union of
        the masks. For a TB-negative row, it is an all-zero mask of the same size. These rows
        stay in the samples, so a segmentation test can use them as true negatives. A TB-positive
        row with no finding mask was not annotated. It gets no mask and an empty ``mask_path``
        (D-33).

        ``mask_path`` is the absolute path of the PNG. A PNG that exists is not written again.
        """
        import numpy as np  # type: ignore[import-untyped]
        from PIL import Image
        from tqdm import tqdm

        if self.df is None:
            raise RuntimeError("Call harmonize() before preprocess_masks().")
        base_dir = base_image_dir or self.base_dir
        # Without the folder, every TB image would look "not annotated" and drop out (D-35).
        # A table with no TB image needs no mask file.
        masks_dir = os.path.normpath(os.path.join(base_dir, "..", "Annotations-2", "masks"))
        has_tb = "tuberculosis" in self.df.columns and (self.df["tuberculosis"] == 1).any()
        if has_tb and not os.path.isdir(masks_dir):
            raise FileNotFoundError(
                f"Mask folder not found: {masks_dir}. Download Annotations-2/ next to CXR_png/, "
                "or leave mask_output_dir as None (D-35)."
            )
        if "finding_masks" not in self.df.columns:
            raise ValueError("The table has no finding_masks column. Run harmonize() of this class first.")

        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(output_dir, exist_ok=True)

        fused_paths: list[str | None] = []
        for _, row in tqdm(
            self.df.iterrows(), total=len(self.df), desc="Fusing TB-finding masks"
        ):
            stem = os.path.splitext(os.path.basename(row["image_path"]))[0]
            out_path = os.path.join(output_dir, stem + ".png")

            # ``finding_masks`` is missing for TB-negative rows and for 13 TB-positive rows.
            raw = row.get("finding_masks")
            mask_rels: list = []
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        mask_rels = parsed
                except (ValueError, TypeError):
                    pass

            if not mask_rels and row.get("tuberculosis") == 1:
                fused_paths.append(None)  # TB-positive, but nobody drew a mask (D-33)
                continue
            if not os.path.isfile(out_path):
                if mask_rels:
                    # TB-positive: union of the finding masks.
                    union = None
                    for rel in mask_rels:
                        with Image.open(os.path.normpath(os.path.join(base_dir, rel))) as im:
                            arr = np.array(im.convert("L"))
                        union = (arr > 0) if union is None else (union | (arr > 0))
                    fused = union.astype(np.uint8) * 255  # type: ignore[union-attr]
                else:
                    # TB-negative: a black PNG of the image size, as in SIIM-ACR.
                    img_path = os.path.normpath(
                        os.path.join(base_dir, row["image_path"])
                    )
                    with Image.open(img_path) as im:
                        w, h = im.size  # PIL gives (W, H)
                    fused = np.zeros((h, w), dtype=np.uint8)
                Image.fromarray(fused).save(out_path)
            fused_paths.append(out_path)

        self.df["mask_path"] = fused_paths
