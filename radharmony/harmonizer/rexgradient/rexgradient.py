"""Harmonizer for the ReXGradient-160K dataset."""

import json
import os

import pandas as pd

from ..base import BaseHarmonizer


_IMAGE_PATH_PREFIX = "../deid_png/"


def _format_report(indication, comparison, findings, impression) -> str | None:
    """Combine the 4 ReXGradient report sections into a single text block.

    Returns ``None`` when every section is empty / NaN — the dataset's
    convention for "no report".
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

    ReXGradient-160K ships per-split metadata in two parallel forms:

    - ``<split>_metadata.csv`` — one row per *study* (140K / 10K / 10K).
    - ``<split>_metadata_view_position.json`` — one entry per study with
      an ``ImagePath`` list expanding to every image in that study
      (238,968 / 17,007 / 17,029).

    We use the JSON because the per-study CSV alone cannot produce
    per-image rows. The ``csv_path`` argument is kept for interface
    compatibility with :class:`BaseHarmonizer`, but it must point at the
    JSON file — a ``.csv`` path is rejected with an explicit error.

    Concrete example::

        csv_path = ".../ReXGradient-160K/download/metadata/train_metadata_view_position.json"
        base_image_dir = ".../ReXGradient-160K/deid_png/"
        # → image_path values like
        #   "GRDNLZHK1CJMB9DS/GRDNLD4ATLU63FN8/studies/<study>/series/<series>/instances/<sop>.png"

    Reports are pulled from the inline ``Indication`` / ``Comparison`` /
    ``Findings`` / ``Impression`` fields and concatenated with section
    headers — see :func:`_format_report`. There are no structured
    pathology labels in this dataset (``LABEL_COLS`` is empty).

    Args:
        csv_path: Path to the per-split ``*_metadata_view_position.json``
            file. The arg name follows the :class:`BaseHarmonizer`
            convention; pointing it at a ``.csv`` raises ``ValueError``.
        base_image_dir: Root of the extracted PNG tree (e.g.
            ``.../ReXGradient-160K/deid_png/``).
    """

    LABEL_COLS = []
    LABEL_JOIN_COLS = None

    # view_position is populated inline by _build_view_position from the JSON
    # — no separate-CSV merge needed.
    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None

    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None

    # Reports are inline; no separate CSV-of-paths to join.
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    EXTRA_OUTPUT_COLS = ["accession_number", "patient_sex", "patient_age", "study_date"]

    def __init__(self, csv_path: str, base_image_dir: str = None):
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

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        if self.base_image_dir is not None:
            snap["base_image_dir"] = self.base_image_dir
        return snap

    # ------------------------------------------------------------------
    # Custom load: JSON -> per-image DataFrame
    # ------------------------------------------------------------------

    def harmonize(self, *args, **kwargs) -> pd.DataFrame:
        """Read the per-image JSON and emit the standard harmonized slice.

        The base ``harmonize()`` is bypassed because the source file is
        JSON, not CSV, and rows must be exploded from ``ImagePath`` lists
        before any of the per-image columns (view_position, image_path)
        can be set.
        """
        self._harmonized_df_override = None
        if self.csv_path is None or not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"ReXGradient JSON metadata not found: {self.csv_path!r}"
            )

        with open(self.csv_path, "r") as f:
            entries = json.load(f)

        # Build flat per-image rows. One JSON entry → len(ImagePath) rows.
        rows = []
        for study_key, e in entries.items():
            image_paths = e.get("ImagePath") or []
            if not image_paths:
                continue
            view_positions = e.get("ImageViewPosition") or [None] * len(image_paths)
            # Defensive: pad view_positions if the list is shorter than ImagePath.
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
            study_date = e.get("StudyDate")
            for img_path, vp in zip(image_paths, view_positions):
                # JSON paths are written relative to the metadata/ dir as
                # '../deid_png/<...>'; strip that prefix so the result is
                # relative to base_image_dir = .../deid_png/.
                if isinstance(img_path, str) and img_path.startswith(_IMAGE_PATH_PREFIX):
                    rel = img_path[len(_IMAGE_PATH_PREFIX):]
                else:
                    rel = img_path
                rows.append(
                    {
                        "patient_id": patient_id,
                        "study_id": study_id,
                        "image_path": rel,
                        "view_position": vp,
                        "report": report,
                        "accession_number": accession,
                        "patient_sex": sex,
                        "patient_age": age,
                        "study_date": study_date,
                    }
                )

        self.df = pd.DataFrame(rows)

        # The base harmonized-slice helper expects optional columns to be
        # NaN rather than missing entirely; columns we did populate are
        # already present. Drop rows where image_path failed to resolve.
        self.df.dropna(subset=["image_path"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self._select_harmonized_columns(self.df)

    # ------------------------------------------------------------------
    # Required builders (no-ops — columns are populated in harmonize()).
    # Kept so subclasses can hook in if needed.
    # ------------------------------------------------------------------

    def _build_patient_id(self) -> None:  # pragma: no cover - bypassed
        pass

    def _build_study_id(self) -> None:  # pragma: no cover - bypassed
        pass

    def _build_image_path(self) -> None:  # pragma: no cover - bypassed
        pass
