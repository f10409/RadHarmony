"""Harmonizer for the BRAX (Brazilian Chest X-Ray) v1.1.0 release.

PhysioNet credentialed-access release (~24,959 studies, ~40,967 images,
19,351 patients) with both DICOM and PNG variants in the same download.
Labels are CheXpert-aligned (14 classes), NLP-derived from Portuguese
radiology reports using the original CheXpert labeller.  Per the BRAX
documentation, label cells use the encoding **``1`` = positive,
``0`` = negation, ``-1`` = uncertainty**, with blank cells indicating
that the labeller did not extract a mention.

Layout::

    <base_image_dir>/                # the BRAX root, e.g. .../physionet.org/files/brax/1.1.0/
      master_spreadsheet.csv         # primary metadata
      Anonymized_DICOMs/
        id_<PatientID>/Study_<UID>/Series_<UID>/image-<UID>.dcm
      images/
        id_<PatientID>/Study_<UID>/Series_<UID>/image-<UID>.png

Both DICOM and PNG variants are produced from the same harmonizer; the
``image_format`` parameter selects which path column to surface as
``image_path``.  Reports are NOT distributed in the public release;
``output_report`` is unsupported.

Uncertain-label handling — five options.  ``"raw"`` is the default so the
harmonized DataFrame mirrors the source CSV cell-for-cell.  The other
four match the CheXpert-paper ablations and exist for callers who need
training-ready 0/1 tensors:

* ``"raw"`` (default) — preserve cells exactly: ``1`` / ``0`` / ``-1`` /
  ``NaN``.  No coercion, no row drops.  Use when faithfulness to the
  source matters (e.g. building label-uncertainty diagnostics) or when
  your loss function explicitly handles ``-1``.
* ``"u_zeros"`` — ``NaN`` → ``0``, then ``-1`` → ``0``.  Retains every row.
* ``"u_ones"``  — ``NaN`` → ``0``, then ``-1`` → ``1``.  Retains every row.
* ``"u_ignore"`` — ``NaN`` → ``0``, then ``-1`` → ``NaN``.  Retains every
  row; downstream loss must mask the resulting ``NaN`` entries.
* ``"drop"`` — ``NaN`` → ``0``, then ``-1`` → ``NaN``, then drop any row
  whose label vector still contains a ``NaN``.  Matches the legacy
  ``CheXpertHarmonizer(drop_uncertain=True)`` behaviour.
"""

import os
import re

import numpy as np
import pandas as pd

from ..base import BaseHarmonizer


_VALID_IMAGE_FORMATS = ("dicom", "png")
_VALID_UNCERTAIN_STRATEGIES = ("raw", "u_zeros", "u_ones", "u_ignore", "drop")
_SERIES_RE = re.compile(r"Series_[^/\\]+")


class BRAXHarmonizer(BaseHarmonizer):
    """Harmonize BRAX into the standard RadHarmony format.

    Args:
        csv_path: Path to ``master_spreadsheet.csv``.
        base_image_dir: BRAX root (containing ``Anonymized_DICOMs/`` and
            ``images/``).  Used only for ``_harmonizer_init_snapshot``;
            the harmonized DataFrame's ``image_path`` is stored relative
            to this directory.
        image_format: ``"dicom"`` (default) selects ``DicomPath``;
            ``"png"`` selects ``PngPath``.
        uncertain_strategy: How to map ``-1`` (uncertain) cells.  See
            module docstring for the four options.
    """

    LABEL_COLS = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "No Finding",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices",
    ]

    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = "ViewPosition"
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    EXTRA_OUTPUT_COLS = [
        "patient_sex",
        "patient_age",
        "manufacturer",
        "study_date",
        "rows",
        "columns",
    ]

    def __init__(
        self,
        csv_path: str,
        base_image_dir: str = None,
        image_format: str = "dicom",
        uncertain_strategy: str = "raw",
    ):
        if image_format not in _VALID_IMAGE_FORMATS:
            raise ValueError(
                f"image_format must be one of {_VALID_IMAGE_FORMATS}, got {image_format!r}"
            )
        if uncertain_strategy not in _VALID_UNCERTAIN_STRATEGIES:
            raise ValueError(
                f"uncertain_strategy must be one of {_VALID_UNCERTAIN_STRATEGIES}, "
                f"got {uncertain_strategy!r}"
            )

        super().__init__(csv_path=os.path.expanduser(csv_path) if csv_path else csv_path)
        self.base_image_dir = (
            os.path.expanduser(base_image_dir) if base_image_dir else base_image_dir
        )
        self.image_format = image_format
        self.uncertain_strategy = uncertain_strategy

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["base_image_dir"] = self.base_image_dir
        snap["image_format"] = self.image_format
        snap["uncertain_strategy"] = self.uncertain_strategy
        return snap

    # ------------------------------------------------------------------
    # Build hooks
    # ------------------------------------------------------------------

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["PatientID"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["AccessionNumber"].astype(str)

    def _build_image_path(self) -> None:
        path_col = "DicomPath" if self.image_format == "dicom" else "PngPath"
        if path_col not in self.df.columns:
            raise ValueError(
                f"Column {path_col!r} not found in master_spreadsheet.csv "
                f"(image_format={self.image_format!r}). "
                f"Available columns: {list(self.df.columns)[:20]}"
            )
        # Paths in the CSV are relative to the BRAX root and include the
        # 'Anonymized_DICOMs/' or 'images/' prefix.  base_image_dir is the
        # BRAX root itself, so no prefix stripping is needed.
        self.df["image_path"] = self.df[path_col].astype(str)

        # Extract Series_<UID> from the path so series_id is stable across
        # multi-series studies.  Falls back to None if the pattern is absent.
        self.df["series_id"] = self.df["image_path"].apply(
            lambda p: (m.group(0) if (m := _SERIES_RE.search(p)) else None)
        )

    def _build_series_id(self) -> None:
        # Already populated by _build_image_path; suppress the base-class default.
        pass

    def _build_labels(self) -> None:
        # Snake-case the 14 label columns first.  Source CSV uses Title Case
        # ('Atelectasis', 'No Finding', etc.) per the BRAX documentation.
        for col in self.LABEL_COLS:
            if col in self.df.columns:
                self.df.rename(
                    columns={col: col.lower().replace(" ", "_")}, inplace=True
                )

        snake_cols = [c.lower().replace(" ", "_") for c in self.LABEL_COLS]
        present = [c for c in snake_cols if c in self.df.columns]

        # 'raw' preserves the source encoding exactly (1 / 0 / -1 / NaN) per
        # the BRAX paper specification.  No coercion, no fillna, no row drops.
        if self.uncertain_strategy == "raw":
            return

        # All other strategies start with the standard CheXpert convention
        # of treating "mention not extracted" (NaN) as negation (0); they
        # differ only in how the explicit -1 (uncertainty) cells are mapped.
        self.df[present] = self.df[present].fillna(0.0)

        if self.uncertain_strategy == "u_zeros":
            self.df[present] = self.df[present].replace({-1: 0})
        elif self.uncertain_strategy == "u_ones":
            self.df[present] = self.df[present].replace({-1: 1})
        elif self.uncertain_strategy == "u_ignore":
            self.df[present] = self.df[present].replace({-1: np.nan})
        elif self.uncertain_strategy == "drop":
            self.df[present] = self.df[present].replace({-1: np.nan})
            self.df.dropna(subset=present, inplace=True)

    def _build_view_position(self) -> None:
        super()._build_view_position()
        # Some BRAX rows have NaN ViewPosition for legacy series; preserve
        # them as 'Unknown' rather than dropping (we have no clean signal).
        if "view_position" in self.df.columns:
            self.df["view_position"] = self.df["view_position"].fillna("Unknown")

    # ------------------------------------------------------------------
    # Extra demographic/scanner metadata
    # ------------------------------------------------------------------

    def _build_extras(self) -> None:
        """Map the extra metadata columns to the snake_case names declared in EXTRA_OUTPUT_COLS."""
        rename = {
            "PatientSex": "patient_sex",
            "PatientAge": "patient_age",
            "Manufacturer": "manufacturer",
            "StudyDate": "study_date",
            "Rows": "rows",
            "Columns": "columns",
        }
        for src, dst in rename.items():
            if src in self.df.columns:
                self.df.rename(columns={src: dst}, inplace=True)

    def harmonize(self, *args, **kwargs) -> pd.DataFrame:
        # Override to call _build_extras between standard hooks; the base
        # flow doesn't know about them.
        self._harmonized_df_override = None
        self.df = pd.read_csv(self.csv_path)

        self._build_patient_id()
        self._build_study_id()
        self._build_image_path()
        self.LABEL_COLS = sorted(
            self.LABEL_COLS, key=lambda c: c.lower().replace(" ", "_")
        )
        self._build_labels()
        self._build_view_position()
        self._build_extras()
        self._build_mask_path()
        self._build_bbox()
        self._build_report()

        self.df.dropna(subset=["image_path"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self._select_harmonized_columns(self.df)
