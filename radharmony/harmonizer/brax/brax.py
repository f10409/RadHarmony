"""Harmonizer for the BRAX (Brazilian Chest X-Ray) v1.1.0 release.

BRAX has DICOM and PNG images in one download. Labels are CheXpert-aligned
(14 classes) and come from Portuguese reports. A label cell is ``1``
(positive), ``0`` (negation), ``-1`` (uncertain) or blank (not mentioned).
The ``image_format`` argument picks the DICOM or the PNG path column.
Reports are not in the public release, so ``output_report`` is unsupported.
The ``uncertain_strategy`` argument sets how ``-1`` cells are mapped:
``raw`` (default), ``u_zeros``, ``u_ones``, ``u_ignore`` or ``drop``.

Counts, layout and strategy table: ``docs/wiki/reference/datasets/cxr-labels/brax.md``.
"""

import os
import re

import numpy as np
import pandas as pd

from .._ids import warn_shared_study_ids
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
        uncertain_strategy: How to map ``-1`` (uncertain) cells. See the
            module docstring for the five options.
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


    #: PatientAge is the low edge of a 5-year group ("0", "5", ... "80", or
    #: "85 or more" for 2,299 of 40,967 images). See ``_age.py``.
    AGE_SOURCE_COL = "patient_age"
    AGE_SOURCE_UNIT = "years"

    #: PatientSex (M or F), renamed to ``patient_sex``. The base step makes ``sex`` (D-26).
    SEX_SOURCE_COL = "patient_sex"

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

    def _read_source(self) -> pd.DataFrame:
        """Read ``master_spreadsheet.csv``.

        Rename the extra metadata columns (PatientSex, Rows, ...) to the names in ``EXTRA_OUTPUT_COLS``.
        """
        df = super()._read_source()
        rename = {
            "PatientSex": "patient_sex",
            "PatientAge": "patient_age",
            "Manufacturer": "manufacturer",
            "StudyDate": "study_date",
            "Rows": "rows",
            "Columns": "columns",
        }
        return df.rename(columns=rename)

    def _build_age_years(self, df: pd.DataFrame) -> pd.DataFrame:
        """Give the low edge of each 5-year age group, so ``"85 or more"`` gives 85 (D-06).

        ``patient_age`` keeps the source value.
        """
        df = super()._build_age_years(df)
        if "age_years" in df.columns:
            df.loc[df[self.AGE_SOURCE_COL].astype(str).str.strip().eq("85 or more"), "age_years"] = 85.0
        return df

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["PatientID"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["AccessionNumber"].astype(str)
        # BRAX 1.1.0 gives 2 AccessionNumbers to 2 patients each (V-H1-9).
        warn_shared_study_ids(self.df, "BRAX", "AccessionNumber")

    def _build_image_path(self) -> None:
        path_col = "DicomPath" if self.image_format == "dicom" else "PngPath"
        if path_col not in self.df.columns:
            raise ValueError(
                f"Column {path_col!r} not found in master_spreadsheet.csv "
                f"(image_format={self.image_format!r}). "
                f"Available columns: {list(self.df.columns)[:20]}"
            )
        # CSV paths start at the BRAX root and keep the 'Anonymized_DICOMs/'
        # or 'images/' prefix, so base_image_dir needs no stripping.
        self.df["image_path"] = self.df[path_col].astype(str)

        # Take Series_<UID> from the path. It is None if the path has none.
        self.df["series_id"] = self.df["image_path"].apply(
            lambda p: (m.group(0) if (m := _SERIES_RE.search(p)) else None)
        )

    def _build_series_id(self) -> None:
        # _build_image_path sets series_id. Skip the base default.
        pass

    def _build_labels(self) -> None:
        # The CSV uses Title Case names such as 'No Finding'. Make them snake case.
        for col in self.LABEL_COLS:
            if col in self.df.columns:
                self.df.rename(
                    columns={col: col.lower().replace(" ", "_")}, inplace=True
                )

        snake_cols = [c.lower().replace(" ", "_") for c in self.LABEL_COLS]
        present = [c for c in snake_cols if c in self.df.columns]

        # 'raw' keeps the source values (1 / 0 / -1 / NaN) and drops no row.
        if self.uncertain_strategy == "raw":
            return

        # Other strategies map NaN ("not mentioned") to 0, as CheXpert does.
        # They differ only in how they map -1.
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
