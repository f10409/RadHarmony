"""Harmonizer for the Emory CXR v2 (internal) dataset."""

import os
import re
import warnings

import pandas as pd

from ..base import BaseHarmonizer


#: The 14 CheXpert-style finding columns shipped in
#: ``EmoryCXR_v2_FindingLabel_*.csv``, keyed on accession.
_FINDING_LABEL_COLS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

#: Join key shared by the metadata, label and report tables.
_ACCESSION_COL = "AccessionNumber_anon"


def _as_join_key(series: pd.Series) -> pd.Series:
    """Turn an accession column into a string join key.

    ``pandas`` reads ``AccessionNumber_anon`` as ``int64`` in the metadata
    and label CSVs. It reads ``object`` in the report CSV, because the
    multi-line ``deid_report`` field blocks the fast path. A ``merge`` across
    the two types raises an error. ``isin`` does not: it returns all False,
    so a coverage check shows 0% overlap when the true overlap is 100%.
    A string key on both sides avoids both problems.
    """
    return series.astype(str).str.strip()


def _is_blank_report(text) -> bool:
    """True for no report: NaN, or a template with only "REPORT", spaces and underscores (D-03)."""
    return pd.isna(text) or not re.sub(r"[\s_]+", "", re.sub(r"(?i)^\s*report", "", str(text)))


def _drop_where_study_has_another_row(table: pd.DataFrame, drop: pd.Series) -> pd.DataFrame:
    """Remove the rows in ``drop`` whose accession also has a row not in ``drop`` (D-03)."""
    others = set(table.loc[~drop, _ACCESSION_COL])
    return table[~(drop & table[_ACCESSION_COL].isin(others))]


class EmoryCXRHarmonizer(BaseHarmonizer):
    """Harmonize Emory CXR v2 into the standard RadHarmony format.

    An internal Emory corpus of 2,430,209 images, 1,214,535 studies and
    402,809 patients (2008 to 2021, 11 institutions). Every study has a
    report, except 3 studies (7 images) with only a blank template (D-03).

    Three tables under ``TABLES/`` join on ``AccessionNumber_anon``:

    - ``EmoryCXR_v2_Metadata_<date>.csv``: one row for each image. It has
      ``ImagePath`` (relative to the PNG root), ``ViewPosition``, ``Frontal``
      and the demographic columns.
    - ``EmoryCXR_v2_Report_<date>.csv``: mostly one row for each accession.
      151 accessions have 2 or more (D-03). The text is in ``deid_report``,
      so :meth:`_build_report` is overridden.
    - ``EmoryCXR_v2_FindingLabel_<date>.csv``: 14 CheXpert-style columns.
      Most cells are blank, and ``No Finding`` has no explicit negative.
      A blank is not a 0. ``-1`` (uncertain) is kept.

    .. warning::

       The metadata manifest has versions that are not interchangeable.
       ``08112025`` has 2,486,502 images and 410,449 patients. ``08152025``
       has 2,430,209 and 402,809 after a filter (minimum age 18.0). Pin one
       file. :func:`infer_path` picks the newest known one, which can change
       when a new manifest arrives.

    .. note::

       Set ``base_image_dir`` to ``DEID_PNG/``, not ``ORIGINAL_PNG/``. Both
       hold 16-bit grayscale PNGs of the same size. They were min-max
       normalized with different pixel extrema, so 53% to 99.99% of the pixels
       differ (a monotone rescale). Do not mix them.

    More: ``docs/wiki/reference/datasets/cxr-labels/emory_cxr.md``.

    Args:
        csv_path: Path to ``EmoryCXR_v2_Metadata_<date>.csv``.
        base_image_dir: Root of the PNG tree, for example
            ``/data/EmoryCXRv2/DEID_PNG/``.
            ``image_path`` values are relative to it.
        label_csv_path: Path to ``EmoryCXR_v2_FindingLabel_<date>.csv``.
            ``None`` skips labels.
        report_csv_path: Path to ``EmoryCXR_v2_Report_<date>.csv``.
            ``None`` skips reports.
    """

    LABEL_COLS = _FINDING_LABEL_COLS
    LABEL_JOIN_COLS = [_ACCESSION_COL]

    # ViewPosition is a column of the metadata CSV, so no merge is needed.
    VIEW_POSITION_SOURCE_COL = "ViewPosition"
    VIEW_POSITION_JOIN_COLS = None

    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None

    # The report text is a CSV column, not a file path. _build_report skips
    # the base file read.
    REPORT_JOIN_COLS = [_ACCESSION_COL]
    REPORT_PATH_COL = None

    EXTRA_OUTPUT_COLS = [
        "accession_number",
        "patient_sex",
        "patient_age",
        "study_date",
        "institution",
        "frontal",
    ]

    #: Metadata ``Age`` is in years, renamed to ``patient_age``. See ``_age.py``.
    AGE_SOURCE_COL = "patient_age"
    AGE_SOURCE_UNIT = "years"
    #: Metadata ``Sex``, renamed to ``patient_sex``. The base step makes ``sex`` (D-26).
    SEX_SOURCE_COL = "patient_sex"

    def __init__(
        self,
        csv_path: str,
        base_image_dir: str = None,
        label_csv_path: str = None,
        report_csv_path: str = None,
    ):
        super().__init__(
            csv_path=csv_path,
            label_csv_path=label_csv_path,
            report_csv_path=report_csv_path,
        )
        self.base_image_dir = base_image_dir

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        if self.base_image_dir is not None:
            snap["base_image_dir"] = self.base_image_dir
        return snap

    #: Metadata columns that pass through, with their new names.
    _PASSTHROUGH_COLS = {
        "Sex": "patient_sex",
        "Age": "patient_age",
        "StudyDate_anon": "study_date",
        "InstitutionName_anon": "institution",
        "Frontal": "frontal",
    }

    def _build_patient_id(self) -> None:
        """Use ``empi_anon`` (the first path part) as the patient key.

        It differs from ``study_id`` on purpose. A patient has up to 296
        studies, and 42.7% of patients have more than one. A split on the
        study key would put one patient in train and test.

        This is the first hook that ``harmonize()`` calls after the CSV read.
        So it also adds the passthrough columns, which must exist before
        ``_select_harmonized_columns``.
        """
        self.df["patient_id"] = self.df["empi_anon"].astype(str)

        for src, dst in self._PASSTHROUGH_COLS.items():
            if src in self.df.columns and dst not in self.df.columns:
                self.df[dst] = self.df[src]

    def _build_view_position(self) -> None:
        """Rename ``ViewPosition``. Then ``Frontal`` wins when the two disagree (D-01, Q-14).

        ``Frontal`` = 0 gives ``Lateral`` (the base step makes it ``LATERAL``, Q-13), except with ``ViewPosition``
        AP: most of these 2,202 images are upright frontal or decubitus, so they get NaN.
        ``Frontal`` = 1 with ``Lateral`` gives NaN, because AP or PA is unknown.
        """
        super()._build_view_position()
        if "Frontal" not in self.df.columns or "view_position" not in self.df.columns:
            return
        vp, frontal = self.df["view_position"], self.df["Frontal"]
        new = vp.mask(frontal == 0, "Lateral")
        new = new.mask(((frontal == 0) & (vp == "AP")) | ((frontal == 1) & (vp == "Lateral")), None)
        self.df["view_position"] = new

    def _build_study_id(self) -> None:
        # Make the raw column a string. It is the join key of the later
        # label and report merges, and pandas does not merge int64 with str.
        self.df[_ACCESSION_COL] = _as_join_key(self.df[_ACCESSION_COL])
        self.df["study_id"] = self.df[_ACCESSION_COL]
        self.df["accession_number"] = self.df[_ACCESSION_COL]

    def _build_image_path(self) -> None:
        """``ImagePath`` is already ``<empi>/<accession>/<sop>.png``.

        It stays relative to ``base_image_dir``, so the table works on any mount.
        """
        self.df["image_path"] = self.df["ImagePath"].astype(str)

    def _preprocess_label_df(self, label_df: pd.DataFrame) -> pd.DataFrame:
        """Make the accession a string and keep only the label columns.

        A label row with only ``No Finding = 1`` is dropped when the study has
        another row (D-03). Other conflicts keep the first row, with a warning.

        Blanks stay NaN on purpose. A blank means "the report did not
        mention it", not "absent". ``No Finding`` has 597,010 positives and
        no negatives. Filling NaN with 0 would invent 726,298 negatives.
        """
        label_df = label_df.copy()
        label_df[_ACCESSION_COL] = _as_join_key(label_df[_ACCESSION_COL])
        labels = [c for c in self.LABEL_COLS if c in label_df.columns]
        label_df = label_df[[_ACCESSION_COL] + labels]
        # A row with only "No Finding" = 1 comes from a blank report template.
        # It loses to another row of the same study (D-03).
        if "No Finding" in labels:
            nf_only = (label_df[labels].notna().sum(axis=1) == 1) & (label_df["No Finding"] == 1)
            label_df = _drop_where_study_has_another_row(label_df, nf_only)
        self._warn_conflicting_rows(label_df, "label CSV")
        return label_df.drop_duplicates(subset=[_ACCESSION_COL])

    def _build_labels(self) -> None:
        """Left-join the finding table, so unlabeled studies stay.

        The base class uses an inner merge. It would drop every image whose
        accession is not in the label CSV.
        """
        if self.label_csv_path is not None:
            label_df = pd.read_csv(self.label_csv_path)
            label_df = self._preprocess_label_df(label_df)
            self.df = self.df.merge(
                label_df, on=self.LABEL_JOIN_COLS, how="left"
            )

        for col in self.LABEL_COLS:
            if col in self.df.columns:
                self.df.rename(
                    columns={col: col.lower().replace(" ", "_")}, inplace=True
                )

    def _build_report(self) -> None:
        """Merge inline report text keyed on accession.

        The base class reads files named in ``REPORT_PATH_COL``. Emory has
        the text itself in ``deid_report``.
        """
        if self.report_csv_path is None:
            return

        report_df = pd.read_csv(self.report_csv_path, dtype=str)
        report_df = self._preprocess_report_df(report_df)
        report_df = report_df[[_ACCESSION_COL, "report"]]
        self.df = self.df.merge(report_df, on=self.REPORT_JOIN_COLS, how="left")

    def _preprocess_report_df(self, report_df: pd.DataFrame) -> pd.DataFrame:
        """Make the join key a string and rename ``deid_report`` to ``report``.

        A blank report template counts as no report. It is dropped when the
        study has another row (D-03). Other conflicts keep the first row, with
        a warning.
        """
        report_df = report_df.copy()
        report_df[_ACCESSION_COL] = _as_join_key(report_df[_ACCESSION_COL])
        if "deid_report" in report_df.columns:
            report_df = report_df.rename(columns={"deid_report": "report"})
        if "report" in report_df.columns:
            # A blank template is no report. A real report of the study wins (D-03).
            blank = report_df["report"].map(_is_blank_report)
            report_df.loc[blank, "report"] = None
            report_df = _drop_where_study_has_another_row(report_df, blank)
            self._warn_conflicting_rows(report_df[[_ACCESSION_COL, "report"]], "report CSV")
        return report_df.drop_duplicates(subset=[_ACCESSION_COL])

    def _warn_conflicting_rows(self, table: pd.DataFrame, name: str) -> None:
        """Warn when an accession of the image table has 2 or more different rows in ``table`` (V-H1-12).

        The callers first drop a blank report template and a "No Finding only"
        label row when the study has another row (D-03). For the rest, for
        example 2 real reports, ``drop_duplicates`` keeps the first row.
        """
        rows = table.drop_duplicates()
        conflict = set(rows.loc[rows[_ACCESSION_COL].duplicated(), _ACCESSION_COL])
        n = len(conflict & set(self.df[_ACCESSION_COL]))
        if n:
            warnings.warn(
                f"Emory CXR: {n} accessions have 2 or more different rows in the {name}. "
                "The table keeps the first row of each.",
                UserWarning,
                stacklevel=2,
            )

class EmoryCXRDeidHarmonizer(EmoryCXRHarmonizer):
    """Emory CXR v2 bound to the de-identified PNG tree.

    The same as :class:`EmoryCXRHarmonizer`, but it refuses an
    ``ORIGINAL_PNG`` root. The two trees differ (see the parent docstring).
    Use the de-identified one for data that leaves the site.
    """

    def __init__(self, csv_path: str, base_image_dir: str = None, **kwargs):
        if base_image_dir and "ORIGINAL_PNG" in os.path.normpath(base_image_dir):
            raise ValueError(
                "EmoryCXRDeidHarmonizer expects the de-identified PNG tree "
                "(.../DEID_PNG/), got an ORIGINAL_PNG path: "
                f"{base_image_dir!r}. ORIGINAL_PNG carries burned-in "
                "annotations and a different intensity normalisation; use "
                "EmoryCXRHarmonizer explicitly if you really want it."
            )
        super().__init__(csv_path=csv_path, base_image_dir=base_image_dir, **kwargs)


#: Path part that marks the DICOM root. The code removes it and everything
#: before it, so ``image_path`` stays relative.
_DICOM_ROOT_MARKER = "DEID_DICOM/"

#: Metadata CSV column with the original SOP Instance UID. It is the only
#: key that reaches the DICOM tree (see :class:`EmoryCXRDicomHarmonizer`).
_SOP_COL = "SOP"


class EmoryCXRDicomHarmonizer(EmoryCXRHarmonizer):
    """Emory CXR v2 bound to the de-identified DICOM tree.

    .. warning::

       These DICOMs are 8-bit. All five MD.ai batch manifests show
       ``BitsStored = BitsAllocated = 8`` and ``MONOCHROME2``. The PNGs in
       ``DEID_PNG/`` are 16-bit. The tree is a re-encoded MD.ai annotation
       export, not the acquisition source. It has less pixel information than
       the PNGs. Use this class only for DICOM headers or the annotation
       round trip. For training pixels use :class:`EmoryCXRHarmonizer`.

    The DICOM export was re-identified. ``AccessionNumber`` is blank in every
    MD.ai row and ``PatientID`` is new for each image, so no cohort key is
    inside the files. The only link back is
    ``TABLES/mapping_deid_dcm_sopid.csv``. It maps each ``dcm_path`` to the
    original ``SOPInstanceUID``, which is also the ``SOP`` column of the
    metadata CSV and the name of the PNG files. This 1 GB file has no backup.
    Without it the DICOM tree has no link to a patient.

    All 2,430,209 SOPs of the ``08152025`` manifest and all 2,486,502 of
    ``08112025`` have a mapping row. The mapping has 2,636,568 rows and the
    MD.ai manifests name 2,636,571 files. The 3 extra files were never
    delivered and have no mapping row. The inner join drops them.

    Args:
        csv_path: Path to ``EmoryCXR_v2_Metadata_<date>.csv``.
        base_image_dir: Root of the DICOM tree, for example
            ``<EmoryCXRv2>/DEID_DICOM/``. Its children are the batch folders
            ``B1/`` to ``B5/``.
        mapping_csv_path: Path to ``mapping_deid_dcm_sopid.csv``.
            Required. It is the only link from a DICOM to a patient.
        label_csv_path: Path to ``EmoryCXR_v2_FindingLabel_<date>.csv``.
        report_csv_path: Path to ``EmoryCXR_v2_Report_<date>.csv``.
    """

    #: ViewPosition stays as in the source (Frank Li, 2026-11-01). No shared names (Q-13).
    VIEW_POSITION_NAMES = False

    def __init__(
        self,
        csv_path: str,
        base_image_dir: str = None,
        mapping_csv_path: str = None,
        label_csv_path: str = None,
        report_csv_path: str = None,
    ):
        if mapping_csv_path is None:
            raise ValueError(
                "mapping_csv_path is required for the DICOM variant. The "
                "MD.ai export blanked AccessionNumber and regenerated "
                "PatientID, so 'mapping_deid_dcm_sopid.csv' is the only "
                "route from a .dcm back to a patient or a report."
            )
        super().__init__(
            csv_path=csv_path,
            base_image_dir=base_image_dir,
            label_csv_path=label_csv_path,
            report_csv_path=report_csv_path,
        )
        self.mapping_csv_path = mapping_csv_path

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["mapping_csv_path"] = self.mapping_csv_path
        return snap

    # No Frontal rule here (Frank Li, 2026-11-01): view_position stays the DICOM ViewPosition, as before.
    _build_view_position = BaseHarmonizer._build_view_position

    def _build_image_path(self) -> None:
        """Join the SOP to dcm_path mapping and make the path relative.

        The join is inner on purpose, unlike the label and report merges.
        A row with no mapping has no file. Its NaN path would fail later in
        the MONAI pipeline instead of here.
        """
        mapping = pd.read_csv(self.mapping_csv_path, dtype=str)
        missing = {"dcm_path", "SOPInstanceUID"} - set(mapping.columns)
        if missing:
            raise ValueError(
                f"{self.mapping_csv_path!r} is missing column(s) {sorted(missing)}; "
                "expected the two-column 'dcm_path,SOPInstanceUID' mapping."
            )
        mapping = mapping.rename(columns={"SOPInstanceUID": _SOP_COL})
        mapping[_SOP_COL] = mapping[_SOP_COL].astype(str).str.strip()
        mapping = mapping.drop_duplicates(subset=[_SOP_COL])

        if _SOP_COL not in self.df.columns:
            raise ValueError(
                "The metadata CSV has no 'SOP' column, so it cannot be "
                "joined to the DICOM tree. The dated manifests all carry "
                "it; check that csv_path points at one of them."
            )
        self.df[_SOP_COL] = self.df[_SOP_COL].astype(str).str.strip()

        before = len(self.df)
        self.df = self.df.merge(mapping, on=_SOP_COL, how="inner")
        if len(self.df) == 0 and before > 0:
            raise ValueError(
                "The SOP join produced zero rows. Check that the mapping "
                "CSV and the metadata manifest describe the same cohort."
            )

        # Remove the absolute prefix. image_path is relative to base_image_dir, as in the PNG variant.
        self.df["image_path"] = (
            self.df["dcm_path"]
            .astype(str)
            .str.split(_DICOM_ROOT_MARKER, n=1)
            .str[-1]
        )
