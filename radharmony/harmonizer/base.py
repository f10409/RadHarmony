"""Base harmonizer for radiological datasets."""

import functools
import os
import pickle
import warnings

import pandas as pd


from ._age import to_years  # noqa: E402

_OPTIONAL_COLS = ["series_id", "view_position", "mask_path", "bbox", "bbox_labels", "report"]

_SAVE_VERSION = 1

#: Source sex value (stripped, lower case) -> ``sex`` value (D-26).
_SEX_VALUES = {"m": "M", "male": "M", "f": "F", "female": "F"}
#: Source values for "unknown" or "other". They give NaN with no warning (D-26).
_SEX_UNKNOWN = {"", "o", "u", "unknown", "other"}

#: Source view (stripped, lower case) -> shared 2-D name (Q-13), as for sex (D-26).
_VIEW_VALUES = {"pa": "PA", "postero_anterior": "PA", "ap": "AP", "antero_posterior": "AP",
                "lateral": "LATERAL", "lat": "LATERAL", "l": "LATERAL", "ll": "LATERAL", "rl": "LATERAL"}
#: Source values for an unknown view. They give NaN (Q-13).
_VIEW_UNKNOWN = {"", "unknown", "unk", "n/a", "exclude", "none"}


def _filter_by_ids(df: pd.DataFrame, study_ids=None, patient_ids=None) -> pd.DataFrame:
    """Keep rows whose ``study_id`` / ``patient_id`` is in the given list (``None`` = no filter)."""
    for col, ids in (("study_id", study_ids), ("patient_id", patient_ids)):
        if ids is not None:
            df = df[df[col].astype(str).isin({str(i) for i in ids})].reset_index(drop=True)
    return df


#: The image-copy arguments of ``BaseHarmonizer.harmonize()`` and their defaults. Keep them the same
#: as its signature (V-H2-27).
_PREPROCESS_ARGS = {"preprocess_output_dir": None, "preprocessor": None, "base_image_dir": None,
                    "num_workers": 1, "sample_n": None}


def _filter_after(self, df, study_ids, patient_ids):
    """Filter ``self.df``, the set table and ``df`` by the ids, with a warning (no speedup)."""
    warnings.warn(
        f"{type(self).__name__} has its own harmonize(), so study_ids / patient_ids "
        "were applied after harmonizing (no speedup).",
        UserWarning,
        stacklevel=3,
    )
    if self.df is not None and {"study_id", "patient_id"} <= set(self.df.columns):
        self.df = _filter_by_ids(self.df, study_ids, patient_ids)
    if getattr(self, "_harmonized_df_override", None) is not None:
        self._harmonized_df_override = _filter_by_ids(self._harmonized_df_override, study_ids, patient_ids)
    return _filter_by_ids(df, study_ids, patient_ids) if isinstance(df, pd.DataFrame) else df


def _copy_images_after(self, pre):
    """Run ``preprocess()`` after an own ``harmonize()``, and return the new table (V-H2-27).

    Args:
        pre: the values of the ``_PREPROCESS_ARGS`` of the call.
    """
    options = {k: v for k, v in pre.items() if k != "preprocess_output_dir"}
    override = getattr(self, "_harmonized_df_override", None)
    if override is None:
        self.preprocess(output_dir=pre["preprocess_output_dir"], **options)
    else:
        # The table is the set table, not self.df, so copy the images of that table.
        source_df, self.df = self.df, override
        try:
            self.preprocess(output_dir=pre["preprocess_output_dir"], **options)
            self._harmonized_df_override = self.df
        finally:
            self.df = source_df
    return self.harmonized_df


def _wrap_harmonize(harmonize):
    """Wrap a subclass ``harmonize()`` so that the ids and the image-copy arguments are not silently ignored.

    ``self._id_filter`` holds the state of one call: the ids, ``applied`` (the base
    ``harmonize()`` ran) and ``preprocessed`` (``preprocess()`` ran). The base
    ``harmonize()`` reads the ids from it and filters early. If the subclass never
    reaches the base ``harmonize()``, the result is filtered afterwards with a
    warning (no speedup).

    If the subclass never reaches the base ``harmonize()`` and never calls
    ``preprocess()``, the wrapper runs ``preprocess()`` after it, when
    ``preprocess_output_dir`` is given (V-H2-27). The subclass makes its masks
    first, so a mask keeps the size of the source image (V-H2-21). A ``sample_n``
    with no ``preprocess_output_dir`` does nothing, so the wrapper then warns.
    The wrapper reads only keyword arguments.
    """

    @functools.wraps(harmonize)
    def wrapper(self, *args, study_ids=None, patient_ids=None, **kwargs):
        pre = {k: kwargs.get(k, v) for k, v in _PREPROCESS_ARGS.items()}
        asked = pre["preprocess_output_dir"] is not None or pre["sample_n"] is not None
        has_ids = study_ids is not None or patient_ids is not None
        # No filter and no image copies, or an inner call in a harmonize() chain: run as is.
        if not (has_ids or asked) or getattr(self, "_id_filter", None):
            return harmonize(self, *args, **kwargs)
        self._id_filter = {"study_ids": study_ids, "patient_ids": patient_ids, "applied": False,
                           "preprocessed": False}
        try:
            df = harmonize(self, *args, **kwargs)
            applied, preprocessed = self._id_filter["applied"], self._id_filter["preprocessed"]
        finally:
            self._id_filter = None
        if applied:
            return df
        if has_ids:
            df = _filter_after(self, df, study_ids, patient_ids)
        if not asked or preprocessed:
            return df
        if pre["preprocess_output_dir"] is None:
            warnings.warn(
                f"{type(self).__name__}.harmonize() ignores sample_n without preprocess_output_dir. "
                "It keeps all rows.",
                UserWarning,
                stacklevel=2,
            )
            return df
        return _copy_images_after(self, pre)

    return wrapper


class BaseHarmonizer:
    """Abstract base class for dataset harmonizers.

    Subclasses must implement :meth:`_build_patient_id`, :meth:`_build_study_id`,
    and :meth:`_build_image_path`.  Optional hooks (:meth:`_build_labels`,
    :meth:`_build_view_position`, :meth:`_build_mask_path`, :meth:`_build_bbox`)
    default to no-ops and can be overridden as needed.

    The final harmonized DataFrame always contains ``patient_id``, ``study_id``,
    and ``image_path`` (relative to ``base_image_dir``, D-04).  Any optional
    columns that were populated are appended in the order defined by
    ``_OPTIONAL_COLS`` followed by ``LABEL_COLS`` (snake_cased).

    Args:
        csv_path: Path to the dataset's primary CSV file.
        label_csv_path: Path to a separate label CSV.  Leave ``None`` when
            labels are already in the main CSV.
        view_position_csv_path: Path to a separate CSV containing the view
            position column.  Leave ``None`` when it is already in the main CSV.
        mask_csv_path: Path to a separate CSV containing mask paths.
            Leave ``None`` when it is already in the main CSV.
        bbox_csv_path: Path to a separate CSV containing bounding boxes.
            Leave ``None`` when it is already in the main CSV.
        report_csv_path: Path to a CSV containing report file paths.
            Must have columns matching ``REPORT_JOIN_COLS`` and ``REPORT_PATH_COL``.
            Leave ``None`` to skip report loading.
        report_base_dir: Root directory prepended to relative report file paths.
            Leave ``None`` when paths in the CSV are already absolute.
    """

    LABEL_COLS: list = []
    LABEL_JOIN_COLS: list = []

    #: Regression target columns. Optional. Only regression datasets populate
    #: this, for example RSNA Bone Age, where ``age_months`` is a continuous
    #: target. It is separate from ``LABEL_COLS``, so downstream code can tell
    #: one-hot labels from continuous targets without a check of the dtypes.
    REG_COLS: list = []

    #: Dataset-specific metadata columns to keep in
    #: :meth:`_select_harmonized_columns`, in addition to labels, reg and
    #: ``_OPTIONAL_COLS``. Use it for covariates that are not targets (for
    #: example sex, age or device manufacturer), so users do not need the raw CSV.
    EXTRA_OUTPUT_COLS: list = []

    #: Source column that holds the age of this dataset, and the unit of that column.
    #: The source units are different. For example, TAIX gives days, ReXGradient gives
    #: DICOM AS strings, and RSNA Bone Age gives months. When both are set,
    #: :meth:`_build_age_years` emits an ``age_years`` float column. The unit
    #: is DECLARED, never guessed from the size of the values. A guess would
    #: give wrong ages for pediatric cohorts, where they matter most.
    AGE_SOURCE_COL: str | None = None
    AGE_SOURCE_UNIT: str | None = None

    #: Source column that holds the sex of the patient. :meth:`_build_sex`
    #: maps its values to the ``sex`` column: "M", "F" or NaN (D-26).
    SEX_SOURCE_COL: str | None = None

    #: Source column to copy into ``series_id`` (cast to str). Leave ``None``
    #: for datasets with no series (most CXR datasets). Then the harmonized output
    #: has no ``series_id`` column. Multi-series datasets (DICOM stacks, for
    #: example RSNA 2024 Lumbar Spine) set this. Then callers can join coords or
    #: QC data, and group splits, without a parse of ``image_path``.
    SERIES_ID_SOURCE_COL: str = None

    VIEW_POSITION_JOIN_COLS: list = []
    #: Map ``view_position`` to the shared 2-D names (Q-13). A 3-D class, whose
    #: ``view_position`` is a plane or series name, sets ``False``.
    VIEW_POSITION_NAMES: bool = True
    VIEW_POSITION_SOURCE_COL: str = None

    MASK_JOIN_COLS: list = []
    MASK_SOURCE_COL: str = None

    BBOX_JOIN_COLS: list = []
    BBOX_SOURCE_COL: str = None

    REPORT_JOIN_COLS: list = []
    REPORT_PATH_COL: str = None  # column in report CSV containing the file path

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # A subclass harmonize() can skip the base one. The wrapper then applies
        # the ids and makes the image copies (V-H2-27).
        if "harmonize" in cls.__dict__:
            cls.harmonize = _wrap_harmonize(cls.__dict__["harmonize"])

    def __init__(
        self,
        csv_path: str,
        label_csv_path: str = None,
        view_position_csv_path: str = None,
        mask_csv_path: str = None,
        bbox_csv_path: str = None,
        report_csv_path: str = None,
        report_base_dir: str = None,
    ):
        self.csv_path = csv_path
        self.label_csv_path = label_csv_path
        self.view_position_csv_path = view_position_csv_path
        self.mask_csv_path = mask_csv_path
        self.bbox_csv_path = bbox_csv_path
        self.report_csv_path = report_csv_path
        self.report_base_dir = report_base_dir
        self.df: pd.DataFrame = None
        self._harmonized_df_override: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Harmonized DataFrame getter / setter
    # ------------------------------------------------------------------

    def _label_columns_for_output(self, df: pd.DataFrame) -> list[str]:
        """Columns to treat as labels when building the public harmonized slice."""
        return [
            c.lower().replace(" ", "_")
            for c in self.LABEL_COLS
            if c.lower().replace(" ", "_") in df.columns
        ]

    def _reg_columns_for_output(self, df: pd.DataFrame) -> list[str]:
        """Columns to treat as regression targets when building the harmonized slice."""
        return [
            c.lower().replace(" ", "_")
            for c in self.REG_COLS
            if c.lower().replace(" ", "_") in df.columns
        ]

    def _build_age_years(self, df: pd.DataFrame) -> pd.DataFrame:
        """Populate ``age_years`` (float) from the declared source column/unit.

        No-op when the subclass has not declared both, so datasets without age
        are unaffected.
        """
        col, unit = self.AGE_SOURCE_COL, self.AGE_SOURCE_UNIT
        if not col or not unit or col not in df.columns:
            return df
        df = df.copy()
        df["age_years"] = to_years(df[col], unit)
        return df

    def _build_sex(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make the ``sex`` column ("M", "F" or NaN) from ``SEX_SOURCE_COL`` (D-26).

        No-op when the subclass does not declare a source column. The values
        for unknown or other (O, U, Unknown, empty) give NaN. Any other value
        that is not in the shared table gives NaN and a warning.
        """
        col = self.SEX_SOURCE_COL
        if not col or col not in df.columns:
            return df
        raw = df[col].map(lambda v: v.strip().lower() if isinstance(v, str) else v)
        sex = raw.map(_SEX_VALUES)
        odd = raw[raw.notna() & sex.isna() & ~raw.isin(_SEX_UNKNOWN)]
        if len(odd):
            warnings.warn(
                f"{type(self).__name__}: {len(odd)} rows have a {col!r} value that is not a "
                f"known sex, for example {sorted(map(str, odd.unique()))[:5]}. They get sex NaN.",
                UserWarning,
                stacklevel=2,
            )
        df = df.copy()
        df["sex"] = sex
        return df

    def _build_view_position_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Give ``view_position`` the shared 2-D names PA, AP and LATERAL (Q-13).

        Unknown words (blank, Unknown, UNK, N/A, EXCLUDE, None) give NaN. Any other value keeps
        its raw value in upper case, for example ``AP_HORIZONTAL``. No-op when the class
        sets ``VIEW_POSITION_NAMES = False`` or the table has no ``view_position``.
        """
        if not self.VIEW_POSITION_NAMES or "view_position" not in df.columns:
            return df
        raw = df["view_position"].map(lambda v: v.strip() if isinstance(v, str) else v)
        low = raw.map(lambda v: v.lower() if isinstance(v, str) else v)
        shared = low.map(_VIEW_VALUES)
        names = shared.where(shared.notna(), raw.map(lambda v: v.upper() if isinstance(v, str) else v))
        df = df.copy()
        df["view_position"] = names.where(~low.isin(_VIEW_UNKNOWN), None)
        return df

    def _select_harmonized_columns(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return the standard harmonized column subset (copy)."""
        df = self.df if df is None else df
        if df is None:
            raise ValueError("No DataFrame loaded; call harmonize() first.")
        df = self._build_age_years(df)
        df = self._build_sex(df)
        df = self._build_view_position_names(df)
        label_cols = self._label_columns_for_output(df)
        reg_cols = self._reg_columns_for_output(df)
        sex_col = ["sex"] if self.SEX_SOURCE_COL else []
        optional = [
            c for c in _OPTIONAL_COLS + ["age_years"] + sex_col + label_cols + reg_cols
            + list(self.EXTRA_OUTPUT_COLS)
            if c in df.columns
        ]
        return df[["patient_id", "study_id", "image_path"] + optional].copy()

    @property
    def harmonized_df(self) -> pd.DataFrame:
        """Harmonized table: explicit override, or derived from :attr:`df` after harmonize."""
        if self._harmonized_df_override is not None:
            return self._harmonized_df_override.copy()
        if self.df is not None:
            return self._select_harmonized_columns(self.df)
        raise ValueError(
            "No harmonized DataFrame available. Call harmonize(), assign "
            "harmonized_df, or use load_from_saved()."
        )

    @harmonized_df.setter
    def harmonized_df(self, value: pd.DataFrame | None) -> None:
        if value is None:
            self._harmonized_df_override = None
        else:
            self._harmonized_df_override = value.copy()

    def set_harmonized_df(self, df: pd.DataFrame | None) -> None:
        """Set or clear the harmonized DataFrame without running CSV merge logic."""
        self.harmonized_df = df

    def get_label_cols(self) -> list[str]:
        """Effective label column names (instance override or class default)."""
        if "LABEL_COLS" in self.__dict__:
            return list(self.__dict__["LABEL_COLS"])
        return list(type(self).LABEL_COLS or [])

    def _harmonizer_init_snapshot(self) -> dict:
        """Keyword arguments needed to reconstruct this harmonizer (for save/load)."""
        snap = {"csv_path": self.csv_path}
        for name in (
            "label_csv_path",
            "view_position_csv_path",
            "mask_csv_path",
            "bbox_csv_path",
            "report_csv_path",
            "report_base_dir",
            "dicom_base_dir",
            "base_image_dir",
        ):
            v = getattr(self, name, None)
            if v is not None:
                snap[name] = v
        return snap

    def save(self, path: str) -> None:
        """Pickle the harmonized DataFrame, label columns, and init snapshot to *path*."""
        df = self.harmonized_df
        label_cols = list(self.get_label_cols())
        payload = {
            "version": _SAVE_VERSION,
            "harmonizer_module": self.__class__.__module__,
            "harmonizer_class": self.__class__.__qualname__,
            "harmonized_df": df,
            "label_cols": label_cols,
            "init": self._harmonizer_init_snapshot(),
        }
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=4)

    @classmethod
    def load_from_saved(cls, path: str, **init_overrides):
        """Load a harmonizer written by :meth:`save`.

        Any ``init_overrides`` are merged on top of the stored init dict (for example,
        new paths after you move the data).

        Raises:
            ValueError: If the file was saved by a different harmonizer class than *cls*.
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)
        expected = f"{cls.__module__}.{cls.__qualname__}"
        got = f"{payload['harmonizer_module']}.{payload['harmonizer_class']}"
        if got != expected:
            raise ValueError(
                f"Saved harmonizer is {got}, but loading with {expected}. "
                "Instantiate the matching class, e.g. "
                f"{payload['harmonizer_module']}.{payload['harmonizer_class']}"
                ".load_from_saved(path)."
            )
        merged = {**payload["init"], **init_overrides}
        inst = cls(**merged)
        df = payload["harmonized_df"]
        label_cols = list(payload["label_cols"])
        inst.LABEL_COLS = sorted(label_cols, key=lambda c: c.lower().replace(" ", "_"))
        inst.set_harmonized_df(df)
        return inst

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _build_patient_id(self) -> None:
        """Populate ``self.df["patient_id"]``."""
        raise NotImplementedError

    def _build_study_id(self) -> None:
        """Populate ``self.df["study_id"]``."""
        raise NotImplementedError

    def _build_image_path(self) -> None:
        """Set ``self.df["image_path"]`` to each image path, relative to ``base_image_dir`` (D-04)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional hooks (no-op by default)
    # ------------------------------------------------------------------

    def _read_source(self) -> pd.DataFrame:
        """Read the source files into one table. Default: the CSV at ``csv_path`` (D-30)."""
        return pd.read_csv(self.csv_path)

    @staticmethod
    def _check_one_line_per_key(src: pd.DataFrame, on: list, what: str, hook: str) -> None:
        """Raise ``ValueError`` if ``src`` has more than one line for a key.

        A join would then give more than one row for an image (D-03, V-H2-17).
        """
        repeated = src[src.duplicated(on, keep=False)]
        if len(repeated):
            key = repeated.iloc[0][on].to_dict()
            raise ValueError(
                f"The {what} file has more than one line for {key}. "
                f"Join them into one line per image in {hook} first (D-03)."
            )

    def _merge_one_line_per_key(self, src: pd.DataFrame, on: list, what: str, hook: str) -> pd.DataFrame:
        """Join ``src`` to ``self.df`` on ``on``, after :meth:`_check_one_line_per_key`."""
        self._check_one_line_per_key(src, on, what, hook)
        return self.df.merge(src, on=on)

    def _build_labels(self) -> None:
        """Merge label CSV (if provided) and rename LABEL_COLS to snake_case."""
        if self.label_csv_path is not None:
            label_df = pd.read_csv(self.label_csv_path)
            label_df = self._preprocess_label_df(label_df)
            self.df = self._merge_one_line_per_key(
                label_df, self.LABEL_JOIN_COLS, "label", "_preprocess_label_df"
            )

        for col in self.LABEL_COLS:
            if col in self.df.columns:
                self.df.rename(
                    columns={col: col.lower().replace(" ", "_")}, inplace=True
                )

    def _preprocess_label_df(self, label_df: pd.DataFrame) -> pd.DataFrame:
        """Hook to clean the label CSV before it is merged into ``self.df``.

        Override in subclasses to apply dataset-specific changes (for
        example, fill NaNs or drop uncertain labels). The default
        implementation returns the DataFrame unchanged.
        """
        return label_df

    def _build_series_id(self) -> None:
        """Set ``series_id`` in ``self.df`` from ``SERIES_ID_SOURCE_COL`` (cast to str).

        Does nothing when the source column is not set or not in the table.
        Datasets with no series (most CXR sets) keep ``SERIES_ID_SOURCE_COL = None``,
        so their harmonized output has no ``series_id`` column.
        """
        src = self.SERIES_ID_SOURCE_COL
        if src and src in self.df.columns:
            self.df["series_id"] = self.df[src].astype(str)
            if src != "series_id":
                self.df.drop(columns=[src], inplace=True)

    def _build_view_position(self) -> None:
        """Merge view-position CSV (if provided) and rename to ``view_position``."""
        if self.view_position_csv_path is not None:
            vp_df = pd.read_csv(self.view_position_csv_path)
            vp_df = self._preprocess_view_position_df(vp_df)
            self.df = self._merge_one_line_per_key(
                vp_df[self.VIEW_POSITION_JOIN_COLS + [self.VIEW_POSITION_SOURCE_COL]],
                self.VIEW_POSITION_JOIN_COLS, "view-position", "_preprocess_view_position_df",
            )
        if (
            self.VIEW_POSITION_SOURCE_COL
            and self.VIEW_POSITION_SOURCE_COL in self.df.columns
        ):
            self.df.rename(
                columns={self.VIEW_POSITION_SOURCE_COL: "view_position"}, inplace=True
            )

    def _preprocess_view_position_df(self, vp_df: pd.DataFrame) -> pd.DataFrame:
        """Hook to clean the view-position CSV before it is merged."""
        return vp_df

    def _build_mask_path(self) -> None:
        """Merge mask CSV (if provided) and rename to ``mask_path``."""
        if self.mask_csv_path is not None:
            mask_df = pd.read_csv(self.mask_csv_path)
            mask_df = self._preprocess_mask_df(mask_df)
            self.df = self._merge_one_line_per_key(
                mask_df[self.MASK_JOIN_COLS + [self.MASK_SOURCE_COL]],
                self.MASK_JOIN_COLS, "mask", "_preprocess_mask_df",
            )
        if self.MASK_SOURCE_COL and self.MASK_SOURCE_COL in self.df.columns:
            self.df.rename(columns={self.MASK_SOURCE_COL: "mask_path"}, inplace=True)

    def _preprocess_mask_df(self, mask_df: pd.DataFrame) -> pd.DataFrame:
        """Hook to clean the mask CSV before it is merged."""
        return mask_df

    def _build_bbox(self) -> None:
        """Merge bbox CSV (if provided) and rename to ``bbox``."""
        if self.bbox_csv_path is not None:
            bbox_df = pd.read_csv(self.bbox_csv_path)
            bbox_df = self._preprocess_bbox_df(bbox_df)
            self.df = self._merge_one_line_per_key(
                bbox_df[self.BBOX_JOIN_COLS + [self.BBOX_SOURCE_COL]],
                self.BBOX_JOIN_COLS, "box", "_preprocess_bbox_df",
            )
        if self.BBOX_SOURCE_COL and self.BBOX_SOURCE_COL in self.df.columns:
            self.df.rename(columns={self.BBOX_SOURCE_COL: "bbox"}, inplace=True)

    def _preprocess_bbox_df(self, bbox_df: pd.DataFrame) -> pd.DataFrame:
        """Hook to clean the bbox CSV before it is merged."""
        return bbox_df

    def _build_report(self) -> None:
        """Load report text from disk and store in ``self.df["report"]``.

        Reads ``report_csv_path``, a CSV with join keys and the path column
        ``REPORT_PATH_COL``. Then reads the text file at each path. Then
        left-merges the result into ``self.df`` on ``REPORT_JOIN_COLS``.

        ``report_base_dir`` is prepended to relative paths.  Files that are
        not found are stored as ``None``. Raise ``ValueError`` if the report
        CSV has more than one line for a key, as the other merges do (D-03).
        """
        if (
            self.report_csv_path is None
            or not self.REPORT_PATH_COL
            or not self.REPORT_JOIN_COLS
        ):
            return

        report_df = pd.read_csv(self.report_csv_path, dtype=str)
        report_df = self._preprocess_report_df(report_df)
        self._check_one_line_per_key(report_df, self.REPORT_JOIN_COLS, "report", "_preprocess_report_df")
        # Only open report files for studies still in self.df (the merge below is a
        # left merge, so other rows would be dropped anyway).
        keys = self.df[self.REPORT_JOIN_COLS].drop_duplicates()
        report_df = report_df.merge(keys, on=self.REPORT_JOIN_COLS, how="inner")

        def _read(path: str) -> str | None:
            fp = (
                os.path.join(self.report_base_dir, path)
                if self.report_base_dir
                else path
            )
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    return f.read()
            except FileNotFoundError:
                return None

        report_df["report"] = report_df[self.REPORT_PATH_COL].apply(_read)
        report_df = report_df[self.REPORT_JOIN_COLS + ["report"]]
        self.df = self.df.merge(report_df, on=self.REPORT_JOIN_COLS, how="left")

    def _preprocess_report_df(self, report_df: pd.DataFrame) -> pd.DataFrame:
        """Hook to clean the report CSV before report files are read."""
        return report_df

    # ------------------------------------------------------------------
    # Mask preprocessing
    # ------------------------------------------------------------------

    def _decode_mask(self, rle: str, width: int, height: int):  # noqa: ARG002
        """Decode a mask string to a (H, W) uint8 numpy array.

        Override in subclasses that store masks as encoded strings (for example RLE).
        Return ``None`` to skip this entry. The default implementation always
        returns ``None`` (no mask encoding defined).
        """
        return None  # A hook for subclasses. The base does not use the parameters.

    def preprocess_masks(
        self,
        output_dir: str,
        base_image_dir: str = None,
        num_cores: int = 1,
    ) -> None:
        """Decode ``mask_path`` strings and save combined masks as PNG files.

        One PNG is written per unique ``image_path``.  Multiple rows sharing
        the same image (multiple annotators) are unioned before saving.
        An image whose ``mask_path`` values are all empty (NaN or ``""``) was not
        annotated. It gets no PNG, and its ``mask_path`` stays empty (D-33).

        The PNG path copies the relative path of the image, for example
        ``<output_dir>/subject/study/image_id.png``.

        Args:
            output_dir: Root directory to write PNG masks into.
            base_image_dir: The folder of the source images, read for their
                size. Default: ``self.base_image_dir``, if the harmonizer has it (D-04).
            num_cores: Number of parallel worker threads.
        """
        import numpy as np  # type: ignore[import-untyped]
        import SimpleITK as sitk  # type: ignore[import-untyped]
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from PIL import Image
        from tqdm import tqdm

        if self.df is None:
            raise RuntimeError("Call harmonize() before preprocess_masks().")
        if "mask_path" not in self.df.columns:
            raise ValueError("No mask_path column found. Call harmonize() first.")

        base_image_dir = base_image_dir or getattr(self, "base_image_dir", None)
        groups = self.df.groupby("image_path")["mask_path"].apply(list).reset_index()

        def _save(row) -> tuple[str, str] | None:
            """Returns (image_path, out_path) on success, None if no mask."""
            if all(pd.api.types.is_scalar(m) and (pd.isna(m) or m == "") for m in row["mask_path"]):
                return None  # nobody annotated this image (D-33)
            # Strip a leading "/": os.path.join drops output_dir before an absolute path (V-H2-21).
            rel = str(row["image_path"]).lstrip("/\\")
            out_path = os.path.join(output_dir, os.path.splitext(rel)[0] + ".png")
            if os.path.isfile(out_path):
                return row["image_path"], out_path  # already processed

            full_path = (
                os.path.join(base_image_dir, row["image_path"])
                if base_image_dir
                else row["image_path"]
            )
            arr = sitk.GetArrayFromImage(sitk.ReadImage(full_path))
            h, w = arr.shape[-2], arr.shape[-1]

            mask = np.zeros((h, w), dtype=np.uint8)
            for rle in row["mask_path"]:
                decoded = self._decode_mask(rle, width=w, height=h)
                if decoded is not None:
                    mask |= decoded

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            Image.fromarray(mask * 255).save(out_path)
            return row["image_path"], out_path

        records = groups.to_dict("records")
        if num_cores == 1:
            results = [_save(rec) for rec in tqdm(records, desc="Preprocessing masks")]
        else:
            with ThreadPoolExecutor(max_workers=num_cores) as ex:
                futures = {ex.submit(_save, rec): rec for rec in records}
                results = [
                    f.result()
                    for f in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc="Preprocessing masks",
                    )
                ]

        # Set mask_path to the saved PNG paths. Keep the old value where no mask was written.
        path_map = dict(r for r in results if r is not None)
        self.df["mask_path"] = (
            self.df["image_path"].map(path_map).fillna(self.df["mask_path"])
        )

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    def preprocess(
        self,
        output_dir: str,
        preprocessor,
        base_image_dir: str = None,
        num_workers: int = 32,
        sample_n: int = None,
    ) -> None:
        """Run the preprocessor over all images and write results to ``output_dir``.

        Rewrites ``image_path`` in ``self.df`` to the path of each output file
        (see ``output_path`` in ``radharmony/preprocessor/base.py``). After this
        step, ``image_path`` is the absolute path of the copy when ``output_dir``
        is absolute (an exception to D-04). It adds no new column. Requires
        ``self.df`` to be populated (call :meth:`harmonize` first).

        Args:
            output_dir: Directory where preprocessed files are written.
            preprocessor: An instance of :class:`~src.preprocessor.RadiologicPreprocessor`.
            base_image_dir: The folder of the source images. Default:
                ``self.base_image_dir``, if the harmonizer has it (D-04).
            num_workers: Number of parallel worker processes (default: 32).
            sample_n: If set, process only a random sample of this size.
        """
        if self.df is None:
            raise RuntimeError("Call harmonize() before preprocess().")
        # A wrapped harmonize() must not run preprocess() a second time (V-H2-27).
        if getattr(self, "_id_filter", None):
            self._id_filter["preprocessed"] = True

        df_to_process = self.df.sample(n=sample_n) if sample_n else self.df
        base_image_dir = base_image_dir or getattr(self, "base_image_dir", None)

        preprocessor.run_batch(
            df=df_to_process,
            base_output_dir=output_dir,
            base_image_dir=base_image_dir,
            num_workers=num_workers,
        )

        self.df.loc[df_to_process.index, "image_path"] = df_to_process[
            "image_path"
        ].apply(lambda p: _preprocessed_path(p, output_dir))

        if sample_n:
            self.df = self.df.loc[df_to_process.index]

    # ------------------------------------------------------------------
    # Verify image existence
    # ------------------------------------------------------------------

    def verify_images(
        self,
        base_image_dir: str = None,
        drop_missing: bool = True,
    ) -> pd.DataFrame:
        """Check that every ``image_path`` in the harmonized DataFrame exists on disk.

        Args:
            base_image_dir: The folder that ``image_path`` is relative to (D-04).
                Default: ``self.base_image_dir``, if the harmonizer has it.
            drop_missing: If ``True`` (default), rows with missing files are
                removed from the harmonized DataFrame.  If ``False``, the
                DataFrame is left unchanged and only the missing-files table
                is returned.

        Returns:
            DataFrame of rows whose image file was not found (empty if all
            files exist).

        Raises:
            ValueError: If ``image_path`` is relative and no ``base_image_dir``
                is known. Without it, every row looks missing (V-H2-23).
        """
        df = self.harmonized_df
        base_image_dir = base_image_dir or getattr(self, "base_image_dir", None)
        if not base_image_dir and not df["image_path"].astype(str).map(os.path.isabs).all():
            raise ValueError(
                "image_path is relative to base_image_dir (D-04). Pass "
                "verify_images(base_image_dir=...), or every row looks missing."
            )

        def _exists(p):
            fp = os.path.join(base_image_dir, p) if base_image_dir else p
            return os.path.isfile(fp)

        from tqdm import tqdm
        tqdm.pandas(desc="Verifying images")
        mask = df["image_path"].progress_apply(_exists)
        missing_df = df[~mask].copy()

        if not missing_df.empty and drop_missing:
            kept = df[mask].reset_index(drop=True)
            self.set_harmonized_df(kept)
            print(
                f"Dropped {len(missing_df)} rows with missing images "
                f"({len(kept)} remaining)."
            )
        elif missing_df.empty:
            print("All image files exist.")
        else:
            print(f"Found {len(missing_df)} rows with missing images (not dropped).")

        return missing_df

    # ------------------------------------------------------------------
    # Harmonize
    # ------------------------------------------------------------------

    def harmonize(
        self,
        preprocess_output_dir: str = None,
        preprocessor=None,
        base_image_dir: str = None,
        num_workers: int = 1,
        sample_n: int = None,
        mask_output_dir: str = None,
        mask_num_cores: int = 1,
        study_ids=None,
        patient_ids=None,
    ) -> pd.DataFrame:
        """Load the CSV, build all standardised columns, and optionally preprocess.

        If ``preprocess_output_dir`` and ``preprocessor`` are given, images are
        preprocessed and ``image_path`` is updated to point to the output files
        before the DataFrame is returned. A subclass ``harmonize()`` that never
        reaches this one gets the same step from ``_wrap_harmonize`` (V-H2-27).

        If ``mask_output_dir`` is given, RLE masks are decoded and saved as PNG
        files, and ``mask_path`` is updated to point to the saved PNGs.

        Args:
            preprocess_output_dir: Directory to write preprocessed image files to.
                Skip image preprocessing if ``None``.
            preprocessor: An instance of :class:`~src.preprocessor.RadiologicPreprocessor`.
                Required when ``preprocess_output_dir`` is set.
            base_image_dir: Passed to :meth:`preprocess` and :meth:`preprocess_masks`.
                Both use ``self.base_image_dir`` when it is ``None``.
            num_workers: Passed to :meth:`preprocess` (default: 1).
            sample_n: Passed to :meth:`preprocess` (default: None). Without
                ``preprocess_output_dir``, it does nothing and gives a warning
                (D-30, V-H2-27).
            mask_output_dir: Directory to write PNG mask files to.
                Skip mask preprocessing if ``None``.
            mask_num_cores: Passed to :meth:`preprocess_masks` (default: 1).
            study_ids: Keep only rows whose ``study_id`` is in this list. The filter
                runs right after the IDs are built, so the labels, reports and
                other columns are built only for these rows. Not the recommended
                way: normally harmonize everything, then filter ``harmonized_df``.
                Use it only to save time when you need a few studies from a large
                dataset (for example MIMIC reports). Harmonizers whose own ``harmonize()`` never reaches this
                one get their result filtered afterwards, with a warning (no speedup).
            patient_ids: Same as ``study_ids``, for ``patient_id``.

        Returns:
            DataFrame with at minimum ``patient_id``, ``study_id``, ``image_path``
            plus any optional columns that were populated.
        """
        self._harmonized_df_override = None
        self.df = self._read_source()

        self._build_patient_id()
        self._build_study_id()
        # Optional early filter (see ``study_ids`` / ``patient_ids`` above). A wrapped
        # subclass harmonize() hands the ids over via self._id_filter.
        pending = getattr(self, "_id_filter", None)
        if pending:
            pending["applied"] = True
            # A call for image copies only has no ids. Then keep the ids that a subclass gives here.
            if pending["study_ids"] is not None or pending["patient_ids"] is not None:
                study_ids, patient_ids = pending["study_ids"], pending["patient_ids"]
        self.df = _filter_by_ids(self.df, study_ids, patient_ids)
        self._build_series_id()
        self._build_image_path()
        self.LABEL_COLS = sorted(self.LABEL_COLS, key=lambda c: c.lower().replace(" ", "_"))
        self._build_labels()
        self._build_view_position()
        self._build_mask_path()
        self._build_bbox()
        self._build_report()

        # Drop rows with no image path (for example, the glob or merge found no
        # DICOM). A NaN path would go into the data dicts and crash the MONAI
        # transform pipeline.
        self.df.dropna(subset=["image_path"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Masks first: preprocess() points image_path to the resized copies (V-H2-21).
        if mask_output_dir is not None:
            self.preprocess_masks(
                output_dir=mask_output_dir,
                base_image_dir=base_image_dir,
                num_cores=mask_num_cores,
            )

        if preprocess_output_dir is not None:
            self.preprocess(
                output_dir=preprocess_output_dir,
                preprocessor=preprocessor,
                base_image_dir=base_image_dir,
                num_workers=num_workers,
                sample_n=sample_n,
            )
        elif sample_n is not None:
            warnings.warn(
                f"{type(self).__name__}.harmonize() ignores sample_n without preprocess_output_dir. "
                "It keeps all rows.",
                UserWarning,
                stacklevel=2,
            )

        return self._select_harmonized_columns(self.df)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _preprocessed_path(image_path: str, output_dir: str) -> str:
    """Return the file that ``RadiologicPreprocessor.run_batch`` writes for ``image_path``."""
    from radharmony.preprocessor.base import output_path

    return output_path(image_path, output_dir)
