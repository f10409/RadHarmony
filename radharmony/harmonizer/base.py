"""Base harmonizer for radiological datasets."""

import functools
import os
import pickle
import warnings

import pandas as pd


_OPTIONAL_COLS = ["series_id", "view_position", "mask_path", "bbox", "bbox_labels", "report"]

_HARMONIZED_CORE_COLS = frozenset(
    {"patient_id", "study_id", "series_id", "image_path", "view_position", "mask_path", "bbox", "report"}
)

_SAVE_VERSION = 1


def _filter_by_ids(df: pd.DataFrame, study_ids=None, patient_ids=None) -> pd.DataFrame:
    """Keep rows whose ``study_id`` / ``patient_id`` is in the given list (``None`` = no filter)."""
    for col, ids in (("study_id", study_ids), ("patient_id", patient_ids)):
        if ids is not None:
            df = df[df[col].astype(str).isin({str(i) for i in ids})].reset_index(drop=True)
    return df


def _wrap_harmonize(harmonize):
    """Wrap a subclass ``harmonize()`` so ``study_ids`` / ``patient_ids`` are never silently ignored.

    The ids are taken out of the call and handed to the base ``harmonize()`` via
    ``self._id_filter``; it filters early. If the subclass never reaches the base
    ``harmonize()``, the result is filtered afterwards with a warning (no speedup).
    """

    @functools.wraps(harmonize)
    def wrapper(self, *args, study_ids=None, patient_ids=None, **kwargs):
        # No filter, or an inner call in a harmonize() chain: run as is.
        if (study_ids is None and patient_ids is None) or getattr(self, "_id_filter", None):
            return harmonize(self, *args, **kwargs)
        self._id_filter = {"study_ids": study_ids, "patient_ids": patient_ids, "applied": False}
        try:
            df = harmonize(self, *args, **kwargs)
            applied = self._id_filter["applied"]
        finally:
            self._id_filter = None
        if applied:
            return df
        warnings.warn(
            f"{type(self).__name__} has its own harmonize(), so study_ids / patient_ids "
            "were applied after harmonizing (no speedup).",
            stacklevel=2,
        )
        if self.df is not None and {"study_id", "patient_id"} <= set(self.df.columns):
            self.df = _filter_by_ids(self.df, study_ids, patient_ids)
        if getattr(self, "_harmonized_df_override", None) is not None:
            self._harmonized_df_override = _filter_by_ids(
                self._harmonized_df_override, study_ids, patient_ids
            )
        return _filter_by_ids(df, study_ids, patient_ids) if isinstance(df, pd.DataFrame) else df

    return wrapper


class BaseHarmonizer:
    """Abstract base class for dataset harmonizers.

    Subclasses must implement :meth:`_build_patient_id`, :meth:`_build_study_id`,
    and :meth:`_build_image_path`.  Optional hooks (:meth:`_build_labels`,
    :meth:`_build_view_position`, :meth:`_build_mask_path`, :meth:`_build_bbox`)
    default to no-ops and can be overridden as needed.

    The final harmonized DataFrame always contains ``patient_id``, ``study_id``,
    and ``image_path`` (absolute).  Any optional columns that were populated are
    appended in the order defined by ``_OPTIONAL_COLS`` followed by
    ``LABEL_COLS`` (snake_cased).

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

    #: Regression target columns.  Optional — only regression datasets (e.g.
    #: RSNA Bone Age, where ``age_months`` is a continuous target) populate
    #: this.  Kept separate from ``LABEL_COLS`` so downstream code can tell
    #: one-hot labels apart from continuous targets without inspecting dtypes.
    REG_COLS: list = []

    #: Dataset-specific metadata columns to carry through
    #: :meth:`_select_harmonized_columns` in addition to labels + reg +
    #: ``_OPTIONAL_COLS``.  Useful for non-target covariates (sex, age,
    #: device manufacturer, etc.) that downstream users may want without
    #: reaching for the raw CSV.
    EXTRA_OUTPUT_COLS: list = []

    #: Source column to copy into ``series_id`` (cast to str).  Leave ``None``
    #: for datasets without a series concept (most CXR datasets) — the column
    #: simply won't appear in the harmonized output.  Multi-series datasets
    #: (DICOM stacks, e.g. RSNA 2024 Lumbar Spine) populate this so callers
    #: can join in coords/QC and group splits without parsing ``image_path``.
    SERIES_ID_SOURCE_COL: str = None

    VIEW_POSITION_JOIN_COLS: list = []
    VIEW_POSITION_SOURCE_COL: str = None

    MASK_JOIN_COLS: list = []
    MASK_SOURCE_COL: str = None

    BBOX_JOIN_COLS: list = []
    BBOX_SOURCE_COL: str = None

    REPORT_JOIN_COLS: list = []
    REPORT_PATH_COL: str = None  # column in report CSV containing the file path

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # A subclass harmonize() may never call the base one; wrap it so
        # study_ids / patient_ids are not silently ignored.
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

    def _select_harmonized_columns(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return the standard harmonized column subset (copy)."""
        df = self.df if df is None else df
        if df is None:
            raise ValueError("No DataFrame loaded; call harmonize() first.")
        label_cols = self._label_columns_for_output(df)
        reg_cols = self._reg_columns_for_output(df)
        optional = [
            c for c in _OPTIONAL_COLS + label_cols + reg_cols + list(self.EXTRA_OUTPUT_COLS)
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
            "csv_path"
        ):
            v = getattr(self, name, None)
            if v is not None:
                snap[name] = v
        return snap

    def save(self, path: str) -> None:
        """Pickle the harmonized DataFrame, label columns, and init snapshot to *path*."""
        df = self.harmonized_df
        label_cols = self.get_label_cols()
        if not label_cols:
            label_cols = [c for c in df.columns if c not in _HARMONIZED_CORE_COLS]
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

        Any ``init_overrides`` are merged on top of the stored init dict (e.g. updated
        paths after moving data).

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
        if not label_cols:
            label_cols = [c for c in df.columns if c not in _HARMONIZED_CORE_COLS]
        inst.LABEL_COLS = sorted(label_cols)
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
        """Populate ``self.df["image_path"]`` with absolute file paths."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional hooks (no-op by default)
    # ------------------------------------------------------------------

    def _build_labels(self) -> None:
        """Merge label CSV (if provided) and rename LABEL_COLS to snake_case."""
        if self.label_csv_path is not None:
            label_df = pd.read_csv(self.label_csv_path)
            label_df = self._preprocess_label_df(label_df)
            self.df = self.df.merge(label_df, on=self.LABEL_JOIN_COLS)

        for col in self.LABEL_COLS:
            if col in self.df.columns:
                self.df.rename(
                    columns={col: col.lower().replace(" ", "_")}, inplace=True
                )

    def _preprocess_label_df(self, label_df: pd.DataFrame) -> pd.DataFrame:
        """Hook to clean the label CSV before it is merged into ``self.df``.

        Override in subclasses to apply dataset-specific transformations
        (e.g. filling NaNs, dropping uncertain labels).  The default
        implementation returns the DataFrame unchanged.
        """
        return label_df

    def _build_series_id(self) -> None:
        """Stamp ``series_id`` onto ``self.df`` from ``SERIES_ID_SOURCE_COL`` (cast to str).

        No-op when the source column is unset or absent — datasets without a
        series concept (most CXR sets) leave ``SERIES_ID_SOURCE_COL = None``
        and simply don't get the column in their harmonized output.
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
            self.df = self.df.merge(
                vp_df[self.VIEW_POSITION_JOIN_COLS + [self.VIEW_POSITION_SOURCE_COL]],
                on=self.VIEW_POSITION_JOIN_COLS,
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
            self.df = self.df.merge(
                mask_df[self.MASK_JOIN_COLS + [self.MASK_SOURCE_COL]],
                on=self.MASK_JOIN_COLS,
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
            self.df = self.df.merge(
                bbox_df[self.BBOX_JOIN_COLS + [self.BBOX_SOURCE_COL]],
                on=self.BBOX_JOIN_COLS,
            )
        if self.BBOX_SOURCE_COL and self.BBOX_SOURCE_COL in self.df.columns:
            self.df.rename(columns={self.BBOX_SOURCE_COL: "bbox"}, inplace=True)

    def _preprocess_bbox_df(self, bbox_df: pd.DataFrame) -> pd.DataFrame:
        """Hook to clean the bbox CSV before it is merged."""
        return bbox_df

    def _build_report(self) -> None:
        """Load report text from disk and store in ``self.df["report"]``.

        Reads ``report_csv_path`` (a CSV with join keys and a path column
        defined by ``REPORT_PATH_COL``), reads the text file at each path,
        and left-merges the result into ``self.df`` on ``REPORT_JOIN_COLS``.

        ``report_base_dir`` is prepended to relative paths.  Files that are
        not found are stored as ``None``.
        """
        if (
            self.report_csv_path is None
            or not self.REPORT_PATH_COL
            or not self.REPORT_JOIN_COLS
        ):
            return

        report_df = pd.read_csv(self.report_csv_path, dtype=str)
        report_df = self._preprocess_report_df(report_df)
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

        Override in subclasses that store masks as encoded strings (e.g. RLE).
        Return ``None`` to skip this entry.  The default implementation always
        returns ``None`` (no mask encoding defined).
        """
        return None  # params intentionally unused — hook for subclasses

    def preprocess_masks(
        self,
        output_dir: str,
        base_image_dir: str = None,
        num_cores: int = 1,
    ) -> None:
        """Decode ``mask_path`` strings and save combined masks as PNG files.

        One PNG is written per unique ``image_path``.  Multiple rows sharing
        the same image (multiple annotators) are unioned before saving.
        Entries where :meth:`_decode_mask` returns ``None`` are skipped; images
        whose combined mask is all-zero are not written.

        The PNG mirrors the image's relative path, e.g.
        ``<output_dir>/subject/study/image_id.png``.

        Args:
            output_dir: Root directory to write PNG masks into.
            base_image_dir: Prepended to relative ``image_path`` values when
                reading images for their dimensions.
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

        groups = self.df.groupby("image_path")["mask_path"].apply(list).reset_index()

        def _save(row) -> tuple[str, str] | None:
            """Returns (image_path, out_path) on success, None if no mask."""
            out_path = os.path.join(
                output_dir, os.path.splitext(row["image_path"])[0] + ".png"
            )
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

        # Update mask_path to saved PNG paths; keep original value where no mask was written
        path_map = {img: out for img, out in results if img is not None}
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

        Adds a ``preprocessed_path`` column to ``self.df`` with the path of
        each output file.  Requires ``self.df`` to be populated (call
        :meth:`harmonize` first).

        Args:
            output_dir: Directory where preprocessed files are written.
            preprocessor: An instance of :class:`~src.preprocessor.RadiologicPreprocessor`.
            base_image_dir: Root directory prepended to relative image paths.
                Pass ``None`` when the CSV already contains absolute paths.
            num_workers: Number of parallel worker processes (default: 32).
            sample_n: If set, process only a random sample of this size.
        """
        if self.df is None:
            raise RuntimeError("Call harmonize() before preprocess().")

        df_to_process = self.df.sample(n=sample_n) if sample_n else self.df

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
            base_image_dir: Prepended to relative ``image_path`` values.
                Leave ``None`` when paths are already absolute.
            drop_missing: If ``True`` (default), rows with missing files are
                removed from the harmonized DataFrame.  If ``False``, the
                DataFrame is left unchanged and only the missing-files table
                is returned.

        Returns:
            DataFrame of rows whose image file was not found (empty if all
            files exist).
        """
        df = self.harmonized_df

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
        before the DataFrame is returned.

        If ``mask_output_dir`` is given, RLE masks are decoded and saved as PNG
        files, and ``mask_path`` is updated to point to the saved PNGs.

        Args:
            preprocess_output_dir: Directory to write preprocessed image files to.
                Skip image preprocessing if ``None``.
            preprocessor: An instance of :class:`~src.preprocessor.RadiologicPreprocessor`.
                Required when ``preprocess_output_dir`` is set.
            base_image_dir: Root directory prepended to relative image paths.
                Pass ``None`` when the CSV already contains absolute paths.
            num_workers: Passed to :meth:`preprocess` (default: 1).
            sample_n: Passed to :meth:`preprocess` (default: None).
            mask_output_dir: Directory to write PNG mask files to.
                Skip mask preprocessing if ``None``.
            mask_num_cores: Passed to :meth:`preprocess_masks` (default: 1).
            study_ids: Keep only rows whose ``study_id`` is in this list, right
                after the IDs are built, so labels, reports, etc. are only built
                for these rows. Not the recommended way: normally harmonize
                everything, then filter ``harmonized_df``. Use it only to save time
                when you need a few studies from a large dataset (e.g. MIMIC
                reports). Harmonizers whose own ``harmonize()`` never reaches this
                one get their result filtered afterwards, with a warning (no speedup).
            patient_ids: Same as ``study_ids``, for ``patient_id``.

        Returns:
            DataFrame with at minimum ``patient_id``, ``study_id``, ``image_path``
            plus any optional columns that were populated.
        """
        self._harmonized_df_override = None
        self.df = pd.read_csv(self.csv_path)

        self._build_patient_id()
        self._build_study_id()
        # Optional early filter (see ``study_ids`` / ``patient_ids`` above). A wrapped
        # subclass harmonize() hands the ids over via self._id_filter.
        pending = getattr(self, "_id_filter", None)
        if pending:
            study_ids, patient_ids = pending["study_ids"], pending["patient_ids"]
            pending["applied"] = True
        self.df = _filter_by_ids(self.df, study_ids, patient_ids)
        self._build_series_id()
        self._build_image_path()
        self.LABEL_COLS = sorted(self.LABEL_COLS, key=lambda c: c.lower().replace(" ", "_"))
        self._build_labels()
        self._build_view_position()
        self._build_mask_path()
        self._build_bbox()
        self._build_report()

        # Drop rows whose image file could not be resolved (e.g. DICOM not
        # found during glob/merge).  Keeping them would propagate NaN paths
        # into the data dicts and crash the MONAI transform pipeline.
        self.df.dropna(subset=["image_path"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        if preprocess_output_dir is not None:
            self.preprocess(
                output_dir=preprocess_output_dir,
                preprocessor=preprocessor,
                base_image_dir=base_image_dir,
                num_workers=num_workers,
                sample_n=sample_n,
            )

        if mask_output_dir is not None:
            self.preprocess_masks(
                output_dir=mask_output_dir,
                base_image_dir=base_image_dir,
                num_cores=mask_num_cores,
            )

        return self._select_harmonized_columns(self.df)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _preprocessed_path(image_path: str, output_dir: str) -> str:
    basename = os.path.basename(image_path)
    stem = (
        basename[:-7] if basename.endswith(".nii.gz") else os.path.splitext(basename)[0]
    )
    return os.path.join(output_dir, stem + ".nii")
