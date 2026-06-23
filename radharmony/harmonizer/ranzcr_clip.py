"""Harmonizer for the RANZCR CLiP catheter & line classification dataset.

Source: Kaggle competition ``ranzcr-clip-catheter-line-classification`` (the
Royal Australian and New Zealand College of Radiologists 2021 challenge).
Kaggle is the primary distribution channel for this dataset.

Layout after ``kagglehub.competition_download``::

    <base_image_dir>/
      train.csv                  # StudyInstanceUID, PatientID, 11 label cols
      train_annotations.csv      # StudyInstanceUID, label, data (polyline pts)
      sample_submission.csv      # test SOP UIDs (no labels)
      train/
        <StudyInstanceUID>.jpg   # ~30k frontal CXRs
      test/
        <StudyInstanceUID>.jpg   # competition holdout, no labels

Train and test rows are both emitted; a ``split`` column distinguishes them.
Test rows have NaN for every label column (no public labels).

The 11 binary multi-label targets describe whether each catheter / line is
positioned correctly.  Raw CSV column names use ``"CVC - Abnormal"`` style;
the base-class default snake-casing (``lower().replace(" ", "_")``) would
produce ``"cvc_-_abnormal"`` which preserves the dash.  We rename explicitly
to clean snake form (``"cvc_abnormal"``) in :meth:`_build_labels` and define
``LABEL_COLS`` as the clean names so ``_label_columns_for_output`` matches.
"""

import json
import os

import pandas as pd

from .base import BaseHarmonizer


# Mapping from raw CSV column name -> clean snake_case label name.
_RAW_TO_SNAKE = {
    "CVC - Abnormal": "cvc_abnormal",
    "CVC - Borderline": "cvc_borderline",
    "CVC - Normal": "cvc_normal",
    "ETT - Abnormal": "ett_abnormal",
    "ETT - Borderline": "ett_borderline",
    "ETT - Normal": "ett_normal",
    "NGT - Abnormal": "ngt_abnormal",
    "NGT - Borderline": "ngt_borderline",
    "NGT - Incompletely Imaged": "ngt_incompletely_imaged",
    "NGT - Normal": "ngt_normal",
    "Swan Ganz Catheter Present": "swan_ganz_catheter_present",
}


class RANZCRClipHarmonizer(BaseHarmonizer):
    """Harmonize RANZCR CLiP into the standard RadHarmony format.

    Args:
        csv_path: Path to ``train.csv``.
        base_image_dir: Root of the Kaggle download (contains ``train/`` and
            ``test/`` JPEG dirs plus the CSVs).
        annotations_csv_path: Path to ``train_annotations.csv``.  Required
            only when masks are needed; pass ``None`` to skip mask building.
        mask_line_thickness: Pixel width passed to ``cv2.polylines`` when
            rasterizing polylines into masks.  Default 15 mirrors the kernel
            used by the official challenge starter notebook.
        include_test_split: When ``True`` (default), append rows for every
            JPEG under ``<base_image_dir>/test/`` with ``split="test"`` and
            NaN labels.
    """

    LABEL_COLS = [
        "cvc_abnormal",
        "cvc_borderline",
        "cvc_normal",
        "ett_abnormal",
        "ett_borderline",
        "ett_normal",
        "ngt_abnormal",
        "ngt_borderline",
        "ngt_incompletely_imaged",
        "ngt_normal",
        "swan_ganz_catheter_present",
    ]

    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    EXTRA_OUTPUT_COLS = ["split"]

    def __init__(
        self,
        csv_path: str,
        base_image_dir: str = None,
        annotations_csv_path: str = None,
        mask_line_thickness: int = 15,
        include_test_split: bool = True,
    ):
        super().__init__(csv_path=os.path.expanduser(csv_path) if csv_path else csv_path)
        self.base_image_dir = (
            os.path.expanduser(base_image_dir) if base_image_dir else base_image_dir
        )
        self.annotations_csv_path = (
            os.path.expanduser(annotations_csv_path)
            if annotations_csv_path
            else annotations_csv_path
        )
        self.mask_line_thickness = mask_line_thickness
        self.include_test_split = include_test_split

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["base_image_dir"] = self.base_image_dir
        snap["annotations_csv_path"] = self.annotations_csv_path
        snap["mask_line_thickness"] = self.mask_line_thickness
        snap["include_test_split"] = self.include_test_split
        return snap

    # ------------------------------------------------------------------
    # Build hooks
    # ------------------------------------------------------------------

    def _build_patient_id(self) -> None:
        # Train rows have a real PatientID; test rows have it as NaN and
        # fall back to study_id.  fillna must run *before* astype(str) — a
        # NaN cast to str becomes "<NA>" / "nan" and then fillna is a no-op.
        if "PatientID" in self.df.columns:
            self.df["patient_id"] = (
                self.df["PatientID"]
                .fillna(self.df["StudyInstanceUID"])
                .astype(str)
            )
        else:
            self.df["patient_id"] = self.df["StudyInstanceUID"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["StudyInstanceUID"].astype(str)

    def _build_image_path(self) -> None:
        # split column is stamped before this hook runs in our custom harmonize().
        self.df["image_path"] = (
            self.df["split"] + "/" + self.df["StudyInstanceUID"].astype(str) + ".jpg"
        )

    def _build_labels(self) -> None:
        # Rename raw CSV cols (e.g. "CVC - Abnormal") to clean snake form so
        # base._label_columns_for_output matches LABEL_COLS exactly.
        rename = {raw: snake for raw, snake in _RAW_TO_SNAKE.items() if raw in self.df.columns}
        if rename:
            self.df.rename(columns=rename, inplace=True)

    def _build_mask_path(self) -> None:
        """Aggregate polylines per StudyInstanceUID into a JSON string in ``mask_path``.

        ``train_annotations.csv`` has one row per (StudyInstanceUID, label,
        polyline).  We collapse them into a list of lists of ``[x, y]`` pairs
        per study and JSON-encode so the value can be carried as a single
        string column through merges and ``_decode_mask``.
        """
        if self.annotations_csv_path is None:
            return

        ann = pd.read_csv(self.annotations_csv_path)
        # data column is a stringified Python list, e.g. "[[123, 456], [125, 460]]"
        ann["polyline"] = ann["data"].apply(json.loads)
        agg = (
            ann.groupby("StudyInstanceUID")["polyline"]
            .apply(list)  # list[list[[x, y], ...]]
            .reset_index()
            .rename(columns={"polyline": "_polyline_lists"})
        )
        agg["mask_path"] = agg["_polyline_lists"].apply(json.dumps)
        agg = agg[["StudyInstanceUID", "mask_path"]]

        self.df = self.df.merge(agg, on="StudyInstanceUID", how="left")

    def _decode_mask(self, mask_str: str, width: int, height: int):
        """Rasterize aggregated polylines into a binary (H, W) uint8 mask.

        ``mask_str`` is the JSON produced by :meth:`_build_mask_path` —
        a list of polylines, each a list of ``[x, y]`` integer points in
        original image pixel space.  Returns ``None`` for rows without
        annotations (NaN / empty).
        """
        if not isinstance(mask_str, str) or not mask_str.strip() or mask_str == "nan":
            return None

        import cv2  # type: ignore[import-untyped]
        import numpy as np  # type: ignore[import-untyped]

        try:
            polylines = json.loads(mask_str)
        except (ValueError, TypeError):
            return None
        if not polylines:
            return None

        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in polylines:
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                mask,
                [pts],
                isClosed=False,
                color=1,
                thickness=self.mask_line_thickness,
            )
        return mask

    # ------------------------------------------------------------------
    # Test split discovery
    # ------------------------------------------------------------------

    def _build_test_rows(self) -> pd.DataFrame:
        """Walk ``<base_image_dir>/test/`` and return one row per JPEG.

        All label columns are populated as NaN so the resulting rows align
        with the train DataFrame's schema after concat.
        """
        test_dir = os.path.join(self.base_image_dir, "test") if self.base_image_dir else None
        if not test_dir or not os.path.isdir(test_dir):
            return pd.DataFrame()

        files = [f for f in os.listdir(test_dir) if f.endswith(".jpg")]
        if not files:
            return pd.DataFrame()

        study_ids = [os.path.splitext(f)[0] for f in files]
        rows = pd.DataFrame({
            "StudyInstanceUID": study_ids,
            "PatientID": [pd.NA] * len(study_ids),
        })
        for raw_col in _RAW_TO_SNAKE:
            rows[raw_col] = pd.NA
        rows["split"] = "test"
        return rows

    # ------------------------------------------------------------------
    # Custom harmonize: combine train + test, build standard columns
    # ------------------------------------------------------------------

    def harmonize(
        self,
        mask_output_dir: str = None,
        mask_num_cores: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        """Read train.csv, optionally append test rows, build standard columns.

        If ``mask_output_dir`` is given and ``annotations_csv_path`` was set,
        polylines are rasterized and written as PNGs and ``mask_path`` is
        rewritten to point at the saved files (mirrors the base class flow
        for RLE-mask datasets).
        """
        self._harmonized_df_override = None

        train_df = pd.read_csv(self.csv_path)
        train_df["split"] = "train"

        if self.include_test_split:
            test_df = self._build_test_rows()
            if not test_df.empty:
                train_df = pd.concat([train_df, test_df], ignore_index=True)

        self.df = train_df

        self._build_patient_id()
        self._build_study_id()
        self._build_image_path()
        self._build_labels()
        self._build_view_position()
        self._build_mask_path()
        self._build_bbox()
        self._build_report()

        self.df.dropna(subset=["image_path"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        if mask_output_dir is not None and "mask_path" in self.df.columns:
            self.preprocess_masks(
                output_dir=mask_output_dir,
                num_cores=mask_num_cores,
            )

        return self._select_harmonized_columns(self.df)

    # ------------------------------------------------------------------
    # Mask preprocessing convenience
    # ------------------------------------------------------------------

    def preprocess_masks(
        self,
        output_dir: str,
        base_image_dir: str = None,
        num_cores: int = 1,
    ) -> None:
        """Decode polyline JSON strings to PNGs, defaulting base_image_dir."""
        super().preprocess_masks(
            output_dir=output_dir,
            base_image_dir=base_image_dir or self.base_image_dir,
            num_cores=num_cores,
        )
