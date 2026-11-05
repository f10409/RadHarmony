"""Harmonizer for the RANZCR CLiP catheter and line classification dataset.

Source: the Kaggle competition ``ranzcr-clip-catheter-line-classification`` (RANZCR 2021).
Kaggle is the primary channel. Layout after ``kagglehub.competition_download``::

    <base_image_dir>/
      train.csv                  # StudyInstanceUID, PatientID, 11 label columns
      train_annotations.csv      # StudyInstanceUID, label, data (polyline points)
      sample_submission.csv      # test SOP UIDs (no labels)
      train/<StudyInstanceUID>.jpg   # about 30K frontal CXRs
      test/<StudyInstanceUID>.jpg    # competition holdout, no labels

By default, the class gives the train rows and the test rows. The ``split`` column tells
them apart. A test row has NaN in every label column. This mixes the splits, so do not use it
for a benchmark. Use :class:`RANZCRClipTrainHarmonizer` (labeled rows) and
:class:`RANZCRClipTestHarmonizer` (unlabeled test images, D-25).

The 11 binary labels say whether each catheter or line is placed right. The CSV names look like
``"CVC - Abnormal"``. The base snake-case rule would give ``"cvc_-_abnormal"`` (with the dash).
:meth:`_build_labels` renames them to ``"cvc_abnormal"``, and ``LABEL_COLS`` has the clean names.
"""

import json
import os

import pandas as pd

from ..base import BaseHarmonizer


# CSV column name and the clean label name.
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
        base_image_dir: Root of the Kaggle download (the ``train/`` and ``test/`` JPEG folders
            and the CSVs).
        annotations_csv_path: Path to ``train_annotations.csv``. Needed only for masks. Use
            ``None`` for no masks.
        mask_line_thickness: Line width in pixels for ``cv2.polylines`` when the polylines
            become masks. The default 15 is the value of the official starter notebook.
        include_test_split: If ``True`` (default), add a row with ``split="test"`` and NaN
            labels for each JPEG in ``<base_image_dir>/test/``.
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

    def _build_patient_id(self) -> None:
        # A test row has no PatientID and uses the study ID. Run fillna before astype(str),
        # because a NaN turns into the text "nan".
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
        # _read_source() sets the split column.
        self.df["image_path"] = (
            self.df["split"] + "/" + self.df["StudyInstanceUID"].astype(str) + ".jpg"
        )

    def _build_labels(self) -> None:
        # The names must match LABEL_COLS for ``_label_columns_for_output``.
        rename = {raw: snake for raw, snake in _RAW_TO_SNAKE.items() if raw in self.df.columns}
        if rename:
            self.df.rename(columns=rename, inplace=True)

    def _build_mask_path(self) -> None:
        """Put the polylines of each StudyInstanceUID into ``mask_path`` as JSON text.

        ``train_annotations.csv`` has one row for each (StudyInstanceUID, label, polyline).
        The harmonizer makes one list of ``[x, y]`` lists for each study. One text column
        carries it through the merges and into ``_decode_mask``.
        """
        if self.annotations_csv_path is None:
            return

        ann = pd.read_csv(self.annotations_csv_path)
        # ``data`` is list text, for example "[[123, 456], [125, 460]]".
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
        """Draw the polylines as a binary (H, W) uint8 mask.

        ``mask_str`` is the JSON of :meth:`_build_mask_path`: polylines of ``[x, y]`` integer
        points in pixels of the original image. A row with no annotation (NaN or empty)
        returns ``None``.
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

    def _build_test_rows(self) -> pd.DataFrame:
        """Return one row for each JPEG in ``<base_image_dir>/test/``.

        All label columns are NaN, so the rows fit the train table.
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

    def _read_source(self) -> pd.DataFrame:
        """Read ``train.csv``. Add the test rows if ``include_test_split``."""
        train_df = pd.read_csv(self.csv_path)
        train_df["split"] = "train"

        if self.include_test_split:
            test_df = self._build_test_rows()
            if not test_df.empty:
                train_df = pd.concat([train_df, test_df], ignore_index=True)

        return train_df

    def preprocess_masks(
        self,
        output_dir: str,
        base_image_dir: str = None,
        num_cores: int = 1,
    ) -> None:
        """Decode the polyline JSON to PNG files. ``base_image_dir`` defaults to the own one.

        With no ``annotations_csv_path``, the table has no ``mask_path``, so it does nothing.
        """
        if self.annotations_csv_path is None:
            return
        super().preprocess_masks(
            output_dir=output_dir,
            base_image_dir=base_image_dir or self.base_image_dir,
            num_cores=num_cores,
        )
