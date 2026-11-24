"""Harmonizer for the 2021 SIIM-FISABIO-RSNA COVID-19 Detection Challenge.

Kaggle competition ``siim-covid19-detection``. Layout (after ``kagglehub.competition_download``)::

    <base_image_dir>/                      # train/
      <StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm   # one chest X-ray each
    train_study_level.csv                  # id=<study_uid>_study + 4 one-hot classes
    train_image_level.csv                  # id=<sop_uid>_image, boxes, label, StudyInstanceUID

The table has one row for each image (SOPInstanceUID): 6,334 rows in 6,054 studies.
``patient_id`` is the DICOM ``PatientID`` tag (3,261 patients), so the studies of a patient stay
together. The study labels join on ``StudyInstanceUID``. The boxes come from the ``boxes``
column, a Python list text of ``{x, y, width, height}`` dicts in DICOM pixels. The code divides
them by ``Rows`` and ``Columns`` of the DICOM headers, to get values from 0 to 1.

``sex`` comes from the DICOM tag ``PatientSex``. The CSV of the release has no sex (D-06, D-26).
The headers have no ``PatientAge``, so the table has no ``age_years``.
"""

import ast
import os

import pandas as pd

from .._bbox import clip_boxes
from .._dicom_tags import read_dicom_tags, tag_values
from ..base import BaseHarmonizer


class SIIMCOVID19Harmonizer(BaseHarmonizer):
    """Harmonize SIIM-FISABIO-RSNA COVID-19 Detection Challenge into RadHarmony format.

    Args:
        csv_path: Path to ``train_study_level.csv`` (4-class appearance labels).
        image_csv_path: Path to ``train_image_level.csv`` (bboxes + per-image mapping).
        base_image_dir: Root of the DICOM tree, usually
            ``<kagglehub_cache>/competitions/siim-covid19-detection/train/``. Required. The code
            walks the tree once to find ``SeriesInstanceUID``. It reads ``Rows`` and
            ``Columns`` to normalize the boxes. It reads ``PatientID`` for ``patient_id``.
            It reads ``PatientSex`` for ``sex``.
        num_workers: Number of threads for the DICOM header reads (default 8).
    """

    LABEL_COLS = [
        "Atypical Appearance",
        "Indeterminate Appearance",
        "Negative for Pneumonia",
        "Typical Appearance",
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

    #: The DICOM tag, read in _read_source(). The base step makes ``sex`` (D-26).
    SEX_SOURCE_COL = "PatientSex"

    def __init__(
        self,
        csv_path: str,
        image_csv_path: str = None,
        base_image_dir: str = None,
        num_workers: int = 8,
    ):
        super().__init__(csv_path=os.path.expanduser(csv_path) if csv_path else csv_path)
        self.image_csv_path = (
            os.path.expanduser(image_csv_path) if image_csv_path else image_csv_path
        )
        self.base_image_dir = (
            os.path.expanduser(base_image_dir) if base_image_dir else base_image_dir
        )
        self.num_workers = num_workers

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["image_csv_path"] = self.image_csv_path
        snap["base_image_dir"] = self.base_image_dir
        snap["num_workers"] = self.num_workers
        return snap

    def _build_patient_id(self) -> None:
        # The CSVs have no patient column. The DICOM PatientID links the studies of
        # one patient (1,200 train patients have more than one study). If the tag is
        # missing, use the study.
        self.df["patient_id"] = self.df["PatientID"].fillna(self.df["study_id"]).astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["study_id"].astype(str)

    def _build_image_path(self) -> None:
        # _read_source() sets image_path from the folder walk.
        pass

    def _build_sop_to_path(self) -> dict[str, str]:
        """Walk ``base_image_dir`` once and return ``{sop_uid: "<study>/<series>/<sop>.dcm"}``."""
        if not self.base_image_dir or not os.path.isdir(self.base_image_dir):
            raise ValueError(
                f"base_image_dir {self.base_image_dir!r} is required and must exist."
            )
        mapping: dict[str, str] = {}
        for study in os.listdir(self.base_image_dir):
            study_path = os.path.join(self.base_image_dir, study)
            if not os.path.isdir(study_path):
                continue
            for series in os.listdir(study_path):
                series_path = os.path.join(study_path, series)
                if not os.path.isdir(series_path):
                    continue
                for fname in os.listdir(series_path):
                    if fname.endswith(".dcm"):
                        sop = fname[:-4]
                        mapping[sop] = f"{study}/{series}/{fname}"
        return mapping

    @staticmethod
    def _parse_boxes(raw) -> list[dict]:
        """Parse the ``boxes`` column: list text of xywh dicts, or NaN."""
        if not isinstance(raw, str) or not raw.strip():
            return []
        try:
            parsed = ast.literal_eval(raw)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            return []

    def _read_source(self) -> pd.DataFrame:
        """Read both CSVs, find the image paths, read the DICOM headers and merge the labels.

        The headers give ``PatientID`` for ``patient_id``, so they are read for all rows.
        ``_dim`` holds the ``(Rows, Columns)`` of each image for :meth:`_build_bbox`.
        """
        if self.image_csv_path is None:
            raise ValueError(
                "image_csv_path is required (train_image_level.csv): it provides "
                "the list of images + per-image bboxes."
            )

        # Walk the image folder: SOP UID to relative path.
        sop_to_path = self._build_sop_to_path()

        study_df = pd.read_csv(self.csv_path)
        study_df["study_uid"] = study_df["id"].str.replace("_study", "", regex=False)
        study_df = study_df.drop(columns=["id"])

        image_df = pd.read_csv(self.image_csv_path)
        image_df["sop_uid"] = image_df["id"].str.replace("_image", "", regex=False)

        image_df["image_path"] = image_df["sop_uid"].map(sop_to_path)
        # Drop the rows whose DICOM is not on disk.
        dropped = image_df["image_path"].isna().sum()
        if dropped:
            print(
                f"[siim_covid19] {dropped} image rows have no matching .dcm under "
                f"{self.base_image_dir}; dropping."
            )
        image_df = image_df.dropna(subset=["image_path"]).reset_index(drop=True)

        # Merge the study labels.
        df = image_df.merge(
            study_df, left_on="StudyInstanceUID", right_on="study_uid", how="left",
        )
        df = df.rename(columns={"StudyInstanceUID": "study_id"})
        df = df.drop(columns=["study_uid", "id"])

        headers = read_dicom_tags(
            self.base_image_dir, df["image_path"].tolist(),
            ["Rows", "Columns", "PatientID", "PatientSex"], self.num_workers,
        )
        # An empty PatientID counts as missing.
        df["PatientID"] = df["image_path"].map(
            lambda p: (headers.get(p) or {}).get("PatientID") or None
        )
        df["PatientSex"] = tag_values(headers, df["image_path"], "PatientSex")
        dims = {}
        for p, h in headers.items():
            try:
                dims[p] = (int(h["Rows"]), int(h["Columns"]))
            except (TypeError, ValueError):
                continue
        df["_dim"] = df["image_path"].map(dims)
        return df

    def _build_bbox(self) -> None:
        """Divide the pixel boxes by the image size, then clip them (D-07)."""
        def _normalize(raw, dim) -> tuple[list, list]:
            raw_boxes = self._parse_boxes(raw)
            if not raw_boxes or not isinstance(dim, tuple):
                return [], []
            H, W = dim
            boxes, labels = [], []
            for b in raw_boxes:
                # Skip a bad entry.
                try:
                    x, y, w, h = b["x"], b["y"], b["width"], b["height"]
                except (KeyError, TypeError):
                    continue
                # One source box starts at x = -3.56 px. clip_boxes clips it below (D-07).
                boxes.append([float(v) for v in (y / H, (y + h) / H, x / W, (x + w) / W)])
                labels.append("opacity")
            return boxes, labels

        # A list, not DataFrame.apply, so a table with no row (an id filter) still gets the columns.
        pairs = [_normalize(raw, dim) for raw, dim in zip(self.df["boxes"], self.df["_dim"], strict=True)]
        self.df["bbox"] = pd.Series([p[0] for p in pairs], index=self.df.index, dtype=object)
        self.df["bbox_labels"] = pd.Series([p[1] for p in pairs], index=self.df.index, dtype=object)
        clip_boxes(self.df, "SIIM-COVID19")  # D-07
