"""Harmonizer for the 2021 SIIM-FISABIO-RSNA COVID-19 Detection Challenge.

Kaggle competition slug: ``siim-covid19-detection``.

Source layout (after ``kagglehub.competition_download``)::

    <base_image_dir>/                      # train/
      <StudyInstanceUID>/
        <SeriesInstanceUID>/
          <SOPInstanceUID>.dcm             # one chest X-ray per file

    train_study_level.csv                  # id=<study_uid>_study + 4 one-hot classes
    train_image_level.csv                  # id=<sop_uid>_image, boxes, label, StudyInstanceUID

One row is emitted per image (SOPInstanceUID) — 6,334 rows across 6,054 studies.
Study-level labels are joined in on ``StudyInstanceUID``; image-level bboxes are
parsed from the ``boxes`` column (Python-literal list of ``{x, y, width, height}``
dicts in *original DICOM pixel space*) and normalised to ``[0, 1]`` using
``Rows``/``Columns`` from the DICOM headers.
"""

import ast
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ..base import BaseHarmonizer


class SIIMCOVID19Harmonizer(BaseHarmonizer):
    """Harmonize SIIM-FISABIO-RSNA COVID-19 Detection Challenge into RadHarmony format.

    Args:
        csv_path: Path to ``train_study_level.csv`` (4-class appearance labels).
        image_csv_path: Path to ``train_image_level.csv`` (bboxes + per-image mapping).
        base_image_dir: Root of the DICOM tree (typically
            ``<kagglehub_cache>/competitions/siim-covid19-detection/train/``).
            Required: used to (a) resolve ``SeriesInstanceUID`` by walking the
            tree once, and (b) read ``Rows``/``Columns`` for bbox normalisation.
        num_workers: Parallel threads for DICOM header reads (default 8).
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

    # ------------------------------------------------------------------
    # Build hooks
    # ------------------------------------------------------------------

    def _build_patient_id(self) -> None:
        # No distinct patient column; collapsing to StudyInstanceUID keeps
        # all images of a study in the same fold.
        self.df["patient_id"] = self.df["study_id"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["study_id"].astype(str)

    def _build_image_path(self) -> None:
        # image_path is populated during harmonize() from the directory walk —
        # nothing to do here.
        pass

    # ------------------------------------------------------------------
    # Directory walk → SOP UID → relative path
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Bbox parsing + normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_boxes(raw) -> list[dict]:
        """Parse the `boxes` column — a str repr of a list of xywh dicts (or NaN)."""
        if not isinstance(raw, str) or not raw.strip():
            return []
        try:
            parsed = ast.literal_eval(raw)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            return []

    def _read_image_dims(
        self, image_paths: list[str]
    ) -> dict[str, tuple[int, int]]:
        """Return ``{image_path_rel: (H, W)}`` by reading DICOM headers in parallel."""
        import pydicom

        def _read_one(rel_path: str) -> tuple[str, tuple[int, int] | None]:
            full = os.path.join(self.base_image_dir, rel_path)
            try:
                ds = pydicom.dcmread(full, stop_before_pixels=True)
                return rel_path, (int(ds.Rows), int(ds.Columns))
            except Exception:
                return rel_path, None

        dims: dict[str, tuple[int, int]] = {}
        if self.num_workers and self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                for rel, dim in ex.map(_read_one, image_paths):
                    if dim is not None:
                        dims[rel] = dim
        else:
            for rel in image_paths:
                _, dim = _read_one(rel)
                if dim is not None:
                    dims[rel] = dim
        return dims

    # ------------------------------------------------------------------
    # Custom harmonize — joins both CSVs + walks image dir + normalises bboxes
    # ------------------------------------------------------------------

    def harmonize(self, **kwargs) -> pd.DataFrame:
        """Load both CSVs, resolve per-image paths, merge study labels, normalise bboxes."""
        self._harmonized_df_override = None

        if self.image_csv_path is None:
            raise ValueError(
                "image_csv_path is required (train_image_level.csv): it provides "
                "the list of images + per-image bboxes."
            )

        # --- 1. Walk image dir to resolve SOP UID -> relative path -----------
        sop_to_path = self._build_sop_to_path()

        # --- 2. Load CSVs ---------------------------------------------------
        study_df = pd.read_csv(self.csv_path)
        study_df["study_uid"] = study_df["id"].str.replace("_study", "", regex=False)
        study_df = study_df.drop(columns=["id"])

        image_df = pd.read_csv(self.image_csv_path)
        image_df["sop_uid"] = image_df["id"].str.replace("_image", "", regex=False)

        # --- 3. Build image_path from the walk -------------------------------
        image_df["image_path"] = image_df["sop_uid"].map(sop_to_path)
        # Drop rows whose DICOM can't be found on disk (no valid image_path).
        dropped = image_df["image_path"].isna().sum()
        if dropped:
            print(
                f"[siim_covid19] {dropped} image rows have no matching .dcm under "
                f"{self.base_image_dir}; dropping."
            )
        image_df = image_df.dropna(subset=["image_path"]).reset_index(drop=True)

        # --- 4. Merge study-level appearance labels --------------------------
        df = image_df.merge(
            study_df, left_on="StudyInstanceUID", right_on="study_uid", how="left",
        )
        df = df.rename(columns={"StudyInstanceUID": "study_id"})
        df = df.drop(columns=["study_uid", "id"])

        # --- 5. Parse + normalise bboxes -------------------------------------
        df["_boxes_parsed"] = df["boxes"].apply(self._parse_boxes)
        has_box_mask = df["_boxes_parsed"].apply(len) > 0
        paths_with_box = df.loc[has_box_mask, "image_path"].tolist()

        dims = self._read_image_dims(paths_with_box)

        def _normalize(row):
            raw_boxes = row["_boxes_parsed"]
            if not raw_boxes:
                return pd.Series({"bbox": [], "bbox_labels": []})
            dim = dims.get(row["image_path"])
            if dim is None:
                return pd.Series({"bbox": [], "bbox_labels": []})
            H, W = dim
            boxes, labels = [], []
            for b in raw_boxes:
                # Guard against malformed entries.
                try:
                    x, y, w, h = b["x"], b["y"], b["width"], b["height"]
                except (KeyError, TypeError):
                    continue
                boxes.append([y / H, (y + h) / H, x / W, (x + w) / W])
                labels.append("opacity")
            return pd.Series({"bbox": boxes, "bbox_labels": labels})

        bbox_cols = df.apply(_normalize, axis=1)
        df["bbox"] = bbox_cols["bbox"]
        df["bbox_labels"] = bbox_cols["bbox_labels"]
        df = df.drop(columns=["_boxes_parsed", "boxes", "label"])

        # --- 6. Rest of the base build pipeline ------------------------------
        self.df = df
        self._build_patient_id()
        self._build_study_id()
        self._build_image_path()
        self._build_labels()  # rename the 4 study-level cols to snake_case
        self._build_view_position()
        self._build_mask_path()
        self._build_report()

        # Keep LABEL_COLS alphabetically sorted by snake_case (base class would
        # also do this, but harmonize() is custom here).
        self.LABEL_COLS = sorted(
            self.LABEL_COLS, key=lambda c: c.lower().replace(" ", "_")
        )
        self.df.dropna(subset=["image_path"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self._select_harmonized_columns(self.df)
