"""Harmonizer for the VinDr-CXR test dataset."""

import os

import pandas as pd

from ..base import BaseHarmonizer


class VinDrCXRTestHarmonizer(BaseHarmonizer):
    """Harmonize VinDr-CXR test set into the standard RadHarmony format.

    Unlike the training set (3 radiologists, majority vote), the test set
    was verified by 5 radiologists and labels/bboxes are already aggregated
    — no ``rad_id`` column, no majority-vote step needed.

    Bounding boxes come from a separate annotations CSV
    (``annotations_test.csv`` at
    ``.../vindr-cxr/1.0.0/annotations/annotations_test.csv``).
    Each image may have multiple boxes (one per finding).  They are
    normalized to ``[0, 1]`` using the original DICOM dimensions and
    stored as a list of ``[y_min, y_max, x_min, x_max]`` per image.

    All images are PA view, so no ``view_position`` column is produced.

    Note: the test CSV uses ``"Other disease"`` (singular); the harmonizer
    renames it to ``"Other diseases"`` to match the train set convention.

    Args:
        csv_path: Path to ``image_labels_test.csv``
            (e.g. ``.../vindr-cxr/1.0.0/annotations/image_labels_test.csv``).
        bbox_csv_path: Path to ``annotations_test.csv`` (optional).
        base_image_dir: Root directory of DICOM files for bbox normalization.
    """

    LABEL_COLS = [
        "Aortic enlargement",
        "Atelectasis",
        "COPD",
        "Calcification",
        "Cardiomegaly",
        "Clavicle fracture",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Enlarged PA",
        "ILD",
        "Infiltration",
        "Lung Opacity",
        "Lung cavity",
        "Lung cyst",
        "Lung tumor",
        "Mediastinal shift",
        "No finding",
        "Nodule/Mass",
        "Other diseases",
        "Other lesion",
        "Pleural effusion",
        "Pleural thickening",
        "Pneumonia",
        "Pneumothorax",
        "Pulmonary fibrosis",
        "Rib fracture",
        "Tuberculosis",
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

    def __init__(self, csv_path: str, bbox_csv_path: str = None, base_image_dir: str = None):
        super().__init__(csv_path=csv_path, bbox_csv_path=bbox_csv_path)
        self.base_image_dir = base_image_dir

    def _build_patient_id(self) -> None:
        # VinDr-CXR has no patient concept; use image_id as patient_id
        self.df["patient_id"] = self.df["image_id"].astype(str)

    def _build_study_id(self) -> None:
        # One image per study; use image_id as study_id
        self.df["study_id"] = self.df["image_id"].astype(str)

    def _build_image_path(self) -> None:
        # DICOM files named <image_id>.dicom under base_image_dir
        self.df["image_path"] = self.df["image_id"] + ".dicom"

    def _build_labels(self) -> None:
        # Labels already aggregated in the CSV; just rename to snake_case.
        # Convention: lower + spaces→underscores (keep "/" as-is).
        # CSV has "Other disease" (singular); rename to match train's "Other diseases"
        if "Other disease" in self.df.columns:
            self.df.rename(columns={"Other disease": "Other diseases"}, inplace=True)
        for col in self.LABEL_COLS:
            if col in self.df.columns:
                snake = col.lower().replace(" ", "_")
                self.df.rename(columns={col: snake}, inplace=True)

    def _build_bbox(self) -> None:
        """Normalize bounding boxes from the annotations CSV.

        Unlike the train set, test bboxes are already aggregated (no
        ``rad_id``).  Each row is one finding per image.  Coordinates are
        normalized to ``[0, 1]`` using DICOM header dimensions and stored
        as ``[y_min, y_max, x_min, x_max]``.

        Two columns are produced:

        * ``bbox`` — list of normalized 4-element coordinate lists per image.
        * ``bbox_labels`` — parallel list of class-name strings (snake_case).
        """
        if self.bbox_csv_path is None:
            return
        bbox_df = pd.read_csv(self.bbox_csv_path)
        # "No finding" rows have NaN coordinates — drop them
        bbox_df = bbox_df.dropna(subset=["x_min", "y_min", "x_max", "y_max"])
        if bbox_df.empty:
            return

        # Read DICOM headers for normalization
        image_ids_with_bbox = bbox_df["image_id"].unique()
        img_sizes = {}
        if self.base_image_dir:
            import pydicom
            for iid in image_ids_with_bbox:
                path = os.path.join(self.base_image_dir, iid + ".dicom")
                if os.path.exists(path):
                    dcm = pydicom.dcmread(path, stop_before_pixels=True)
                    img_sizes[iid] = (int(dcm.Rows), int(dcm.Columns))

        # Normalize to [y_min/H, y_max/H, x_min/W, x_max/W] + collect labels
        def _normalize_group(g):
            iid = g.name
            if iid not in img_sizes:
                return pd.Series({"bbox": [], "bbox_labels": []})
            H, W = img_sizes[iid]
            boxes, labels = [], []
            for _, row in g.iterrows():
                boxes.append([
                    row["y_min"] / H,
                    row["y_max"] / H,
                    row["x_min"] / W,
                    row["x_max"] / W,
                ])
                labels.append(row["class_name"].lower().replace(" ", "_"))
            return pd.Series({"bbox": boxes, "bbox_labels": labels})

        bbox_agg = (
            bbox_df.groupby("image_id")
            .apply(_normalize_group, include_groups=False)
            .reset_index()
        )
        self.df = self.df.merge(bbox_agg, on="image_id", how="left")
        # Images without annotations get empty lists
        self.df["bbox"] = self.df["bbox"].apply(
            lambda x: x if isinstance(x, list) else []
        )
        self.df["bbox_labels"] = self.df["bbox_labels"].apply(
            lambda x: x if isinstance(x, list) else []
        )

    def harmonize(self, **kwargs) -> pd.DataFrame:
        """Read labels CSV and build standard columns.

        Labels are already aggregated (verified by 5 radiologists), so no
        majority-vote step is needed — unlike the train harmonizer.
        """
        self._harmonized_df_override = None
        self.df = pd.read_csv(self.csv_path)

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

        return self._select_harmonized_columns(self.df)
