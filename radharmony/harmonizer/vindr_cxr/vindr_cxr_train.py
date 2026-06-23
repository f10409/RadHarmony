"""Harmonizer for the VinDr-CXR training dataset."""

import os

import pandas as pd

from ..base import BaseHarmonizer


class VinDrCXRTrainHarmonizer(BaseHarmonizer):
    """Harmonize VinDr-CXR training set into the standard RadHarmony format.

    Three radiologists independently label each chest X-ray.  Labels are
    aggregated via majority vote: a finding is positive when >= 2 of 3
    annotators agree.

    Bounding boxes come from a separate annotations CSV
    (``annotations_train.csv`` at
    ``.../vindr-cxr/1.0.0/annotations/annotations_train.csv``).
    Each image may have multiple boxes (one per finding per radiologist).
    They are stored as a list of ``[x_min, y_min, x_max, y_max]``
    pixel-coordinate lists in the ``bbox`` column.

    All images are PA view, so no ``view_position`` column is produced.

    Args:
        csv_path: Path to ``image_labels_train.csv``
            (e.g. ``.../vindr-cxr/1.0.0/annotations/image_labels_train.csv``).
        bbox_csv_path: Path to ``annotations_train.csv`` (optional).
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
        # Majority vote already applied in harmonize(); rename to snake_case
        # Use same convention as BaseHarmonizer._label_columns_for_output:
        # lower + spaces→underscores (keep "/" as-is so the base class can match)
        for col in self.LABEL_COLS:
            if col in self.df.columns:
                snake = col.lower().replace(" ", "_")
                self.df.rename(columns={col: snake}, inplace=True)

    def _build_bbox(self) -> None:
        """Aggregate per-image bounding boxes from the annotations CSV.

        Three radiologists independently draw bounding boxes per finding.
        Aggregation mirrors the majority-vote label strategy: for each
        ``(image_id, class_name)`` pair, keep the finding only when >= 2
        radiologists drew a box, then average the coordinates across those
        annotators to produce one consensus box per finding.

        All bboxes are normalized to ``[0, 1]`` using the original DICOM
        image dimensions and converted to the pipeline format
        ``[y_min, y_max, x_min, x_max]``
        (i.e. ``[dim0_min, dim0_max, dim1_min, dim1_max]``).

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

        # --- Majority vote: keep (image_id, class_name) with >= 2 rads ---
        vote_counts = (
            bbox_df.groupby(["image_id", "class_name"])["rad_id"]
            .nunique()
            .reset_index(name="n_rads")
        )
        keep = vote_counts[vote_counts["n_rads"] >= 2][["image_id", "class_name"]]
        bbox_df = bbox_df.merge(keep, on=["image_id", "class_name"], how="inner")

        if bbox_df.empty:
            return

        # --- Average coordinates across radiologists per finding ---
        agg_df = (
            bbox_df.groupby(["image_id", "class_name"])[
                ["x_min", "y_min", "x_max", "y_max"]
            ]
            .mean()
            .reset_index()
        )

        # Read DICOM headers to get original image dimensions for normalization
        image_ids_with_bbox = agg_df["image_id"].unique()
        img_sizes = {}
        if self.base_image_dir:
            import pydicom
            for iid in image_ids_with_bbox:
                path = os.path.join(self.base_image_dir, iid + ".dicom")
                if os.path.exists(path):
                    dcm = pydicom.dcmread(path, stop_before_pixels=True)
                    img_sizes[iid] = (int(dcm.Rows), int(dcm.Columns))

        # Normalize and convert to [y_min/H, y_max/H, x_min/W, x_max/W],
        # also collect the class label for each box (snake_case).
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
            agg_df.groupby("image_id")
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
        """Read labels CSV, aggregate via majority vote, build standard columns."""
        self._harmonized_df_override = None
        self.df = pd.read_csv(self.csv_path)

        # Majority vote across 3 radiologists: positive when >= 2 agree
        self.df = (
            self.df.drop(columns=["rad_id"])
            .groupby("image_id")
            .sum()
            .ge(2)
            .astype(int)
            .reset_index()
        )

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
