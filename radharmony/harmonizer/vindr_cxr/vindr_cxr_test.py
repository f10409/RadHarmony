"""Harmonizer for the VinDr-CXR test dataset."""

import pandas as pd

from .._bbox import box_label, clip_boxes
from .._dicom_tags import add_patient_tags
from ..base import BaseHarmonizer
from ._age import _AgeZeroIsUnknownMixin
from ._sizes import image_sizes


class VinDrCXRTestHarmonizer(_AgeZeroIsUnknownMixin, BaseHarmonizer):
    """Harmonize VinDr-CXR test set into the standard RadHarmony format.

    The train set has 3 radiologists and a majority vote. In the test set, 5 radiologists
    checked the data, and the labels and boxes are already joined. There is no ``rad_id``
    column and no vote.

    The boxes are in a separate CSV, ``annotations_test.csv``
    (``.../vindr-cxr/1.0.0/annotations/annotations_test.csv``). An image can have several boxes
    (one for each finding). They are fractions from 0 to 1 (with the DICOM image size), stored
    as a list of ``[y_min, y_max, x_min, x_max]`` for each image.

    All images are PA view, so there is no ``view_position`` column.

    ``sex`` and ``age_years`` come from the DICOM tags ``PatientSex`` and ``PatientAge``. The
    CSV of the release has neither (D-06, D-26). Many headers have an empty tag, the sex ``O``
    or a bad age (for example ``Y``). These give NaN. An age of 0 (``000Y`` or ``000D``) means
    unknown here and gives NaN too. The harmonizer reads the headers only if ``base_image_dir``
    is set.

    The test CSV has ``"Other disease"`` (singular). The harmonizer renames it to
    ``"Other diseases"``, as in the train set.

    Args:
        csv_path: Path to ``image_labels_test.csv``, for example
            ``.../vindr-cxr/1.0.0/annotations/image_labels_test.csv``.
        bbox_csv_path: Path to ``annotations_test.csv`` (optional).
        base_image_dir: Root folder of the DICOM files, for the box fractions.
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

    # DICOM tags, read by _build_patient_tags() (D-06, D-26).
    AGE_SOURCE_COL = "PatientAge"
    AGE_SOURCE_UNIT = "dicom_as"
    SEX_SOURCE_COL = "PatientSex"

    def __init__(self, csv_path: str, bbox_csv_path: str = None, base_image_dir: str = None):
        super().__init__(csv_path=csv_path, bbox_csv_path=bbox_csv_path)
        self.base_image_dir = base_image_dir

    def _build_patient_id(self) -> None:
        # The release has no patient. The image ID is the patient ID.
        self.df["patient_id"] = self.df["image_id"].astype(str)

    def _build_study_id(self) -> None:
        # One image for each study.
        self.df["study_id"] = self.df["image_id"].astype(str)

    def _build_image_path(self) -> None:
        # The files are <image_id>.dicom in base_image_dir.
        self.df["image_path"] = self.df["image_id"] + ".dicom"
        self._build_patient_tags()

    def _build_patient_tags(self) -> None:
        """Read ``PatientSex`` and ``PatientAge`` from the DICOM headers (D-06)."""
        if self.base_image_dir:
            self.df = add_patient_tags(self.df, self.base_image_dir)

    def _build_labels(self) -> None:
        # The CSV labels are already joined. Rename to snake_case (lower case, "_" for spaces,
        # "/" stays). The CSV has "Other disease" (singular). Rename it as in train.
        if "Other disease" in self.df.columns:
            self.df.rename(columns={"Other disease": "Other diseases"}, inplace=True)
        for col in self.LABEL_COLS:
            if col in self.df.columns:
                snake = col.lower().replace(" ", "_")
                self.df.rename(columns={col: snake}, inplace=True)

    def _build_bbox(self) -> None:
        """Convert the boxes of the annotations CSV to fractions.

        The test boxes are already joined (no ``rad_id``). A row is one finding of an image. The
        coordinates are fractions from 0 to 1 (with the DICOM header size), stored as
        ``[y_min, y_max, x_min, x_max]``.

        It makes 2 columns:

        * ``bbox``: a list of 4-value fraction lists for each image.
        * ``bbox_labels``: the class names (snake_case), in the same order.

        A box is clipped to the image. A box with no area left is dropped, with a warning (D-07).
        """
        if self.bbox_csv_path is None:
            return
        bbox_df = pd.read_csv(self.bbox_csv_path)
        # A "No finding" row has NaN coordinates.
        bbox_df = bbox_df.dropna(subset=["x_min", "y_min", "x_max", "y_max"])
        if bbox_df.empty:
            return

        # The DICOM headers give the image size.
        image_ids_with_bbox = bbox_df["image_id"].unique()
        img_sizes = image_sizes(self.base_image_dir, image_ids_with_bbox, "VinDr-CXR test")  # warns about lost boxes

        # Fractions [y_min/H, y_max/H, x_min/W, x_max/W] and the labels.
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
                labels.append(box_label(row["class_name"]))
            return pd.Series({"bbox": boxes, "bbox_labels": labels})

        bbox_agg = (
            bbox_df.groupby("image_id")
            .apply(_normalize_group, include_groups=False)
            .reset_index()
        )
        self.df = self.df.merge(bbox_agg, on="image_id", how="left")
        # An image with no box gets empty lists.
        self.df["bbox"] = self.df["bbox"].apply(
            lambda x: x if isinstance(x, list) else []
        )
        self.df["bbox_labels"] = self.df["bbox_labels"].apply(
            lambda x: x if isinstance(x, list) else []
        )
        clip_boxes(self.df, "VinDr-CXR test")  # D-07
