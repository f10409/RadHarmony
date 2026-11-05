"""Harmonizer for the VinDr-CXR training dataset."""

import numpy as np
import pandas as pd

from .._bbox import box_label, clip_boxes
from .._dicom_tags import add_patient_tags
from ..base import BaseHarmonizer
from ._age import _AgeZeroIsUnknownMixin
from ._sizes import image_sizes

_COORDS = ["x_min", "y_min", "x_max", "y_max"]


def _iou(a, b) -> float:
    """Intersection over union of 2 ``[x_min, y_min, x_max, y_max]`` boxes."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0) * max(h, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _group_boxes(g: pd.DataFrame, iou: float = 0.5) -> list[list[float]]:
    """Return one mean box for each lesion of one image and class (D-36).

    In file order, a box joins the first group whose mean box has an IoU
    above ``iou`` with it, or starts a new group. A group with boxes of 2 or
    more radiologists (``rad_id``) is a lesion.

    Args:
        g: the annotation rows of one ``(image_id, class_name)`` pair.
        iou: the overlap above which 2 boxes are the same lesion.

    Returns:
        The mean ``[x_min, y_min, x_max, y_max]`` of each lesion.
    """
    groups = []  # each: [list of boxes, set of rad_id]
    for box, rad in zip(g[_COORDS].to_numpy(float).tolist(), g["rad_id"], strict=True):
        for boxes, rads in groups:
            if _iou(np.mean(boxes, axis=0), box) > iou:
                boxes.append(box)
                rads.add(rad)
                break
        else:
            groups.append(([box], {rad}))
    return [np.mean(boxes, axis=0).tolist() for boxes, rads in groups if len(rads) >= 2]


class VinDrCXRTrainHarmonizer(_AgeZeroIsUnknownMixin, BaseHarmonizer):
    """Harmonize VinDr-CXR training set into the standard RadHarmony format.

    3 radiologists label each chest X-ray on their own. The labels use a majority vote: a
    finding is positive if 2 or more of the 3 agree.

    The boxes are in a separate CSV, ``annotations_train.csv``
    (``.../vindr-cxr/1.0.0/annotations/annotations_train.csv``). One radiologist can draw
    several boxes of one class. The overlapping boxes of one class form a lesion. A lesion of
    2 or more radiologists gives one box (D-36). ``bbox`` has ``[y_min, y_max, x_min, x_max]``
    as fractions of the image size (D-07).

    All images are PA view, so there is no ``view_position`` column.

    ``sex`` and ``age_years`` come from the DICOM tags ``PatientSex`` and ``PatientAge``. The
    CSV of the release has neither (D-06, D-26). Many headers have an empty tag, the sex ``O``
    or a bad age (for example ``Y``). These give NaN. An age of 0 (``000Y`` or ``000D``) means
    unknown here and gives NaN too. The harmonizer reads the headers only if ``base_image_dir``
    is set.

    Args:
        csv_path: Path to ``image_labels_train.csv``, for example
            ``.../vindr-cxr/1.0.0/annotations/image_labels_train.csv``.
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

    # DICOM tags, read by _build_patient_tags() (D-06, D-26).
    AGE_SOURCE_COL = "PatientAge"
    AGE_SOURCE_UNIT = "dicom_as"
    SEX_SOURCE_COL = "PatientSex"

    def __init__(self, csv_path: str, bbox_csv_path: str = None, base_image_dir: str = None):
        super().__init__(csv_path=csv_path, bbox_csv_path=bbox_csv_path)
        self.base_image_dir = base_image_dir

    def _read_source(self) -> pd.DataFrame:
        """Read the labels CSV and apply the majority vote: positive if 2 or more of the 3 agree."""
        return (
            pd.read_csv(self.csv_path)
            .drop(columns=["rad_id"])
            .groupby("image_id")
            .sum()
            .ge(2)
            .astype(int)
            .reset_index()
        )

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
        # _read_source() did the majority vote. Rename as in
        # BaseHarmonizer._label_columns_for_output: lower case and "_" for spaces. A "/" stays,
        # so the base class finds the column.
        for col in self.LABEL_COLS:
            if col in self.df.columns:
                snake = col.lower().replace(" ", "_")
                self.df.rename(columns={col: snake}, inplace=True)

    def _build_bbox(self) -> None:
        """Build the boxes of each image from the annotations CSV.

        3 radiologists draw boxes for each finding, and one radiologist can draw several boxes
        of one class (D-36). For each ``(image_id, class_name)`` pair, the boxes that overlap
        (IoU above 0.5) form one lesion (:func:`_group_boxes`). A lesion of 2 or more
        radiologists stays, as in the majority vote of the labels. Its box is the mean of its
        boxes.

        A box is clipped to the image. A box with no area left is dropped, with a warning
        (D-07). The boxes are fractions from 0 to 1 (with the DICOM image size), in the pipeline
        format ``[y_min, y_max, x_min, x_max]`` (``[dim0_min, dim0_max, dim1_min, dim1_max]``).

        It makes 2 columns:

        * ``bbox``: a list of 4-value fraction lists for each image.
        * ``bbox_labels``: the class names (snake_case), in the same order.
        """
        if self.bbox_csv_path is None:
            return
        bbox_df = pd.read_csv(self.bbox_csv_path)
        # A "No finding" row has NaN coordinates.
        bbox_df = bbox_df.dropna(subset=_COORDS)
        if bbox_df.empty:
            return

        # One box for each lesion of 2 or more radiologists (D-36).
        lesions = [
            {"image_id": iid, "class_name": cls, **dict(zip(_COORDS, box, strict=True))}
            for (iid, cls), g in bbox_df.groupby(["image_id", "class_name"], sort=True)
            for box in _group_boxes(g)
        ]
        if not lesions:
            return
        agg_df = pd.DataFrame(lesions)

        # The DICOM headers give the original image size.
        image_ids_with_bbox = agg_df["image_id"].unique()
        img_sizes = image_sizes(self.base_image_dir, image_ids_with_bbox, "VinDr-CXR train")  # warns about lost boxes

        # Fractions [y_min/H, y_max/H, x_min/W, x_max/W] and the class label of each box.
        # clip_boxes clips them below (D-07).
        def _normalize_group(g):
            iid = g.name
            if iid not in img_sizes:
                return pd.Series({"bbox": [], "bbox_labels": []})
            H, W = img_sizes[iid]
            boxes = [[float(v) for v in (row["y_min"] / H, row["y_max"] / H, row["x_min"] / W, row["x_max"] / W)]
                     for _, row in g.iterrows()]
            labels = [box_label(row["class_name"]) for _, row in g.iterrows()]
            return pd.Series({"bbox": boxes, "bbox_labels": labels})

        bbox_agg = (
            agg_df.groupby("image_id")
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
        clip_boxes(self.df, "VinDr-CXR train")  # D-07
