"""Harmonizer for the RSNA Pneumonia Detection Challenge dataset.

The source is the adjudicated JSON export, not CSV files. It has annotations of several
annotator teams. The default label group is ``Calculated`` (the final consensus). It has
3 classes: Normal (9,790 images), No Lung Opacity / Not Normal (12,788 images) and
Lung Opacity (11,390 boxes).

A box is a pixel ``(x, y, width, height)`` on a 1024x1024 image. The harmonizer converts it to
fractions ``[dim0_min, dim0_max, dim1_min, dim1_max]``. The images are DICOM files in
``<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm``.

Details: ``docs/wiki/reference/datasets/cxr-boxes-masks/rsna_pneumonia.md``.
"""

import json
import os

import pandas as pd

from .._bbox import box_label, clip_boxes
from ..base import BaseHarmonizer


# Label IDs for the "Calculated" group in the adjudicated JSON export.
_CALCULATED_LABEL_IDS = {
    "L_o8w": "Normal",
    "L_yd0": "No Lung Opacity / Not Normal",
    "L_v8n": "Lung Opacity",
}


class RSNAPneumoniaHarmonizer(BaseHarmonizer):
    """Harmonize RSNA Pneumonia (adjudicated JSON export) into RadHarmony format.

    This harmonizer reads JSON, not CSV. ``csv_path`` is the JSON file. The name stays the
    same because :class:`BaseHarmonizer` uses it.

    Args:
        csv_path: Path to the adjudicated JSON file, for example
            ``pneumonia-challenge-annotations-adjudicated-kaggle_2018.json``.
        base_image_dir: Root of the image export with ``<StudyUID>/<SeriesUID>/<SOPUID>.dcm``,
            for example ``~/Downloads/rsna/``.
        label_group: The label group to use. Default ``"Calculated"`` (final consensus).
    """

    LABEL_COLS = [
        "Lung Opacity",
        "No Lung Opacity / Not Normal",
        "Normal",
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
        base_image_dir: str = None,
        label_group: str = "Calculated",
    ):
        super().__init__(csv_path=os.path.expanduser(csv_path) if csv_path else csv_path)
        self.base_image_dir = os.path.expanduser(base_image_dir) if base_image_dir else base_image_dir
        self.label_group = label_group

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["base_image_dir"] = self.base_image_dir
        snap["label_group"] = self.label_group
        return snap

    def _resolve_label_ids(self, raw: dict) -> dict[str, str]:
        """Return ``{label_id: label_name}`` for the chosen label group.

        For ``"Calculated"``, it falls back to ``_CALCULATED_LABEL_IDS`` if the JSON has no
        matching ``labelGroups`` entry.
        """
        for lg in raw.get("labelGroups", []):
            if lg["name"] == self.label_group:
                return {
                    label["id"]: label["name"]
                    for label in lg["labels"]
                }
        if self.label_group == "Calculated":
            return _CALCULATED_LABEL_IDS
        raise ValueError(
            f"Label group '{self.label_group}' not found in JSON. "
            f"Available groups: {[lg['name'] for lg in raw.get('labelGroups', [])]}"
        )

    def _parse_json(self, raw: dict) -> pd.DataFrame:
        """Flatten the JSON into one row per image (``SOPInstanceUID``).

        Columns: the 3 UIDs, one 0/1 column for each label in :attr:`LABEL_COLS`, and
        ``bbox`` (list of normalized boxes) and ``bbox_labels``.
        """
        label_ids = self._resolve_label_ids(raw)
        dataset = raw["datasets"][0]
        annotations = dataset["annotations"]

        group_anns = [
            a for a in annotations
            if a.get("labelId") in label_ids and a.get("SOPInstanceUID")
        ]

        image_data: dict[str, dict] = {}
        for ann in group_anns:
            sop = ann["SOPInstanceUID"]
            if sop not in image_data:
                image_data[sop] = {
                    "SOPInstanceUID": sop,
                    "StudyInstanceUID": ann["StudyInstanceUID"],
                    "SeriesInstanceUID": ann["SeriesInstanceUID"],
                    "Lung Opacity": 0,
                    "No Lung Opacity / Not Normal": 0,
                    "Normal": 0,
                    "bbox": [],
                    "bbox_labels": [],
                }

            label_name = label_ids[ann["labelId"]]
            image_data[sop][label_name] = 1

            # Only the local annotations have a box.
            data = ann.get("data")
            img_h = ann.get("height")
            img_w = ann.get("width")
            if data and img_h and img_w:
                x, y = data["x"], data["y"]
                w, h = data["width"], data["height"]
                # Fractions [dim0_min, dim0_max, dim1_min, dim1_max].
                image_data[sop]["bbox"].append([
                    y / img_h,          # dim0_min (row top)
                    (y + h) / img_h,    # dim0_max (row bottom)
                    x / img_w,          # dim1_min (col left)
                    (x + w) / img_w,    # dim1_max (col right)
                ])
                image_data[sop]["bbox_labels"].append(box_label(label_name))

        return pd.DataFrame(list(image_data.values()))

    def _build_patient_id(self) -> None:
        """Use StudyInstanceUID as the patient ID (one study for each patient)."""
        self.df["patient_id"] = self.df["StudyInstanceUID"].astype(str)

    def _build_study_id(self) -> None:
        """Use StudyInstanceUID as the study ID."""
        self.df["study_id"] = self.df["StudyInstanceUID"].astype(str)

    def _build_image_path(self) -> None:
        """Set ``image_path`` to ``<StudyUID>/<SeriesUID>/<SOPUID>.dcm``."""
        self.df["image_path"] = (
            self.df["StudyInstanceUID"]
            + "/"
            + self.df["SeriesInstanceUID"]
            + "/"
            + self.df["SOPInstanceUID"]
            + ".dcm"
        )

    def _build_labels(self) -> None:
        """Rename the label columns to snake_case.

        :meth:`_parse_json` already made the 0/1 columns. The rename is the same as in
        :meth:`BaseHarmonizer._label_columns_for_output` (``lower().replace(" ", "_")``).
        It keeps the slash in ``no_lung_opacity_/_not_normal``, so that
        ``_select_harmonized_columns`` finds the column.
        """
        for col in self.LABEL_COLS:
            snake = col.lower().replace(" ", "_")
            if col in self.df.columns and snake != col:
                self.df.rename(columns={col: snake}, inplace=True)

    def _read_source(self) -> pd.DataFrame:
        """Read the JSON (not a CSV) and return one row per image with labels and boxes."""
        with open(self.csv_path, "r") as f:
            raw = json.load(f)
        return self._parse_json(raw)

    def _build_bbox(self) -> None:
        # _parse_json() made the boxes. This step only clips them.
        clip_boxes(self.df, "RSNA Pneumonia")  # D-07
