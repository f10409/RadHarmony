"""Harmonizer for the RSNA Pneumonia Detection Challenge dataset.

This dataset uses the **adjudicated JSON annotation export** instead of
standard CSV files.  The JSON contains annotations from multiple annotator
teams; by default the ``Calculated`` label group is used (final consensus
labels).

The Calculated group has three classes:
  - **Normal** (L_o8w, global) — 9,790 images
  - **No Lung Opacity / Not Normal** (L_yd0, global) — 12,788 images
  - **Lung Opacity** (L_v8n, local/bbox) — 11,390 bbox annotations

Bounding boxes are pixel-space ``(x, y, width, height)`` on 1024×1024 images,
normalised to ``[dim0_min, dim0_max, dim1_min, dim1_max]`` fractional coords.

Images are DICOM files in a nested directory structure:
``<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm``
(e.g. ``~/Downloads/rsna/``).
"""

import json
import os
from collections import defaultdict

import pandas as pd

from ..base import BaseHarmonizer


# Label IDs for the "Calculated" group in the adjudicated JSON export.
_CALCULATED_LABEL_IDS = {
    "L_o8w": "Normal",
    "L_yd0": "No Lung Opacity / Not Normal",
    "L_v8n": "Lung Opacity",
}


class RSNAPneumoniaHarmonizer(BaseHarmonizer):
    """Harmonize RSNA Pneumonia (adjudicated JSON export) into RadHarmony format.

    Unlike most harmonizers that read CSVs, this one parses the adjudicated
    JSON directly.  The ``csv_path`` parameter actually points to the JSON
    file — we reuse the name for compatibility with :class:`BaseHarmonizer`.

    Args:
        csv_path: Path to the adjudicated JSON annotation file
            (e.g. ``pneumonia-challenge-annotations-adjudicated-kaggle_2018.json``).
        base_image_dir: Root of the image export containing
            ``<StudyUID>/<SeriesUID>/<SOPUID>.dcm`` (e.g. ``~/Downloads/rsna/``).
        label_group: Name of the label group to use.  Default ``"Calculated"``
            for the final consensus labels.
    """

    LABEL_COLS = [
        "Lung Opacity",
        "No Lung Opacity / Not Normal",
        "Normal",
    ]

    # --- Join / source column configuration ---
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

    # ------------------------------------------------------------------
    # JSON parsing helpers
    # ------------------------------------------------------------------

    def _resolve_label_ids(self, raw: dict) -> dict[str, str]:
        """Build ``{label_id: label_name}`` for the chosen label group.

        Falls back to the hardcoded ``_CALCULATED_LABEL_IDS`` when the
        requested group is ``"Calculated"`` and the JSON doesn't contain
        a matching ``labelGroups`` entry (shouldn't happen, but safe).
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
        """Flatten the adjudicated JSON into one row per unique image (SOPInstanceUID).

        Columns produced:
        - ``SOPInstanceUID``, ``StudyInstanceUID``, ``SeriesInstanceUID``
        - One-hot label columns matching :attr:`LABEL_COLS`
        - ``bbox`` (list of normalised boxes) and ``bbox_labels``
        """
        label_ids = self._resolve_label_ids(raw)
        dataset = raw["datasets"][0]
        annotations = dataset["annotations"]

        # Filter to only annotations from the chosen label group.
        group_anns = [
            a for a in annotations
            if a.get("labelId") in label_ids and a.get("SOPInstanceUID")
        ]

        # Aggregate per image (SOPInstanceUID).
        image_data: dict[str, dict] = {}
        for ann in group_anns:
            sop = ann["SOPInstanceUID"]
            if sop not in image_data:
                image_data[sop] = {
                    "SOPInstanceUID": sop,
                    "StudyInstanceUID": ann["StudyInstanceUID"],
                    "SeriesInstanceUID": ann["SeriesInstanceUID"],
                    # One-hot labels, initialised to 0
                    "Lung Opacity": 0,
                    "No Lung Opacity / Not Normal": 0,
                    "Normal": 0,
                    # Bounding boxes (list of [dim0_min, dim0_max, dim1_min, dim1_max])
                    "bbox": [],
                    "bbox_labels": [],
                }

            label_name = label_ids[ann["labelId"]]
            image_data[sop][label_name] = 1

            # Collect bounding box if present (only for local/bbox annotations).
            data = ann.get("data")
            img_h = ann.get("height")
            img_w = ann.get("width")
            if data and img_h and img_w:
                x, y = data["x"], data["y"]
                w, h = data["width"], data["height"]
                # Normalise to [dim0_min, dim0_max, dim1_min, dim1_max]
                image_data[sop]["bbox"].append([
                    y / img_h,          # dim0_min (row top)
                    (y + h) / img_h,    # dim0_max (row bottom)
                    x / img_w,          # dim1_min (col left)
                    (x + w) / img_w,    # dim1_max (col right)
                ])
                image_data[sop]["bbox_labels"].append(
                    label_name.lower().replace(" ", "_")
                )

        return pd.DataFrame(list(image_data.values()))

    # ------------------------------------------------------------------
    # Build hooks (called by custom harmonize)
    # ------------------------------------------------------------------

    def _build_patient_id(self) -> None:
        """Use StudyInstanceUID as patient identifier (one study per patient)."""
        self.df["patient_id"] = self.df["StudyInstanceUID"].astype(str)

    def _build_study_id(self) -> None:
        """Use StudyInstanceUID as study identifier."""
        self.df["study_id"] = self.df["StudyInstanceUID"].astype(str)

    def _build_image_path(self) -> None:
        """Construct image path from the nested directory structure.

        The image export uses a three-level UID hierarchy::

            <base_image_dir>/
              <StudyInstanceUID>/
                <SeriesInstanceUID>/
                  <SOPInstanceUID>.dcm

        For example::

            ~/Downloads/rsna/1.2.276.0...114974/1.2.276.0...114973/1.2.276.0...114975.dcm
        """
        self.df["image_path"] = (
            self.df["StudyInstanceUID"]
            + "/"
            + self.df["SeriesInstanceUID"]
            + "/"
            + self.df["SOPInstanceUID"]
            + ".dcm"
        )

    def _build_labels(self) -> None:
        """Rename label columns to snake_case.

        Labels are already one-hot encoded during JSON parsing in
        :meth:`_parse_json`, so we only need to rename.

        Uses the same transform as :meth:`BaseHarmonizer._label_columns_for_output`
        (``lower().replace(" ", "_")``) to keep the slash in
        ``no_lung_opacity_/_not_normal`` — this ensures
        ``_select_harmonized_columns`` can find the column.
        """
        for col in self.LABEL_COLS:
            snake = col.lower().replace(" ", "_")
            if col in self.df.columns and snake != col:
                self.df.rename(columns={col: snake}, inplace=True)

    # ------------------------------------------------------------------
    # Main harmonize (custom — reads JSON instead of CSV)
    # ------------------------------------------------------------------

    def harmonize(self, **kwargs) -> pd.DataFrame:
        """Parse adjudicated JSON, build per-image rows with labels and bboxes.

        Overrides the base ``harmonize()`` because the source is JSON,
        not CSV.  Reads the JSON once, flattens annotations from the
        chosen label group, one-hot encodes labels, aggregates bounding
        boxes, and resolves image paths.
        """
        self._harmonized_df_override = None

        # Read the JSON file (reuses csv_path for the file path).
        with open(self.csv_path, "r") as f:
            raw = json.load(f)

        self.df = self._parse_json(raw)

        self._build_patient_id()
        self._build_study_id()
        self._build_image_path()
        self._build_labels()
        # No separate view_position, mask, or report sources.
        self._build_view_position()
        self._build_mask_path()
        self._build_report()

        self.df.dropna(subset=["image_path"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self._select_harmonized_columns(self.df)
