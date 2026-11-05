"""Harmonizer for the CheXlocalize dataset."""

import json
import os

import numpy as np

from ..base import BaseHarmonizer

# The 10 findings that have masks, in snake case. The mask file calls Lung Opacity "Airspace Opacity".
_MASKED_FINDINGS = ["atelectasis", "cardiomegaly", "consolidation", "edema", "enlarged_cardiomediastinum",
                    "lung_lesion", "lung_opacity", "pleural_effusion", "pneumothorax", "support_devices"]


class CheXlocalizeHarmonizer(BaseHarmonizer):
    """Harmonizer for the CheXlocalize dataset (CheXpert ``test`` split).

    The CSV has no ID columns. The code takes ``patient_id`` and ``study_id``
    from the ``Path`` column. Example: ``test/patient64741/study1/view1_frontal.jpg``
    gives patient_id ``patient64741`` and study_id ``patient64741_study1``.
    This split has no ``-1`` (uncertain) labels.

    ``view_position`` comes from the file name (``frontal`` or ``lateral``).
    The table writes ``FRONTAL`` or ``LATERAL`` (Q-13).

    Segmentation masks are optional. They come from
    ``gt_segmentations_test.json`` (COCO-RLE, keyed by
    ``<study_id>_<image_basename>``). The file has up to 10 masks per image
    (one for each pathology) for 499 of the 668 test images. The mask
    pipeline needs one mask per image, so the code joins them with a bitwise
    OR (SIIM-ACR does the same). Pass ``mask_json_path`` to turn this on.
    With ``None`` the code does not load the JSON (several MB).

    Args:
        csv_path: Path to ``test_labels.csv``.
        mask_json_path: Path to ``gt_segmentations_test.json``. Leave
            ``None`` to omit the ``mask_path`` column entirely.
        base_image_dir: Root of the CheXpert ``test/`` image tree. It is
            the default ``base_image_dir`` of ``preprocess_masks``.
    """

    LABEL_COLS = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "No Finding",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices",
    ]

    LABEL_JOIN_COLS = []
    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = []
    REPORT_PATH_COL = None

    # val_labels.csv has the CheXpert columns Sex and Age (years). test_labels.csv
    # has neither, so the test table has no sex and no age_years (D-06, D-26).
    AGE_SOURCE_COL = "Age"
    AGE_SOURCE_UNIT = "years"
    SEX_SOURCE_COL = "Sex"

    # Prefix of the Path column before the patient folder (the split folder).
    _PATH_PREFIX = r"^test/"

    def __init__(self, csv_path: str, mask_json_path: str = None, base_image_dir: str = None):
        super().__init__(csv_path=csv_path)
        self.mask_json_path = mask_json_path
        self.base_image_dir = base_image_dir
        self._mask_json = None
        if mask_json_path is not None:
            with open(mask_json_path) as f:
                self._mask_json = json.load(f)

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["mask_json_path"] = self.mask_json_path
        snap["base_image_dir"] = self.base_image_dir
        return snap

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["Path"].str.extract(r"(patient\d+)")

    def _build_study_id(self) -> None:
        study_raw = self.df["Path"].str.extract(r"patient\d+/(study\d+)")[0]
        self.df["study_id"] = self.df["patient_id"] + "_" + study_raw

    def _build_image_path(self) -> None:
        # base_image_dir is the "test/" folder, so remove that prefix from
        # paths such as "test/patient64741/study1/view1_frontal.jpg".
        self.df["image_path"] = self.df["Path"].str.replace(self._PATH_PREFIX, "", regex=True)

    def _build_labels(self) -> None:
        for col in self.LABEL_COLS:
            if col in self.df.columns:
                self.df.rename(columns={col: col.lower().replace(" ", "_")}, inplace=True)

    def _build_view_position(self) -> None:
        basename = self.df["image_path"].apply(os.path.basename)
        self.df["view_position"] = basename.apply(
            lambda b: "Lateral" if "lateral" in b.lower() else "Frontal"
        )

    def _build_mask_path(self) -> None:
        if self._mask_json is None:
            return  # no masks requested

        # Column steps, not DataFrame.apply: apply on an empty table gives a table (V-H1-38).
        stems = self.df["image_path"].map(lambda p: os.path.splitext(os.path.basename(p))[0])
        keys = self.df["study_id"].astype(str) + "_" + stems
        # A labeled negative (0 for all 10 masked findings) is annotated with no finding. Its key is not
        # in the mask file, so preprocess_masks writes an all-zero mask for it (D-33, Q-10).
        negative = False
        if all(c in self.df.columns for c in _MASKED_FINDINGS):
            negative = (self.df[_MASKED_FINDINGS] == 0).all(axis=1)
        self.df["mask_path"] = keys.where(keys.isin(self._mask_json.keys()) | negative, None)

    def _decode_mask(self, key, width: int, height: int):
        """Union every per-pathology COCO-RLE mask for one image into a single binary mask."""
        if key is None or self._mask_json is None or key not in self._mask_json:
            return None
        from pycocotools import mask as mask_utils

        combined = np.zeros((height, width), dtype=np.uint8)
        for rle in self._mask_json[key].values():
            decoded = mask_utils.decode(rle).astype(np.uint8)
            combined |= decoded
        return combined

    def preprocess_masks(self, output_dir: str, base_image_dir: str = None, num_cores: int = 1) -> None:
        """Decode + union per-pathology RLE masks into PNGs, defaulting to the test image dir."""
        super().preprocess_masks(
            output_dir=output_dir,
            base_image_dir=base_image_dir or self.base_image_dir,
            num_cores=num_cores,
        )
