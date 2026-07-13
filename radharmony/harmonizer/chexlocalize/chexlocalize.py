"""Harmonizer for the CheXlocalize dataset."""

import json
import os

import numpy as np

from ..base import BaseHarmonizer


class CheXlocalizeHarmonizer(BaseHarmonizer):
    """Harmonizer for the CheXlocalize dataset (CheXpert ``test`` split).

    ``patient_id`` and ``study_id`` are extracted from the ``Path`` column
    (e.g. ``test/patient64741/study1/view1_frontal.jpg`` -> patient_id
    ``patient64741``, study_id ``patient64741_study1``); there is no
    dedicated ID column in the CSV. Labels live in the same CSV and this
    split is fully verified (no ``-1`` uncertainty values).

    ``view_position`` is derived from the image filename (``frontal`` /
    ``lateral``), since there is no dedicated CSV column.

    Segmentation masks (optional) come from CheXlocalize's
    ``gt_segmentations_test.json`` -- COCO-RLE, keyed by
    ``<study_id>_<image_basename>``, with one mask PER PATHOLOGY (up to 10)
    per image, covering 499 of 668 test images. RadHarmony's mask pipeline
    (``BaseHarmonizer.preprocess_masks``) supports only a single combined
    mask per image, so all per-pathology masks for an image are unioned
    (bitwise OR) into one binary "any abnormal region" mask -- the same
    pattern SIIM-ACR uses to union multiple annotator rows. Pass
    ``mask_json_path`` to enable; leave it ``None`` to skip loading the
    (several-MB) JSON when only classification labels are needed.

    Args:
        csv_path: Path to ``test_labels.csv``.
        mask_json_path: Path to ``gt_segmentations_test.json``. Leave
            ``None`` to omit the ``mask_path`` column entirely.
        base_image_dir: Root of the CheXpert ``test/`` image tree. Only
            used as the default ``base_image_dir`` for ``preprocess_masks``.
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
        # Path values look like "test/patient64741/study1/view1_frontal.jpg";
        # base_image_dir points at the "test/" directory, so strip that prefix.
        self.df["image_path"] = self.df["Path"].str.replace(r"^test/", "", regex=True)

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
            return  # mask support not requested

        def _key(row) -> str:
            base = os.path.splitext(os.path.basename(row["image_path"]))[0]
            return f"{row['study_id']}_{base}"

        keys = self.df.apply(_key, axis=1)
        self.df["mask_path"] = keys.where(keys.isin(self._mask_json.keys()), None)

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
