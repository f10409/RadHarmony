"""Harmonizer for the 10 MedSeg dataset1 ``val_im`` slices (no masks, D-25)."""

import os

import pandas as pd

from .medseg_covid_ct import MedSegCOVIDCTHarmonizer, _axial, _load_volume, _window_to_uint8, slice_folder


class MedSegCOVIDCTValHarmonizer(MedSegCOVIDCTHarmonizer):
    """MedSeg Covid Dataset 1 ``val_im.nii.gz``: 10 axial slices, with no labels and no masks.

    The release gives no masks for these slices, so the table has no label
    columns and no ``mask_path`` (D-25). The slice map of dataset1 covers only
    ``tr_im``, so each slice is its own patient, with no age or sex. The slices go to
    ``<slice_output_dir>/dataset1_val_hu<low>_<high>/images/``.

    Args:
        base_dir: The folder that holds ``val_im.nii.gz``.
        slice_output_dir: The folder for the slice PNGs.
        hu_window: Hounsfield window for the 8-bit PNGs.
    """

    LABEL_COLS = []
    SLICE_FOLDER = "dataset1_val"

    def __init__(self, base_dir: str, slice_output_dir: str = None, hu_window=(-1000.0, 400.0)):
        super().__init__(base_dir=base_dir, slice_output_dir=slice_output_dir, subset="dataset1",
                         hu_window=hu_window)

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap.pop("subset", None)
        snap.pop("csv_path", None)
        return snap

    def _materialise(self) -> pd.DataFrame:
        from PIL import Image

        if not self.slice_output_dir:
            raise ValueError("slice_output_dir is required: one PNG per slice is written there.")
        im = os.path.join(self.base_dir, "val_im.nii.gz")
        if not os.path.isfile(im):
            raise FileNotFoundError(f"expected val_im.nii.gz under {self.base_dir}")
        img_d = os.path.join(self.slice_output_dir, slice_folder(self.SLICE_FOLDER, self.hu_window), "images")
        os.makedirs(img_d, exist_ok=True)
        vol = _load_volume(im)
        rows = []
        for k in range(vol.shape[2]):
            stem = f"{self.SLICE_FOLDER}_{k:04d}"
            ip = os.path.join(img_d, stem + ".png")
            if not os.path.isfile(ip):
                Image.fromarray(_window_to_uint8(_axial(vol, k), *self.hu_window)).save(ip)
            rows.append({"stem": stem, "image_path": os.path.basename(ip), "subset": self.subset,
                         "volume": "val", "slice_index": k})
        return pd.DataFrame(rows)
